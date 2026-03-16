#!/usr/bin/env python3
"""Analyze GT instance statistics across all CellMap crops at multiple resolutions.

Uses multiprocessing to parallelize across crops.

Output:
  gt_stats/instance_stats.csv   — raw per-instance measurements
  gt_stats/summary.txt          — aggregated statistics
  gt_stats/recommended_config.json — derived config per class × resolution
"""

import os
import csv
import json
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import zarr
from scipy.ndimage import center_of_mass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────

DATA_ROOT = "/nrs/cellmap/data"
INSTANCE_CLASSES = ["nuc", "ves", "endo", "lyso", "ld", "perox", "mito", "mt", "cell"]
TARGET_RESOLUTIONS_NM = [8, 16, 32, 64]
MAX_WORKERS = 64

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gt_stats")


# ── Helpers ────────────────────────────────────────────────────────────────

def get_scale_info(zarr_grp_path):
    """Read multiscale metadata from a zarr group."""
    grp = zarr.open(zarr_grp_path, mode="r")
    attrs = grp.attrs
    if "multiscales" not in attrs:
        return {}
    scales = {}
    for s in attrs["multiscales"][0]["datasets"]:
        path = s["path"]
        res = s["coordinateTransformations"][0]["scale"]
        scales[path] = {"resolution": res}
    return scales


def find_scale_for_resolution(scales, target_nm, max_ratio=2.0):
    """Find the scale level closest to target_nm (matching on Y axis)."""
    best = None
    best_diff = float("inf")
    for path, info in scales.items():
        y_res = info["resolution"][1]
        ratio = max(y_res / target_nm, target_nm / y_res)
        if ratio <= max_ratio:
            diff = abs(y_res - target_nm)
            if diff < best_diff:
                best_diff = diff
                best = (path, y_res)
    return best


# ── Per-crop worker ────────────────────────────────────────────────────────

def process_crop(crop_info):
    """Process a single crop — runs in a worker process.

    Returns list of dicts (one per instance) or empty list on error.
    """
    dataset_name, crop_name, crop_dir, available_classes = crop_info
    results = []

    for cls_name in available_classes:
        cls_path = os.path.join(crop_dir, cls_name)

        try:
            scales = get_scale_info(cls_path)
        except Exception:
            continue

        for target_nm in TARGET_RESOLUTIONS_NM:
            match = find_scale_for_resolution(scales, target_nm)
            if match is None:
                continue

            scale_path, actual_res = match

            try:
                data = zarr.open(os.path.join(cls_path, scale_path), "r")[:]
            except Exception:
                continue

            inst_ids = np.unique(data)
            inst_ids = inst_ids[inst_ids > 0]
            if len(inst_ids) == 0:
                continue

            for inst_id in inst_ids:
                inst_mask = data == inst_id
                vol = int(inst_mask.sum())
                if vol == 0:
                    continue

                where = np.where(inst_mask)
                bd = int(where[0].max() - where[0].min() + 1)
                bh = int(where[1].max() - where[1].min() + 1)
                bw = int(where[2].max() - where[2].min() + 1)

                com = center_of_mass(inst_mask)

                results.append({
                    "dataset": dataset_name,
                    "crop": crop_name,
                    "class": cls_name,
                    "resolution_nm": actual_res,
                    "inst_id": int(inst_id),
                    "volume": vol,
                    "bbox_d": bd, "bbox_h": bh, "bbox_w": bw,
                    "bbox_max": max(bd, bh, bw),
                    "com_z": round(com[0], 1),
                    "com_y": round(com[1], 1),
                    "com_x": round(com[2], 1),
                })

    return results


# ── Discovery ──────────────────────────────────────────────────────────────

def discover_crops(data_root):
    """Find all dataset/crop combinations with instance GT using glob."""
    import glob as globmod

    crops = []
    pattern = os.path.join(data_root, "*", "*.zarr", "recon-1", "labels", "groundtruth", "*")
    crop_dirs = sorted(globmod.glob(pattern))

    for crop_dir in crop_dirs:
        if not os.path.isdir(crop_dir):
            continue

        crop_name = os.path.basename(crop_dir)
        gt_base = os.path.dirname(crop_dir)
        zarr_dir = os.path.dirname(os.path.dirname(os.path.dirname(gt_base)))
        dataset_name = os.path.basename(zarr_dir)

        available_classes = []
        for cls in INSTANCE_CLASSES:
            if os.path.isdir(os.path.join(crop_dir, cls)):
                available_classes.append(cls)

        if available_classes:
            crops.append((dataset_name, crop_name, crop_dir, available_classes))

    return crops


# ── Aggregation & recommendations ──────────────────────────────────────────

def compute_nn_distances(all_stats):
    """Nearest-neighbor COM distance for instances of same class in same crop."""
    groups = defaultdict(list)
    for s in all_stats:
        key = (s["dataset"], s["crop"], s["class"], s["resolution_nm"])
        groups[key].append(s)

    nn_dists = {}
    for key, instances in groups.items():
        if len(instances) < 2:
            continue
        coms = np.array([[s["com_z"], s["com_y"], s["com_x"]] for s in instances])
        dists = []
        for i in range(len(coms)):
            d = np.linalg.norm(coms - coms[i], axis=1)
            d[i] = np.inf
            dists.append(d.min())
        nn_dists[key] = dists

    return nn_dists


def write_results(all_stats):
    """Write CSV, summary, and recommended config."""
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Raw CSV ────────────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, "instance_stats.csv")
    fields = ["dataset", "crop", "class", "resolution_nm", "inst_id",
              "volume", "bbox_d", "bbox_h", "bbox_w", "bbox_max",
              "com_z", "com_y", "com_x"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_stats)
    log.info("Wrote %s (%d rows)", csv_path, len(all_stats))

    # ── NN distances ───────────────────────────────────────────────────
    nn_dists = compute_nn_distances(all_stats)

    # ── Group by class × resolution ────────────────────────────────────
    grouped = defaultdict(list)
    for s in all_stats:
        best_target = min(TARGET_RESOLUTIONS_NM, key=lambda t: abs(t - s["resolution_nm"]))
        grouped[(s["class"], best_target)].append(s)

    nn_grouped = defaultdict(list)
    for key, dists in nn_dists.items():
        _, _, cls_name, res = key
        best_target = min(TARGET_RESOLUTIONS_NM, key=lambda t: abs(t - res))
        nn_grouped[(cls_name, best_target)].extend(dists)

    # ── Summary + recommendations ──────────────────────────────────────
    summary_path = os.path.join(OUT_DIR, "summary.txt")
    reco_path = os.path.join(OUT_DIR, "recommended_config.json")
    recommendations = {}

    with open(summary_path, "w") as f:
        def p(msg=""):
            f.write(msg + "\n")
            print(msg)

        p("=" * 90)
        p("GT Instance Statistics - Per Class x Resolution")
        p("=" * 90)

        for cls_name in INSTANCE_CLASSES:
            p(f"\n{'─' * 90}")
            p(f"  {cls_name.upper()}")
            p(f"{'─' * 90}")

            for target_nm in TARGET_RESOLUTIONS_NM:
                key = (cls_name, target_nm)
                instances = grouped.get(key, [])
                if not instances:
                    continue

                volumes = np.array([s["volume"] for s in instances])
                bbox_maxes = np.array([s["bbox_max"] for s in instances])
                nn = np.array(nn_grouped.get(key, []))

                n_crops = len(set((s["dataset"], s["crop"]) for s in instances))
                p(f"\n  Resolution: ~{target_nm}nm  |  {len(instances)} instances from {n_crops} crops")
                p(f"    Volume (voxels):   min={volumes.min():>8d}  p5={int(np.percentile(volumes, 5)):>8d}  "
                  f"median={int(np.median(volumes)):>8d}  p95={int(np.percentile(volumes, 95)):>8d}  max={volumes.max():>8d}")
                p(f"    Bbox max extent:   min={bbox_maxes.min():>8d}  p5={int(np.percentile(bbox_maxes, 5)):>8d}  "
                  f"median={int(np.median(bbox_maxes)):>8d}  p95={int(np.percentile(bbox_maxes, 95)):>8d}  max={bbox_maxes.max():>8d}")
                if len(nn) > 0:
                    p(f"    NN COM distance:   min={nn.min():>8.1f}  p5={np.percentile(nn, 5):>8.1f}  "
                      f"median={np.median(nn):>8.1f}  p95={np.percentile(nn, 95):>8.1f}  max={nn.max():>8.1f}")
                else:
                    p(f"    NN COM distance:   (only single-instance crops)")

                # Recommendations
                rec_min_size = max(5, int(np.percentile(volumes, 5) * 0.5))
                rec_n_steps = max(20, int(np.percentile(bbox_maxes, 95) * 2))

                if len(nn) > 0:
                    nn_bound = np.percentile(nn, 5) / 2.0
                    rec_conv_radius = round(min(nn_bound, np.median(bbox_maxes) / 3.0), 1)
                    rec_conv_radius = max(1.5, rec_conv_radius)
                else:
                    rec_conv_radius = round(max(1.5, np.median(bbox_maxes) / 3.0), 1)

                p(f"    -> Recommended:  min_size={rec_min_size}  n_steps={rec_n_steps}  convergence_radius={rec_conv_radius}")

                reco_key = f"{cls_name}_{target_nm}nm"
                recommendations[reco_key] = {
                    "class": cls_name,
                    "resolution_nm": target_nm,
                    "n_instances": len(instances),
                    "n_crops": n_crops,
                    "volume_p5": int(np.percentile(volumes, 5)),
                    "volume_median": int(np.median(volumes)),
                    "volume_p95": int(np.percentile(volumes, 95)),
                    "bbox_max_p5": int(np.percentile(bbox_maxes, 5)),
                    "bbox_max_median": int(np.median(bbox_maxes)),
                    "bbox_max_p95": int(np.percentile(bbox_maxes, 95)),
                    "nn_dist_p5": round(float(np.percentile(nn, 5)), 1) if len(nn) > 0 else None,
                    "nn_dist_median": round(float(np.median(nn)), 1) if len(nn) > 0 else None,
                    "recommended": {
                        "min_size": rec_min_size,
                        "n_steps": rec_n_steps,
                        "convergence_radius": rec_conv_radius,
                    },
                }

        p(f"\n{'=' * 90}")

    log.info("Wrote %s", summary_path)

    with open(reco_path, "w") as f:
        json.dump(recommendations, f, indent=2)
    log.info("Wrote %s", reco_path)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    log.info("Discovering crops in %s ...", DATA_ROOT)
    crops = discover_crops(DATA_ROOT)
    log.info("Found %d crops with instance GT", len(crops))

    all_stats = []
    n_done = 0
    n_errors = 0

    log.info("Processing with %d workers...", MAX_WORKERS)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_crop, crop): crop for crop in crops}

        for future in as_completed(futures):
            n_done += 1
            crop = futures[future]
            dataset_name, crop_name, _, _ = crop

            try:
                results = future.result()
                all_stats.extend(results)
                if n_done % 50 == 0:
                    log.info("  [%d/%d] %d instances so far...", n_done, len(crops), len(all_stats))
            except Exception as e:
                log.warning("  FAILED %s/%s: %s", dataset_name, crop_name, e)
                n_errors += 1

    log.info("Done. %d instances from %d crops (%d errors).", len(all_stats), n_done - n_errors, n_errors)

    if all_stats:
        write_results(all_stats)
    else:
        log.error("No instances found!")


if __name__ == "__main__":
    main()

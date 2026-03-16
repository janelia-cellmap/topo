#!/usr/bin/env python3
"""Paper-style comparison: 5 instance segmentation methods on tiled inference.

Methods compared:
1. **Our flow** (topo diffusion) — heat-equation flows + Euler tracking + convergence clustering
2. **Cellpose flow** — cellpose gradient tracking + clustering
3. **Binary CC** — binarize GT, then connected components (baseline)
4. **EDT watershed** — distance transform → seeds → watershed (classical)
5. **Mutex watershed** (Funke lab affinities) — affinity graph → mutex watershed

Benchmark:
- Load GT mito crop (~16nm, crop155, 5 instances)
- Split into overlapping sub-crops
- For each method: generate representation per sub-crop → stitch → postprocess
- Evaluate: instance counts, Variation of Information, angular error (flow methods)

Generates: topo/paper/imgs/compare_methods.png
"""

import os
import time
import numpy as np
import zarr
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from collections import defaultdict

from scipy.ndimage import (
    distance_transform_edt,
    label as nd_label,
    center_of_mass,
)
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

# ── Our flow ─────────────────────────────────────────────────────────────
from topo import (
    generate_diffusion_flows,
    postprocess_single,
    compute_subcrop_slices,
    build_spatial_mask,
    stitch_flows,
    stitch_labels,
    stitch_volumes,
)
from topo.config import get_instance_class_config, get_postprocess_config

# ── Cellpose ─────────────────────────────────────────────────────────────
from cellpose.dynamics import masks_to_flows_gpu_3d, compute_masks

# ── Mutex watershed (Funke lab) ──────────────────────────────────────────
import mwatershed

# ── Data ─────────────────────────────────────────────────────────────────
# DATA_ROOT = "/nrs/cellmap/data/jrc_hela-2/jrc_hela-2.zarr/recon-1/labels/groundtruth/crop155"
# MITO_PATH = f"{DATA_ROOT}/mito/s3"  # ~16nm
MITO_PATH = "/groups/cellmap/cellmap/zouinkhim/OrganelleNet/topo/examples/example.zarr/mito"
OUT_DIR = os.path.join(os.path.dirname(__file__), "imgs")
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Sub-cropping utilities — imported from topo.stitch
# ═══════════════════════════════════════════════════════════════════════════

def compact_relabel(instances):
    """Relabel to contiguous 1..N."""
    unique_ids = np.unique(instances)
    if len(unique_ids) <= 1:
        return np.zeros_like(instances)
    remap = np.zeros(unique_ids.max() + 1, dtype=np.int32)
    for new_id, old_id in enumerate(unique_ids[1:], start=1):
        remap[old_id] = new_id
    return remap[instances]


# ═══════════════════════════════════════════════════════════════════════════
# Method 1: Our diffusion flow
# ═══════════════════════════════════════════════════════════════════════════

def method_our_flow(mito_mask, subcrop_slices, mito_flow_cfg, mito_post_cfg):
    """Our topo diffusion flows + Euler tracking + convergence clustering."""
    D, H, W = mito_mask.shape

    # Full-crop: no spatial mask — volume boundary is real, not a crop edge
    t0 = time.time()
    full_flow = generate_diffusion_flows(
        mito_mask,
        n_iter=mito_flow_cfg.get("diffusion_iters", 200),
        spatial_mask=None,
    )
    full_time = time.time() - t0

    fg = mito_mask > 0
    t0 = time.time()
    full_result = postprocess_single(
        sem_mask=fg, flow=full_flow,
        n_steps=mito_post_cfg["n_steps"],
        step_size=mito_post_cfg["step_size"],
        convergence_radius=mito_post_cfg["convergence_radius"],
        min_size=mito_post_cfg["min_size"],
        group=mito_post_cfg["group"],
    )
    full_post_time = time.time() - t0

    # Per sub-crop → stitch
    subcrop_flows = []
    t0 = time.time()
    for slc in subcrop_slices:
        sub_mask = mito_mask[slc[0], slc[1], slc[2]]
        sub_spatial = build_spatial_mask(sub_mask.shape)
        sub_flow = generate_diffusion_flows(
            sub_mask,
            n_iter=mito_flow_cfg.get("diffusion_iters", 200),
            spatial_mask=sub_spatial,
        )
        subcrop_flows.append(sub_flow)
    stitch_gen_time = time.time() - t0

    t0 = time.time()
    stitched_flow = stitch_flows((D, H, W), subcrop_slices, subcrop_flows)
    stitch_time = time.time() - t0

    t0 = time.time()
    stitched_result = postprocess_single(
        sem_mask=fg, flow=stitched_flow,
        n_steps=mito_post_cfg["n_steps"],
        step_size=mito_post_cfg["step_size"],
        convergence_radius=mito_post_cfg["convergence_radius"],
        min_size=mito_post_cfg["min_size"],
        group=mito_post_cfg["group"],
    )
    stitch_post_time = time.time() - t0

    return {
        "name": "Our Flow\n(topo diffusion)",
        "full_result": full_result,
        "stitched_result": stitched_result,
        "full_flow": full_flow,
        "stitched_flow": stitched_flow,
        "times": {
            "full_gen": full_time, "full_post": full_post_time,
            "stitch_gen": stitch_gen_time, "stitch_blend": stitch_time,
            "stitch_post": stitch_post_time,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Method 2: Cellpose flow
# ═══════════════════════════════════════════════════════════════════════════

def cellpose_flows_from_mask(instance_mask, device=None):
    """Generate cellpose 3D flows from instance mask."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if instance_mask.max() == 0:
        return np.zeros((3,) + instance_mask.shape, dtype=np.float32)
    # Cellpose requires contiguous labels 1..N (find_objects returns None for gaps)
    relabeled = compact_relabel(instance_mask.astype(np.int32)).astype(np.int64)
    try:
        flows = masks_to_flows_gpu_3d(relabeled, device=device)
        return flows.astype(np.float32)
    except Exception:
        # Cellpose can fail on certain small/edge-case masks
        return np.zeros((3,) + instance_mask.shape, dtype=np.float32)


def cellpose_postprocess(flow, fg_mask, niter=200, min_size=15):
    """Cellpose postprocessing: follow flows → compute masks."""
    cellprob = fg_mask.astype(np.float32)
    result = compute_masks(
        flow.astype(np.float32), cellprob,
        do_3D=True, niter=niter, min_size=min_size,
        cellprob_threshold=0.0, flow_threshold=0.8,
    )
    return result.astype(np.int32)


def method_cellpose(mito_mask, subcrop_slices):
    """Cellpose gradient tracking."""
    D, H, W = mito_mask.shape
    fg = mito_mask > 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Full-crop
    t0 = time.time()
    full_flow = cellpose_flows_from_mask(mito_mask, device=device)
    full_time = time.time() - t0

    t0 = time.time()
    full_result = cellpose_postprocess(full_flow, fg)
    full_post_time = time.time() - t0

    # Per sub-crop → stitch flows → postprocess
    subcrop_flows = []
    t0 = time.time()
    for slc in subcrop_slices:
        sub_mask = mito_mask[slc[0], slc[1], slc[2]]
        sub_flow = cellpose_flows_from_mask(sub_mask, device=device)
        subcrop_flows.append(sub_flow)
    stitch_gen_time = time.time() - t0

    t0 = time.time()
    stitched_flow = stitch_flows((D, H, W), subcrop_slices, subcrop_flows)
    stitch_time = time.time() - t0

    t0 = time.time()
    stitched_result = cellpose_postprocess(stitched_flow, fg)
    stitch_post_time = time.time() - t0

    return {
        "name": "Cellpose\nFlow",
        "full_result": full_result,
        "stitched_result": stitched_result,
        "full_flow": full_flow,
        "stitched_flow": stitched_flow,
        "times": {
            "full_gen": full_time, "full_post": full_post_time,
            "stitch_gen": stitch_gen_time, "stitch_blend": stitch_time,
            "stitch_post": stitch_post_time,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Method 3: Binary connected components (baseline)
# ═══════════════════════════════════════════════════════════════════════════

def method_binary_cc(mito_mask, subcrop_slices):
    """Binarize → connected components. Simplest baseline."""
    D, H, W = mito_mask.shape
    fg = mito_mask > 0

    # Full-crop
    t0 = time.time()
    full_result, _ = nd_label(fg)
    full_result = full_result.astype(np.int32)
    full_time = time.time() - t0

    # Per sub-crop → stitch labels
    subcrop_labels = []
    t0 = time.time()
    for slc in subcrop_slices:
        sub_fg = fg[slc[0], slc[1], slc[2]]
        sub_labels, _ = nd_label(sub_fg)
        subcrop_labels.append(sub_labels.astype(np.int32))
    stitch_gen_time = time.time() - t0

    t0 = time.time()
    stitched_result = compact_relabel(stitch_labels((D, H, W), subcrop_slices, subcrop_labels))
    stitch_time = time.time() - t0

    return {
        "name": "Binary\nConn. Comp.",
        "full_result": full_result,
        "stitched_result": stitched_result,
        "full_flow": None,
        "stitched_flow": None,
        "times": {
            "full_gen": full_time, "full_post": 0.0,
            "stitch_gen": stitch_gen_time, "stitch_blend": stitch_time,
            "stitch_post": 0.0,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Method 4: EDT watershed
# ═══════════════════════════════════════════════════════════════════════════

def edt_watershed(fg_mask, min_distance=5, min_size=50):
    """EDT → local maxima seeds → watershed."""
    if not fg_mask.any():
        return np.zeros_like(fg_mask, dtype=np.int32)

    edt = distance_transform_edt(fg_mask).astype(np.float32)
    coords = peak_local_max(edt, min_distance=min_distance, labels=fg_mask)

    if len(coords) == 0:
        labels, _ = nd_label(fg_mask)
        return labels.astype(np.int32)

    markers = np.zeros_like(fg_mask, dtype=np.int32)
    for i, c in enumerate(coords):
        markers[c[0], c[1], c[2]] = i + 1

    ws = watershed(-edt, markers, mask=fg_mask)

    # Filter small
    for uid in np.unique(ws):
        if uid == 0:
            continue
        if (ws == uid).sum() < min_size:
            ws[ws == uid] = 0

    return compact_relabel(ws.astype(np.int32))


def method_edt_watershed(mito_mask, subcrop_slices, min_distance=8, min_size=240):
    """Classical EDT watershed."""
    D, H, W = mito_mask.shape
    fg = mito_mask > 0

    # Full-crop
    t0 = time.time()
    full_result = edt_watershed(fg, min_distance=min_distance, min_size=min_size)
    full_time = time.time() - t0

    # Per sub-crop → stitch labels
    subcrop_labels = []
    t0 = time.time()
    for slc in subcrop_slices:
        sub_fg = fg[slc[0], slc[1], slc[2]]
        sub_labels = edt_watershed(sub_fg, min_distance=min_distance, min_size=min_size)
        subcrop_labels.append(sub_labels)
    stitch_gen_time = time.time() - t0

    t0 = time.time()
    stitched_result = compact_relabel(stitch_labels((D, H, W), subcrop_slices, subcrop_labels))
    stitch_time = time.time() - t0

    return {
        "name": "EDT\nWatershed",
        "full_result": full_result,
        "stitched_result": stitched_result,
        "full_flow": None,
        "stitched_flow": None,
        "times": {
            "full_gen": full_time, "full_post": 0.0,
            "stitch_gen": stitch_gen_time, "stitch_blend": stitch_time,
            "stitch_post": 0.0,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Method 5: Mutex watershed (Funke lab affinities)
# ═══════════════════════════════════════════════════════════════════════════

# Standard 3D offsets: short-range (direct neighbors) + long-range
AFFINITY_OFFSETS = [
    # Direct neighbors (attractive)
    [1, 0, 0], [0, 1, 0], [0, 0, 1],
    # Medium range
    [3, 0, 0], [0, 3, 0], [0, 0, 3],
    # Long range (repulsive in mutex watershed)
    [9, 0, 0], [0, 9, 0], [0, 0, 9],
    [18, 0, 0], [0, 18, 0], [0, 0, 18],
]


def compute_affinities(instance_mask, offsets):
    """Compute GT affinities: 1 if source and target share same instance."""
    D, H, W = instance_mask.shape
    affinities = np.zeros((len(offsets), D, H, W), dtype=np.float64)

    for i, (dz, dy, dx) in enumerate(offsets):
        sz = slice(max(0, -dz), D - max(0, dz))
        sy = slice(max(0, -dy), H - max(0, dy))
        sx = slice(max(0, -dx), W - max(0, dx))
        tz = slice(max(0, dz), D - max(0, -dz))
        ty = slice(max(0, dy), H - max(0, -dy))
        tx = slice(max(0, dx), W - max(0, -dx))

        source = instance_mask[sz, sy, sx]
        target = instance_mask[tz, ty, tx]
        same = (source == target) & (source > 0)
        affinities[i, sz, sy, sx] = same.astype(np.float64)

    return affinities


def mutex_watershed_from_affinities(affinities, offsets, fg_mask):
    """Run mutex watershed on affinity volume."""
    # Shift affinities to [-0.5, 0.5] range (mwatershed convention)
    result = mwatershed.agglom(affinities - 0.5, offsets)
    # Mask to foreground
    result[~fg_mask] = 0
    return compact_relabel(result.astype(np.int32))


def method_affinities(mito_mask, subcrop_slices, min_size=240):
    """Funke lab affinity prediction → mutex watershed."""
    D, H, W = mito_mask.shape
    fg = mito_mask > 0
    offsets = AFFINITY_OFFSETS

    # Full-crop
    t0 = time.time()
    full_aff = compute_affinities(mito_mask, offsets)
    full_gen_time = time.time() - t0

    t0 = time.time()
    full_result = mutex_watershed_from_affinities(full_aff, offsets, fg)
    # Filter small
    for uid in np.unique(full_result):
        if uid == 0:
            continue
        if (full_result == uid).sum() < min_size:
            full_result[full_result == uid] = 0
    full_result = compact_relabel(full_result)
    full_post_time = time.time() - t0

    # Per sub-crop → stitch affinities → mutex watershed
    subcrop_affs = []
    t0 = time.time()
    for slc in subcrop_slices:
        sub_mask = mito_mask[slc[0], slc[1], slc[2]]
        sub_aff = compute_affinities(sub_mask, offsets)
        subcrop_affs.append(sub_aff)
    stitch_gen_time = time.time() - t0

    t0 = time.time()
    stitched_aff = stitch_volumes((D, H, W), subcrop_slices, subcrop_affs, n_channels=len(offsets))
    stitch_blend_time = time.time() - t0

    t0 = time.time()
    stitched_result = mutex_watershed_from_affinities(stitched_aff, offsets, fg)
    for uid in np.unique(stitched_result):
        if uid == 0:
            continue
        if (stitched_result == uid).sum() < min_size:
            stitched_result[stitched_result == uid] = 0
    stitched_result = compact_relabel(stitched_result)
    stitch_post_time = time.time() - t0

    return {
        "name": "Mutex WS\n(affinities)",
        "full_result": full_result,
        "stitched_result": stitched_result,
        "full_flow": None,
        "stitched_flow": None,
        "times": {
            "full_gen": full_gen_time, "full_post": full_post_time,
            "stitch_gen": stitch_gen_time, "stitch_blend": stitch_blend_time,
            "stitch_post": stitch_post_time,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation metrics
# ═══════════════════════════════════════════════════════════════════════════

def variation_of_information(gt, pred):
    """Compute Variation of Information (VI = VI_split + VI_merge).

    Lower is better. VI=0 means perfect match.
    Returns (vi_split, vi_merge, vi_total).
    """
    # Only evaluate on foreground
    fg = gt > 0
    gt_fg = gt[fg]
    pred_fg = pred[fg]
    n = len(gt_fg)
    if n == 0:
        return 0.0, 0.0, 0.0

    # Contingency table
    gt_ids = np.unique(gt_fg)
    pred_ids = np.unique(pred_fg)

    gt_map = {gid: i for i, gid in enumerate(gt_ids)}
    pred_map = {pid: i for i, pid in enumerate(pred_ids)}

    contingency = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
    for gv, pv in zip(gt_fg, pred_fg):
        contingency[gt_map[gv], pred_map[pv]] += 1

    contingency /= n

    # Marginals
    p_gt = contingency.sum(axis=1)
    p_pred = contingency.sum(axis=0)

    # Entropy
    h_gt = -np.sum(p_gt[p_gt > 0] * np.log2(p_gt[p_gt > 0]))
    h_pred = -np.sum(p_pred[p_pred > 0] * np.log2(p_pred[p_pred > 0]))

    # Mutual information
    mi = 0.0
    for i in range(len(gt_ids)):
        for j in range(len(pred_ids)):
            if contingency[i, j] > 0:
                mi += contingency[i, j] * np.log2(
                    contingency[i, j] / (p_gt[i] * p_pred[j])
                )

    vi_split = h_gt - mi   # GT entropy not explained by pred (under-segmentation)
    vi_merge = h_pred - mi  # Pred entropy not explained by GT (over-segmentation)

    return vi_split, vi_merge, vi_split + vi_merge


def count_instances(labels):
    """Count non-zero unique instance IDs."""
    return len(np.unique(labels[labels > 0]))


def angular_error_on_fg(flow_a, flow_b, fg_mask):
    """Mean angular error (degrees) between two flow fields on foreground."""
    if flow_a is None or flow_b is None:
        return float("nan")
    dot = np.clip((flow_a * flow_b).sum(axis=0), -1, 1)
    angle = np.degrees(np.arccos(dot))
    return float(angle[fg_mask].mean())


# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

def _random_cmap(n_labels):
    """Generate a random colormap for instance labels."""
    np.random.seed(42)
    colors = np.random.rand(n_labels + 1, 3)
    colors[0] = 0  # background = black
    return colors


def _label_to_rgb(labels, max_id=None):
    """Convert label volume to RGB using random colors."""
    if max_id is None:
        max_id = labels.max()
    colors = _random_cmap(max_id)
    rgb = colors[np.clip(labels, 0, max_id)]
    return rgb


def _flow_to_hsv_2d(fy, fx, fg_mask):
    """Convert 2D flow to HSV RGB image."""
    angle = np.arctan2(fy, fx)
    hue = (angle + np.pi) / (2 * np.pi)
    mag = np.sqrt(fy**2 + fx**2)
    mag_norm = mag / (mag.max() + 1e-8)

    H, W = fy.shape
    hsv = np.zeros((H, W, 3), dtype=np.float32)
    hsv[..., 0] = hue
    hsv[..., 1] = 1.0
    hsv[..., 2] = mag_norm
    rgb = hsv_to_rgb(hsv)
    rgb[~fg_mask] = 0.0
    return rgb


def _slice_volume(vol, axis, idx):
    """Take a 2D slice from a 3D volume along the given axis."""
    if axis == 0:
        return vol[idx]              # Z-slice → YX plane
    elif axis == 1:
        return vol[:, idx, :]        # Y-slice → ZX plane
    else:
        return vol[:, :, idx]        # X-slice → ZY plane


def _slice_flow_2d(flow_4d, axis, idx):
    """Extract 2D flow components for HSV visualization.

    For a Z-slice (axis=0): show (fy, fx) = flow[1], flow[2]
    For a Y-slice (axis=1): show (fz, fx) = flow[0], flow[2]
    For a X-slice (axis=2): show (fz, fy) = flow[0], flow[1]
    """
    if flow_4d is None:
        return None, None
    if axis == 0:
        return flow_4d[1, idx], flow_4d[2, idx]
    elif axis == 1:
        return flow_4d[0, :, idx, :], flow_4d[2, :, idx, :]
    else:
        return flow_4d[0, :, :, idx], flow_4d[1, :, :, idx]


AXIS_NAMES = {0: "Z", 1: "Y", 2: "X"}
PLANE_NAMES = {0: "YX", 1: "ZX", 2: "ZY"}


def plot_comparison(gt_mask, results, axis, idx, out_path):
    """Generate paper-quality comparison figure for one slice.

    Args:
        gt_mask: [D, H, W] instance labels.
        results: list of method result dicts.
        axis: 0=Z, 1=Y, 2=X — which axis to slice along.
        idx: slice index along that axis.
        out_path: output PNG path.
    """
    n_methods = len(results)
    fig, axes = plt.subplots(3, n_methods + 1, figsize=(4 * (n_methods + 1), 12))

    fg = gt_mask > 0
    fg_2d = _slice_volume(fg, axis, idx)
    gt_2d = _slice_volume(gt_mask, axis, idx)
    max_gt = gt_mask.max()

    axis_name = AXIS_NAMES[axis]
    plane_name = PLANE_NAMES[axis]

    # ── Column 0: Ground truth ──
    gt_rgb = _label_to_rgb(gt_2d, max_id=max_gt)
    for row in range(3):
        ax = axes[row, 0]
        ax.imshow(gt_rgb, interpolation="nearest")
        if row == 0:
            ax.set_title(f"Ground Truth\n({count_instances(gt_mask)} inst.)",
                         fontsize=11, fontweight="bold")
        ax.axis("off")

    axes[0, 0].set_ylabel("Full crop", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Stitched", fontsize=11, fontweight="bold")
    axes[2, 0].set_ylabel("Flow / Repr.", fontsize=11, fontweight="bold")

    # ── Columns 1..N: Methods ──
    all_metrics = []
    for col, res in enumerate(results, start=1):
        full_2d = _slice_volume(res["full_result"], axis, idx)
        stitch_2d = _slice_volume(res["stitched_result"], axis, idx)

        # Row 0: Full-crop result
        ax = axes[0, col]
        ax.imshow(_label_to_rgb(full_2d), interpolation="nearest")
        n_full = count_instances(res["full_result"])
        _, _, vi_t = variation_of_information(gt_mask, res["full_result"])
        ax.set_title(f"{res['name']}\nFull: {n_full} inst. | VI={vi_t:.2f}",
                     fontsize=10)
        ax.axis("off")

        # Row 1: Stitched result
        ax = axes[1, col]
        ax.imshow(_label_to_rgb(stitch_2d), interpolation="nearest")
        n_stitch = count_instances(res["stitched_result"])
        _, _, vi_t2 = variation_of_information(gt_mask, res["stitched_result"])
        ax.set_title(f"Stitched: {n_stitch} inst. | VI={vi_t2:.2f}", fontsize=10)
        ax.axis("off")

        # Row 2: Flow HSV or representation
        ax = axes[2, col]
        fy, fx = _slice_flow_2d(res.get("stitched_flow"), axis, idx)
        if fy is not None:
            rgb = _flow_to_hsv_2d(fy, fx, fg_2d)
            ax.imshow(rgb, interpolation="nearest")
            ang_err = angular_error_on_fg(
                res.get("full_flow"), res.get("stitched_flow"), fg
            )
            ax.set_title(f"Stitched flow (HSV)\nAng. err: {ang_err:.1f}°", fontsize=10)
        else:
            ax.imshow(_label_to_rgb(full_2d), interpolation="nearest", alpha=0.5)
            ax.imshow(fg_2d.astype(float), cmap="gray", alpha=0.3, interpolation="nearest")
            ax.set_title("(no flow field)", fontsize=10, fontstyle="italic")
        ax.axis("off")

        # Collect metrics
        total_time_full = sum(v for k, v in res["times"].items() if "full" in k)
        total_time_stitch = sum(v for k, v in res["times"].items() if "stitch" in k)
        all_metrics.append({
            "name": res["name"].replace("\n", " "),
            "n_full": n_full,
            "n_stitch": n_stitch,
            "vi_full": vi_t,
            "vi_stitch": vi_t2,
            "time_full": total_time_full,
            "time_stitch": total_time_stitch,
        })

    fig.suptitle(
        "Instance Segmentation Methods — Full Crop vs. Tiled/Stitched\n"
        f"Mito (crop155, 100³ @ 16nm, GT={count_instances(gt_mask)} inst.)  "
        f"— {plane_name} plane, {axis_name}={idx}",
        fontsize=14, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    return all_metrics


def plot_metrics_table(gt_mask, all_metrics, out_path):
    """Plot a summary metrics table as a figure."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    n_gt = count_instances(gt_mask)
    names = [m["name"] for m in all_metrics]
    x = np.arange(len(names))

    # Instance count comparison
    ax = axes[0]
    width = 0.35
    bars1 = ax.bar(x - width/2, [m["n_full"] for m in all_metrics], width,
                   label="Full crop", color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x + width/2, [m["n_stitch"] for m in all_metrics], width,
                   label="Stitched", color="#DD8452", alpha=0.85)
    ax.axhline(y=n_gt, color="red", linestyle="--", linewidth=1.5, label=f"GT ({n_gt})")
    ax.set_ylabel("Instance count")
    ax.set_title("Instance Count", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=20, ha="right")
    ax.legend(fontsize=8)

    # VI comparison
    ax = axes[1]
    bars1 = ax.bar(x - width/2, [m["vi_full"] for m in all_metrics], width,
                   label="Full crop", color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x + width/2, [m["vi_stitch"] for m in all_metrics], width,
                   label="Stitched", color="#DD8452", alpha=0.85)
    ax.set_ylabel("Variation of Information")
    ax.set_title("VI (lower = better)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=20, ha="right")
    ax.legend(fontsize=8)

    # Time comparison
    ax = axes[2]
    bars1 = ax.bar(x - width/2, [m["time_full"] for m in all_metrics], width,
                   label="Full crop", color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x + width/2, [m["time_stitch"] for m in all_metrics], width,
                   label="Stitched", color="#DD8452", alpha=0.85)
    ax.set_ylabel("Time (s)")
    ax.set_title("Compute Time", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=20, ha="right")
    ax.legend(fontsize=8)

    fig.suptitle("Quantitative Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Paper Comparison: 5 Instance Segmentation Methods")
    print("=" * 70)

    # ── Load data ──
    print("\n[1] Loading mito (~16nm, crop155)...")
    mito_mask = zarr.open(MITO_PATH, "r")[:]
    D, H, W = mito_mask.shape
    print(f"  Shape: {D}x{H}x{W}")
    print(f"  GT instances: {np.unique(mito_mask[mito_mask > 0])}")

    mito_flow_cfg = get_instance_class_config(16)["mito"]
    mito_post_cfg = get_postprocess_config(16)["mito"]

    # ── Sub-crop setup ──
    crop_size = (D // 2 + 4, H // 2 + 4, W // 2 + 4)
    crop_size = tuple(min(cs, vs) for cs, vs in zip(crop_size, (D, H, W)))
    overlap = tuple(cs // 3 for cs in crop_size)
    subcrop_slices = compute_subcrop_slices((D, H, W), crop_size, overlap)
    print(f"\n[2] Sub-crop setup: {len(subcrop_slices)} tiles, "
          f"size={crop_size}, overlap={overlap}")

    # ── Run all methods ──
    results = []

    print("\n[3] Method 1: Our diffusion flow...")
    results.append(method_our_flow(mito_mask, subcrop_slices, mito_flow_cfg, mito_post_cfg))
    print(f"  Full: {count_instances(results[-1]['full_result'])} inst, "
          f"Stitched: {count_instances(results[-1]['stitched_result'])} inst")

    print("\n[4] Method 2: Cellpose flow...")
    results.append(method_cellpose(mito_mask, subcrop_slices))
    print(f"  Full: {count_instances(results[-1]['full_result'])} inst, "
          f"Stitched: {count_instances(results[-1]['stitched_result'])} inst")

    print("\n[5] Method 3: Binary connected components...")
    results.append(method_binary_cc(mito_mask, subcrop_slices))
    print(f"  Full: {count_instances(results[-1]['full_result'])} inst, "
          f"Stitched: {count_instances(results[-1]['stitched_result'])} inst")

    print("\n[6] Method 4: EDT watershed...")
    results.append(method_edt_watershed(mito_mask, subcrop_slices))
    print(f"  Full: {count_instances(results[-1]['full_result'])} inst, "
          f"Stitched: {count_instances(results[-1]['stitched_result'])} inst")

    print("\n[7] Method 5: Mutex watershed (affinities)...")
    results.append(method_affinities(mito_mask, subcrop_slices))
    print(f"  Full: {count_instances(results[-1]['full_result'])} inst, "
          f"Stitched: {count_instances(results[-1]['stitched_result'])} inst")

    # ── Evaluation ──
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    gt_n = count_instances(mito_mask)
    print(f"\n{'Method':<25} {'Full':<8} {'Stitch':<8} "
          f"{'VI_full':<10} {'VI_stitch':<10} "
          f"{'T_full(s)':<10} {'T_stitch(s)':<12}")
    print("-" * 90)

    for res in results:
        name = res["name"].replace("\n", " ")
        n_full = count_instances(res["full_result"])
        n_stitch = count_instances(res["stitched_result"])
        _, _, vi_full = variation_of_information(mito_mask, res["full_result"])
        _, _, vi_stitch = variation_of_information(mito_mask, res["stitched_result"])
        t_full = sum(v for k, v in res["times"].items() if "full" in k)
        t_stitch = sum(v for k, v in res["times"].items() if "stitch" in k)
        print(f"{name:<25} {n_full:<8} {n_stitch:<8} "
              f"{vi_full:<10.3f} {vi_stitch:<10.3f} "
              f"{t_full:<10.2f} {t_stitch:<12.2f}")

    print(f"\n(GT: {gt_n} instances)")

    # ── Plots: 30 views across Z, Y, X planes ──
    print("\n[8] Generating 30 comparison figures (different planes & slices)...")

    # 10 slices per axis, evenly spaced through the volume
    views = []
    for axis, size, name in [(0, D, "z"), (1, H, "y"), (2, W, "x")]:
        for j in range(10):
            idx = int(size * (j + 1) / 11)  # evenly spaced, skip edges
            views.append((axis, idx, f"{name}{idx}"))

    # Extra requested slices
    views.append((1, 67, "y67"))   # [:, 67, :] — XZ plane at Y=67

    for i, (axis, idx, label) in enumerate(views):
        out_path = os.path.join(OUT_DIR, f"compare_methods_{i+1:02d}_{label}.png")
        plot_comparison(mito_mask, results, axis, idx, out_path)

    # Also save the metrics bar chart
    all_metrics = plot_comparison(
        mito_mask, results, 0, D // 2,
        os.path.join(OUT_DIR, "compare_methods.png"),
    )
    plot_metrics_table(
        mito_mask, all_metrics,
        os.path.join(OUT_DIR, "compare_metrics.png"),
    )

    print("\nDone. Generated 30 slice views + 1 summary + 1 metrics chart.")


if __name__ == "__main__":
    main()

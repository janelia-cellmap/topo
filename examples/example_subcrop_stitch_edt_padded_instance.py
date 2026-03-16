#!/usr/bin/env python3
"""Simulate tiled inference with per-instance open-boundary EDT flows.

Same pipeline as example_subcrop_stitch.py but uses direct flows
(per-instance EDT with open-boundary padding) instead of diffusion flows.

1. Load a GT crop
2. Split into overlapping sub-crops
3. Generate per-instance direct flows per sub-crop (open-boundary EDT)
4. Stitch flows back together (cosine-weighted blending in overlaps)
5. Postprocess the stitched flow field
6. Compare: GT vs full-crop flow vs stitched flow vs postprocessed

Visualize all intermediate results in Neuroglancer.
"""

import numpy as np
import zarr
import neuroglancer
from scipy.ndimage import distance_transform_edt, label as ndlabel
from skimage.segmentation import watershed

from topo import (
    generate_direct_flows,
    postprocess_single,
    compute_subcrop_slices,
    build_spatial_mask,
    cosine_blend_weight,
    stitch_flows,
)
from topo.stitch import _boundary_flags
from topo.config import get_postprocess_config

# ── Data ──────────────────────────────────────────────────────────────────
MITO_PATH = "/groups/cellmap/cellmap/zouinkhim/OrganelleNet/topo/examples/example.zarr/mito"


def view_in_neuroglancer(**kwargs):
    """Launch Neuroglancer viewer with provided volumes as layers."""
    neuroglancer.set_server_bind_address("0.0.0.0")
    viewer = neuroglancer.Viewer()

    with viewer.txn() as s:
        for name, array in kwargs.items():
            if (
                array.dtype in (float, np.float32, np.float64)
                or "raw" in name
                or "__img" in name
            ):
                s.layers[name] = neuroglancer.ImageLayer(
                    source=neuroglancer.LocalVolume(data=array),
                )
            else:
                s.layers[name] = neuroglancer.SegmentationLayer(
                    source=neuroglancer.LocalVolume(data=array),
                )

    print(f"\nNeuroglancer URL: {viewer.get_viewer_url()}")
    return viewer


# ── Sub-cropping utilities (imported from topo.stitch) ────────────────────


def compute_per_instance_edt(instance_mask, spatial_mask):
    """Compute per-instance EDT volume (open-boundary padding).

    Returns [D,H,W] float32 where each instance's voxels contain
    that instance's open-boundary EDT values.
    """
    D, H, W = instance_mask.shape
    edt_vol = np.zeros((D, H, W), dtype=np.float32)

    inst_ids = np.unique(instance_mask)
    inst_ids = inst_ids[inst_ids > 0]

    for iid in inst_ids:
        mask = instance_mask == iid
        if not mask.any():
            continue

        # Open-boundary EDT: pad with 1 where instance exits crop
        padded = np.pad(mask.astype(np.uint8), 1, mode='constant', constant_values=0)
        sp_padded = np.pad((spatial_mask > 0.5).astype(np.uint8), 1,
                           mode='constant', constant_values=0)

        for axis in range(3):
            for side in [0, -1]:
                face_slc = [slice(1, -1)] * 3
                face_slc[axis] = 1 if side == 0 else padded.shape[axis] - 2
                face_mask = padded[tuple(face_slc)]
                face_spatial = sp_padded[tuple(face_slc)]

                exits = (face_mask > 0) & (face_spatial == 0)
                if exits.any():
                    pad_slc = [slice(1, -1)] * 3
                    pad_slc[axis] = 0 if side == 0 else padded.shape[axis] - 1
                    padded[tuple(pad_slc)] = np.where(exits, 1, padded[tuple(pad_slc)])

        dist = distance_transform_edt(padded)
        dist = dist[1:-1, 1:-1, 1:-1]
        edt_vol[mask] = dist[mask]

    return edt_vol


def saturate_edt(edt, sigma=5.0):
    """Saturate EDT: 0 at boundary, rises to ~1 within sigma pixels.

    Formula: 1 - exp(-edt / sigma)
    """
    return (1.0 - np.exp(-edt / sigma)).astype(np.float32)


# ── Watershed on saturated EDT ────────────────────────────────────────────

def watershed_on_edt(sat_edt, fg_mask, seed_threshold=0.5, min_size=50):
    """Marker-controlled watershed on the saturated EDT.

    1. Find seeds: connected components where sat_edt > seed_threshold
    2. Watershed on -sat_edt using those seeds as markers
    3. Mask to foreground, filter small instances

    Returns instance labels (uint32).
    """
    # Seeds: regions deep inside instances (high sat_edt)
    seed_mask = (sat_edt > seed_threshold) & fg_mask
    markers, n_seeds = ndlabel(seed_mask)
    print(f"    Found {n_seeds} seeds (threshold={seed_threshold})")

    if n_seeds == 0:
        return np.zeros_like(fg_mask, dtype=np.uint32)

    # Watershed on inverted EDT (valleys become ridges)
    # Only flood within foreground
    result = watershed(-sat_edt, markers=markers, mask=fg_mask)

    # Filter small instances
    for uid in np.unique(result):
        if uid == 0:
            continue
        if (result == uid).sum() < min_size:
            result[result == uid] = 0

    # Compact relabel
    final = np.zeros_like(result, dtype=np.uint32)
    for new_id, uid in enumerate(np.unique(result[result > 0]), start=1):
        final[result == uid] = new_id

    return final


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Sub-crop stitching — Per-instance open-boundary EDT flows")
    print("=" * 60)

    # ── 1. Load full crop ─────────────────────────────────────────────
    print("\n[1/6] Loading mito...")
    mito_mask = zarr.open(MITO_PATH, "r")[:]
    D, H, W = mito_mask.shape
    print(f"  Shape: {D}x{H}x{W}")
    print(f"  Instances: {np.unique(mito_mask[mito_mask > 0])}")

    mito_post_cfg = get_postprocess_config(16)["mito"]

    # ── 2. Generate full-crop direct flows (reference) ────────────────
    # No spatial mask for the full volume — the volume boundary is real,
    # not a crop edge, so use Dirichlet BC to keep flow inward.
    print("\n[2/6] Generating full-crop direct flows (per-instance EDT, reference)...")
    full_flows = generate_direct_flows(mito_mask, spatial_mask=None)
    full_edt = compute_per_instance_edt(mito_mask, np.ones(mito_mask.shape, dtype=np.float32))
    full_sat_edt = saturate_edt(full_edt, sigma=5.0)
    print(f"  Flow shape: {full_flows.shape}")

    # ── 3. Split into sub-crops ───────────────────────────────────────
    crop_size = (D // 2 + 4, H // 2 + 4, W // 2 + 4)
    crop_size = tuple(min(cs, vs) for cs, vs in zip(crop_size, (D, H, W)))
    overlap = tuple(cs // 3 for cs in crop_size)

    subcrop_slices = compute_subcrop_slices((D, H, W), crop_size, overlap)
    n_subcrops = len(subcrop_slices)
    print(f"\n[3/6] Splitting into {n_subcrops} sub-crops...")
    print(f"  Crop size: {crop_size}, overlap: {overlap}")

    subcrop_map = np.zeros((D, H, W), dtype=np.uint32)
    for i, slc in enumerate(subcrop_slices):
        subcrop_map[slc[0], slc[1], slc[2]] = i + 1

    # ── 4. Generate per-instance direct flows per sub-crop ────────────
    print("\n[4/6] Generating per-instance direct flows per sub-crop...")
    subcrop_flows = []
    subcrop_edts = []
    for i, slc in enumerate(subcrop_slices):
        sub_mask = mito_mask[slc[0], slc[1], slc[2]]
        sub_spatial = build_spatial_mask(sub_mask.shape)

        sub_flow = generate_direct_flows(sub_mask, spatial_mask=sub_spatial)
        sub_edt = compute_per_instance_edt(sub_mask, sub_spatial)
        subcrop_flows.append(sub_flow)
        subcrop_edts.append(sub_edt)

        n_inst = len(np.unique(sub_mask[sub_mask > 0]))
        print(f"  Sub-crop {i+1}/{n_subcrops}: shape={sub_mask.shape}, "
              f"instances={n_inst}")

    # ── 5. Stitch flows ──────────────────────────────────────────────
    print("\n[5/6] Stitching sub-crop flows (cosine blending)...")
    stitched_flows = stitch_flows((D, H, W), subcrop_slices, subcrop_flows)

    # Also stitch EDT for visualization
    edt_sum = np.zeros((D, H, W), dtype=np.float64)
    edt_weight = np.zeros((D, H, W), dtype=np.float64)
    for slc, edt in zip(subcrop_slices, subcrop_edts):
        w = cosine_blend_weight(edt.shape, at_volume_boundary=_boundary_flags(slc, (D, H, W)))
        edt_sum[slc[0], slc[1], slc[2]] += edt * w
        edt_weight[slc[0], slc[1], slc[2]] += w
    edt_weight = np.clip(edt_weight, 1e-8, None)
    stitched_edt = (edt_sum / edt_weight).astype(np.float32)
    stitched_sat_edt = saturate_edt(stitched_edt, sigma=5.0)

    # Compare with full-crop flow
    fg = mito_mask > 0
    if fg.any():
        dot = np.clip((full_flows * stitched_flows).sum(axis=0), -1, 1)
        angle_err = np.degrees(np.arccos(dot))
        angle_err_fg = angle_err[fg]
        print(f"  Angular error (full vs stitched) on FG:")
        print(f"    Mean: {angle_err_fg.mean():.1f}deg, "
              f"Median: {np.median(angle_err_fg):.1f}deg, "
              f"P95: {np.percentile(angle_err_fg, 95):.1f}deg")

    # ── 6. Postprocess ───────────────────────────────────────────────
    print("\n[6/6] Postprocessing...")

    post_full = postprocess_single(
        sem_mask=fg,
        flow=full_flows,
        n_steps=mito_post_cfg["n_steps"],
        step_size=mito_post_cfg["step_size"],
        convergence_radius=mito_post_cfg["convergence_radius"],
        min_size=mito_post_cfg["min_size"],
        group=mito_post_cfg["group"],
    )
    print(f"  Full-crop:  GT={mito_mask.max()} instances, "
          f"recovered={post_full.max()}")

    post_stitched = postprocess_single(
        sem_mask=fg,
        flow=stitched_flows,
        n_steps=mito_post_cfg["n_steps"],
        step_size=mito_post_cfg["step_size"],
        convergence_radius=mito_post_cfg["convergence_radius"],
        min_size=mito_post_cfg["min_size"],
        group=mito_post_cfg["group"],
    )
    print(f"  Stitched:   GT={mito_mask.max()} instances, "
          f"recovered={post_stitched.max()}")

    # Watershed on full-crop saturated EDT
    print("  Running watershed on EDT (full-crop)...")
    ws_full = watershed_on_edt(full_sat_edt, fg)
    print(f"  WS full:    GT={mito_mask.max()} instances, "
          f"recovered={ws_full.max()}")

    # Watershed on stitched saturated EDT
    print("  Running watershed on EDT (stitched)...")
    ws_stitched = watershed_on_edt(stitched_sat_edt, fg)
    print(f"  WS stitch:  GT={mito_mask.max()} instances, "
          f"recovered={ws_stitched.max()}")

    # ── Derived layers ────────────────────────────────────────────────
    full_mag = np.sqrt((full_flows ** 2).sum(axis=0)).astype(np.float32)
    stitch_mag = np.sqrt((stitched_flows ** 2).sum(axis=0)).astype(np.float32)
    angle_err_vol = angle_err.astype(np.float32) if fg.any() else np.zeros_like(full_mag)

    weight_viz = np.zeros((D, H, W), dtype=np.float32)
    for slc in subcrop_slices:
        cs = tuple(s.stop - s.start for s in slc)
        w = cosine_blend_weight(cs, at_volume_boundary=_boundary_flags(slc, (D, H, W)))
        weight_viz[slc[0], slc[1], slc[2]] += w

    # ── Launch Neuroglancer ───────────────────────────────────────────
    print("\nLaunching Neuroglancer...")
    viewer = view_in_neuroglancer(
        gt_instances=mito_mask.astype(np.uint32),
        # subcrop_regions=subcrop_map,
        # blend_weights__img=weight_viz,

        # # Per-instance EDT
        # full_edt__img=full_edt,
        # full_sat_edt__img=full_sat_edt,
        # stitched_edt__img=stitched_edt,
        # stitched_sat_edt__img=stitched_sat_edt,

        # # Full-crop flow
        # full_flow_dz__img=full_flows[0],
        # full_flow_dy__img=full_flows[1],
        # full_flow_dx__img=full_flows[2],
        # full_flow_mag__img=full_mag,

        # # Stitched flow
        # stitch_flow_dz__img=stitched_flows[0],
        # stitch_flow_dy__img=stitched_flows[1],
        # stitch_flow_dx__img=stitched_flows[2],
        # stitch_flow_mag__img=stitch_mag,

        # # Comparison
        # # angular_error__img=angle_err_vol,

        # # Postprocessed results — flow tracking
        # post_full_crop=post_full.astype(np.uint32),
        # post_stitched=post_stitched.astype(np.uint32),

        # Postprocessed results — watershed on EDT
        ws_full_crop=ws_full,
        ws_stitched=ws_stitched,
    )

    print("\nLayers:")
    print("  gt_instances      — ground truth instance labels")
    print("  subcrop_regions   — which sub-crop each voxel belongs to")
    print("  blend_weights     — sum of cosine blend weights")
    print("  full_edt          — per-instance open-boundary EDT (full crop)")
    print("  full_sat_edt      — saturated EDT sigma=5 (full crop)")
    print("  stitched_edt      — per-instance open-boundary EDT (stitched)")
    print("  stitched_sat_edt  — saturated EDT sigma=5 (stitched)")
    print("  full_flow_*       — direct flow from full-crop (reference)")
    print("  stitch_flow_*     — direct flow stitched from sub-crops")
    print("  angular_error     — angle between full and stitched (degrees)")
    print("  post_full_crop    — instances from full-crop flow (Euler tracking)")
    print("  post_stitched     — instances from stitched flow (Euler tracking)")
    print("  ws_full_crop      — instances from full-crop EDT (watershed)")
    print("  ws_stitched       — instances from stitched EDT (watershed)")

    print("\nPress Enter to exit.")
    input()


if __name__ == "__main__":
    main()

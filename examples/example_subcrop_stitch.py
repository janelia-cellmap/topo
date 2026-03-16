#!/usr/bin/env python3
"""Simulate tiled inference: split → per-tile flow → stitch → postprocess.

Demonstrates the full pipeline as it would work at inference time:
1. Load a GT crop (mito ~16nm)
2. Split into overlapping sub-crops
3. Generate flows independently per sub-crop (with spatial masks)
4. Stitch flows back together (cosine-weighted blending in overlaps)
5. Postprocess the stitched flow field
6. Compare: GT vs full-crop flow vs stitched flow vs postprocessed

Visualize all intermediate results in Neuroglancer.
"""

import numpy as np
import zarr
import neuroglancer

from topo import (
    generate_diffusion_flows,
    postprocess_single,
    compute_subcrop_slices,
    build_spatial_mask,
    cosine_blend_weight,
    stitch_flows,
)
from topo.config import get_instance_class_config, get_postprocess_config
from topo.stitch import _boundary_flags

# ── Data ──────────────────────────────────────────────────────────────────
# DATA_ROOT = "/nrs/cellmap/data/jrc_hela-2/jrc_hela-2.zarr/recon-1/labels/groundtruth/crop155"
# MITO_PATH = f"{DATA_ROOT}/mito/s3"  # ~16nm
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



# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Sub-crop stitching example")
    print("=" * 60)

    # ── 1. Load full crop ─────────────────────────────────────────────
    print("\n[1/6] Loading mito (~16nm, crop155)...")
    mito_mask = zarr.open(MITO_PATH, "r")[:]

    D, H, W = mito_mask.shape
    print(f"  Shape: {D}x{H}x{W}")
    print(f"  Instances: {np.unique(mito_mask[mito_mask > 0])}")

    mito_flow_cfg = get_instance_class_config(16)["mito"]
    mito_post_cfg = get_postprocess_config(16)["mito"]

    # ── 2. Generate full-crop flow (reference) ────────────────────────
    # No spatial mask for the full volume — the volume boundary is real,
    # not a crop edge, so diffusion should use Dirichlet BC (flow stays
    # inward) rather than Neumann (which would let flow leak outward and
    # cause overmerging of instances near the edge).
    print("\n[2/6] Generating full-crop diffusion flows (reference)...")
    full_flows = generate_diffusion_flows(
        mito_mask,
        n_iter=mito_flow_cfg.get("diffusion_iters", 200),
        spatial_mask=None,
    )
    print(f"  Flow shape: {full_flows.shape}")

    # ── 3. Split into sub-crops ───────────────────────────────────────
    # Use ~50% overlap for good blending
    crop_size = (D // 2 + 4, H // 2 + 4, W // 2 + 4)
    # Clamp to volume size
    crop_size = tuple(min(cs, vs) for cs, vs in zip(crop_size, (D, H, W)))
    overlap = tuple(cs // 3 for cs in crop_size)

    subcrop_slices = compute_subcrop_slices((D, H, W), crop_size, overlap)
    n_subcrops = len(subcrop_slices)
    print(f"\n[3/6] Splitting into {n_subcrops} sub-crops...")
    print(f"  Crop size: {crop_size}, overlap: {overlap}")

    # Visualize sub-crop boundaries
    subcrop_map = np.zeros((D, H, W), dtype=np.uint32)
    for i, slc in enumerate(subcrop_slices):
        subcrop_map[slc[0], slc[1], slc[2]] = i + 1

    # ── 4. Generate flows per sub-crop ────────────────────────────────
    print("\n[4/6] Generating flows per sub-crop...")
    subcrop_flows = []
    for i, slc in enumerate(subcrop_slices):
        sub_mask = mito_mask[slc[0], slc[1], slc[2]]
        sub_spatial = build_spatial_mask(sub_mask.shape)

        sub_flow = generate_diffusion_flows(
            sub_mask,
            n_iter=mito_flow_cfg.get("diffusion_iters", 200),
            spatial_mask=sub_spatial,
        )
        subcrop_flows.append(sub_flow)

        n_inst = len(np.unique(sub_mask[sub_mask > 0]))
        print(f"  Sub-crop {i+1}/{n_subcrops}: shape={sub_mask.shape}, "
              f"instances={n_inst}")

    # ── 5. Stitch flows ──────────────────────────────────────────────
    print("\n[5/6] Stitching sub-crop flows (cosine blending)...")
    stitched_flows = stitch_flows((D, H, W), subcrop_slices, subcrop_flows)

    # Compare with full-crop flow
    fg = mito_mask > 0
    if fg.any():
        # Angular error between full and stitched
        dot = np.clip(
            (full_flows * stitched_flows).sum(axis=0),
            -1, 1,
        )
        angle_err = np.degrees(np.arccos(dot))
        angle_err_fg = angle_err[fg]
        print(f"  Angular error (full vs stitched) on FG:")
        print(f"    Mean: {angle_err_fg.mean():.1f}°, "
              f"Median: {np.median(angle_err_fg):.1f}°, "
              f"P95: {np.percentile(angle_err_fg, 95):.1f}°")

    # ── 6. Postprocess ───────────────────────────────────────────────
    print("\n[6/6] Postprocessing...")

    # From full-crop flow
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

    # From stitched flow
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

    # ── Compute derived layers ────────────────────────────────────────
    full_mag = np.sqrt((full_flows ** 2).sum(axis=0)).astype(np.float32)
    stitch_mag = np.sqrt((stitched_flows ** 2).sum(axis=0)).astype(np.float32)
    angle_err_vol = angle_err.astype(np.float32) if fg.any() else np.zeros_like(full_mag)

    # Blend weight visualization (sum of cosine weights)
    weight_viz = np.zeros((D, H, W), dtype=np.float32)
    for slc in subcrop_slices:
        cs = tuple(s.stop - s.start for s in slc)
        w = cosine_blend_weight(cs, at_volume_boundary=_boundary_flags(slc, (D, H, W)))
        weight_viz[slc[0], slc[1], slc[2]] += w

    # ── Launch Neuroglancer ───────────────────────────────────────────
    print("\nLaunching Neuroglancer...")
    viewer = view_in_neuroglancer(
        # Ground truth
        gt_instances=mito_mask.astype(np.uint32),

        # Sub-crop layout
        subcrop_regions=subcrop_map,
        blend_weights__img=weight_viz,

        # Full-crop flow
        full_flow_dz__img=full_flows[0],
        full_flow_dy__img=full_flows[1],
        full_flow_dx__img=full_flows[2],
        full_flow_mag__img=full_mag,

        # Stitched flow
        stitch_flow_dz__img=stitched_flows[0],
        stitch_flow_dy__img=stitched_flows[1],
        stitch_flow_dx__img=stitched_flows[2],
        stitch_flow_mag__img=stitch_mag,

        # Comparison
        angular_error__img=angle_err_vol,

        # Postprocessed results
        post_full_crop=post_full.astype(np.uint32),
        post_stitched=post_stitched.astype(np.uint32),
    )

    print("\nLayers:")
    print("  gt_instances      — ground truth instance labels")
    print("  subcrop_regions   — which sub-crop each voxel belongs to")
    print("  blend_weights     — sum of cosine blend weights (bright=overlap)")
    print("  full_flow_*       — flow from full-crop generation (reference)")
    print("  stitch_flow_*     — flow stitched from sub-crops")
    print("  angular_error     — angle between full and stitched flows (degrees)")
    print("  post_full_crop    — instances from full-crop flow")
    print("  post_stitched     — instances from stitched flow")

    print("\nPress Enter to exit.")
    input()


if __name__ == "__main__":
    main()

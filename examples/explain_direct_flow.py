#!/usr/bin/env python3
"""Step-by-step visualization of direct flow generation.

Shows every stage of the pipeline for the nuc instance at ~64nm:
1. Raw instance mask
2. Spatial mask (crop boundary)
3. Instance touches crop boundary? (dilation check)
4. Center finding: COM vs DT-interior-peak vs DT-open-boundary peak
5. Padded mask for open-boundary EDT
6. Quiver comparisons (COM, interior DT, open-boundary DT)
7. HSV flow comparisons
8. Angular difference maps

Generates: explain_direct_flow.png
"""

import os
import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from scipy.ndimage import center_of_mass, distance_transform_edt, binary_dilation

# ── Data ──────────────────────────────────────────────────────────────────
DATA_ROOT = "/nrs/cellmap/data/jrc_hela-2/jrc_hela-2.zarr/recon-1/labels/groundtruth/crop155"
NUC_PATH = f"{DATA_ROOT}/nuc/s5"  # ~64nm
OUT_DIR = os.path.join(os.path.dirname(__file__), "imgs")
os.makedirs(OUT_DIR, exist_ok=True)


def _flow_to_hsv(fy, fx, fg_mask):
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


def _compute_open_boundary_edt(mask, spatial_mask):
    """EDT with crop faces treated as open (pad with 1 where instance exits).

    This pushes the peak toward the crop edge, so that when two adjacent
    crops share the same instance, both peaks are near the shared face.
    """
    padded = np.pad(mask.astype(np.uint8), 1, mode='constant', constant_values=0)
    sp_padded = np.pad((spatial_mask > 0.5).astype(np.uint8), 1,
                       mode='constant', constant_values=0)

    for axis in range(mask.ndim):
        for side in [0, -1]:
            # Face of the original volume (offset by 1 for padding)
            face_slc = [slice(1, -1)] * mask.ndim
            face_slc[axis] = 1 if side == 0 else padded.shape[axis] - 2
            face_mask = padded[tuple(face_slc)]
            face_spatial = sp_padded[tuple(face_slc)]

            # Instance present AND spatial_mask=0 → instance exits here
            exits = (face_mask > 0) & (face_spatial == 0)
            if exits.any():
                pad_slc = [slice(1, -1)] * mask.ndim
                pad_slc[axis] = 0 if side == 0 else padded.shape[axis] - 1
                padded[tuple(pad_slc)] = np.where(
                    exits, 1, padded[tuple(pad_slc)])

    dist = distance_transform_edt(padded)
    return dist[1:-1, 1:-1, 1:-1], padded


def _unit_vectors_2d(center_yz, mask_2d, coords_y, coords_x):
    """Compute 2D unit vectors toward a center point."""
    dy = center_yz[0] - coords_y
    dx = center_yz[1] - coords_x
    mag = np.sqrt(dy**2 + dx**2)
    mag_safe = np.clip(mag, 1e-8, None)
    uy = np.where(mask_2d, dy / mag_safe, 0)
    ux = np.where(mask_2d, dx / mag_safe, 0)
    return uy, ux, mag


def main():
    # Load data
    instance_mask = zarr.open(NUC_PATH, "r")[:]
    nuc_2 = instance_mask.copy()  # keep original for plotting
    nuc_2 = nuc_2.transpose(1, 2, 0)  # ZYX → XYZ for plotting
    nuc_2 *=10
    instance_mask = np.maximum(nuc_2, instance_mask)  # combine to get more instances
    D, H, W = instance_mask.shape
    mid_z = D // 2
    print(f"Nuc shape: {D}x{H}x{W}, mid slice: z={mid_z}")
    print(f"Instance IDs: {np.unique(instance_mask)}")

    mask = instance_mask > 0
    mask_2d = mask[mid_z]

    # ── Spatial mask ──────────────────────────────────────────────────
    spatial_mask = np.ones((D, H, W), dtype=np.float32)
    spatial_mask[0, :, :] = 0; spatial_mask[-1, :, :] = 0
    spatial_mask[:, 0, :] = 0; spatial_mask[:, -1, :] = 0
    spatial_mask[:, :, 0] = 0; spatial_mask[:, :, -1] = 0
    spatial_2d = spatial_mask[mid_z]

    # ── Boundary detection ────────────────────────────────────────────
    dilated = binary_dilation(mask, iterations=1)
    boundary_region = dilated & ~mask
    touches = boundary_region & (spatial_mask < 0.5)
    is_cropped = bool(np.any(touches))
    print(f"Instance touches crop boundary: {is_cropped}")

    # ── Per-instance computation ────────────────────────────────────────
    instance_ids = np.unique(instance_mask)
    instance_ids = instance_ids[instance_ids > 0]  # skip background
    print(f"Computing per-instance for {len(instance_ids)} instances...")

    coords = np.mgrid[0:D, 0:H, 0:W].astype(np.float32)
    coords_y = coords[1, mid_z]
    coords_x = coords[2, mid_z]

    # Accumulators for composite fields
    dist_naive_vol = np.zeros((D, H, W), dtype=np.float32)
    dist_interior_vol = np.zeros((D, H, W), dtype=np.float32)
    dist_open_vol = np.zeros((D, H, W), dtype=np.float32)
    uy_com = np.zeros((H, W), dtype=np.float32)
    ux_com = np.zeros((H, W), dtype=np.float32)
    mag_com = np.zeros((H, W), dtype=np.float32)
    uy_int = np.zeros((H, W), dtype=np.float32)
    ux_int = np.zeros((H, W), dtype=np.float32)
    mag_int = np.zeros((H, W), dtype=np.float32)
    uy_open = np.zeros((H, W), dtype=np.float32)
    ux_open = np.zeros((H, W), dtype=np.float32)
    mag_open = np.zeros((H, W), dtype=np.float32)

    for iid in instance_ids:
        inst_mask = instance_mask == iid
        inst_mask_2d = inst_mask[mid_z]
        if not inst_mask_2d.any():
            continue

        # Method A: COM
        com_i = np.array(center_of_mass(inst_mask), dtype=np.float32)
        uy_i, ux_i, mag_i = _unit_vectors_2d(
            [com_i[1], com_i[2]], inst_mask_2d, coords_y, coords_x)
        uy_com[inst_mask_2d] = uy_i[inst_mask_2d]
        ux_com[inst_mask_2d] = ux_i[inst_mask_2d]
        mag_com[inst_mask_2d] = mag_i[inst_mask_2d]

        # Method B: Interior DT
        interior_i = inst_mask & (spatial_mask > 0.5)
        if interior_i.any():
            dist_int_i = distance_transform_edt(interior_i)
        else:
            dist_int_i = distance_transform_edt(inst_mask)
        dist_interior_vol[inst_mask] = dist_int_i[inst_mask]
        peak_int_3d = np.unravel_index(np.argmax(dist_int_i), dist_int_i.shape)
        peak_int_i = np.array(peak_int_3d, dtype=np.float32)
        uy_i, ux_i, mag_i = _unit_vectors_2d(
            [peak_int_i[1], peak_int_i[2]], inst_mask_2d, coords_y, coords_x)
        uy_int[inst_mask_2d] = uy_i[inst_mask_2d]
        ux_int[inst_mask_2d] = ux_i[inst_mask_2d]
        mag_int[inst_mask_2d] = mag_i[inst_mask_2d]

        # Method C: Open-boundary DT
        dist_open_i, _ = _compute_open_boundary_edt(inst_mask, spatial_mask)
        dist_open_vol[inst_mask] = dist_open_i[inst_mask]
        peak_open_3d = np.unravel_index(np.argmax(dist_open_i), dist_open_i.shape)
        peak_open_i = np.array(peak_open_3d, dtype=np.float32)
        uy_i, ux_i, mag_i = _unit_vectors_2d(
            [peak_open_i[1], peak_open_i[2]], inst_mask_2d, coords_y, coords_x)
        uy_open[inst_mask_2d] = uy_i[inst_mask_2d]
        ux_open[inst_mask_2d] = ux_i[inst_mask_2d]
        mag_open[inst_mask_2d] = mag_i[inst_mask_2d]

        # Naive EDT per instance
        dist_naive_i = distance_transform_edt(inst_mask)
        dist_naive_vol[inst_mask] = dist_naive_i[inst_mask]

        print(f"  Instance {iid}: COM=({com_i[1]:.1f},{com_i[2]:.1f}), "
              f"IntDT=({peak_int_i[1]:.0f},{peak_int_i[2]:.0f}), "
              f"OpenDT=({peak_open_i[1]:.0f},{peak_open_i[2]:.0f})")

    # 2D slices of composite distance fields
    dist_interior_2d = dist_interior_vol[mid_z]
    dist_open_2d = dist_open_vol[mid_z]

    # ── Quiver helpers ────────────────────────────────────────────────
    step = 2
    yy, xx = np.mgrid[0:H, 0:W]
    step_mask = np.zeros_like(mask_2d)
    step_mask[::step, ::step] = True
    show = mask_2d & step_mask

    # ── Plot: 4 rows × 3 columns ─────────────────────────────────────
    fig, axes = plt.subplots(4, 3, figsize=(20, 24))

    # ===== Row 1: Setup =====
    # (0,0) Instance mask
    ax = axes[0, 0]
    inst_2d = instance_mask[mid_z]
    im = ax.imshow(inst_2d, cmap="nipy_spectral", interpolation="nearest")
    # Draw boundaries where neighboring pixels have different instance IDs
    boundary = np.zeros_like(inst_2d, dtype=bool)
    boundary[:-1, :] |= inst_2d[:-1, :] != inst_2d[1:, :]
    boundary[1:, :]  |= inst_2d[:-1, :] != inst_2d[1:, :]
    boundary[:, :-1] |= inst_2d[:, :-1] != inst_2d[:, 1:]
    boundary[:, 1:]  |= inst_2d[:, :-1] != inst_2d[:, 1:]
    ax.imshow(np.ma.masked_where(~boundary, np.ones_like(inst_2d, dtype=float)),
              cmap="gray_r", interpolation="nearest", alpha=1.0, vmin=0, vmax=1)
    ax.set_title("1. Instance mask\n(real values, black=boundary)")
    fig.colorbar(im, ax=ax, shrink=0.6, label="instance ID")

    # (0,1) Spatial mask
    ax = axes[0, 1]
    im = ax.imshow(spatial_2d, cmap="RdYlGn", interpolation="nearest", vmin=0, vmax=1)
    ax.imshow(mask_2d, cmap="gray", alpha=0.3, interpolation="nearest")
    ax.set_title("2. Spatial mask\n(green=interior, red=crop boundary)")
    fig.colorbar(im, ax=ax, shrink=0.6)

    # (0,2) Boundary touch detection
    ax = axes[0, 2]
    ax.imshow(boundary, cmap="gray", interpolation="nearest")
    ax.set_title("3. Instance boundaries\n(white=boundary between instances)")

    # ===== Row 2: Per-instance distance transforms (3 methods) =====
    # (1,0) Naive EDT per instance
    ax = axes[1, 0]
    dt_display = np.where(mask_2d, dist_naive_vol[mid_z], np.nan)
    im = ax.imshow(dt_display, cmap="hot", interpolation="nearest")
    ax.set_title("4a. Per-instance EDT\n(naive, no spatial awareness)")
    fig.colorbar(im, ax=ax, shrink=0.6, label="dist (vox)")

    # (1,1) Interior DT per instance — crop boundary = hard wall
    ax = axes[1, 1]
    dt_display = np.where(mask_2d, dist_interior_2d, np.nan)
    im = ax.imshow(dt_display, cmap="hot", interpolation="nearest")
    ax.set_title("4b. Per-instance EDT(mask & spatial)\n(crop edge = wall)")
    fig.colorbar(im, ax=ax, shrink=0.6, label="dist (vox)")

    # (1,2) Open-boundary DT per instance — crop boundary = open
    ax = axes[1, 2]
    dt_display = np.where(mask_2d, dist_open_2d, np.nan)
    im = ax.imshow(dt_display, cmap="hot", interpolation="nearest")
    ax.set_title("4c. Per-instance EDT(padded)\n(crop edge = open)")
    fig.colorbar(im, ax=ax, shrink=0.6, label="dist (vox)")

    # ===== Row 3: Quiver arrows (3 methods) =====
    for col, (uy, ux, mag_v, label) in enumerate([
        (uy_com, ux_com, mag_com, "A) COM"),
        (uy_int, ux_int, mag_int, "B) Interior DT"),
        (uy_open, ux_open, mag_open, "C) Open-boundary DT"),
    ]):
        ax = axes[2, col]
        ax.imshow(mask_2d, cmap="gray", interpolation="nearest", alpha=0.3)
        ax.quiver(xx[show], yy[show], ux[show], -uy[show],
                  mag_v[show], cmap="viridis", scale=20, width=0.005)
        ax.set_title(f"5. Quiver → {label}")

    # ===== Row 4: HSV flows (3 methods) =====
    for col, (uy, ux, label) in enumerate([
        (uy_com, ux_com, "A) COM"),
        (uy_int, ux_int, "B) Interior DT"),
        (uy_open, ux_open, "C) Open-boundary DT"),
    ]):
        ax = axes[3, col]
        rgb = _flow_to_hsv(uy, ux, mask_2d)
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(f"6. HSV flow → {label}")

    for ax in axes.flat:
        ax.set_ylabel("Y"); ax.set_xlabel("X")

    fig.suptitle(
        "Direct Flow — Per-Instance, 3 Center-Finding Methods\n"
        f"Nuc ({D}×{H}×{W} @ ~64nm), z={mid_z}, {len(instance_ids)} instances, cropped={is_cropped}",
        fontsize=14, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = os.path.join(OUT_DIR, "explain_direct_flow.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

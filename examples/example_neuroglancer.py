#!/usr/bin/env python3
"""Visualize flow generation steps in Neuroglancer (3D).

Loads the nuc instance at ~64nm and generates layers for each stage:
- Instance mask
- Spatial mask
- Distance transforms (interior DT, open-boundary DT)
- Flow components (dz, dy, dx) as image layers
- Flow magnitude
- Center points as segmentation markers

Launch, then open the printed URL in your browser.
"""

import numpy as np
import zarr
import neuroglancer
from scipy.ndimage import center_of_mass, distance_transform_edt, binary_dilation

from topo import generate_direct_flows
from topo.config import get_postprocess_config

# ── Data ──────────────────────────────────────────────────────────────────
DATA_ROOT = "/nrs/cellmap/data/jrc_hela-2/jrc_hela-2.zarr/recon-1/labels/groundtruth/crop155"
MITO_PATH = f"{DATA_ROOT}/mito/s3"  # ~16nm
NUC_PATH = f"{DATA_ROOT}/nuc/s5"   # ~64nm


def view_in_neuroglancer(**kwargs):
    """Launch Neuroglancer viewer with provided volumes as layers."""
    neuroglancer.set_server_bind_address("0.0.0.0")
    viewer = neuroglancer.Viewer()

    with viewer.txn() as s:
        for array_name, array in kwargs.items():
            if (
                array.dtype in (float, np.float32, np.float64)
                or "raw" in array_name
                or "__img" in array_name
            ):
                s.layers[array_name] = neuroglancer.ImageLayer(
                    source=neuroglancer.LocalVolume(data=array),
                )
            else:
                s.layers[array_name] = neuroglancer.SegmentationLayer(
                    source=neuroglancer.LocalVolume(data=array),
                )

    print(f"\nNeuroglancer URL: {viewer.get_viewer_url()}")
    return viewer


def build_spatial_mask(shape):
    """Spatial mask: 1 in interior, 0 at crop boundary faces."""
    sp = np.ones(shape, dtype=np.float32)
    sp[0, :, :] = 0; sp[-1, :, :] = 0
    sp[:, 0, :] = 0; sp[:, -1, :] = 0
    sp[:, :, 0] = 0; sp[:, :, -1] = 0
    return sp


def compute_open_boundary_edt(mask, spatial_mask):
    """EDT with large padding at open crop faces."""
    pad_width = max(mask.shape)
    padded = np.pad(mask.astype(np.uint8), pad_width,
                    mode='constant', constant_values=0)
    sp = spatial_mask > 0.5

    for axis in range(mask.ndim):
        for side in [0, -1]:
            face_slc = [slice(None)] * mask.ndim
            face_slc[axis] = 0 if side == 0 else mask.shape[axis] - 1
            exits = mask[tuple(face_slc)] & ~sp[tuple(face_slc)]

            if exits.any():
                if side == 0:
                    pad_range = range(0, pad_width)
                else:
                    pad_range = range(padded.shape[axis] - pad_width,
                                      padded.shape[axis])
                for i in pad_range:
                    slc = [slice(pad_width, -pad_width)] * mask.ndim
                    slc[axis] = i
                    padded[tuple(slc)] = np.where(exits, 1, padded[tuple(slc)])

    dist = distance_transform_edt(padded)
    s = tuple(slice(pad_width, -pad_width) for _ in range(mask.ndim))
    return dist[s]


def make_center_marker(shape, center, radius=1):
    """Create a small sphere marker volume at the given center."""
    marker = np.zeros(shape, dtype=np.uint32)
    z0, y0, x0 = [int(round(c)) for c in center]
    D, H, W = shape
    for dz in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dz**2 + dy**2 + dx**2 <= radius**2:
                    zz = np.clip(z0 + dz, 0, D - 1)
                    yy = np.clip(y0 + dy, 0, H - 1)
                    xx = np.clip(x0 + dx, 0, W - 1)
                    marker[zz, yy, xx] = 1
    return marker


def main():
    # ── NUC ───────────────────────────────────────────────────────────
    print("Loading nuc (~64nm)...")
    nuc_mask = zarr.open(NUC_PATH, "r")[:]
    D, H, W = nuc_mask.shape
    print(f"  Shape: {D}x{H}x{W}")

    mask = nuc_mask > 0
    spatial_mask = build_spatial_mask(nuc_mask.shape)

    # Centers
    com = np.array(center_of_mass(mask), dtype=np.float32)
    print(f"  COM: z={com[0]:.1f}, y={com[1]:.1f}, x={com[2]:.1f}")

    # Interior DT
    interior = mask & (spatial_mask > 0.5)
    dist_interior = distance_transform_edt(interior).astype(np.float32)
    peak_int = np.unravel_index(np.argmax(dist_interior), dist_interior.shape)
    print(f"  Interior DT peak: {peak_int}")

    # Open-boundary DT
    dist_open = compute_open_boundary_edt(mask, spatial_mask).astype(np.float32)
    peak_open = np.unravel_index(np.argmax(dist_open), dist_open.shape)
    print(f"  Open DT peak: {peak_open}")

    # Flows with open-boundary center
    print("  Generating direct flows (with spatial mask)...")
    nuc_flows = generate_direct_flows(nuc_mask, spatial_mask=spatial_mask)

    # Flow components as image layers
    flow_dz = nuc_flows[0].astype(np.float32)
    flow_dy = nuc_flows[1].astype(np.float32)
    flow_dx = nuc_flows[2].astype(np.float32)
    flow_mag = np.sqrt(flow_dz**2 + flow_dy**2 + flow_dx**2).astype(np.float32)

    # Center markers (different IDs for different methods)
    centers_marker = np.zeros(nuc_mask.shape, dtype=np.uint32)
    # COM = label 1
    com_marker = make_center_marker(nuc_mask.shape, com, radius=1)
    centers_marker += com_marker * 1
    # Interior DT = label 2
    int_marker = make_center_marker(nuc_mask.shape, peak_int, radius=1)
    centers_marker = np.where(int_marker > 0, 2, centers_marker)
    # Open DT = label 3
    open_marker = make_center_marker(nuc_mask.shape, peak_open, radius=1)
    centers_marker = np.where(open_marker > 0, 3, centers_marker)

    # Boundary detection
    dilated = binary_dilation(mask, iterations=1)
    boundary_touch = (dilated & ~mask & (spatial_mask < 0.5)).astype(np.uint32)

    # ── MITO ──────────────────────────────────────────────────────────
    print("\nLoading mito (~16nm)...")
    mito_mask = zarr.open(MITO_PATH, "r")[:]
    print(f"  Shape: {mito_mask.shape}")

    mito_spatial = build_spatial_mask(mito_mask.shape)

    from topo import generate_diffusion_flows
    from topo.config import get_instance_class_config
    mito_cfg = get_instance_class_config(16)["mito"]
    print(f"  Generating diffusion flows (iters={mito_cfg.get('diffusion_iters', 200)})...")
    mito_flows = generate_diffusion_flows(
        mito_mask, n_iter=mito_cfg.get("diffusion_iters", 200),
        spatial_mask=mito_spatial,
    )
    mito_flow_dz = mito_flows[0].astype(np.float32)
    mito_flow_dy = mito_flows[1].astype(np.float32)
    mito_flow_dx = mito_flows[2].astype(np.float32)
    mito_flow_mag = np.sqrt(
        mito_flow_dz**2 + mito_flow_dy**2 + mito_flow_dx**2
    ).astype(np.float32)

    # ── Launch Neuroglancer ───────────────────────────────────────────
    print("\nLaunching Neuroglancer...")
    viewer = view_in_neuroglancer(
        # NUC layers
        nuc_instances=nuc_mask.astype(np.uint32),
        nuc_spatial_mask__img=spatial_mask,
        nuc_boundary_touch=boundary_touch,
        nuc_edt_interior__img=dist_interior,
        nuc_edt_open__img=dist_open,
        nuc_flow_dz__img=flow_dz,
        nuc_flow_dy__img=flow_dy,
        nuc_flow_dx__img=flow_dx,
        nuc_flow_magnitude__img=flow_mag,
        nuc_centers=centers_marker,
        # MITO layers
        mito_instances=mito_mask.astype(np.uint32),
        mito_flow_dz__img=mito_flow_dz,
        mito_flow_dy__img=mito_flow_dy,
        mito_flow_dx__img=mito_flow_dx,
        mito_flow_magnitude__img=mito_flow_mag,
    )

    print("\nLayers:")
    print("  NUC:")
    print("    nuc_instances        — instance segmentation")
    print("    nuc_spatial_mask     — crop boundary (0) vs interior (1)")
    print("    nuc_boundary_touch   — where instance exits crop")
    print("    nuc_edt_interior     — EDT with crop edge as wall")
    print("    nuc_edt_open         — EDT with crop edge as open")
    print("    nuc_flow_dz/dy/dx    — flow vector components")
    print("    nuc_flow_magnitude   — flow vector magnitude")
    print("    nuc_centers          — 1=COM, 2=Interior DT, 3=Open DT")
    print("  MITO:")
    print("    mito_instances       — instance segmentation")
    print("    mito_flow_dz/dy/dx   — diffusion flow components")
    print("    mito_flow_magnitude  — flow magnitude")

    print("\nPress Ctrl+C to stop the viewer.")
    input("(or press Enter to exit)\n")


if __name__ == "__main__":
    main()

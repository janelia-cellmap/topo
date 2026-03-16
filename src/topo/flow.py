"""CPU flow field generation using NumPy/SciPy.

Two strategies:
- Direct flows: unit vectors pointing from each voxel to instance center of mass.
  Works well for convex shapes (nuclei, vesicles, etc.).
- Diffusion flows: solve the heat equation inside each instance, then take the
  gradient. Flows follow the object topology, handling non-convex shapes like
  mitochondria that would fail with direct center-of-mass vectors.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import center_of_mass, distance_transform_edt, binary_dilation
from typing import Optional

from .config import EVALUATED_INSTANCE_CLASSES, get_instance_class_config


def _touches_crop_boundary(mask: np.ndarray, spatial_mask: np.ndarray) -> bool:
    """Check if an instance touches the annotation boundary (spatial_mask=0)."""
    dilated = binary_dilation(mask, iterations=1)
    boundary_region = dilated & ~mask
    return bool(np.any(boundary_region & (spatial_mask < 0.5)))


def _find_center(
    mask: np.ndarray, spatial_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Find the instance center: COM for fully visible instances,
    distance transform peak for cropped instances.

    For cropped instances, we pad the mask with 1s at crop faces where
    the instance exits (pretending it continues beyond the crop).  This
    way the EDT only sees real instance boundaries, not crop boundaries,
    and the peak shifts toward the crop edge.  At inference, when two
    adjacent crops each contain half of the same instance, both peaks
    will be near the shared crop face → easy to merge.
    """
    if spatial_mask is not None and _touches_crop_boundary(mask, spatial_mask):
        # Pad with 0 (default), then fill crop faces with 1 where
        # the instance touches the annotation boundary
        padded = np.pad(mask.astype(np.uint8), 1, mode='constant', constant_values=0)
        sp_padded = np.pad((spatial_mask > 0.5).astype(np.uint8), 1,
                           mode='constant', constant_values=0)

        # For each face: if the instance touches a crop boundary there,
        # extend the mask into the padding (as if the instance continues)
        for axis in range(mask.ndim):
            for side in [0, -1]:
                # Slice at the face of the original volume (offset by 1 for padding)
                face_slc = [slice(1, -1)] * mask.ndim
                face_slc[axis] = 1 if side == 0 else padded.shape[axis] - 2
                face_mask = padded[tuple(face_slc)]
                face_spatial = sp_padded[tuple(face_slc)]

                # Where instance is present AND spatial_mask=0 → crop boundary
                exits = (face_mask > 0) & (face_spatial == 0)
                if exits.any():
                    pad_slc = [slice(1, -1)] * mask.ndim
                    pad_slc[axis] = 0 if side == 0 else padded.shape[axis] - 1
                    padded[tuple(pad_slc)] = np.where(exits, 1, padded[tuple(pad_slc)])

        dist = distance_transform_edt(padded)
        # Crop back to original size
        dist = dist[1:-1, 1:-1, 1:-1]
        peak = np.unravel_index(np.argmax(dist), dist.shape)
        return np.array(peak, dtype=np.float32)
    else:
        return np.array(center_of_mass(mask), dtype=np.float32)


def generate_direct_flows(
    instance_mask: np.ndarray,
    spatial_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Generate unit vectors from each voxel to its instance's center.

    For instances fully inside the annotated region, the center is the
    center of mass.  For instances touching the crop boundary
    (spatial_mask=0), the center is the distance-transform peak — the
    point farthest from any edge — which is robust to cropping.

    Args:
        instance_mask: [D, H, W] int array of instance IDs (0 = background).
        spatial_mask: [D, H, W] float/bool — 1 inside annotated region.
            If None, center of mass is used for all instances.

    Returns:
        flows: [3, D, H, W] float32 — unit flow vectors (dz, dy, dx).
    """
    D, H, W = instance_mask.shape
    flows = np.zeros((3, D, H, W), dtype=np.float32)
    coords = np.mgrid[0:D, 0:H, 0:W].astype(np.float32)

    inst_ids = np.unique(instance_mask)
    inst_ids = inst_ids[inst_ids > 0]

    for inst_id in inst_ids:
        mask = instance_mask == inst_id
        if mask.sum() == 0:
            continue

        com = _find_center(mask, spatial_mask)

        dz = com[0] - coords[0]
        dy = com[1] - coords[1]
        dx = com[2] - coords[2]
        mag = np.sqrt(dz ** 2 + dy ** 2 + dx ** 2)
        mag = np.clip(mag, 1e-8, None)

        flows[0][mask] = (dz / mag)[mask]
        flows[1][mask] = (dy / mag)[mask]
        flows[2][mask] = (dx / mag)[mask]

        # Sink at center
        z0 = np.clip(int(round(com[0])), 0, D - 1)
        y0 = np.clip(int(round(com[1])), 0, H - 1)
        x0 = np.clip(int(round(com[2])), 0, W - 1)
        if mask[z0, y0, x0]:
            flows[:, z0, y0, x0] = 0.0

    return flows


def generate_diffusion_flows(
    instance_mask: np.ndarray,
    n_iter: int = 200,
    adaptive_iters: bool = True,
    adaptive_factor: int = 6,
    spatial_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Generate topology-aware flows via heat equation diffusion.

    For each instance:
    1. Initialize a scalar field with coordinate values inside the mask.
    2. Iteratively diffuse (Laplacian smoothing) within the instance.
    3. Take the spatial gradient to obtain the flow direction.

    Boundary conditions:
    - Annotation boundaries (spatial_mask=0): Neumann (free diffusion), so
      cut instances don't pile up against the crop edge.
    - True background (spatial_mask=1, instance=0): Dirichlet=0, so flow
      points inward at real boundaries.

    Args:
        instance_mask: [D, H, W] int array of instance IDs (0 = background).
        n_iter: Maximum diffusion iterations (cap for adaptive).
        adaptive_iters: If True, scale iterations with instance extent.
        adaptive_factor: Multiplier for adaptive iteration count.
        spatial_mask: [D, H, W] float/bool — 1 inside annotated region.
            If None, Dirichlet=0 is used everywhere.

    Returns:
        flows: [3, D, H, W] float32 — unit flow vectors (dz, dy, dx).
    """
    D, H, W = instance_mask.shape
    flows = np.zeros((3, D, H, W), dtype=np.float32)

    inst_ids = np.unique(instance_mask)
    inst_ids = inst_ids[inst_ids > 0]

    for inst_id in inst_ids:
        mask = instance_mask == inst_id
        if mask.sum() == 0:
            continue

        # Crop to bounding box with padding for efficiency
        where_mask = np.where(mask)
        pad = 2
        z0 = max(0, where_mask[0].min() - pad)
        z1 = min(D, where_mask[0].max() + pad + 1)
        y0 = max(0, where_mask[1].min() - pad)
        y1 = min(H, where_mask[1].max() + pad + 1)
        x0 = max(0, where_mask[2].min() - pad)
        x1 = min(W, where_mask[2].max() + pad + 1)

        crop_mask = mask[z0:z1, y0:y1, x0:x1]
        cd, ch, cw = crop_mask.shape
        crop_where = np.where(crop_mask)

        # Build update mask
        if spatial_mask is not None:
            crop_spatial = spatial_mask[z0:z1, y0:y1, x0:x1] > 0.5
            update_mask = crop_mask | ~crop_spatial
        else:
            update_mask = crop_mask

        # Adaptive iteration count
        if adaptive_iters:
            max_extent = max(cd, ch, cw)
            inst_n_iter = min(adaptive_factor * max_extent, n_iter)
        else:
            inst_n_iter = n_iter

        inst_flows = np.zeros((3, cd, ch, cw), dtype=np.float64)

        for axis in range(3):
            offsets = [z0, y0, x0]
            field = np.zeros((cd, ch, cw), dtype=np.float64)
            field[crop_mask] = (crop_where[axis] + offsets[axis]).astype(np.float64)

            for _ in range(inst_n_iter):
                fp = np.pad(field, 1, mode='edge')
                new_field = field.copy()

                for ax in range(3):
                    slc_f = [slice(1, -1)] * 3
                    slc_b = [slice(1, -1)] * 3
                    slc_f[ax] = slice(2, None)
                    slc_b[ax] = slice(None, -2)

                    laplacian = (
                        fp[tuple(slc_f)]
                        + fp[tuple(slc_b)]
                        - 2.0 * field
                    )
                    new_field += (1.0 / 6.0) * laplacian

                field[update_mask] = new_field[update_mask]

            inst_flows[axis] = np.gradient(field, axis=axis)

        # Normalize to unit vectors
        mag = np.sqrt(inst_flows[0] ** 2 + inst_flows[1] ** 2 + inst_flows[2] ** 2)
        mag = np.clip(mag, 1e-8, None)

        for axis in range(3):
            flows[axis, z0:z1, y0:y1, x0:x1][crop_mask] = (
                inst_flows[axis] / mag
            )[crop_mask].astype(np.float32)

    return flows


def compute_boundary_map(
    instance_mask: np.ndarray, dilation_width: int = 2
) -> np.ndarray:
    """Compute inter-instance boundary map.

    Returns 1.0 where adjacent voxels have different non-zero instance IDs,
    dilated by ``dilation_width``.

    Args:
        instance_mask: [D, H, W] int array of instance IDs.
        dilation_width: Boundary dilation in voxels.

    Returns:
        boundary: [D, H, W] float32 boundary map.
    """
    boundary = np.zeros(instance_mask.shape, dtype=np.float32)
    for axis in range(instance_mask.ndim):
        slc_a = [slice(None)] * instance_mask.ndim
        slc_b = [slice(None)] * instance_mask.ndim
        slc_a[axis] = slice(None, -1)
        slc_b[axis] = slice(1, None)

        a = instance_mask[tuple(slc_a)]
        b = instance_mask[tuple(slc_b)]
        is_bnd = (a != b) & (a > 0) & (b > 0)
        boundary[tuple(slc_a)] = np.maximum(boundary[tuple(slc_a)], is_bnd)
        boundary[tuple(slc_b)] = np.maximum(boundary[tuple(slc_b)], is_bnd)

    if dilation_width > 1:
        struct = np.ones([3] * instance_mask.ndim)
        boundary = binary_dilation(
            boundary > 0, struct, iterations=dilation_width - 1
        ).astype(np.float32)

    return boundary


def compute_flow_targets(
    instance_ids: np.ndarray,
    class_names: Optional[list[str]] = None,
    class_config: Optional[dict] = None,
    resolution_nm: Optional[int] = None,
    diffusion_iters: int = 200,
    adaptive_iters: bool = True,
    adaptive_factor: int = 6,
    spatial_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-class flow targets from instance ID volumes.

    Each class gets 3 flow channels (dz, dy, dx). Background voxels get
    flow = (0, 0, 0). Flow type (direct vs. diffusion) is determined by
    class_config.

    Args:
        instance_ids: [N, D, H, W] int32 — instance IDs per class.
        class_names: List of N class names. Defaults to EVALUATED_INSTANCE_CLASSES.
        class_config: Per-class config dict. Defaults to resolution-based config.
        resolution_nm: Resolution in nm, used to select default class_config.
        diffusion_iters: Max diffusion iterations.
        adaptive_iters: Scale iterations with instance extent.
        adaptive_factor: Multiplier for adaptive iteration count.
        spatial_mask: [D, H, W] float/bool — 1 inside annotated region.

    Returns:
        flows: [N*3, D, H, W] float32 — per-class flow unit vectors.
        class_fg: [N, D, H, W] float32 — per-class foreground masks.
    """
    if class_names is None:
        class_names = EVALUATED_INSTANCE_CLASSES
    if class_config is None:
        class_config = get_instance_class_config(resolution_nm)

    N = instance_ids.shape[0]
    D, H, W = instance_ids.shape[1:]

    flows = np.zeros((N * 3, D, H, W), dtype=np.float32)
    class_fg = np.zeros((N, D, H, W), dtype=np.float32)

    for c, cls_name in enumerate(class_names):
        ids = instance_ids[c]
        fg = ids > 0
        class_fg[c] = fg.astype(np.float32)

        if not fg.any():
            continue

        cfg = class_config.get(cls_name, {"flow_type": "direct"})
        flow_type = cfg.get("flow_type", "direct")

        if flow_type == "diffusion":
            n_iter = cfg.get("diffusion_iters", diffusion_iters)
            cls_flows = generate_diffusion_flows(
                ids, n_iter,
                adaptive_iters=adaptive_iters,
                adaptive_factor=adaptive_factor,
                spatial_mask=spatial_mask,
            )
        else:
            cls_flows = generate_direct_flows(ids, spatial_mask=spatial_mask)

        flows[c * 3 : c * 3 + 3] = cls_flows

        # boundary_only: restrict training to near boundaries
        if cfg.get("boundary_only", False):
            bw = cfg.get("boundary_width", 20)
            sp = (spatial_mask > 0.5) if spatial_mask is not None else np.ones_like(fg)
            annotated_bg = sp & ~fg
            if annotated_bg.any():
                dist_to_boundary = distance_transform_edt(~annotated_bg)
                near_boundary = dist_to_boundary <= bw
                class_fg[c] = (fg & near_boundary).astype(np.float32)
            else:
                class_fg[c] = 0.0

    return flows, class_fg

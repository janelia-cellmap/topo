"""Stitching utilities for tiled inference.

Split a volume into overlapping sub-crops, process each independently,
then blend results back into the full volume using boundary-aware cosine
weighting.  The key insight is that cosine tapering should only be applied
at *internal* subcrop seams (where two tiles overlap), not at volume
boundaries where there is no neighbor to blend with.
"""

from __future__ import annotations

import numpy as np
from itertools import product
from typing import Optional


def compute_subcrop_slices(
    volume_shape: tuple[int, ...],
    crop_size: tuple[int, ...],
    overlap: tuple[int, ...],
) -> list[tuple[slice, ...]]:
    """Compute sub-crop slices with overlap for each axis.

    Args:
        volume_shape: (D, H, W) of the full volume.
        crop_size: (cd, ch, cw) size of each sub-crop.
        overlap: (od, oh, ow) overlap between adjacent sub-crops.

    Returns:
        List of tuples of slices [(slice_d, slice_h, slice_w), ...].
    """
    slices_per_axis = []
    for size, cs, ov in zip(volume_shape, crop_size, overlap):
        step = cs - ov
        starts = list(range(0, size - cs + 1, step))
        # Make sure we cover the end
        if starts[-1] + cs < size:
            starts.append(size - cs)
        slices_per_axis.append([slice(s, s + cs) for s in starts])

    return list(product(*slices_per_axis))


def build_spatial_mask(shape: tuple[int, ...]) -> np.ndarray:
    """Spatial mask: 1 in interior, 0 at boundary faces.

    Used to mark subcrop boundaries so diffusion uses Neumann BC there
    (allowing flow to continue past the crop edge).

    Note: Do NOT use this for the full volume if the volume boundary is
    real (not a crop edge).  Pass ``spatial_mask=None`` to flow generation
    instead, so that Dirichlet BC keeps flow pointing inward.
    """
    sp = np.ones(shape, dtype=np.float32)
    sp[0, :, :] = 0; sp[-1, :, :] = 0
    sp[:, 0, :] = 0; sp[:, -1, :] = 0
    sp[:, :, 0] = 0; sp[:, :, -1] = 0
    return sp


def _boundary_flags(
    slc: tuple[slice, ...],
    volume_shape: tuple[int, ...],
) -> list[bool]:
    """Return 6 bools indicating which subcrop faces sit at the volume boundary.

    Order: [z_lo, z_hi, y_lo, y_hi, x_lo, x_hi].
    """
    D, H, W = volume_shape
    return [
        slc[0].start == 0, slc[0].stop == D,
        slc[1].start == 0, slc[1].stop == H,
        slc[2].start == 0, slc[2].stop == W,
    ]


def cosine_blend_weight(
    shape: tuple[int, ...],
    at_volume_boundary: Optional[list[bool]] = None,
) -> np.ndarray:
    """Cosine-ramp weight volume for blending overlapping sub-crops.

    At internal subcrop seams the weight tapers from 1 (center) to 0 (edge),
    enabling smooth blending with the neighboring tile.  At volume boundaries
    (no neighbor), the weight stays at 1 so the flow is fully preserved.

    Args:
        shape: (d, h, w) of the subcrop.
        at_volume_boundary: List of 6 bools — [z_lo, z_hi, y_lo, y_hi, x_lo, x_hi].
            True means that face sits at the volume boundary (no overlap).
            If None, all faces are tapered (original behavior).

    Returns:
        [d, h, w] float32 weight volume.
    """
    if at_volume_boundary is None:
        at_volume_boundary = [False] * 6

    weights = []
    for ax, s in enumerate(shape):
        lo_boundary = at_volume_boundary[ax * 2]
        hi_boundary = at_volume_boundary[ax * 2 + 1]

        if lo_boundary and hi_boundary:
            # Both faces at volume boundary → no tapering needed
            w = np.ones(s)
        elif lo_boundary:
            # Only taper the high end: 1 → 0
            ramp = np.linspace(0, np.pi, s)
            w = (1 + np.cos(ramp)) / 2
        elif hi_boundary:
            # Only taper the low end: 0 → 1
            ramp = np.linspace(0, np.pi, s)
            w = (1 - np.cos(ramp)) / 2
        else:
            # Both faces internal → full cosine taper: 0 → 1 → 0
            ramp = np.linspace(0, np.pi, s)
            w = (1 - np.cos(ramp)) / 2

        weights.append(w)

    w3d = weights[0][:, None, None] * weights[1][None, :, None] * weights[2][None, None, :]
    return w3d.astype(np.float32)


def stitch_flows(
    volume_shape: tuple[int, int, int],
    subcrop_slices: list[tuple[slice, ...]],
    subcrop_flows: list[np.ndarray],
) -> np.ndarray:
    """Stitch per-subcrop flows into a full volume using cosine blending.

    Args:
        volume_shape: (D, H, W) of the full volume.
        subcrop_slices: List of (slice_d, slice_h, slice_w) tuples.
        subcrop_flows: List of [3, cd, ch, cw] flow arrays.

    Returns:
        stitched: [3, D, H, W] float32 blended flow field.
    """
    D, H, W = volume_shape
    flow_sum = np.zeros((3, D, H, W), dtype=np.float64)
    weight_sum = np.zeros((D, H, W), dtype=np.float64)

    for slc, flow in zip(subcrop_slices, subcrop_flows):
        crop_shape = flow.shape[1:]
        at_boundary = _boundary_flags(slc, volume_shape)
        w = cosine_blend_weight(crop_shape, at_volume_boundary=at_boundary)

        flow_sum[:, slc[0], slc[1], slc[2]] += flow * w[None, :, :, :]
        weight_sum[slc[0], slc[1], slc[2]] += w

    # Normalize
    weight_sum = np.clip(weight_sum, 1e-8, None)
    stitched = (flow_sum / weight_sum[None, :, :, :]).astype(np.float32)

    # Re-normalize to unit vectors where there's signal
    mag = np.sqrt((stitched ** 2).sum(axis=0))
    mag = np.clip(mag, 1e-8, None)
    stitched = stitched / mag[None, :, :, :]
    stitched[:, weight_sum < 1e-6] = 0  # zero where no data

    return stitched


def stitch_labels(
    volume_shape: tuple[int, int, int],
    subcrop_slices: list[tuple[slice, ...]],
    subcrop_labels: list[np.ndarray],
) -> np.ndarray:
    """Stitch per-subcrop label volumes using highest-weight-wins in overlaps.

    Each voxel gets the label from whichever subcrop has the highest cosine
    blend weight at that position (i.e. the subcrop whose center is closest).
    Labels are relabeled to avoid collisions across sub-crops.

    Args:
        volume_shape: (D, H, W) of the full volume.
        subcrop_slices: List of (slice_d, slice_h, slice_w) tuples.
        subcrop_labels: List of [cd, ch, cw] int label arrays.

    Returns:
        [D, H, W] int32 stitched labels.
    """
    D, H, W = volume_shape
    best_label = np.zeros((D, H, W), dtype=np.int32)
    best_weight = np.zeros((D, H, W), dtype=np.float32)

    offset = 0
    for slc, labels in zip(subcrop_slices, subcrop_labels):
        # Relabel to avoid collisions across sub-crops
        relabeled = labels.copy()
        relabeled[relabeled > 0] += offset
        offset = relabeled.max()

        crop_shape = labels.shape
        at_boundary = _boundary_flags(slc, volume_shape)
        w = cosine_blend_weight(crop_shape, at_volume_boundary=at_boundary)

        region = best_weight[slc[0], slc[1], slc[2]]
        update = w > region
        best_label[slc[0], slc[1], slc[2]] = np.where(
            update, relabeled, best_label[slc[0], slc[1], slc[2]]
        )
        best_weight[slc[0], slc[1], slc[2]] = np.maximum(region, w)

    return best_label


def stitch_volumes(
    volume_shape: tuple[int, int, int],
    subcrop_slices: list[tuple[slice, ...]],
    subcrop_volumes: list[np.ndarray],
    n_channels: int = 1,
) -> np.ndarray:
    """Stitch per-subcrop multi-channel volumes (e.g. affinities) using cosine blending.

    Args:
        volume_shape: (D, H, W) of the full volume.
        subcrop_slices: List of (slice_d, slice_h, slice_w) tuples.
        subcrop_volumes: List of [C, cd, ch, cw] arrays (or [cd, ch, cw] if n_channels=1).
        n_channels: Number of channels.

    Returns:
        [C, D, H, W] float64 blended volume (or [D, H, W] if n_channels=1).
    """
    D, H, W = volume_shape
    vol_sum = np.zeros((n_channels, D, H, W), dtype=np.float64)
    weight_sum = np.zeros((D, H, W), dtype=np.float64)

    for slc, vol in zip(subcrop_slices, subcrop_volumes):
        if vol.ndim == 3:
            vol = vol[None, ...]
        crop_shape = vol.shape[1:]
        at_boundary = _boundary_flags(slc, volume_shape)
        w = cosine_blend_weight(crop_shape, at_volume_boundary=at_boundary)

        vol_sum[:, slc[0], slc[1], slc[2]] += vol * w[None]
        weight_sum[slc[0], slc[1], slc[2]] += w

    weight_sum = np.clip(weight_sum, 1e-8, None)
    result = (vol_sum / weight_sum[None]).astype(np.float64)

    if n_channels == 1:
        return result[0]
    return result

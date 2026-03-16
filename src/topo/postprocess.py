"""Instance segmentation postprocessing: flow tracking + convergence clustering.

Converts predicted flow fields into instance labels via:
1. Euler integration — track each foreground voxel along the flow field
2. Convergence clustering — group voxels by where they end up (grid hash + union-find)
3. Optional cleanup — split disconnected components, orphan recovery, flow QC

Three morphology groups with different strategies:
- Group 1 (convex): standard tracking + clustering
- Group 2 (elongated): tracking + clustering + split disconnected components
- Group 3 (thin): tracking + clustering + split + orphan recovery
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
from scipy.ndimage import (
    center_of_mass as scipy_com,
    distance_transform_edt,
    label as nd_label,
)

from .config import EVALUATED_INSTANCE_CLASSES, get_postprocess_config

logger = logging.getLogger(__name__)


# ── Default postprocessing config ──────────────────────────────────────────




# ── Main entry point ──────────────────────────────────────────────────────

def run_instance_segmentation(
    semantic_pred: np.ndarray,
    flow_pred: np.ndarray,
    class_names: Optional[list[str]] = None,
    class_config: Optional[Dict[str, dict]] = None,
    flow_error_threshold: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Convert semantic predictions + flow fields into instance labels.

    Args:
        semantic_pred: [N_classes, D, H, W] per-class probabilities.
        flow_pred: [N*3, D, H, W] per-class flow predictions.
        class_names: List of N class names. Defaults to EVALUATED_INSTANCE_CLASSES.
        class_config: Per-class postprocessing config. Defaults to resolution-based config.
        flow_error_threshold: If > 0, discard masks whose recomputed flows
            differ from predicted flows by more than this MSE threshold.

    Returns:
        Dict mapping class name -> [D, H, W] int32 instance labels.
    """
    if class_names is None:
        class_names = EVALUATED_INSTANCE_CLASSES
    if class_config is None:
        class_config = get_postprocess_config()

    D, H, W = flow_pred.shape[1:]
    results = {}

    for cls_idx, cls_name in enumerate(class_names):
        if cls_name not in class_config:
            continue

        cfg = class_config[cls_name]
        mask = semantic_pred[cls_idx] > 0.5

        if mask.sum() == 0:
            results[cls_name] = np.zeros((D, H, W), dtype=np.int32)
            continue

        ch_start = cls_idx * 3
        class_flow = flow_pred[ch_start:ch_start + 3]

        g = cfg["group"]
        if g == 1:
            instances = _process_group1(mask, class_flow, cfg)
        elif g == 2:
            instances = _process_group2(mask, class_flow, cfg)
        elif g == 3:
            instances = _process_group3(mask, class_flow, cfg)
        else:
            instances = _process_group1(mask, class_flow, cfg)

        if flow_error_threshold > 0 and instances.max() > 0:
            instances = remove_bad_flow_masks(
                instances, class_flow, flow_error_threshold
            )

        results[cls_name] = instances

        n = instances.max()
        if n > 0:
            logger.debug("  %s (group %d): %d instances", cls_name, g, n)

    return results


# ── Single-class postprocessing ────────────────────────────────────────────

def postprocess_single(
    sem_mask: np.ndarray,
    flow: np.ndarray,
    n_steps: int = 100,
    step_size: float = 1.0,
    convergence_radius: float = 4.0,
    min_size: int = 50,
    group: int = 1,
) -> np.ndarray:
    """Postprocess a single class: track flows → cluster → cleanup.

    Args:
        sem_mask: [D, H, W] bool — foreground mask.
        flow: [3, D, H, W] float — flow field (dz, dy, dx).
        n_steps: Euler integration steps.
        step_size: Step size per iteration.
        convergence_radius: Merge clusters within this radius.
        min_size: Remove instances smaller than this.
        group: Morphology group (1=convex, 2=elongated, 3=thin).

    Returns:
        instances: [D, H, W] int32 — instance labels.
    """
    cfg = {
        "n_steps": n_steps,
        "step_size": step_size,
        "convergence_radius": convergence_radius,
        "min_size": min_size,
    }

    if group == 1:
        return _process_group1(sem_mask, flow, cfg)
    elif group == 2:
        return _process_group2(sem_mask, flow, cfg)
    elif group == 3:
        return _process_group3(sem_mask, flow, cfg)
    else:
        return _process_group1(sem_mask, flow, cfg)


# ── Flow tracking ─────────────────────────────────────────────────────────

def track_flows(
    sem_mask: np.ndarray,
    flow: np.ndarray,
    n_steps: int,
    step_size: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Euler integration: move each foreground voxel along the flow field.

    Args:
        sem_mask: [D, H, W] bool — foreground mask.
        flow: [3, D, H, W] float — predicted flow (dz, dy, dx).
        n_steps: Number of integration steps.
        step_size: Distance per step.

    Returns:
        fg_coords: [N, 3] int — original voxel positions.
        final_positions: [N, 3] float64 — where particles ended up.
    """
    D, H, W = sem_mask.shape
    fg_coords = np.argwhere(sem_mask)

    if len(fg_coords) == 0:
        return fg_coords, fg_coords.astype(np.float64)

    positions = fg_coords.astype(np.float64)

    for _ in range(n_steps):
        zi = np.clip(positions[:, 0], 0, D - 1).astype(int)
        yi = np.clip(positions[:, 1], 0, H - 1).astype(int)
        xi = np.clip(positions[:, 2], 0, W - 1).astype(int)

        positions[:, 0] += step_size * flow[0, zi, yi, xi]
        positions[:, 1] += step_size * flow[1, zi, yi, xi]
        positions[:, 2] += step_size * flow[2, zi, yi, xi]

        positions[:, 0] = np.clip(positions[:, 0], 0, D - 1)
        positions[:, 1] = np.clip(positions[:, 1], 0, H - 1)
        positions[:, 2] = np.clip(positions[:, 2], 0, W - 1)

    return fg_coords, positions


# ── Convergence clustering ─────────────────────────────────────────────────

def cluster_convergence(
    fg_coords: np.ndarray,
    final_positions: np.ndarray,
    volume_shape: tuple,
    convergence_radius: float,
    min_size: int,
) -> np.ndarray:
    """Cluster particles by final positions using grid hashing + union-find.

    Args:
        fg_coords: [N, 3] int — original voxel coords.
        final_positions: [N, 3] float — converged positions.
        volume_shape: (D, H, W).
        convergence_radius: Merge clusters within this radius.
        min_size: Remove clusters smaller than this.

    Returns:
        instances: [D, H, W] int32 — instance labels.
    """
    D, H, W = volume_shape

    if len(fg_coords) == 0:
        return np.zeros((D, H, W), dtype=np.int32)

    # Pass 1: grid-based initial clustering
    grid_res = max(1.0, convergence_radius / 2.0)
    grid_pos = np.round(final_positions / grid_res).astype(np.int64)

    gz = np.clip(grid_pos[:, 0], 0, int(D / grid_res))
    gy = np.clip(grid_pos[:, 1], 0, int(H / grid_res))
    gx = np.clip(grid_pos[:, 2], 0, int(W / grid_res))

    gh = int(H / grid_res) + 1
    gw = int(W / grid_res) + 1
    hashes = gz * gh * gw + gy * gw + gx

    unique_hashes, inverse = np.unique(hashes, return_inverse=True)
    n_clusters = len(unique_hashes)

    # Compute cluster centers
    cluster_centers = np.zeros((n_clusters, 3), dtype=np.float64)
    cluster_counts = np.zeros(n_clusters, dtype=np.int64)

    for i in range(len(fg_coords)):
        cid = inverse[i]
        cluster_centers[cid] += final_positions[i]
        cluster_counts[cid] += 1

    for c in range(n_clusters):
        if cluster_counts[c] > 0:
            cluster_centers[c] /= cluster_counts[c]

    # Pass 2: merge nearby clusters (union-find with grid locality)
    parent = list(range(n_clusters))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    grid_to_clusters = defaultdict(list)
    for c in range(n_clusters):
        key = tuple(np.round(cluster_centers[c] / convergence_radius).astype(int))
        grid_to_clusters[key].append(c)

    for key, clusters in grid_to_clusters.items():
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    neighbor_key = (key[0] + dz, key[1] + dy, key[2] + dx)
                    if neighbor_key not in grid_to_clusters:
                        continue
                    for ci in clusters:
                        for cj in grid_to_clusters[neighbor_key]:
                            if ci >= cj:
                                continue
                            ri, rj = find(ci), find(cj)
                            if ri == rj:
                                continue
                            dist = np.linalg.norm(
                                cluster_centers[ci] - cluster_centers[cj]
                            )
                            if dist < convergence_radius:
                                parent[rj] = ri

    # Remap cluster IDs
    root_to_id = {}
    next_id = 1
    final_ids = np.zeros(n_clusters, dtype=np.int32)
    for c in range(n_clusters):
        root = find(c)
        if root not in root_to_id:
            root_to_id[root] = next_id
            next_id += 1
        final_ids[c] = root_to_id[root]

    # Build output volume
    instances = np.zeros((D, H, W), dtype=np.int32)
    for i in range(len(fg_coords)):
        z, y, x = fg_coords[i]
        instances[z, y, x] = final_ids[inverse[i]]

    return filter_small(instances, min_size)


# ── Group processors ───────────────────────────────────────────────────────

def _process_group1(sem_mask, flow, cfg):
    """Group 1 (convex): standard tracking + clustering."""
    fg_coords, final_pos = track_flows(
        sem_mask, flow, n_steps=cfg["n_steps"], step_size=cfg["step_size"]
    )
    return cluster_convergence(
        fg_coords, final_pos, sem_mask.shape,
        cfg["convergence_radius"], cfg["min_size"]
    )


def _merge_adjacent_clusters(
    instances: np.ndarray,
    flow: np.ndarray,
    dot_threshold: float = 0.0,
) -> np.ndarray:
    """Merge adjacent clusters only when flow agrees at their boundary.

    After clustering, two clusters from the same elongated object may get
    different IDs (distant sinks).  Their voxel regions touch, AND the flow
    vectors on both sides of the boundary point in the same direction
    (positive dot product) — they're part of the same stream.

    Two distinct instances that happen to touch will have opposing flows
    at the boundary (each pointing toward its own sink), giving a negative
    dot product → they stay separate.

    Args:
        instances: [D, H, W] int32 — cluster labels.
        flow: [3, D, H, W] float — flow field used for clustering.
        dot_threshold: Minimum mean dot product at the boundary to merge.
            Default 0.0 means flows must be at least non-opposing.
    """
    max_inst = instances.max()
    if max_inst <= 1:
        return instances

    # Collect flow dot products at boundaries between each pair
    pair_dots = defaultdict(list)

    for axis in range(instances.ndim):
        slc_a = [slice(None)] * instances.ndim
        slc_b = [slice(None)] * instances.ndim
        slc_a[axis] = slice(None, -1)
        slc_b[axis] = slice(1, None)

        a = instances[tuple(slc_a)]
        b = instances[tuple(slc_b)]

        boundary = (a > 0) & (b > 0) & (a != b)
        if not boundary.any():
            continue

        # Get flow vectors on both sides of the boundary
        # Side A: use slc_a indexing into flow
        flow_slc_a = [slice(None)] + slc_a  # [3, ...] prefix
        flow_slc_b = [slice(None)] + slc_b
        fa = flow[tuple(flow_slc_a)]  # [3, ...]
        fb = flow[tuple(flow_slc_b)]  # [3, ...]

        # Dot product at each boundary voxel
        dot = (fa * fb).sum(axis=0)  # [...]

        ids_a = a[boundary]
        ids_b = b[boundary]
        dots = dot[boundary]

        # Accumulate per-pair
        for ia, ib, d in zip(ids_a, ids_b, dots):
            key = (min(ia, ib), max(ia, ib))
            pair_dots[key].append(d)

    # Union-find: only merge pairs with positive mean dot product
    parent = list(range(max_inst + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (id_a, id_b), dots in pair_dots.items():
        mean_dot = np.mean(dots)
        if mean_dot > dot_threshold:
            ra, rb = find(id_a), find(id_b)
            if ra != rb:
                parent[rb] = ra

    # Remap
    remap = {}
    next_id = 1
    output = np.zeros_like(instances)
    for inst_id in range(1, max_inst + 1):
        root = find(inst_id)
        if root not in remap:
            remap[root] = next_id
            next_id += 1
        output[instances == inst_id] = remap[root]

    return output


def _process_group2(sem_mask, flow, cfg):
    """Group 2 (elongated): tracking + clustering + merge adjacent + split."""
    fg_coords, final_pos = track_flows(
        sem_mask, flow, n_steps=cfg["n_steps"], step_size=cfg["step_size"]
    )
    instances = cluster_convergence(
        fg_coords, final_pos, sem_mask.shape,
        cfg["convergence_radius"], cfg["min_size"]
    )
    # Merge clusters whose voxel regions directly touch —
    # fixes over-splitting from multiple diffusion sinks in elongated objects.
    # This won't merge truly separate instances because the flow-based
    # clustering already placed a background gap (unlabeled voxels) between them.
    instances = _merge_adjacent_clusters(instances, flow)
    return split_disconnected(instances)


def _process_group3(sem_mask, flow, cfg):
    """Group 3 (thin): tracking + clustering + split + orphan recovery."""
    fg_coords, final_pos = track_flows(
        sem_mask, flow, n_steps=cfg["n_steps"], step_size=cfg["step_size"]
    )
    instances = cluster_convergence(
        fg_coords, final_pos, sem_mask.shape,
        cfg["convergence_radius"], cfg["min_size"]
    )
    # instances = split_disconnected(instances)

    # Orphan recovery: assign missed foreground voxels to nearest instance
    orphans = sem_mask & (instances == 0)
    if orphans.sum() > 0 and instances.max() > 0:
        _, nidx = distance_transform_edt(instances == 0, return_indices=True)
        instances[orphans] = instances[tuple(nidx[:, orphans])]

    return instances


# ── Utilities ──────────────────────────────────────────────────────────────

def split_disconnected(
    instances: np.ndarray,
    min_fragment_ratio: float = 0.1,
) -> np.ndarray:
    """Split disconnected components, absorbing small fragments into nearest.

    When an elongated instance (e.g. mito) gets clustered correctly but has
    a thin bridge that breaks connectivity, naive splitting creates spurious
    fragments.  Instead, we only keep components above ``min_fragment_ratio``
    of the parent instance size.  Smaller fragments are reassigned to the
    nearest large component via distance transform.

    Args:
        instances: [D, H, W] int32 — instance labels.
        min_fragment_ratio: Components smaller than this fraction of the
            parent instance are absorbed rather than split off.

    Returns:
        Relabeled instance volume.
    """
    output = np.zeros_like(instances)
    next_id = 1

    for inst_id in np.unique(instances):
        if inst_id == 0:
            continue
        mask = instances == inst_id
        total_size = mask.sum()
        components, n_comp = nd_label(mask)

        if n_comp == 1:
            output[mask] = next_id
            next_id += 1
            continue

        # Measure component sizes
        comp_sizes = {}
        for comp_id in range(1, n_comp + 1):
            comp_sizes[comp_id] = (components == comp_id).sum()

        min_size = max(1, int(total_size * min_fragment_ratio))

        # Separate large (keep) vs small (absorb) components
        large_comps = [c for c, s in comp_sizes.items() if s >= min_size]
        small_comps = [c for c, s in comp_sizes.items() if s < min_size]

        if not large_comps:
            # All fragments are small — keep them as one instance
            output[mask] = next_id
            next_id += 1
            continue

        # Assign each large component its own ID
        comp_to_id = {}
        for comp_id in large_comps:
            comp_to_id[comp_id] = next_id
            output[components == comp_id] = next_id
            next_id += 1

        # Absorb small fragments into nearest large component
        if small_comps:
            # Build a mask of large components
            large_mask = np.zeros_like(mask)
            for comp_id in large_comps:
                large_mask |= (components == comp_id)
            # Distance to nearest large component voxel
            _, nearest_idx = distance_transform_edt(
                ~large_mask, return_indices=True
            )
            for comp_id in small_comps:
                comp_mask = components == comp_id
                # Find which large component each small voxel is nearest to
                nz, ny, nx = nearest_idx[:, comp_mask]
                output[comp_mask] = output[nz, ny, nx]

    return output


def remove_bad_flow_masks(
    instances: np.ndarray,
    pred_flow: np.ndarray,
    threshold: float = 0.4,
) -> np.ndarray:
    """Discard masks whose recomputed flows disagree with predicted flows.

    For each instance, compute unit vectors from voxels to center of mass,
    then compare to the network's predicted flows. Masks with MSE > threshold
    are removed.

    Args:
        instances: [D, H, W] int32 — instance labels.
        pred_flow: [3, D, H, W] float — predicted flow field.
        threshold: MSE threshold above which masks are discarded.

    Returns:
        instances: [D, H, W] int32 — filtered instance labels.
    """
    max_id = instances.max()
    if max_id == 0:
        return instances

    bad_ids = []
    for inst_id in range(1, max_id + 1):
        mask = instances == inst_id
        n_vox = mask.sum()
        if n_vox == 0:
            continue

        com = np.array(scipy_com(mask), dtype=np.float64)
        coords = np.argwhere(mask)
        diff = com - coords
        mag = np.linalg.norm(diff, axis=1, keepdims=True)
        mag = np.clip(mag, 1e-8, None)
        gt_flow = diff / mag

        z, y, x = coords[:, 0], coords[:, 1], coords[:, 2]
        pf = np.stack([pred_flow[0, z, y, x],
                       pred_flow[1, z, y, x],
                       pred_flow[2, z, y, x]], axis=1)

        mse = np.mean((gt_flow - pf) ** 2)
        if mse > threshold:
            bad_ids.append(inst_id)

    if bad_ids:
        logger.debug("  flow QC: removing %d/%d masks (threshold=%.2f)",
                      len(bad_ids), max_id, threshold)
        for bid in bad_ids:
            instances[instances == bid] = 0
        instances = compact_relabel(instances)

    return instances


def compact_relabel(instances: np.ndarray) -> np.ndarray:
    """Relabel instance IDs to be contiguous starting from 1."""
    unique_ids = np.unique(instances)
    if len(unique_ids) <= 1:
        return np.zeros_like(instances)
    remap = np.zeros(unique_ids.max() + 1, dtype=np.int32)
    for new_id, old_id in enumerate(unique_ids[1:], start=1):
        remap[old_id] = new_id
    return remap[instances]


def filter_small(instances: np.ndarray, min_size: int) -> np.ndarray:
    """Remove instances smaller than min_size and compact relabel."""
    if min_size <= 0 or instances.max() == 0:
        return instances.astype(np.int32)

    flat = instances.ravel()
    counts = np.bincount(flat)
    remap = np.zeros(len(counts), dtype=np.int32)
    nxt = 1
    for iid in range(1, len(counts)):
        if counts[iid] >= min_size:
            remap[iid] = nxt
            nxt += 1
    return remap[flat].reshape(instances.shape)

"""GPU-accelerated instance segmentation postprocessing.

Drop-in replacement for postprocess.run_instance_segmentation that runs
Euler integration on GPU (torch tensors). Clustering remains on CPU
(union-find is inherently sequential).

Usage:
    from topo import run_instance_segmentation_gpu, postprocess_single_gpu

    instances = run_instance_segmentation_gpu(semantic_pred, flow_pred, device="cuda")
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from .config import EVALUATED_INSTANCE_CLASSES, get_postprocess_config
from .postprocess import (
    cluster_convergence,
    split_disconnected,
    remove_bad_flow_masks,
    _merge_adjacent_clusters,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_instance_segmentation_gpu(
    semantic_pred: np.ndarray,
    flow_pred: np.ndarray,
    device: str = "cuda",
    class_names: Optional[list[str]] = None,
    class_config: Optional[Dict[str, dict]] = None,
    flow_error_threshold: float = 0.0,
) -> Dict[str, np.ndarray]:
    """GPU-accelerated instance segmentation from flows.

    Args:
        semantic_pred: [N_classes, D, H, W] per-class probabilities (numpy).
        flow_pred: [N*3, D, H, W] per-class flow predictions (numpy).
        device: torch device string.
        class_names: List of N class names. Defaults to EVALUATED_INSTANCE_CLASSES.
        class_config: Per-class postprocessing config. Defaults to resolution-based config.
        flow_error_threshold: discard masks with flow MSE > threshold (0=disabled).

    Returns:
        Dict mapping class name -> [D, H, W] int32 instance labels.
    """
    if class_names is None:
        class_names = EVALUATED_INSTANCE_CLASSES
    if class_config is None:
        class_config = get_postprocess_config()

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    D, H, W = flow_pred.shape[1:]
    results = {}

    # Move full flow field to GPU once
    flow_t = torch.from_numpy(flow_pred).to(dev, dtype=torch.float32)

    for cls_idx, cls_name in enumerate(class_names):
        if cls_name not in class_config:
            continue

        cfg = class_config[cls_name]
        mask = semantic_pred[cls_idx] > 0.5

        if mask.sum() == 0:
            results[cls_name] = np.zeros((D, H, W), dtype=np.int32)
            continue

        ch_start = cls_idx * 3
        class_flow_t = flow_t[ch_start:ch_start + 3]  # [3, D, H, W]

        g = cfg["group"]
        instances = _process_group_gpu(mask, class_flow_t, dev, cfg, g)

        if flow_error_threshold > 0 and instances.max() > 0:
            class_flow_np = flow_pred[ch_start:ch_start + 3]
            instances = remove_bad_flow_masks(instances, class_flow_np, flow_error_threshold)

        results[cls_name] = instances

        n = instances.max()
        if n > 0:
            logger.debug("  %s (group %d): %d instances", cls_name, g, n)

    return results


# ---------------------------------------------------------------------------
# Single-class postprocessing (GPU)
# ---------------------------------------------------------------------------

def postprocess_single_gpu(
    sem_mask: np.ndarray,
    flow: np.ndarray,
    device: str = "cuda",
    n_steps: int = 100,
    step_size: float = 1.0,
    convergence_radius: float = 4.0,
    min_size: int = 50,
    group: int = 1,
) -> np.ndarray:
    """GPU postprocess a single class: track flows on GPU -> cluster on CPU -> cleanup.

    Args:
        sem_mask: [D, H, W] bool — foreground mask.
        flow: [3, D, H, W] float — flow field (dz, dy, dx).
        device: torch device string.
        n_steps: Euler integration steps.
        step_size: Step size per iteration.
        convergence_radius: Merge clusters within this radius.
        min_size: Remove instances smaller than this.
        group: Morphology group (1=convex, 2=elongated, 3=thin).

    Returns:
        instances: [D, H, W] int32 — instance labels.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    flow_t = torch.from_numpy(flow).to(dev, dtype=torch.float32)

    cfg = {
        "n_steps": n_steps,
        "step_size": step_size,
        "convergence_radius": convergence_radius,
        "min_size": min_size,
        "group": group,
    }

    return _process_group_gpu(sem_mask, flow_t, dev, cfg, group)


# ---------------------------------------------------------------------------
# GPU flow tracking
# ---------------------------------------------------------------------------

@torch.no_grad()
def track_flows_gpu(
    sem_mask: np.ndarray,
    flow: torch.Tensor,
    device: torch.device,
    n_steps: int,
    step_size: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Euler integration on GPU — all particles update in parallel.

    Args:
        sem_mask: [D, H, W] bool numpy — foreground mask.
        flow: [3, D, H, W] float torch tensor on device.
        device: torch device.
        n_steps: number of integration steps.
        step_size: distance per step.

    Returns:
        fg_coords: [N, 3] int numpy — original voxel positions.
        final_positions: [N, 3] float64 numpy — converged positions.
    """
    D, H, W = sem_mask.shape
    fg_coords_np = np.argwhere(sem_mask)  # [N, 3]

    if len(fg_coords_np) == 0:
        return fg_coords_np, fg_coords_np.astype(np.float64)

    # Move particle positions to GPU
    positions = torch.from_numpy(fg_coords_np).to(device, dtype=torch.float32)  # [N, 3]
    limits = torch.tensor([D - 1, H - 1, W - 1], device=device, dtype=torch.float32)

    for _ in range(n_steps):
        # Clamp and convert to integer indices for lookup
        idx = positions.clamp(min=0)
        idx = torch.min(idx, limits).long()  # [N, 3]

        zi = idx[:, 0]
        yi = idx[:, 1]
        xi = idx[:, 2]

        # Gather flow vectors for all particles at once
        dz = flow[0, zi, yi, xi]
        dy = flow[1, zi, yi, xi]
        dx = flow[2, zi, yi, xi]

        # Update positions
        positions[:, 0] += step_size * dz
        positions[:, 1] += step_size * dy
        positions[:, 2] += step_size * dx

        # Clamp to volume
        positions.clamp_(min=0)
        torch.min(positions, limits, out=positions)

    return fg_coords_np, positions.cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Group processors (GPU tracking + CPU clustering)
# ---------------------------------------------------------------------------

def _process_group_gpu(
    sem_mask: np.ndarray,
    flow_t: torch.Tensor,
    device: torch.device,
    cfg: dict,
    group: int,
) -> np.ndarray:
    """Process a single class with GPU tracking and CPU clustering."""
    fg_coords, final_pos = track_flows_gpu(
        sem_mask, flow_t, device,
        n_steps=cfg["n_steps"],
        step_size=cfg.get("step_size", 1.0),
    )

    instances = cluster_convergence(
        fg_coords, final_pos, sem_mask.shape,
        cfg["convergence_radius"], cfg["min_size"],
    )

    if group == 2:
        flow_np = flow_t.cpu().numpy()
        instances = _merge_adjacent_clusters(instances, flow_np)
        instances = split_disconnected(instances)
    elif group == 3:
        # Orphan recovery: assign missed foreground voxels to nearest instance
        orphans = sem_mask & (instances == 0)
        if orphans.sum() > 0 and instances.max() > 0:
            _, nidx = distance_transform_edt(instances == 0, return_indices=True)
            instances[orphans] = instances[tuple(nidx[:, orphans])]

    return instances

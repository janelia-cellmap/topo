"""GPU-accelerated flow field generation using PyTorch conv3d.

Drop-in replacement for the CPU flow functions, running ~50-100x faster
by expressing the heat equation diffusion as 3D convolutions on GPU.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional

from .config import EVALUATED_INSTANCE_CLASSES, INSTANCE_CLASS_CONFIG


# ---------------------------------------------------------------------------
# Laplacian kernel (cached per device)
# ---------------------------------------------------------------------------

_kernel_cache: dict[str, torch.Tensor] = {}


def _build_laplacian_kernel(device: torch.device) -> torch.Tensor:
    """6-connected 3D Laplacian kernel."""
    kernel = torch.zeros(1, 1, 3, 3, 3, device=device)
    kernel[0, 0, 1, 1, 1] = -6.0
    kernel[0, 0, 0, 1, 1] = 1.0
    kernel[0, 0, 2, 1, 1] = 1.0
    kernel[0, 0, 1, 0, 1] = 1.0
    kernel[0, 0, 1, 2, 1] = 1.0
    kernel[0, 0, 1, 1, 0] = 1.0
    kernel[0, 0, 1, 1, 2] = 1.0
    return kernel


def _get_kernel(device: torch.device) -> torch.Tensor:
    key = str(device)
    if key not in _kernel_cache:
        _kernel_cache[key] = _build_laplacian_kernel(device)
    return _kernel_cache[key]


# ---------------------------------------------------------------------------
# Core diffusion
# ---------------------------------------------------------------------------


def _diffuse_field_gpu(
    field: torch.Tensor,
    update_mask: torch.Tensor,
    n_iter: int,
    kernel: torch.Tensor,
) -> torch.Tensor:
    """Run heat equation diffusion on GPU.

    Args:
        field: [1, 1, D, H, W] float32.
        update_mask: [1, 1, D, H, W] bool — where field may evolve.
        n_iter: number of iterations.
        kernel: [1, 1, 3, 3, 3] Laplacian kernel.

    Returns:
        Diffused field [1, 1, D, H, W].
    """
    dirichlet_mask = ~update_mask

    for _ in range(n_iter):
        padded = F.pad(field, (1, 1, 1, 1, 1, 1), mode='replicate')
        laplacian = F.conv3d(padded, kernel)
        field = field + (1.0 / 6.0) * laplacian
        field[dirichlet_mask] = 0.0

    return field


# ---------------------------------------------------------------------------
# Per-class flow generators
# ---------------------------------------------------------------------------


def generate_direct_flows_gpu(
    instance_mask: torch.Tensor,
) -> torch.Tensor:
    """GPU direct flows: unit vectors to center of mass.

    Args:
        instance_mask: [D, H, W] int — instance IDs (0 = background).

    Returns:
        flows: [3, D, H, W] float32.
    """
    device = instance_mask.device
    D, H, W = instance_mask.shape
    flows = torch.zeros(3, D, H, W, device=device)

    inst_ids = instance_mask.unique()
    inst_ids = inst_ids[inst_ids > 0]
    if len(inst_ids) == 0:
        return flows

    coords_z = torch.arange(D, device=device, dtype=torch.float32).view(-1, 1, 1).expand(D, H, W)
    coords_y = torch.arange(H, device=device, dtype=torch.float32).view(1, -1, 1).expand(D, H, W)
    coords_x = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, -1).expand(D, H, W)

    for inst_id in inst_ids:
        mask = instance_mask == inst_id
        if not mask.any():
            continue

        n_vox = mask.sum().float()
        com_z = coords_z[mask].sum() / n_vox
        com_y = coords_y[mask].sum() / n_vox
        com_x = coords_x[mask].sum() / n_vox

        dz = com_z - coords_z
        dy = com_y - coords_y
        dx = com_x - coords_x
        mag = torch.sqrt(dz ** 2 + dy ** 2 + dx ** 2).clamp(min=1e-8)

        flows[0][mask] = (dz / mag)[mask]
        flows[1][mask] = (dy / mag)[mask]
        flows[2][mask] = (dx / mag)[mask]

        # Sink at center
        z0 = com_z.round().long().clamp(0, D - 1)
        y0 = com_y.round().long().clamp(0, H - 1)
        x0 = com_x.round().long().clamp(0, W - 1)
        if mask[z0, y0, x0]:
            flows[:, z0, y0, x0] = 0.0

    return flows


def generate_diffusion_flows_gpu(
    instance_mask: torch.Tensor,
    n_iter: int = 200,
    adaptive_iters: bool = True,
    adaptive_factor: int = 6,
    spatial_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """GPU diffusion flows: topology-aware via heat equation.

    Args:
        instance_mask: [D, H, W] int — instance IDs (0 = background).
        n_iter: Max diffusion iterations.
        adaptive_iters: Scale iterations with instance extent.
        adaptive_factor: Multiplier for adaptive iteration count.
        spatial_mask: [D, H, W] float or None — 1 inside annotated region.

    Returns:
        flows: [3, D, H, W] float32.
    """
    device = instance_mask.device
    D, H, W = instance_mask.shape
    kernel = _get_kernel(device)
    flows = torch.zeros(3, D, H, W, device=device)

    inst_ids = instance_mask.unique()
    inst_ids = inst_ids[inst_ids > 0]
    if len(inst_ids) == 0:
        return flows

    for inst_id in inst_ids:
        mask = instance_mask == inst_id
        if not mask.any():
            continue

        # Crop to bounding box
        where_z, where_y, where_x = torch.where(mask)
        pad = 2
        z0 = max(0, where_z.min().item() - pad)
        z1 = min(D, where_z.max().item() + pad + 1)
        y0 = max(0, where_y.min().item() - pad)
        y1 = min(H, where_y.max().item() + pad + 1)
        x0 = max(0, where_x.min().item() - pad)
        x1 = min(W, where_x.max().item() + pad + 1)

        crop_mask = mask[z0:z1, y0:y1, x0:x1]
        cd, ch, cw = crop_mask.shape

        if spatial_mask is not None:
            crop_spatial = spatial_mask[z0:z1, y0:y1, x0:x1] > 0.5
            update_mask = crop_mask | ~crop_spatial
        else:
            update_mask = crop_mask

        if adaptive_iters:
            max_extent = max(cd, ch, cw)
            inst_n_iter = min(adaptive_factor * max_extent, n_iter)
        else:
            inst_n_iter = n_iter

        update_5d = update_mask.unsqueeze(0).unsqueeze(0)
        inst_flows = torch.zeros(3, cd, ch, cw, device=device)

        for axis in range(3):
            field = torch.zeros(1, 1, cd, ch, cw, device=device)
            if axis == 0:
                coords = torch.arange(z0, z0 + cd, device=device, dtype=torch.float32).view(-1, 1, 1).expand(cd, ch, cw)
            elif axis == 1:
                coords = torch.arange(y0, y0 + ch, device=device, dtype=torch.float32).view(1, -1, 1).expand(cd, ch, cw)
            else:
                coords = torch.arange(x0, x0 + cw, device=device, dtype=torch.float32).view(1, 1, -1).expand(cd, ch, cw)

            field[0, 0][crop_mask] = coords[crop_mask]
            field = _diffuse_field_gpu(field, update_5d, inst_n_iter, kernel)

            # Central difference gradient
            f = field[0, 0]
            grad = torch.zeros_like(f)
            if axis == 0 and cd > 1:
                grad[1:-1] = (f[2:] - f[:-2]) / 2.0
                grad[0] = f[1] - f[0]
                grad[-1] = f[-1] - f[-2]
            elif axis == 1 and ch > 1:
                grad[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / 2.0
                grad[:, 0] = f[:, 1] - f[:, 0]
                grad[:, -1] = f[:, -1] - f[:, -2]
            elif axis == 2 and cw > 1:
                grad[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / 2.0
                grad[:, :, 0] = f[:, :, 1] - f[:, :, 0]
                grad[:, :, -1] = f[:, :, -1] - f[:, :, -2]

            inst_flows[axis] = grad

        mag = torch.sqrt(inst_flows[0] ** 2 + inst_flows[1] ** 2 + inst_flows[2] ** 2).clamp(min=1e-8)
        for axis in range(3):
            flows[axis, z0:z1, y0:y1, x0:x1][crop_mask] = (inst_flows[axis] / mag)[crop_mask]

    return flows


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_flow_targets_gpu(
    instance_ids: torch.Tensor,
    class_names: list[str] | None = None,
    class_config: dict | None = None,
    spatial_mask: torch.Tensor | None = None,
    diffusion_iters: int = 50,
    adaptive_iters: bool = True,
    adaptive_factor: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-class flow targets on GPU.

    Drop-in replacement for :func:`topo.compute_flow_targets` but operates
    on GPU tensors.

    Args:
        instance_ids: [B, N, D, H, W] int — instance IDs per class.
        class_names: List of N class names.
        class_config: Per-class config dict.
        spatial_mask: [B, 1, D, H, W] float or None.
        diffusion_iters: Max diffusion iterations.
        adaptive_iters: Scale iterations with instance extent.
        adaptive_factor: Multiplier for adaptive iteration count.

    Returns:
        flows: [B, N*3, D, H, W] float32.
        class_fg: [B, N, D, H, W] float32.
    """
    if class_names is None:
        class_names = EVALUATED_INSTANCE_CLASSES
    if class_config is None:
        class_config = INSTANCE_CLASS_CONFIG

    device = instance_ids.device
    B, N, D, H, W = instance_ids.shape
    kernel = _get_kernel(device)

    flows = torch.zeros(B, N * 3, D, H, W, device=device)
    class_fg = torch.zeros(B, N, D, H, W, device=device)

    for b in range(B):
        sp = spatial_mask[b, 0] if spatial_mask is not None else None

        for c, cls_name in enumerate(class_names):
            ids = instance_ids[b, c]
            fg = ids > 0
            class_fg[b, c] = fg.float()

            if not fg.any():
                continue

            cfg = class_config.get(cls_name, {"flow_type": "direct"})
            flow_type = cfg.get("flow_type", "direct")
            ch_start = c * 3

            if flow_type == "diffusion":
                n_iter = cfg.get("diffusion_iters", diffusion_iters)
                cls_flows = generate_diffusion_flows_gpu(
                    ids, n_iter,
                    adaptive_iters=adaptive_iters,
                    adaptive_factor=adaptive_factor,
                    spatial_mask=sp,
                )
            else:
                cls_flows = generate_direct_flows_gpu(ids)

            flows[b, ch_start:ch_start + 3] = cls_flows

            if cfg.get("boundary_only", False):
                bw = cfg.get("boundary_width", 20)
                annotated = (sp > 0.5) if sp is not None else torch.ones_like(fg)
                annotated_bg = annotated & ~fg
                if annotated_bg.any():
                    from scipy.ndimage import distance_transform_edt
                    dist = distance_transform_edt((~annotated_bg).cpu().numpy())
                    near_boundary = torch.from_numpy(dist <= bw).to(device)
                    class_fg[b, c] = (fg & near_boundary).float()
                else:
                    class_fg[b, c] = 0.0

    return flows, class_fg

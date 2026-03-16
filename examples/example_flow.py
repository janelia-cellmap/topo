#!/usr/bin/env python3
"""topo example: generate flow fields and run postprocessing on real CellMap data.

Loads:
- Mito instances at ~16nm resolution (non-convex → diffusion flows)
- Nuc instance  at ~64nm resolution (convex → direct flows)

Three flow visualization styles:
1. Quiver (arrow) plots
2. HSV color wheel (hue = direction, value = magnitude)
3. Streamlines (particle trajectories)

Saves visualizations to examples/imgs/
"""

import os
import numpy as np
import zarr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, hsv_to_rgb

from topo import generate_direct_flows, generate_diffusion_flows, postprocess_single
from topo.config import get_instance_class_config, get_postprocess_config

# ── paths ──────────────────────────────────────────────────────────────────
DATA_ROOT = "/nrs/cellmap/data/jrc_hela-2/jrc_hela-2.zarr/recon-1/labels/groundtruth/crop155"
MITO_PATH = f"{DATA_ROOT}/mito/s3"  # ~16nm (20.96, 16.0, 16.0)
NUC_PATH = f"{DATA_ROOT}/nuc/s5"  # ~64nm (83.84, 64.0, 64.0)
OUT_DIR = os.path.join(os.path.dirname(__file__), "imgs")

os.makedirs(OUT_DIR, exist_ok=True)


# ── helpers ────────────────────────────────────────────────────────────────

def instance_cmap(n=20):
    """Random but reproducible colormap for instance labels."""
    rng = np.random.RandomState(42)
    colors = np.zeros((n + 1, 4))
    colors[0] = [0, 0, 0, 1]  # background = black
    for i in range(1, n + 1):
        colors[i] = [*rng.rand(3) * 0.7 + 0.3, 1.0]
    return ListedColormap(colors)


def _extract_slice(instance_mask, flows, slice_idx, axis):
    """Extract a 2D slice of instance mask and 2D flow components."""
    if axis == 0:
        inst_slice = instance_mask[slice_idx]
        fy = flows[1, slice_idx]
        fx = flows[2, slice_idx]
        ylabel, xlabel = "Y", "X"
    elif axis == 1:
        inst_slice = instance_mask[:, slice_idx]
        fy = flows[0, :, slice_idx]
        fx = flows[2, :, slice_idx]
        ylabel, xlabel = "Z", "X"
    else:
        inst_slice = instance_mask[:, :, slice_idx]
        fy = flows[0, :, :, slice_idx]
        fx = flows[1, :, :, slice_idx]
        ylabel, xlabel = "Z", "Y"
    return inst_slice, fy, fx, ylabel, xlabel


def _flow_to_hsv(fy, fx, fg_mask):
    """Convert 2D flow to HSV RGB image.

    Hue = direction (angle), Saturation = 1, Value = magnitude.
    Background is black.
    """
    angle = np.arctan2(fy, fx)  # [-pi, pi]
    hue = (angle + np.pi) / (2 * np.pi)  # [0, 1]
    mag = np.sqrt(fy ** 2 + fx ** 2)
    mag_norm = mag / (mag.max() + 1e-8)

    H, W = fy.shape
    hsv = np.zeros((H, W, 3), dtype=np.float32)
    hsv[..., 0] = hue
    hsv[..., 1] = 1.0
    hsv[..., 2] = mag_norm

    rgb = hsv_to_rgb(hsv)
    rgb[~fg_mask] = 0.0
    return rgb


def _draw_color_wheel(ax, size=60):
    """Draw a small HSV color wheel legend in the corner of an axis."""
    angles = np.linspace(0, 2 * np.pi, 256)
    radii = np.linspace(0, 1, size)
    r, theta = np.meshgrid(radii, angles)

    hue = theta / (2 * np.pi)
    sat = np.ones_like(r)
    val = r

    hsv = np.stack([hue, sat, val], axis=-1)
    rgb = hsv_to_rgb(hsv)

    # Convert polar to cartesian image
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Create a small square image
    wh = size * 2 + 1
    img = np.zeros((wh, wh, 3))
    ix = ((x + 1) / 2 * (wh - 1)).astype(int)
    iy = ((y + 1) / 2 * (wh - 1)).astype(int)
    valid = (ix >= 0) & (ix < wh) & (iy >= 0) & (iy < wh)
    img[iy[valid], ix[valid]] = rgb[valid]

    # Place as inset
    bbox = ax.get_position()
    inset = ax.figure.add_axes([bbox.x1 - 0.08, bbox.y1 - 0.12, 0.07, 0.10])
    inset.imshow(img, origin="lower")
    inset.axis("off")
    inset.set_title("dir", fontsize=7, pad=1)


# ── Visualization style 1: Quiver (arrows) ────────────────────────────────

def plot_flow_quiver(
    instance_mask, flows, slice_idx, axis, title, filename,
    arrow_step=3, scale=15,
):
    """Instance mask + sparse quiver arrows."""
    inst_slice, fy, fx, ylabel, xlabel = _extract_slice(
        instance_mask, flows, slice_idx, axis)

    H, W = inst_slice.shape
    cmap = instance_cmap(int(instance_mask.max()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(inst_slice, cmap=cmap, interpolation="nearest",
                   vmin=0, vmax=instance_mask.max())
    axes[0].set_title("Instance IDs")
    axes[0].set_ylabel(ylabel)
    axes[0].set_xlabel(xlabel)

    axes[1].imshow(inst_slice, cmap=cmap, interpolation="nearest",
                   vmin=0, vmax=instance_mask.max(), alpha=0.5)

    yy, xx = np.mgrid[0:H, 0:W]
    fg = inst_slice > 0
    step_mask = np.zeros_like(fg)
    step_mask[::arrow_step, ::arrow_step] = True
    show = fg & step_mask

    axes[1].quiver(
        xx[show], yy[show], fx[show], -fy[show],
        color="white", scale=scale, width=0.004, headwidth=3, headlength=4,
    )
    axes[1].set_title("Flow Field (quiver)")
    axes[1].set_ylabel(ylabel)
    axes[1].set_xlabel(xlabel)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


# ── Visualization style 2: HSV color wheel ────────────────────────────────

def plot_flow_hsv(
    instance_mask, flows, slice_idx, axis, title, filename,
):
    """Instance mask + HSV-encoded flow direction/magnitude."""
    inst_slice, fy, fx, ylabel, xlabel = _extract_slice(
        instance_mask, flows, slice_idx, axis)

    fg = inst_slice > 0
    rgb = _flow_to_hsv(fy, fx, fg)
    cmap = instance_cmap(int(instance_mask.max()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(inst_slice, cmap=cmap, interpolation="nearest",
                   vmin=0, vmax=instance_mask.max())
    axes[0].set_title("Instance IDs")
    axes[0].set_ylabel(ylabel)
    axes[0].set_xlabel(xlabel)

    axes[1].imshow(rgb, interpolation="nearest")
    axes[1].set_title("Flow Field (HSV: hue=direction, brightness=magnitude)")
    axes[1].set_ylabel(ylabel)
    axes[1].set_xlabel(xlabel)
    _draw_color_wheel(axes[1])

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


# ── Visualization style 3: Streamlines ─────────────────────────────────────

def plot_flow_streamlines(
    instance_mask, flows, slice_idx, axis, title, filename,
    density=2.0,
):
    """Instance mask + streamlines showing particle trajectories."""
    inst_slice, fy, fx, ylabel, xlabel = _extract_slice(
        instance_mask, flows, slice_idx, axis)

    H, W = inst_slice.shape
    fg = inst_slice > 0
    cmap = instance_cmap(int(instance_mask.max()))

    # Mask out background for streamplot (set to 0 so streamlines stop)
    fx_masked = np.where(fg, fx, 0.0)
    fy_masked = np.where(fg, fy, 0.0)

    speed = np.sqrt(fx_masked ** 2 + fy_masked ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(inst_slice, cmap=cmap, interpolation="nearest",
                   vmin=0, vmax=instance_mask.max())
    axes[0].set_title("Instance IDs")
    axes[0].set_ylabel(ylabel)
    axes[0].set_xlabel(xlabel)

    axes[1].imshow(inst_slice, cmap=cmap, interpolation="nearest",
                   vmin=0, vmax=instance_mask.max(), alpha=0.3)

    xx = np.arange(W).astype(float)
    yy = np.arange(H).astype(float)

    # streamplot expects (x, y, u, v) where u=dx, v=dy
    # matplotlib y-axis is inverted, so negate fy
    axes[1].streamplot(
        xx, yy, fx_masked, fy_masked,
        color=speed, cmap="plasma",
        density=density, linewidth=0.8, arrowsize=0.8,
        broken_streamlines=True,
    )
    axes[1].set_xlim(0, W - 1)
    axes[1].set_ylim(H - 1, 0)
    axes[1].set_title("Flow Field (streamlines)")
    axes[1].set_ylabel(ylabel)
    axes[1].set_xlabel(xlabel)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


# ── Postprocessing comparison ──────────────────────────────────────────────

def plot_comparison(gt_mask, post_mask, title, filename):
    """Side-by-side: ground truth instances vs postprocessed instances."""
    n_max = max(int(gt_mask.max()), int(post_mask.max()))
    cmap = instance_cmap(n_max)

    mid = gt_mask.shape[0] // 2

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(gt_mask[mid], cmap=cmap, interpolation="nearest",
                   vmin=0, vmax=n_max)
    axes[0].set_title("Ground Truth Instances")

    axes[1].imshow(post_mask[mid], cmap=cmap, interpolation="nearest",
                   vmin=0, vmax=n_max)
    axes[1].set_title("Postprocessed (from GT flows)")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("topo example: flow generation + postprocessing")
    print("=" * 60)

    # ── 1. Mito at ~16nm (diffusion flows) ─────────────────────────────
    print("\n[1/8] Loading mito instances (~16nm, crop155)...")
    mito_mask = zarr.open(MITO_PATH, "r")[:]
    print(f"  Shape: {mito_mask.shape}")
    print(f"  Instances: {np.unique(mito_mask[mito_mask > 0])}")

    mito_flow_cfg = get_instance_class_config(16)["mito"]
    mito_post_cfg = get_postprocess_config(16)["mito"]

    print("\n[2/8] Generating diffusion flows for mito (non-convex)...")
    mito_flows = generate_diffusion_flows(
        mito_mask, n_iter=mito_flow_cfg.get("diffusion_iters", 200),
    )
    print(f"  Flow shape: {mito_flows.shape}")
    fg = mito_mask > 0
    mag = np.sqrt((mito_flows[:, fg] ** 2).sum(axis=0))
    print(f"  Mean flow magnitude on FG: {mag.mean():.4f}")

    mid_z = mito_mask.shape[0] // 2

    print("\n[3/8] Saving mito flow visualizations (3 styles)...")

    # Style 1: Quiver
    plot_flow_quiver(
        mito_mask, mito_flows, mid_z, axis=0,
        title="Mito — Quiver (~16nm, XY slice)",
        filename="mito_flow_quiver.png",
        arrow_step=2, scale=20,
    )

    # Style 2: HSV
    plot_flow_hsv(
        mito_mask, mito_flows, mid_z, axis=0,
        title="Mito — HSV Color Wheel (~16nm, XY slice)",
        filename="mito_flow_hsv.png",
    )

    # Style 3: Streamlines
    plot_flow_streamlines(
        mito_mask, mito_flows, mid_z, axis=0,
        title="Mito — Streamlines (~16nm, XY slice)",
        filename="mito_flow_streamlines.png",
        density=2.5,
    )

    print("\n[4/8] Postprocessing mito (Euler integration + clustering)...")
    mito_post = postprocess_single(
        sem_mask=(mito_mask > 0),
        flow=mito_flows,
        n_steps=mito_post_cfg["n_steps"],
        step_size=mito_post_cfg["step_size"],
        convergence_radius=mito_post_cfg["convergence_radius"],
        min_size=mito_post_cfg["min_size"],
        group=mito_post_cfg["group"],
    )
    print(f"  GT instances: {mito_mask.max()}")
    print(f"  Recovered instances: {mito_post.max()}")
    plot_comparison(
        mito_mask, mito_post,
        title="Mito — GT vs Postprocessed (~16nm)",
        filename="mito_postprocess.png",
    )

    # ── 2. Nuc at ~64nm (direct flows) ─────────────────────────────────
    print("\n[5/8] Loading nuc instance (~64nm, crop155)...")
    nuc_mask = zarr.open(NUC_PATH, "r")[:]
    print(f"  Shape: {nuc_mask.shape}")
    print(f"  Instances: {np.unique(nuc_mask[nuc_mask > 0])}")

    nuc_post_cfg = get_postprocess_config(64)["nuc"]

    # Build spatial mask: 1 in interior, 0 at crop boundary faces
    nuc_spatial = np.ones(nuc_mask.shape, dtype=np.float32)
    nuc_spatial[0, :, :] = 0; nuc_spatial[-1, :, :] = 0
    nuc_spatial[:, 0, :] = 0; nuc_spatial[:, -1, :] = 0
    nuc_spatial[:, :, 0] = 0; nuc_spatial[:, :, -1] = 0

    print("  Generating direct flows for nuc (convex, with spatial mask)...")
    nuc_flows = generate_direct_flows(nuc_mask, spatial_mask=nuc_spatial)
    print(f"  Flow shape: {nuc_flows.shape}")

    mid_z_nuc = nuc_mask.shape[0] // 2

    print("\n[6/8] Saving nuc flow visualizations (3 styles)...")

    plot_flow_quiver(
        nuc_mask, nuc_flows, mid_z_nuc, axis=0,
        title="Nuc — Quiver (~64nm, XY slice)",
        filename="nuc_flow_quiver.png",
        arrow_step=1, scale=10,
    )

    plot_flow_hsv(
        nuc_mask, nuc_flows, mid_z_nuc, axis=0,
        title="Nuc — HSV Color Wheel (~64nm, XY slice)",
        filename="nuc_flow_hsv.png",
    )

    plot_flow_streamlines(
        nuc_mask, nuc_flows, mid_z_nuc, axis=0,
        title="Nuc — Streamlines (~64nm, XY slice)",
        filename="nuc_flow_streamlines.png",
        density=1.5,
    )

    print("\n[7/8] Postprocessing nuc...")
    nuc_post = postprocess_single(
        sem_mask=(nuc_mask > 0),
        flow=nuc_flows,
        n_steps=nuc_post_cfg["n_steps"],
        step_size=nuc_post_cfg["step_size"],
        convergence_radius=nuc_post_cfg["convergence_radius"],
        min_size=nuc_post_cfg.get("min_size", 10),
        group=nuc_post_cfg["group"],
    )
    print(f"  GT instances: {nuc_mask.max()}")
    print(f"  Recovered instances: {nuc_post.max()}")
    plot_comparison(
        nuc_mask, nuc_post,
        title="Nuc — GT vs Postprocessed (~64nm)",
        filename="nuc_postprocess.png",
    )

    print("\n[8/8] Done!")
    print("=" * 60)
    print(f"All images saved to {OUT_DIR}/")
    print("  Flow visualizations (3 styles per class):")
    print("    *_flow_quiver.png       — sparse arrow plots")
    print("    *_flow_hsv.png          — HSV color wheel (hue=dir, bright=mag)")
    print("    *_flow_streamlines.png  — particle trajectory streamlines")
    print("  Postprocessing:")
    print("    *_postprocess.png       — GT vs recovered instances")
    print("=" * 60)


if __name__ == "__main__":
    main()

# GT Instance Statistics Analysis

## Goal

Derive **data-driven postprocessing parameters** (`min_size`, `n_steps`, `convergence_radius`) for the topo flow-based instance segmentation pipeline, per organelle class and per resolution.

## Method

We scan **all available CellMap ground-truth crops** across all datasets and resolutions (8, 16, 32, 64 nm). For each instance in each crop, we measure:

| Metric | Description | Informs |
|--------|-------------|---------|
| **Volume** (voxels) | Total voxel count of the instance | `min_size` — instances smaller than this are filtered as fragments |
| **Bbox max extent** | Max of the bounding box dimensions (D, H, W) in voxels | `n_steps` — Euler integration needs enough steps to traverse the longest axis |
| **NN COM distance** | Nearest-neighbor distance between centers-of-mass of same-class instances in the same crop | `convergence_radius` — upper bound to avoid merging distinct instances |

### Recommendation formulas

```
min_size        = max(5, volume_p5 * 0.5)
n_steps         = max(20, bbox_max_p95 * 2)
convergence_radius = max(1.5, min(nn_dist_p5 / 2, bbox_max_median / 3))
```

## Dataset coverage

| Class | 8nm | 16nm | 32nm | 64nm |
|-------|-----|------|------|------|
| nuc | 392 inst / 93 crops | 493 / 131 | 851 / 146 | 646 / 98 |
| ves | 5426 / 197 | 5413 / 193 | 5209 / 179 | 428 / 66 |
| endo | 2964 / 192 | 3007 / 190 | 3492 / 178 | 1482 / 88 |
| lyso | 2496 / 116 | 7253 / 164 | 4825 / 159 | 3968 / 117 |
| ld | 3119 / 70 | 2278 / 75 | 4006 / 81 | 2168 / 61 |
| perox | 2151 / 65 | 1978 / 66 | 4904 / 76 | 3207 / 60 |
| mito | 5706 / 198 | 12369 / 250 | 9412 / 251 | 6460 / 169 |
| mt | 786 / 82 | 786 / 78 | 1070 / 70 | 49 / 9 |
| cell | 5230 / 281 | 4531 / 277 | 4167 / 264 | 2347 / 149 |

## Results per class

### NUC (nucleus)

Large, convex blobs. Median volume ranges from ~86K (64nm) to ~966K (8nm) voxels. Bbox max extent median: 93-174 voxels depending on resolution.

| Resolution | min_size | n_steps | conv_radius | Notes |
|-----------|----------|---------|-------------|-------|
| 8nm | 5331 | 943 | 1.5 | |
| 16nm | 1154 | 800 | 1.5 | |
| 32nm | 225 | 662 | 1.5 | |
| 64nm | 266 | 336 | 1.5 | |

### VES (vesicle)

Very small, dense organelles. Median volume: 1-163 voxels. At 32nm+ most vesicles are 1-2 voxels — flow-based segmentation has limited value at coarse resolutions.

| Resolution | min_size | n_steps | conv_radius | Notes |
|-----------|----------|---------|-------------|-------|
| 8nm | 14 | 28 | 2.7 | Best resolution for vesicles |
| 16nm | 5 | 20 | 1.5 | |
| 32nm | 5 | 20 | 1.5 | Most vesicles are 1-2 voxels |
| 64nm | 5 | 20 | 1.5 | Too coarse for vesicles |

### ENDO (endosome)

Medium-sized, somewhat irregular shapes. Median bbox max: 2-21 voxels.

| Resolution | min_size | n_steps | conv_radius | Notes |
|-----------|----------|---------|-------------|-------|
| 8nm | 81 | 112 | 5.2 | |
| 16nm | 9 | 56 | 1.5 | |
| 32nm | 5 | 26 | 1.5 | |
| 64nm | 5 | 20 | 1.5 | |

### LYSO (lysosome)

Medium-sized, roughly spherical. Similar to endosomes but slightly larger.

| Resolution | min_size | n_steps | conv_radius | Notes |
|-----------|----------|---------|-------------|-------|
| 8nm | 226 | 110 | 1.5 | |
| 16nm | 20 | 76 | 1.5 | |
| 32nm | 5 | 36 | 1.5 | |
| 64nm | 5 | 20 | 1.5 | |

### LD (lipid droplet)

Spherical, variable size. Can be very large (p95 volume ~855K at 8nm).

| Resolution | min_size | n_steps | conv_radius | Notes |
|-----------|----------|---------|-------------|-------|
| 8nm | 676 | 290 | 1.5 | |
| 16nm | 83 | 152 | 3.9 | |
| 32nm | 14 | 111 | 1.5 | |
| 64nm | 5 | 58 | 2.3 | |

### PEROX (peroxisome)

Small to medium, roughly spherical.

| Resolution | min_size | n_steps | conv_radius | Notes |
|-----------|----------|---------|-------------|-------|
| 8nm | 964 | 214 | 1.5 | |
| 16nm | 121 | 102 | 6.0 | |
| 32nm | 40 | 52 | 1.5 | |
| 64nm | 5 | 26 | 1.7 | |

### MITO (mitochondria)

**Elongated, non-convex** — the most challenging class for flow-based segmentation. Bbox max extent p95 reaches 160-268 voxels, meaning diffusion flows can create multiple sink points within a single instance.

| Resolution | min_size | n_steps | conv_radius | Notes |
|-----------|----------|---------|-------------|-------|
| 8nm | 831 | 536 | 1.5 | |
| 16nm | 239 | 320 | 1.5 | |
| 32nm | 26 | 160 | 1.5 | |
| 64nm | 7 | 90 | 2.3 | |

### MT (microtubule)

Thin, elongated structures. High bbox_max relative to volume (median bbox 70 vs median volume 1027 at 8nm). Nearly disappear at 64nm (49 instances, median 2 voxels).

| Resolution | min_size | n_steps | conv_radius | Notes |
|-----------|----------|---------|-------------|-------|
| 8nm | 83 | 300 | 1.5 | |
| 16nm | 8 | 150 | 1.5 | |
| 32nm | 5 | 74 | 1.5 | |
| 64nm | 5 | 32 | 1.5 | Very few samples |

### CELL

Largest structures. Huge variance — from tiny fragments (volume p5 = 1-3) to massive volumes (p95 up to 2.4M voxels at 8nm). The low p5 suggests many crops contain partial cell boundaries or labeling artifacts.

| Resolution | min_size | n_steps | conv_radius | Notes |
|-----------|----------|---------|-------------|-------|
| 8nm | 5 | 614 | 3.2 | |
| 16nm | 5 | 297 | 1.5 | |
| 32nm | 5 | 150 | 1.5 | |
| 64nm | 5 | 140 | 1.5 | |

## Recommended resolution per class

Based on the statistics, the optimal resolution per organelle balances having enough voxels per instance against computational cost:

| Class | Best Resolution | Reasoning |
|-------|----------------|-----------|
| **nuc** | 32-64nm | Huge volumes even at 64nm (median 86K voxels). No need for fine resolution. |
| **ves** | 8nm | At 16nm median is only 18 voxels, at 32nm+ they're 1-2 voxels. 8nm is the only viable option. |
| **endo** | 8-16nm | Median 194 voxels at 16nm is workable. At 32nm drops to 20 voxels. |
| **lyso** | 16nm | Median 668 voxels at 16nm, good bbox extent (median 14). At 32nm only 81 voxels. |
| **ld** | 16-32nm | Still 1,327 median voxels at 32nm. 16nm gives more shape detail for large droplets. |
| **perox** | 16nm | Median 2,389 voxels, bbox median 22. Good balance. At 32nm (682 voxels) still OK. |
| **mito** | 16nm | Median 13,334 voxels, bbox median 50 — enough to capture elongated shape. At 32nm (1,709) loses too much topology. |
| **mt** | 8nm | Thin structures — at 16nm median volume is only 115, bbox median 35. At 32nm they're nearly gone. |
| **cell** | 32-64nm | Massive structures. Even at 64nm, p95 volume is 22K. Finer resolution is wasteful. |

**Summary**: 8nm is only necessary for **ves** and **mt** (thin/tiny structures). Most organelles work well at **16nm**. Large structures (nuc, cell) can go to **32-64nm**.

## Known issues

### `convergence_radius` is 1.5 almost everywhere

The formula uses `nn_dist_p5 / 2` as an upper bound. But `nn_dist_p5 = 0.0` for most classes — caused by:
- Instances that touch or overlap (NN COM distance = 0)
- Tiny label fragments near larger instances
- Crops where multiple instances share nearly identical centers

This drives `convergence_radius` to the floor value of 1.5, which is **too small** for elongated classes like mito and mt where diffusion creates multiple sink points within a single instance.

**Potential fixes:**
1. Filter out `nn_dist = 0` before computing percentiles (treat as label artifacts)
2. Use `nn_dist_p10` or `nn_dist_p25` instead of `nn_dist_p5`
3. For elongated classes (mito, mt, cell), use `bbox_max_median / 2` as the convergence radius instead of the NN-based bound

### Resolution limits

Some classes become too small at coarse resolutions to benefit from flow-based segmentation:
- **Vesicles** at 32nm+: median 1-2 voxels
- **Microtubules** at 64nm: median 2 voxels
- **Endosomes** at 64nm: median 3 voxels

For these, connected-component labeling may be more appropriate than flow tracking.

## Output files

| File | Description |
|------|-------------|
| `instance_stats.csv` | Raw per-instance measurements (dataset, crop, class, resolution, volume, bbox, COM) |
| `recommended_config.json` | Structured recommendations per class x resolution |
| `summary.txt` | Human-readable aggregated statistics |

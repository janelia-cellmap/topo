"""Default class configurations for flow generation and postprocessing.

All postprocessing parameters are resolution-dependent, derived from
GT instance statistics (see topo/scripts/gt_stats/ANALYSIS.md).
"""

from typing import TypedDict


RESOLUTION: int | None = None  # nm — must be set explicitly before use


class ClassConfig(TypedDict, total=False):
    group: int
    flow_type: str  # "direct" or "diffusion"
    diffusion_iters: int
    boundary_only: bool
    boundary_width: int


EVALUATED_INSTANCE_CLASSES: list[str] = [
    "nuc", "ves", "endo", "lyso", "ld", "perox", "mito", "mt", "cell",
]

# ── Resolution-dependent flow generation configs ──────────────────────────
# diffusion_iters scales with instance extent (more voxels → more iterations).
# boundary_width for cell scales similarly.

_INSTANCE_CLASS_BY_RESOLUTION: dict[int, dict[str, ClassConfig]] = {
    8: {
        "nuc":   {"group": 1, "flow_type": "direct"},
        "ves":   {"group": 1, "flow_type": "direct"},
        "endo":  {"group": 1, "flow_type": "direct"},
        "lyso":  {"group": 1, "flow_type": "direct"},
        "ld":    {"group": 1, "flow_type": "direct"},
        "perox": {"group": 1, "flow_type": "direct"},
        "cell":  {"group": 2, "flow_type": "diffusion", "diffusion_iters": 400,
                  "boundary_only": True, "boundary_width": 40},
        "mito":  {"group": 2, "flow_type": "diffusion", "diffusion_iters": 2000},
        "mt":    {"group": 3, "flow_type": "direct"},
    },
    16: {
        "nuc":   {"group": 1, "flow_type": "direct"},
        "ves":   {"group": 1, "flow_type": "direct"},
        "endo":  {"group": 1, "flow_type": "direct"},
        "lyso":  {"group": 1, "flow_type": "direct"},
        "ld":    {"group": 1, "flow_type": "direct"},
        "perox": {"group": 1, "flow_type": "direct"},
        "cell":  {"group": 2, "flow_type": "diffusion", "diffusion_iters": 200,
                  "boundary_only": True, "boundary_width": 20},
        "mito":  {"group": 2, "flow_type": "diffusion", "diffusion_iters": 1000},
        "mt":    {"group": 3, "flow_type": "direct"},
    },
    32: {
        "nuc":   {"group": 1, "flow_type": "direct"},
        "ves":   {"group": 1, "flow_type": "direct"},
        "endo":  {"group": 1, "flow_type": "direct"},
        "lyso":  {"group": 1, "flow_type": "direct"},
        "ld":    {"group": 1, "flow_type": "direct"},
        "perox": {"group": 1, "flow_type": "direct"},
        "cell":  {"group": 2, "flow_type": "diffusion", "diffusion_iters": 100,
                  "boundary_only": True, "boundary_width": 10},
        "mito":  {"group": 2, "flow_type": "diffusion", "diffusion_iters": 500},
        "mt":    {"group": 3, "flow_type": "direct"},
    },
    64: {
        "nuc":   {"group": 1, "flow_type": "direct"},
        "ves":   {"group": 1, "flow_type": "direct"},
        "endo":  {"group": 1, "flow_type": "direct"},
        "lyso":  {"group": 1, "flow_type": "direct"},
        "ld":    {"group": 1, "flow_type": "direct"},
        "perox": {"group": 1, "flow_type": "direct"},
        "cell":  {"group": 2, "flow_type": "diffusion", "diffusion_iters": 50,
                  "boundary_only": True, "boundary_width": 5},
        "mito":  {"group": 2, "flow_type": "diffusion", "diffusion_iters": 250},
        "mt":    {"group": 3, "flow_type": "direct"},
    },
}


def get_instance_class_config(resolution_nm: int | None = None) -> dict[str, ClassConfig]:
    """Get flow generation config for a given resolution.

    Args:
        resolution_nm: Resolution in nm (8, 16, 32, or 64).
                       Defaults to the module-level RESOLUTION variable.

    Returns:
        Dict mapping class name to flow generation parameters.

    Raises:
        ValueError: If resolution_nm is None and RESOLUTION is not set.
    """
    if resolution_nm is None:
        resolution_nm = RESOLUTION
    print(f"Using resolution {resolution_nm} nm for instance class config.")
    if resolution_nm is None:
        raise ValueError(
            "resolution_nm must be provided, or set topo.config.RESOLUTION first."
        )

    if resolution_nm in _INSTANCE_CLASS_BY_RESOLUTION:
        return _INSTANCE_CLASS_BY_RESOLUTION[resolution_nm]

    available = sorted(_INSTANCE_CLASS_BY_RESOLUTION.keys())
    nearest = min(available, key=lambda r: abs(r - resolution_nm))
    return _INSTANCE_CLASS_BY_RESOLUTION[nearest]


# Convenience: default config at module-level RESOLUTION
# INSTANCE_CLASS_CONFIG: dict[str, ClassConfig] = get_instance_class_config()

# ── Resolution-dependent postprocessing configs ────────────────────────────
# Derived from GT statistics across all CellMap crops.
# Keys: resolution in nm → class name → parameters.

_POSTPROCESS_BY_RESOLUTION: dict[int, dict[str, dict]] = {
    8: {
        "nuc":   {"group": 1, "n_steps": 943,  "step_size": 1.0, "convergence_radius": 58.0,  "min_size": 5331},
        "ves":   {"group": 1, "n_steps": 28,   "step_size": 1.0, "convergence_radius": 2.7,   "min_size": 14},
        "endo":  {"group": 1, "n_steps": 112,  "step_size": 1.0, "convergence_radius": 5.2,   "min_size": 81},
        "lyso":  {"group": 1, "n_steps": 110,  "step_size": 1.0, "convergence_radius": 8.7,   "min_size": 226},
        "ld":    {"group": 1, "n_steps": 290,  "step_size": 1.0, "convergence_radius": 17.0,  "min_size": 676},
        "perox": {"group": 1, "n_steps": 214,  "step_size": 1.0, "convergence_radius": 15.3,  "min_size": 964},
        "cell":  {"group": 2, "n_steps": 614,  "step_size": 1.0, "convergence_radius": 3.2,   "min_size": 5},
        "mito":  {"group": 2, "n_steps": 536,  "step_size": 1.0, "convergence_radius": 26.5,  "min_size": 831},
        "mt":    {"group": 3, "n_steps": 300,  "step_size": 1.0, "convergence_radius": 23.3,  "min_size": 83},
    },
    16: {
        "nuc":   {"group": 1, "n_steps": 800,  "step_size": 1.0, "convergence_radius": 38.0,  "min_size": 1000},
        "ves":   {"group": 1, "n_steps": 20,   "step_size": 1.0, "convergence_radius": 1.5,   "min_size": 5},
        "endo":  {"group": 1, "n_steps": 56,   "step_size": 1.0, "convergence_radius": 3.3,   "min_size": 10},
        "lyso":  {"group": 1, "n_steps": 76,   "step_size": 1.0, "convergence_radius": 4.7,   "min_size": 20},
        "ld":    {"group": 1, "n_steps": 152,  "step_size": 1.0, "convergence_radius": 3.9,   "min_size": 80},
        "perox": {"group": 1, "n_steps": 102,  "step_size": 1.0, "convergence_radius": 6.0,   "min_size": 120},
        "cell":  {"group": 2, "n_steps": 300,  "step_size": 1.0, "convergence_radius": 6.0,   "min_size": 5},
        "mito":  {"group": 2, "n_steps": 320,  "step_size": 1.0, "convergence_radius": 25.0,  "min_size": 240},
        "mt":    {"group": 3, "n_steps": 150,  "step_size": 1.0, "convergence_radius": 12.0,  "min_size": 8},
    },
    32: {
        "nuc":   {"group": 1, "n_steps": 662,  "step_size": 1.0, "convergence_radius": 44.7,  "min_size": 225},
        "ves":   {"group": 1, "n_steps": 20,   "step_size": 1.0, "convergence_radius": 1.5,   "min_size": 5},
        "endo":  {"group": 1, "n_steps": 26,   "step_size": 1.0, "convergence_radius": 1.7,   "min_size": 5},
        "lyso":  {"group": 1, "n_steps": 36,   "step_size": 1.0, "convergence_radius": 2.3,   "min_size": 5},
        "ld":    {"group": 1, "n_steps": 111,  "step_size": 1.0, "convergence_radius": 5.3,   "min_size": 14},
        "perox": {"group": 1, "n_steps": 52,   "step_size": 1.0, "convergence_radius": 5.0,   "min_size": 40},
        "cell":  {"group": 2, "n_steps": 150,  "step_size": 1.0, "convergence_radius": 4.3,   "min_size": 5},
        "mito":  {"group": 2, "n_steps": 160,  "step_size": 1.0, "convergence_radius": 12.0,  "min_size": 26},
        "mt":    {"group": 3, "n_steps": 74,   "step_size": 1.0, "convergence_radius": 4.7,   "min_size": 5},
    },
    64: {
        "nuc":   {"group": 1, "n_steps": 336,  "step_size": 1.0, "convergence_radius": 31.0,  "min_size": 266},
        "ves":   {"group": 1, "n_steps": 20,   "step_size": 1.0, "convergence_radius": 1.5,   "min_size": 5},
        "endo":  {"group": 1, "n_steps": 20,   "step_size": 1.0, "convergence_radius": 1.5,   "min_size": 5},
        "lyso":  {"group": 1, "n_steps": 20,   "step_size": 1.0, "convergence_radius": 1.5,   "min_size": 5},
        "ld":    {"group": 1, "n_steps": 58,   "step_size": 1.0, "convergence_radius": 2.3,   "min_size": 5},
        "perox": {"group": 1, "n_steps": 26,   "step_size": 1.0, "convergence_radius": 1.7,   "min_size": 5},
        "cell":  {"group": 2, "n_steps": 140,  "step_size": 1.0, "convergence_radius": 3.0,   "min_size": 5},
        "mito":  {"group": 2, "n_steps": 90,   "step_size": 1.0, "convergence_radius": 7.5,   "min_size": 7},
        "mt":    {"group": 3, "n_steps": 32,   "step_size": 1.0, "convergence_radius": 1.5,   "min_size": 5},
    },
}


def get_postprocess_config(resolution_nm: int | None = None) -> dict[str, dict]:
    """Get postprocessing config for a given resolution.

    Args:
        resolution_nm: Resolution in nm (8, 16, 32, or 64).
                       Defaults to the module-level RESOLUTION variable.

    Returns:
        Dict mapping class name to postprocessing parameters.

    Raises:
        ValueError: If resolution_nm is None and RESOLUTION is not set.
    """
    if resolution_nm is None:
        resolution_nm = RESOLUTION
    print(f"Getting postprocess config for resolution {resolution_nm} nm...")
    if resolution_nm is None:
        raise ValueError(
            "resolution_nm must be provided, or set topo.config.RESOLUTION first."
        )

    if resolution_nm in _POSTPROCESS_BY_RESOLUTION:
        return _POSTPROCESS_BY_RESOLUTION[resolution_nm]

    # Interpolate: pick the nearest available resolution
    available = sorted(_POSTPROCESS_BY_RESOLUTION.keys())
    nearest = min(available, key=lambda r: abs(r - resolution_nm))
    return _POSTPROCESS_BY_RESOLUTION[nearest]


# Convenience: default config at module-level RESOLUTION
# POSTPROCESS_CONFIG: dict[str, dict] = get_postprocess_config()

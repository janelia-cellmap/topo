"""Default class configurations for flow generation."""

from typing import TypedDict


class ClassConfig(TypedDict, total=False):
    group: int
    flow_type: str  # "direct" or "diffusion"
    diffusion_iters: int
    boundary_only: bool
    boundary_width: int


EVALUATED_INSTANCE_CLASSES: list[str] = [
    "nuc", "ves", "endo", "lyso", "ld", "perox", "mito", "mt", "cell",
]

INSTANCE_CLASS_CONFIG: dict[str, ClassConfig] = {
    # Group 1: Convex — direct flows to center of mass
    "nuc":   {"group": 1, "flow_type": "direct"},
    "ves":   {"group": 1, "flow_type": "direct"},
    "endo":  {"group": 1, "flow_type": "direct"},
    "lyso":  {"group": 1, "flow_type": "direct"},
    "ld":    {"group": 1, "flow_type": "direct"},
    "perox": {"group": 1, "flow_type": "direct"},
    # Group 2: Non-convex / elongated — diffusion-based flows (heat equation)
    "cell":  {"group": 2, "flow_type": "diffusion", "diffusion_iters": 200,
              "boundary_only": True, "boundary_width": 20},
    "mito":  {"group": 2, "flow_type": "diffusion", "diffusion_iters": 200},
    # Group 3: Thin — direct flows
    "mt":    {"group": 3, "flow_type": "direct"},
}

"""topo — Topology-aware diffusion-based flow field generation for 3D instance segmentation.

Generates 3D vector fields that point toward instance centers using either:
- Direct flows: unit vectors to center of mass (for convex shapes)
- Diffusion flows: heat-equation-based flows that follow object topology (for non-convex shapes)
"""

from .flow import (
    compute_flow_targets,
    generate_direct_flows,
    generate_diffusion_flows,
)
from .config import INSTANCE_CLASS_CONFIG, EVALUATED_INSTANCE_CLASSES

__version__ = "0.1.0"

__all__ = [
    "compute_flow_targets",
    "generate_direct_flows",
    "generate_diffusion_flows",
    "INSTANCE_CLASS_CONFIG",
    "EVALUATED_INSTANCE_CLASSES",
]

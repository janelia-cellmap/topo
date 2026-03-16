"""topo — Topology-aware diffusion-based flow field generation for 3D instance segmentation.

Generates 3D vector fields that point toward instance centers using either:
- Direct flows: unit vectors to center of mass (for convex shapes)
- Diffusion flows: heat-equation-based flows that follow object topology (for non-convex shapes)

Includes postprocessing to convert flow fields into instance labels via
Euler integration and convergence clustering.
"""

from .flow import (
    compute_flow_targets,
    generate_direct_flows,
    generate_diffusion_flows,
)
from .postprocess import (
    run_instance_segmentation,
    postprocess_single,
    track_flows,
    cluster_convergence,
    split_disconnected,
)
from .config import (
    EVALUATED_INSTANCE_CLASSES,
    get_instance_class_config,
    get_postprocess_config,
)
from .stitch import (
    compute_subcrop_slices,
    build_spatial_mask,
    cosine_blend_weight,
    stitch_flows,
    stitch_labels,
    stitch_volumes,
)

__version__ = "0.1.0"

__all__ = [
    # Flow generation
    "compute_flow_targets",
    "generate_direct_flows",
    "generate_diffusion_flows",
    # Postprocessing
    "run_instance_segmentation",
    "postprocess_single",
    "track_flows",
    "cluster_convergence",
    "split_disconnected",
    # Config
    "EVALUATED_INSTANCE_CLASSES",
    "get_instance_class_config",
    "get_postprocess_config",
    # Stitching
    "compute_subcrop_slices",
    "build_spatial_mask",
    "cosine_blend_weight",
    "stitch_flows",
    "stitch_labels",
    "stitch_volumes",
]

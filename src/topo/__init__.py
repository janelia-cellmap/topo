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

# GPU modules require torch — import lazily so topo works without it
try:
    from .flow_gpu import (
        compute_flow_targets_gpu,
        generate_direct_flows_gpu,
        generate_diffusion_flows_gpu,
    )
    from .postprocess_gpu import (
        run_instance_segmentation_gpu,
        postprocess_single_gpu,
        track_flows_gpu,
    )
except ImportError:
    pass
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
    # Flow generation (GPU)
    "compute_flow_targets_gpu",
    "generate_direct_flows_gpu",
    "generate_diffusion_flows_gpu",
    # Postprocessing
    "run_instance_segmentation",
    "postprocess_single",
    "track_flows",
    "cluster_convergence",
    "split_disconnected",
    # Postprocessing (GPU)
    "run_instance_segmentation_gpu",
    "postprocess_single_gpu",
    "track_flows_gpu",
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

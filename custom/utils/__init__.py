"""
SAM3D-Objects Utilities Package.

This package provides utilities for the SAM3D-Objects 3D Gaussian splatting
pipeline, including depth processing, Gaussian operations, rendering,
pose refinement, and evaluation.

Module Structure
----------------
- config: Configuration classes (RefinementConfig)
- depth: Depth processing and pointmap generation
- io_utils: File I/O operations (load/save images, masks, meshes)
- tokens: SLAT token caching and manipulation
- gaussian: Gaussian splatting operations
- rendering: Differentiable rendering with gsplat
- refinement: Pose refinement using differentiable rendering
- inference_utils: Inference helpers
- temporal: Temporal point cloud accumulation
- visualization: Plotting and visualization utilities

Backwards Compatibility
-----------------------
All functions previously available in utils.general are re-exported here
for backwards compatibility. Existing imports like:

    from utils.general import load_image, RefinementConfig

will continue to work. However, for new code, prefer importing from
specific submodules:

    from utils.io_utils import load_image
    from utils.config import RefinementConfig
"""

# Config
from .config import RefinementConfig, load_refinement_config

# Depth processing
from .depth import (
    compute_conegs_scaling,
    depth_to_pointmap,
    load_and_process_depth,
    radial_to_z_depth,
    transform_to_pytorch3d_convention,
    verify_reprojection,
)

# Gaussian operations
from .gaussian import (
    C0,
    RGB2SH,
    SH2RGB,
    create_background_gaussians,
    create_gaussians_from_pointmap,
    create_gaussians_object,
    join_gaussians,
    transform_scene_to_r3_convention,
)

# Inference utilities
from .inference_utils import (
    compute_and_cache_frame_tokens,
    ensure_all_frames_have_tokens,
    run_inference_on_masks,
)

# I/O utilities
from .io_utils import (
    get_cache_filename,
    load_image,
    load_masks,
    save_mesh_to_obj,
    setup_paths,
)

# Refinement
from .refinement import (
    apply_pose_to_gaussian,
    decode_per_frame_gaussians,
    refine_pose_for_frame,
    refine_poses_for_sequence,
    refine_poses_global_scale,
)

# Rendering
from .rendering import (
    render_and_compare,
    render_gaussian_params,
    render_gaussians_scene,
    render_gaussians_to_image,
    save_comparison_image,
)

# Temporal point clouds
from .temporal import (
    add_frame_to_temporal_point_cloud,
    finalize_temporal_point_cloud,
    save_temporal_point_cloud,
)

# Tokens
from .tokens import (
    apply_median_scale_to_tokens,
    average_slat_tokens,
    compute_frame_weights_from_error,
    compute_frame_weights_from_masks,
    find_best_canon_frame,
    load_all_frame_tokens,
    load_decoder_inputs_from_cache,
    redecode_slat,
    save_tokens,
)

# Evaluation
from .evaluation import (
    evaluate_standard_mode,
    evaluate_with_canonical_objects,
    print_evaluation_summary,
    process_frame_for_eval,
    process_frame_from_cache,
    process_frame_full_inference,
    process_frame_with_canonical_object,
)

# Visualization
from .visualization import plot_refinement_history

__all__ = [
    # Config
    "RefinementConfig",
    "load_refinement_config",
    # Depth
    "radial_to_z_depth",
    "depth_to_pointmap",
    "transform_to_pytorch3d_convention",
    "load_and_process_depth",
    "verify_reprojection",
    "compute_conegs_scaling",
    # I/O
    "load_image",
    "load_masks",
    "setup_paths",
    "get_cache_filename",
    "save_mesh_to_obj",
    # Gaussian
    "C0",
    "RGB2SH",
    "SH2RGB",
    "create_gaussians_object",
    "join_gaussians",
    "create_gaussians_from_pointmap",
    "create_background_gaussians",
    "transform_scene_to_r3_convention",
    # Rendering
    "render_gaussian_params",
    "render_gaussians_scene",
    "render_gaussians_to_image",
    "render_and_compare",
    "save_comparison_image",
    # Tokens
    "save_tokens",
    "load_decoder_inputs_from_cache",
    "load_all_frame_tokens",
    "average_slat_tokens",
    "apply_median_scale_to_tokens",
    "redecode_slat",
    "find_best_canon_frame",
    "compute_frame_weights_from_masks",
    "compute_frame_weights_from_error",
    # Inference
    "run_inference_on_masks",
    "compute_and_cache_frame_tokens",
    "ensure_all_frames_have_tokens",
    # Refinement
    "apply_pose_to_gaussian",
    "refine_pose_for_frame",
    "refine_poses_global_scale",
    "refine_poses_for_sequence",
    "decode_per_frame_gaussians",
    # Temporal
    "save_temporal_point_cloud",
    "add_frame_to_temporal_point_cloud",
    "finalize_temporal_point_cloud",
    # Evaluation
    "evaluate_standard_mode",
    "evaluate_with_canonical_objects",
    "print_evaluation_summary",
    "process_frame_for_eval",
    "process_frame_from_cache",
    "process_frame_full_inference",
    "process_frame_with_canonical_object",
    # Visualization
    "plot_refinement_history",
]

"""
Sequence Evaluation Script for SAM3D.

This script evaluates the quality of Gaussian reconstructions by:
1. Loading a sequence (Kubric4D or DAVIS)
2. Loading cached SLAT tokens if available, otherwise running full inference
3. Re-decoding tokens and applying saved pose to get Gaussians
4. Rendering the Gaussians back to images
5. Comparing rendered images with ground truth using PSNR, SSIM, and LPIPS metrics

Usage Examples:

    # Evaluate Kubric4D sequence with default stride
    python custom/evaluate_sequence.py --dataset kubric4d --scene-name scn02719
    
    # Evaluate DAVIS sequence with specific stride
    python custom/evaluate_sequence.py --dataset davis --scene-name car-turn --frame-stride 5
    
    # Evaluate single frame
    python custom/evaluate_sequence.py --dataset kubric4d --scene-name scn02719 --frame-index 0
    
    # Evaluate without background (background is on by default)
    python custom/evaluate_sequence.py --dataset kubric4d --scene-name scn02719 --no-background
    
    # Save rendered images for visual inspection
    python custom/evaluate_sequence.py --dataset davis --scene-name car-turn --save-renders
    
    # Force full inference (ignore cached tokens)
    python custom/evaluate_sequence.py --dataset davis --scene-name car-turn --no-cache
    
    # Average tokens across frames for canonical object, then warp with per-frame poses
    python custom/evaluate_sequence.py --dataset kubric4d --scene-name scn02719 --canonicalization average
    
    # Use weighted averaging based on mask area (larger masks = higher weight)
    python custom/evaluate_sequence.py --dataset kubric4d --scene-name scn02719 --canonicalization average --weighting-type mask-area
    
    # Use weighted averaging based on rendering error (lower error = higher weight)
    python custom/evaluate_sequence.py --dataset kubric4d --scene-name scn02719 --canonicalization average --weighting-type mask-error
    
    # Use a single frame's tokens as canonical for all frames
    python custom/evaluate_sequence.py --dataset kubric4d --scene-name scn02719 --canonicalization pickone --canon-frame 0
    
    # Refine per-frame poses using differentiable rendering (with token averaging)
    python custom/evaluate_sequence.py --dataset kubric4d --scene-name scn02719 --canonicalization average --refine-poses --refine-config custom/configs/refinement.yaml

    # Refine per-frame poses in standard mode (each frame's Gaussian refined independently)
    python custom/evaluate_sequence.py --dataset kubric4d --scene-name scn02719 --refine-poses --refine-config custom/configs/refinement.yaml

    # Refine poses with CLI overrides (override YAML config values)
    python custom/evaluate_sequence.py --dataset kubric4d --scene-name scn02719 --refine-poses --refine-config custom/configs/refinement.yaml --refine-iterations 200 --refine-lr-rotation 0.005

    # Refine poses with per-frame scale refinement
    python custom/evaluate_sequence.py --dataset kubric4d --scene-name scn02719 --refine-poses --refine-config custom/configs/refinement.yaml --refine-scale perframe

    # Refine poses with global scale (single scale for all frames, batch-optimized)
    python custom/evaluate_sequence.py --dataset kubric4d --scene-name scn02719 --refine-poses --refine-config custom/configs/refinement.yaml --refine-scale global

    # Global scale refinement with custom batch size
    python custom/evaluate_sequence.py --dataset kubric4d --scene-name scn02719 --refine-poses --refine-config custom/configs/refinement.yaml --refine-scale global --refine-batch-size 4

    # Refine poses with optical flow correspondence loss
    python custom/evaluate_sequence.py --dataset kubric4d --scene-name scn02719 --refine-poses --refine-config custom/configs/refinement.yaml --use-flow
    
    # Use median scale for consistent object sizing across frames
    python custom/evaluate_sequence.py --dataset kubric4d --scene-name scn02719 --median-scale

Canonicalization (--canonicalization):
    Controls how the canonical object representation is created:
    - none:     Standard per-frame mode - each frame uses its own decoded Gaussian (default)
    - average:  Average SLAT tokens across all frames to get a single canonical Gaussian
    - pickone:  Use a specific frame's tokens as the canonical for all frames
                The frame is selected via --canon-frame (default: 0)

Weighting Types (for --canonicalization average mode):
    - uniform:    Simple average of all frame tokens (default)
    - mask-area:  Weight by mask visibility - frames with larger masks contribute more
    - mask-error: Weight by inverse rendering error - frames with lower error contribute more

Median Scale (--median-scale):
    Uses the median of all per-frame predicted scales for each object across the
    sequence. This provides more consistent object sizing when the per-frame scale
    predictions vary significantly.

Scale Refinement (--refine-scale):
    Controls how scale is handled during pose refinement:
    - none:     Do not refine scale, use initial predicted scale (default)
    - perframe: Refine scale independently for each frame
    - global:   Optimize a single scale for all frames in the sequence.
                Uses batch-based optimization where loss is aggregated over
                multiple frames per iteration. The first frame's scale is used
                as the initial value.

Pose Refinement (--refine-poses):
    Optimizes per-frame poses (rotation, translation) using differentiable
    Gaussian rendering. Each frame's Gaussian is treated as a "local space" canonical
    object, and its local-to-world transformation is refined to minimize RGB and
    silhouette loss in the masked region. Scale is only refined if --refine-scale is set.
    
    - With --canonicalization average/pickone: Refines poses relative to a canonical
      Gaussian created by averaging SLAT tokens or picking a single frame's tokens.
    - With --canonicalization none (standard mode): Each frame's per-frame Gaussian is
      used as its own canonical, and its predicted transformation is refined.

Flow Loss (--use-flow):
    When enabled, adds an optical flow correspondence loss during pose refinement.
    Uses SEA-RAFT to compute bidirectional flow between rendered and ground-truth images,
    then penalizes the endpoint error in the masked region. This provides dense
    correspondence constraints for better pose alignment.

    - Requires SEA-RAFT checkpoints in custom/submodules/sea_raft_core/checkpoints/
    - Weight controlled by --flow-weight (default 0.1)

Refinement Configuration:
    IMPORTANT: You must explicitly pass --refine-config to load the YAML config file.
    Without it, hardcoded defaults are used (e.g., silhouette_weight=0, rgb_ssim_weight=0).

    Example: --refine-config custom/configs/refinement.yaml

    Settings can be overridden via CLI arguments. CLI values take precedence over YAML values.

    CLI overrides (all optional, override YAML when specified):
        --refine-iterations N      Number of optimization iterations
        --refine-lr-rotation F     Learning rate for rotation
        --refine-lr-translation F  Learning rate for translation
        --refine-lr-scale F        Learning rate for scale
        --refine-scale MODE        Scale refinement: none, perframe, global
        --refine-batch-size N      Batch size for global scale refinement
        --rgb-loss-type TYPE       RGB loss: l1 or l2
        --rgb-multiscale BOOL      Enable multi-scale RGB loss
        --rgb-ssim-weight F        SSIM loss weight (0 to disable)
        --silhouette-weight F      Master silhouette weight (0 to disable)
        --silhouette-com-weight F  Center-of-mass loss weight
        --silhouette-sdt-weight F  Signed distance transform loss weight
        --silhouette-iou-weight F  Soft IoU loss weight
        --use-regularization BOOL  Enable regularization loss
        --regularization-weight F  Regularization loss weight
        --use-flow BOOL            Enable optical flow loss
        --flow-weight F            Optical flow loss weight
        --refine-verbose BOOL      Verbose logging during refinement
        --refine-log-interval N    Logging interval

Output:
    - Metrics printed to console
    - Optional: Rendered images saved to custom/results/{dataset}/eval/renders/
    - Optional: Metrics saved to JSON file
"""
from dataclasses import dataclass
from typing import Optional, Literal
import os
import sys
import json

import tyro


def apply_cli_overrides(config, args):
    """Apply CLI argument overrides to a RefinementConfig.

    Parameters
    ----------
    config : RefinementConfig
        The config loaded from YAML or defaults.
    args : EvaluationConfig
        The CLI arguments containing potential overrides.

    Returns
    -------
    RefinementConfig
        Updated config with CLI overrides applied.
    """
    # Map CLI argument names to RefinementConfig field names
    cli_to_config = {
        'refine_iterations': 'num_iterations',
        'refine_lr_rotation': 'lr_rotation',
        'refine_lr_translation': 'lr_translation',
        'refine_lr_scale': 'lr_scale',
        'refine_scale': 'refine_scale',
        'refine_batch_size': 'batch_size',
        'rgb_loss_type': 'rgb_loss_type',
        'rgb_multiscale': 'rgb_multiscale',
        'rgb_ssim_weight': 'rgb_ssim_weight',
        'silhouette_weight': 'silhouette_weight',
        'silhouette_com_weight': 'silhouette_com_weight',
        'silhouette_sdt_weight': 'silhouette_sdt_weight',
        'silhouette_iou_weight': 'silhouette_iou_weight',
        'use_regularization': 'use_regularization',
        'regularization_weight': 'regularization_weight',
        'use_flow': 'use_flow',
        'flow_weight': 'flow_weight',
        'refine_verbose': 'verbose',
        'refine_log_interval': 'log_interval',
    }

    # Apply overrides for any non-None CLI values
    for cli_name, config_name in cli_to_config.items():
        cli_value = getattr(args, cli_name, None)
        if cli_value is not None:
            setattr(config, config_name, cli_value)

    return config


@dataclass
class EvaluationConfig:
    """Configuration for SAM3D sequence evaluation."""
    
    # Dataset configuration
    dataset: Literal["kubric4d", "davis"] = "kubric4d"
    """Dataset type to process."""
    
    dataset_path: Optional[str] = None
    """Path to dataset root. Defaults to standard paths per dataset type."""
    
    scene_name: Optional[str] = None
    """Name of the scene to process. Defaults: kubric4d='scn02719', davis='car-turn'."""
    
    frame_index: Optional[int] = None
    """Frame index to process (0-based). If not specified, processes frames with --frame-stride."""
    
    frame_stride: int = 10
    """Stride for iterating over frames when --frame-index is not specified."""
    
    # Processing options
    use_moge: bool = False
    """Use MoGe depth model instead of ground truth depth (required for DAVIS)."""
    
    object_index: Optional[int] = None
    """Only process the object at this index (0-based). If not specified, processes all objects."""
    
    seed: int = 42
    """Random seed for inference."""
    
    background: bool = True
    """Add background Gaussians from non-masked regions."""
    
    use_cache: bool = True
    """Use cached tokens if available (set to False to force full inference)."""
    
    median_scale: bool = False
    """Use median of per-frame predicted scales for each object across the sequence."""
    
    # Canonicalization options
    canonicalization: Literal["none", "average", "pickone"] = "none"
    """Canonicalization mode: 'none' (per-frame Gaussians), 'average' (average tokens across frames), 'pickone' (use single frame's tokens for all)."""
    
    canon_frame: Optional[int] = None
    """Frame index to use as canonical when --canonicalization pickone. If not specified, automatically picks the frame with highest mask coverage for each object."""
    
    weighting_type: Literal["uniform", "mask-area", "mask-error"] = "uniform"
    """Weighting type for token averaging. Only used with --canonicalization average."""
    
    # Pose refinement options
    refine_poses: bool = False
    """Refine per-frame poses using differentiable Gaussian rendering."""

    refine_config: Optional[str] = None
    """Path to YAML config file for pose refinement (e.g., custom/configs/refinement.yaml). Required to use non-default settings."""

    # Pose refinement CLI overrides (these override YAML config values when specified)
    refine_iterations: Optional[int] = None
    """Number of refinement iterations."""

    refine_lr_rotation: Optional[float] = None
    """Learning rate for rotation."""

    refine_lr_translation: Optional[float] = None
    """Learning rate for translation."""

    refine_lr_scale: Optional[float] = None
    """Learning rate for scale."""

    refine_scale: Optional[Literal["none", "perframe", "global"]] = None
    """Scale refinement mode."""

    refine_batch_size: Optional[int] = None
    """Batch size for global scale refinement (0=all frames)."""

    rgb_loss_type: Optional[Literal["l1", "l2"]] = None
    """RGB loss type."""

    rgb_multiscale: Optional[bool] = None
    """Enable multi-scale RGB loss."""

    rgb_ssim_weight: Optional[float] = None
    """SSIM loss weight (0=disabled)."""

    silhouette_weight: Optional[float] = None
    """Master weight for silhouette losses (0=disabled)."""

    silhouette_com_weight: Optional[float] = None
    """Center-of-mass loss weight within silhouette loss."""

    silhouette_sdt_weight: Optional[float] = None
    """Signed distance transform loss weight."""

    silhouette_iou_weight: Optional[float] = None
    """Soft IoU loss weight within silhouette loss."""

    use_regularization: Optional[bool] = None
    """Enable regularization loss."""

    regularization_weight: Optional[float] = None
    """Regularization loss weight."""

    use_flow: Optional[bool] = None
    """Enable optical flow correspondence loss."""

    flow_weight: Optional[float] = None
    """Optical flow loss weight."""

    refine_verbose: Optional[bool] = None
    """Verbose logging during refinement."""

    refine_log_interval: Optional[int] = None
    """Logging interval during refinement."""

    # Output options
    output_dir: Optional[str] = None
    """Directory to save outputs. Defaults to results/<dataset>/eval/ in script directory."""
    
    save_renders: bool = False
    """Save rendered images and comparisons."""
    
    save_metrics: bool = False
    """Save metrics to JSON file."""
    
    save_point_clouds: bool = False
    """Save temporal point clouds for visualization with Open3D."""


def main(args: EvaluationConfig):
    """Main execution function."""
    
    # Import heavy modules only when actually running (not for --help)
    import torch
    
    # Skip sam3d_objects initialization for lightweight usage
    os.environ['LIDRA_SKIP_INIT'] = '1'
    
    # Add parent directory to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    from custom.utils import (
        setup_paths,
        redecode_slat,
        average_slat_tokens,
        apply_median_scale_to_tokens,
        compute_frame_weights_from_error,
        compute_frame_weights_from_masks,
        find_best_canon_frame,
        ensure_all_frames_have_tokens,
        load_all_frame_tokens,
        decode_per_frame_gaussians,
        refine_poses_for_sequence,
        evaluate_with_canonical_objects,
        evaluate_standard_mode,
        print_evaluation_summary,
        plot_refinement_history,
        RefinementConfig,
        load_refinement_config,
    )
    from evaluator import Evaluator
    from inference import Inference
    
    # Get the project root directory
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    
    # Set defaults based on dataset type
    if args.dataset_path is None:
        if args.dataset == "kubric4d":
            args.dataset_path = "/mnt/lustre/work/geiger/gwb987/data/kubric4d"
        else:  # davis
            args.dataset_path = "/mnt/lustre/work/geiger/gwb987/data/DAVIS"
    
    if args.scene_name is None:
        if args.dataset == "kubric4d":
            args.scene_name = "scn02719"
        else:  # davis
            args.scene_name = "car-turn"
    
    # DAVIS always requires MoGe (no GT depth)
    if args.dataset == "davis":
        args.use_moge = True
    
    # Build experiment name from configuration
    exp_name_parts = [args.scene_name]
    
    # Canonicalization mode
    if args.canonicalization == "none":
        exp_name_parts.append("perframe")
    elif args.canonicalization == "average":
        exp_name_parts.append(f"avg_{args.weighting_type}")
    elif args.canonicalization == "pickone":
        if args.canon_frame is not None:
            exp_name_parts.append(f"pick{args.canon_frame}")
        else:
            exp_name_parts.append("pickauto")
    
    # Scale options
    if args.median_scale:
        exp_name_parts.append("medscale")
    
    # Refinement options
    if args.refine_poses:
        exp_name_parts.append("refine")
    else:
        exp_name_parts.append("norefine")
    
    exp_name = "_".join(exp_name_parts)
    
    if args.output_dir is None:
        args.output_dir = os.path.join(SCRIPT_DIR, f"results/{args.dataset}/eval/{exp_name}")
    
    # Tokens directory (where demo.py saves cached tokens)
    tokens_dir = os.path.join(SCRIPT_DIR, f"results/{args.dataset}/tokens")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup paths
    paths = setup_paths(args.dataset_path, args.scene_name, dataset_type=args.dataset)
    
    # Determine which frames to process
    num_frames = len(paths['image_names'])
    if args.frame_index is not None:
        frame_indices = [args.frame_index]
    else:
        frame_indices = list(range(0, num_frames, args.frame_stride))
    
    # Print configuration
    print("=" * 60)
    print("SAM3D Sequence Evaluation")
    print("=" * 60)
    print(f"Dataset:           {args.dataset}")
    print(f"Dataset path:      {args.dataset_path}")
    print(f"Scene name:        {args.scene_name}")
    print(f"Total frames:      {num_frames}")
    if args.frame_index is not None:
        print(f"Frame index:       {args.frame_index}")
    else:
        print(f"Frame stride:      {args.frame_stride}")
        print(f"Frames to eval:    {len(frame_indices)} frames")
    print(f"Use MoGe depth:    {args.use_moge}")
    print(f"Object index:      {args.object_index if args.object_index is not None else 'all'}")
    print(f"With background:   {args.background}")
    print(f"Use cache:         {args.use_cache}")
    print(f"Median scale:      {args.median_scale}")
    print(f"Canonicalization:  {args.canonicalization}")
    if args.canonicalization == "average":
        print(f"  Weighting type:  {args.weighting_type}")
    if args.canonicalization == "pickone":
        print(f"  Canon frame:     {args.canon_frame if args.canon_frame is not None else 'auto (best coverage)'}")
    print(f"Refine poses:      {args.refine_poses}")
    if args.refine_poses:
        print(f"  Config file:     {args.refine_config or 'default'}")
    print(f"Tokens directory:  {tokens_dir}")
    print(f"Output directory:  {args.output_dir}")
    print("=" * 60)
    
    # Initialize inference pipeline
    TAG = "hf"
    config_path = os.path.join(PROJECT_ROOT, "checkpoints", TAG, "pipeline.yaml")
    print(f"\nInitializing inference pipeline from {config_path}")
    inference = Inference(config_path, compile=False)
    
    # Initialize evaluator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = Evaluator(device)
    
    # Handle canonicalization modes (average or pickone)
    if args.canonicalization in ["average", "pickone"]:
        
        mode_name = "AVERAGE TOKENS" if args.canonicalization == "average" else "PICK ONE FRAME"
        print("\n" + "=" * 60)
        print(f"{mode_name} MODE: Creating canonical objects")
        print("=" * 60)
        
        # Ensure all requested frames have tokens (compute missing ones)
        tokens_by_object = ensure_all_frames_have_tokens(
            args, paths, frame_indices, inference, tokens_dir
        )
        
        if not tokens_by_object:
            print("\nError: No tokens available and could not compute them!")
            return
        
        print(f"\nFound tokens for {len(tokens_by_object)} objects")
        
        # Apply median scale if requested
        if args.median_scale:
            print("\nApplying median scale normalization...")
            apply_median_scale_to_tokens(tokens_by_object)
        
        # Update frame_indices to include all frames that now have tokens
        # (should match frame_indices after ensure_all_frames_have_tokens)
        available_frame_indices = set()
        for obj_idx, tokens_list in tokens_by_object.items():
            for fid, _ in tokens_list:
                available_frame_indices.add(fid)
        frame_indices = sorted(available_frame_indices)
        print(f"Will evaluate {len(frame_indices)} frames: {frame_indices}")
        
        # Create canonical Gaussians based on canonicalization mode
        canonical_gaussians = {}
        pipeline = inference._pipeline
        
        for obj_idx in sorted(tokens_by_object.keys()):
            tokens_list = tokens_by_object[obj_idx]
            num_token_frames = len(tokens_list)
            
            print(f"\n  Object {obj_idx}: {num_token_frames} frames with tokens")
            
            if args.canonicalization == "average":
                # Compute weights based on selected method
                weights = None
                if args.weighting_type == "mask-error":
                    # Weight by inverse rendering error (lower error = higher weight)
                    weights = compute_frame_weights_from_error(
                        tokens_by_object, obj_idx,
                        args, paths, inference, tokens_dir
                    )
                elif args.weighting_type == "mask-area":
                    # Weight by mask visibility (larger mask = higher weight)
                    weights = compute_frame_weights_from_masks(
                        tokens_list,
                        args.dataset_path,
                        args.scene_name,
                        args.dataset,
                        obj_idx
                    )
                # else: uniform weighting (weights=None)
                
                # Average the SLAT tokens
                avg_slat = average_slat_tokens(tokens_list, weights=weights)
                
                # Decode the averaged tokens to get canonical Gaussian
                decoded = redecode_slat(pipeline, avg_slat, formats=["gaussian"])
                canonical_gaussians[obj_idx] = decoded['gaussian'][0]
                
                print(f"    Canonical Gaussian (averaged): {canonical_gaussians[obj_idx].get_xyz.shape[0]} points")
            
            elif args.canonicalization == "pickone":
                # Determine which frame to use as canonical
                if args.canon_frame is not None:
                    # User specified a frame
                    canon_frame = args.canon_frame
                else:
                    # Auto-select frame with highest mask coverage for this object
                    canon_frame, mask_area = find_best_canon_frame(
                        tokens_list,
                        args.dataset_path,
                        args.scene_name,
                        args.dataset,
                        obj_idx
                    )
                    print(f"    Auto-selected canon frame {canon_frame} (mask area: {mask_area} pixels)")
                
                # Find the token for the canonical frame
                canon_token = None
                for fid, decoder_input in tokens_list:
                    if fid == canon_frame:
                        canon_token = decoder_input
                        break
                
                if canon_token is None:
                    # If exact frame not found, use the first available frame
                    print(f"    Warning: Canon frame {canon_frame} not found for object {obj_idx}, using first available frame")
                    canon_frame = tokens_list[0][0]
                    canon_token = tokens_list[0][1]
                
                # Decode the canonical frame's tokens
                decoded = redecode_slat(pipeline, canon_token['decoder_input_slat'], formats=["gaussian"])
                canonical_gaussians[obj_idx] = decoded['gaussian'][0]
                
                print(f"    Canonical Gaussian (from frame {canon_frame}): {canonical_gaussians[obj_idx].get_xyz.shape[0]} points")
        
        # Store original tokens for pre-refinement evaluation
        tokens_by_object_original = None
        
        if args.refine_poses:
            
            # Deep copy the tokens_by_object to preserve original poses
            tokens_by_object_original = {}
            for obj_idx, tokens_list in tokens_by_object.items():
                tokens_by_object_original[obj_idx] = [
                    (fid, {
                        'decoder_input_slat': di['decoder_input_slat'],
                        'rotation': di['rotation'].clone(),
                        'translation': di['translation'].clone(),
                        'scale': di['scale'].clone(),
                    })
                    for fid, di in tokens_list
                ]
            
            # Evaluate BEFORE refinement
            print("\n" + "=" * 60)
            print("EVALUATION BEFORE POSE REFINEMENT")
            print("=" * 60)
            
            # Build suffix based on canonicalization mode
            if args.canonicalization == "average":
                canon_suffix = f"_averaged_{args.weighting_type}"
            else:  # pickone
                canon_suffix = f"_pickone_frame{args.canon_frame}"
            
            suffix_before = canon_suffix
            summary_before = evaluate_with_canonical_objects(
                args, paths, frame_indices, inference, tokens_dir,
                canonical_gaussians, tokens_by_object_original, evaluator, device,
                suffix=suffix_before, save_renders=True
            )
            
            if summary_before:
                print_evaluation_summary(summary_before, "Results BEFORE Pose Refinement")
            
            # Now refine poses
            print("\n" + "=" * 60)
            print("REFINING POSES")
            print("=" * 60)

            # Load refinement config from YAML file and apply CLI overrides
            refinement_config = load_refinement_config(args.refine_config)
            refinement_config = apply_cli_overrides(refinement_config, args)
            print(f"\nRefinement config: {args.refine_config or 'default'} (+ CLI overrides)")
            print(f"  Iterations:       {refinement_config.num_iterations}")
            print(f"  LR (rotation):    {refinement_config.lr_rotation}")
            print(f"  LR (translation): {refinement_config.lr_translation}")
            print(f"  LR (scale):       {refinement_config.lr_scale}")
            print(f"  Refine scale:     {refinement_config.refine_scale}")
            print(f"  RGB loss type:    {refinement_config.rgb_loss_type}")
            print(f"  RGB multiscale:   {refinement_config.rgb_multiscale}")
            print(f"  SSIM weight:      {refinement_config.rgb_ssim_weight}")
            print(f"  Silhouette wt:    {refinement_config.silhouette_weight}")
            print(f"  Regularization:   {refinement_config.use_regularization}" + (f" (weight={refinement_config.regularization_weight})" if refinement_config.use_regularization else ""))
            print(f"  Flow loss:        {refinement_config.use_flow}" + (f" (weight={refinement_config.flow_weight})" if refinement_config.use_flow else ""))
            if refinement_config.refine_scale == "global":
                print(f"  Batch size:       {refinement_config.batch_size if refinement_config.batch_size > 0 else 'all frames'}")
            
            tokens_by_object = refine_poses_for_sequence(
                canonical_gaussians,
                tokens_by_object,
                args, paths, inference,
                config=refinement_config,
            )
            
            # Evaluate AFTER refinement
            print("\n" + "=" * 60)
            print("EVALUATION AFTER POSE REFINEMENT")
            print("=" * 60)
            
            suffix_after = f"{canon_suffix}_refined"
            summary_after = evaluate_with_canonical_objects(
                args, paths, frame_indices, inference, tokens_dir,
                canonical_gaussians, tokens_by_object, evaluator, device,
                suffix=suffix_after, save_renders=True
            )
            
            if summary_after:
                print_evaluation_summary(summary_after, "Results AFTER Pose Refinement")
            
            # Print comparison
            if summary_before and summary_after:
                print("\n" + "=" * 60)
                print("COMPARISON: Before vs After Pose Refinement")
                print("=" * 60)
                psnr_diff = summary_after['psnr_mean'] - summary_before['psnr_mean']
                ssim_diff = summary_after['ssim_mean'] - summary_before['ssim_mean']
                lpip_diff = summary_after['lpip_mean'] - summary_before['lpip_mean']
                
                print(f"\nPSNR:  {summary_before['psnr_mean']:.2f} → {summary_after['psnr_mean']:.2f} ({psnr_diff:+.2f} dB)")
                print(f"SSIM:  {summary_before['ssim_mean']:.4f} → {summary_after['ssim_mean']:.4f} ({ssim_diff:+.4f})")
                print(f"LPIPS: {summary_before['lpip_mean']:.4f} → {summary_after['lpip_mean']:.4f} ({lpip_diff:+.4f})")
                
                # Determine improvement
                improved = []
                if psnr_diff > 0:
                    improved.append("PSNR")
                if ssim_diff > 0:
                    improved.append("SSIM")
                if lpip_diff < 0:  # Lower is better for LPIPS
                    improved.append("LPIPS")
                
                if improved:
                    print(f"\n✓ Pose refinement improved: {', '.join(improved)}")
                else:
                    print("\n✗ Pose refinement did not improve metrics")
            
            # Save refinement loss history for plotting
            if args.save_metrics:
                refinement_data = {
                    'dataset': args.dataset,
                    'scene_name': args.scene_name,
                    'config': refinement_config.to_dict(),
                    'objects': {}
                }
                
                for obj_idx, tokens_list in tokens_by_object.items():
                    refinement_data['objects'][obj_idx] = {}
                    for frame_idx, decoder_input in tokens_list:
                        if 'refinement_loss_history' in decoder_input:
                            refinement_data['objects'][obj_idx][frame_idx] = {
                                'loss_history': decoder_input['refinement_loss_history'],
                                'best_iteration': decoder_input['refinement_best_iteration'],
                            }
                
                refinement_path = os.path.join(
                    args.output_dir, 
                    f"{args.scene_name}{canon_suffix}_refinement_history.json"
                )
                with open(refinement_path, 'w') as f:
                    json.dump(refinement_data, f, indent=2)
                print(f"\nSaved refinement loss history to {refinement_path}")
                
                # Plot refinement loss curves
                plot_path = os.path.join(
                    args.output_dir,
                    f"{args.scene_name}{canon_suffix}_refinement_history.png"
                )
                plot_refinement_history(refinement_data, plot_path)
        
        else:
            # No refinement requested - just evaluate once
            print("\n" + "=" * 60)
            print("Rendering frames with canonical objects + per-frame poses")
            print("=" * 60)
            
            # Build suffix based on canonicalization mode
            if args.canonicalization == "average":
                canon_suffix = f"_averaged_{args.weighting_type}"
            else:  # pickone
                canon_suffix = f"_pickone_frame{args.canon_frame}"
            
            suffix = canon_suffix
            summary = evaluate_with_canonical_objects(
                args, paths, frame_indices, inference, tokens_dir,
                canonical_gaussians, tokens_by_object, evaluator, device,
                suffix=suffix, save_renders=True
            )
            
            if summary:
                print_evaluation_summary(summary, "Sequence Evaluation Summary")
    
    else:
        # Standard mode: process each frame independently
        print("\n" + "=" * 60)
        print("STANDARD MODE: Processing each frame independently")
        print("=" * 60)
        
        if args.refine_poses:
            # For standard mode with refinement, we need to:
            # 1. Load all cached tokens
            # 2. Decode per-frame canonical Gaussians
            # 3. Use shared refinement logic

            # Load refinement config from YAML file and apply CLI overrides
            refine_config = load_refinement_config(args.refine_config)
            refine_config = apply_cli_overrides(refine_config, args)

            print(f"\nPose refinement config: {args.refine_config or 'default'} (+ CLI overrides)")
            print(f"  Iterations:       {refine_config.num_iterations}")
            print(f"  LR (rotation):    {refine_config.lr_rotation}")
            print(f"  LR (translation): {refine_config.lr_translation}")
            print(f"  LR (scale):       {refine_config.lr_scale}")
            print(f"  Refine scale:     {refine_config.refine_scale}")
            print(f"  RGB loss type:    {refine_config.rgb_loss_type}")
            print(f"  RGB multiscale:   {refine_config.rgb_multiscale}")
            print(f"  SSIM weight:      {refine_config.rgb_ssim_weight}")
            print(f"  Silhouette wt:    {refine_config.silhouette_weight}")
            print(f"  Regularization:   {refine_config.use_regularization}" + (f" (weight={refine_config.regularization_weight})" if refine_config.use_regularization else ""))
            print(f"  Flow loss:        {refine_config.use_flow}" + (f" (weight={refine_config.flow_weight})" if refine_config.use_flow else ""))
            if refine_config.refine_scale == "global":
                print(f"  Batch size:       {refine_config.batch_size if refine_config.batch_size > 0 else 'all frames'}")
            
            # Load all frame tokens
            print("\nLoading cached tokens for all frames...")
            tokens_by_object = load_all_frame_tokens(
                tokens_dir, args.scene_name, args.object_index, args.background
            )
            
            if not tokens_by_object:
                print("\nError: No cached tokens found. Run demo.py first to cache tokens.")
                return
            
            # Filter to requested frame_indices
            for obj_idx in tokens_by_object:
                tokens_by_object[obj_idx] = [
                    (fid, di) for fid, di in tokens_by_object[obj_idx] 
                    if fid in frame_indices
                ]
            
            # Apply median scale if requested
            if args.median_scale:
                print("\nApplying median scale normalization...")
                apply_median_scale_to_tokens(tokens_by_object)
            
            # Decode per-frame canonical Gaussians
            print("\nDecoding per-frame Gaussians...")
            pipeline = inference._pipeline
            canonical_gaussians = decode_per_frame_gaussians(tokens_by_object, pipeline)
            
            # Store original tokens for pre-refinement evaluation
            tokens_by_object_original = {}
            for obj_idx, tokens_list in tokens_by_object.items():
                tokens_by_object_original[obj_idx] = [
                    (fid, {
                        'decoder_input_slat': di['decoder_input_slat'],
                        'rotation': di['rotation'].clone(),
                        'translation': di['translation'].clone(),
                        'scale': di['scale'].clone(),
                    })
                    for fid, di in tokens_list
                ]
            
            # Evaluate BEFORE refinement
            print("\n" + "=" * 60)
            print("EVALUATION BEFORE POSE REFINEMENT")
            print("=" * 60)
            
            summary_before = evaluate_with_canonical_objects(
                args, paths, frame_indices, inference, tokens_dir,
                canonical_gaussians, tokens_by_object_original, evaluator, device,
                suffix="_before", save_renders=True,
                per_frame_canonical=True
            )
            
            if summary_before:
                print_evaluation_summary(summary_before, "Results BEFORE Pose Refinement")
            
            # Refine poses using shared function
            print("\n" + "=" * 60)
            print("REFINING POSES")
            print("=" * 60)
            
            tokens_by_object = refine_poses_for_sequence(
                canonical_gaussians,
                tokens_by_object,
                args, paths, inference,
                config=refine_config,
                per_frame_canonical=True
            )
            
            # Evaluate AFTER refinement
            print("\n" + "=" * 60)
            print("EVALUATION AFTER POSE REFINEMENT")
            print("=" * 60)
            
            summary_after = evaluate_with_canonical_objects(
                args, paths, frame_indices, inference, tokens_dir,
                canonical_gaussians, tokens_by_object, evaluator, device,
                suffix="_after", save_renders=True,
                per_frame_canonical=True
            )
            
            if summary_after:
                print_evaluation_summary(summary_after, "Results AFTER Pose Refinement")
            
            # Print comparison
            if summary_before and summary_after:
                print("\n" + "=" * 60)
                print("COMPARISON: Before vs After Pose Refinement")
                print("=" * 60)
                psnr_diff = summary_after['psnr_mean'] - summary_before['psnr_mean']
                ssim_diff = summary_after['ssim_mean'] - summary_before['ssim_mean']
                lpip_diff = summary_after['lpip_mean'] - summary_before['lpip_mean']
                
                print(f"\nPSNR:  {summary_before['psnr_mean']:.2f} → {summary_after['psnr_mean']:.2f} ({psnr_diff:+.2f} dB)")
                print(f"SSIM:  {summary_before['ssim_mean']:.4f} → {summary_after['ssim_mean']:.4f} ({ssim_diff:+.4f})")
                print(f"LPIPS: {summary_before['lpip_mean']:.4f} → {summary_after['lpip_mean']:.4f} ({lpip_diff:+.4f})")
                
                # Determine improvement
                improved = []
                if psnr_diff > 0:
                    improved.append("PSNR")
                if ssim_diff > 0:
                    improved.append("SSIM")
                if lpip_diff < 0:  # Lower is better for LPIPS
                    improved.append("LPIPS")
                
                if improved:
                    print(f"\n✓ Pose refinement improved: {', '.join(improved)}")
                else:
                    print("\n✗ Pose refinement did not improve metrics")
            
            # Save refinement loss history and plot
            if args.save_metrics:
                refinement_data = {
                    'dataset': args.dataset,
                    'scene_name': args.scene_name,
                    'config': refine_config.to_dict(),
                    'objects': {}
                }

                for obj_idx, tokens_list in tokens_by_object.items():
                    refinement_data['objects'][obj_idx] = {}
                    for frame_idx, decoder_input in tokens_list:
                        if 'refinement_loss_history' in decoder_input:
                            refinement_data['objects'][obj_idx][frame_idx] = {
                                'loss_history': decoder_input['refinement_loss_history'],
                                'best_iteration': decoder_input['refinement_best_iteration'],
                            }

                refinement_path = os.path.join(
                    args.output_dir,
                    f"{args.scene_name}_standard_refinement_history.json"
                )
                with open(refinement_path, 'w') as f:
                    json.dump(refinement_data, f, indent=2)
                print(f"\nSaved refinement loss history to {refinement_path}")

                # Plot refinement loss curves
                plot_path = os.path.join(
                    args.output_dir,
                    f"{args.scene_name}_standard_refinement_history.png"
                )
                plot_refinement_history(refinement_data, plot_path)
        else:
            # No refinement
            if args.median_scale:
                # With median scale: need to load all tokens, apply median, and use canonical path
                print("\nLoading cached tokens for median scale computation...")
                tokens_by_object = load_all_frame_tokens(
                    tokens_dir, args.scene_name, args.object_index, args.background
                )
                
                if not tokens_by_object:
                    print("\nError: No cached tokens found. Run demo.py first to cache tokens.")
                    return
                
                # Filter to requested frame_indices
                for obj_idx in tokens_by_object:
                    tokens_by_object[obj_idx] = [
                        (fid, di) for fid, di in tokens_by_object[obj_idx] 
                        if fid in frame_indices
                    ]
                
                # Apply median scale
                print("\nApplying median scale normalization...")
                apply_median_scale_to_tokens(tokens_by_object)
                
                # Decode per-frame canonical Gaussians
                print("\nDecoding per-frame Gaussians...")
                pipeline = inference._pipeline
                canonical_gaussians = decode_per_frame_gaussians(tokens_by_object, pipeline)
                
                suffix = "_median_scale"
                summary = evaluate_with_canonical_objects(
                    args, paths, frame_indices, inference, tokens_dir,
                    canonical_gaussians, tokens_by_object, evaluator, device,
                    suffix=suffix, save_renders=True,
                    per_frame_canonical=True
                )
            else:
                # Standard evaluation without median scale
                summary = evaluate_standard_mode(
                    args, paths, frame_indices, inference, tokens_dir,
                    evaluator, device, suffix="", save_renders=True
                )
            
            if summary:
                print_evaluation_summary(summary, "Sequence Evaluation Summary")
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    config = tyro.cli(EvaluationConfig)
    main(config)

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
    python custom/evaluate_sequence.py --dataset kubric4d --scene-name scn02719 --average-tokens
    
    # Use weighted averaging based on mask area (larger masks = higher weight)
    python custom/evaluate_sequence.py --dataset kubric4d --scene-name scn02719 --average-tokens --weighting-type mask-area
    
    # Use weighted averaging based on rendering error (lower error = higher weight)
    python custom/evaluate_sequence.py --dataset kubric4d --scene-name scn02719 --average-tokens --weighting-type mask-error
    
    # Refine per-frame poses using differentiable rendering (with token averaging)
    python custom/evaluate_sequence.py --dataset kubric4d --scene-name scn02719 --average-tokens --refine-poses
    
    # Refine per-frame poses in standard mode (each frame's Gaussian refined independently)
    python custom/evaluate_sequence.py --dataset kubric4d --scene-name scn02719 --refine-poses
    
    # Refine poses with custom settings
    python custom/evaluate_sequence.py --dataset kubric4d --scene-name scn02719 --average-tokens --refine-poses --refine-iterations 200 --refine-lr-rotation 0.005 --refine-lr-translation 0.0005 --refine-lr-scale 0.0005
    
    # Refine poses with scale refinement enabled (perframe)
    python custom/evaluate_sequence.py --dataset kubric4d --scene-name scn02719 --refine-poses --refine-scale perframe
    
    # Use median scale for consistent object sizing across frames
    python custom/evaluate_sequence.py --dataset kubric4d --scene-name scn02719 --median-scale

Weighting Types (for --average-tokens mode):
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
    - global:   Optimize a single scale for all frames (not implemented yet)

Pose Refinement (--refine-poses):
    Optimizes per-frame poses (rotation, translation) using differentiable
    Gaussian rendering. Each frame's Gaussian is treated as a "local space" canonical
    object, and its local-to-world transformation is refined to minimize RGB and
    silhouette loss in the masked region. Scale is only refined if --refine-scale is set.
    
    - With --average-tokens: Refines poses relative to a canonical Gaussian created
      by averaging SLAT tokens across frames.
    - Without --average-tokens (standard mode): Each frame's per-frame Gaussian is
      used as its own canonical, and its predicted transformation is refined.

Output:
    - Metrics printed to console
    - Optional: Rendered images saved to custom/results/{dataset}/eval/renders/
    - Optional: Metrics saved to JSON file
"""
import os
import sys
import json
import argparse
import torch

# Skip sam3d_objects initialization for lightweight usage
os.environ['LIDRA_SKIP_INIT'] = '1'

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import (
    setup_paths,
    redecode_slat,
    average_slat_tokens,
    apply_median_scale_to_tokens,
    compute_frame_weights_from_error,
    compute_frame_weights_from_masks,
    ensure_all_frames_have_tokens,
    load_all_frame_tokens,
    decode_per_frame_gaussians,
    refine_poses_for_sequence,
    evaluate_with_canonical_objects,
    evaluate_standard_mode,
    print_evaluation_summary,
    plot_refinement_history,
    RefinementConfig,
)
from evaluator import Evaluator
from inference import Inference


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate SAM3D Gaussian reconstruction quality on sequences",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["kubric4d", "davis"],
        default="kubric4d",
        help="Dataset type to process",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to dataset root. Defaults to standard paths per dataset type.",
    )
    parser.add_argument(
        "--scene-name",
        type=str,
        default=None,
        help="Name of the scene to process. Defaults: kubric4d='scn02719', davis='car-turn'",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="Frame index to process (0-based). If not specified, processes frames with --frame-stride",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=10,
        help="Stride for iterating over frames when --frame-index is not specified",
    )
    
    # Processing options
    parser.add_argument(
        "--use-moge",
        action="store_true",
        help="Use MoGe depth model instead of ground truth depth (required for DAVIS)",
    )
    parser.add_argument(
        "--object-index",
        type=int,
        default=None,
        help="Only process the object at this index (0-based). If not specified, processes all objects.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for inference",
    )
    parser.add_argument(
        "--no-background",
        action="store_true",
        help="Disable adding background Gaussians from non-masked regions",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cached tokens and run full inference",
    )
    parser.add_argument(
        "--median-scale",
        action="store_true",
        help="Use median of per-frame predicted scales for each object across the sequence",
    )
    
    # Token averaging options
    parser.add_argument(
        "--average-tokens",
        action="store_true",
        help="Average SLAT tokens across frames to get canonical object, then warp with per-frame poses",
    )
    parser.add_argument(
        "--weighting-type",
        type=str,
        choices=["uniform", "mask-area", "mask-error"],
        default="uniform",
        help="Weighting type for token averaging: 'uniform' (simple average), 'mask-area' (weight by mask visibility), 'mask-error' (weight by inverse rendering error). Only used with --average-tokens",
    )
    
    # Pose refinement options
    parser.add_argument(
        "--refine-poses",
        action="store_true",
        help="Refine per-frame poses using differentiable Gaussian rendering",
    )
    parser.add_argument(
        "--refine-scale",
        type=str,
        choices=["none", "perframe", "global"],
        default="none",
        help="Scale refinement mode: 'none' (keep predicted scale), 'perframe' (refine scale independently per frame), 'global' (optimize single scale across sequence, not implemented)",
    )
    parser.add_argument(
        "--refine-iterations",
        type=int,
        default=100,
        help="Number of iterations for pose refinement",
    )
    parser.add_argument(
        "--refine-lr-rotation",
        type=float,
        default=0.01,
        help="Learning rate for rotation refinement",
    )
    parser.add_argument(
        "--refine-lr-translation",
        type=float,
        default=0.001,
        help="Learning rate for translation refinement",
    )
    parser.add_argument(
        "--refine-lr-scale",
        type=float,
        default=0.001,
        help="Learning rate for scale refinement (only used when --refine-scale is 'perframe')",
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save outputs. Defaults to results/<dataset>/eval/ in script directory",
    )
    parser.add_argument(
        "--save-renders",
        action="store_true",
        help="Save rendered images and comparisons",
    )
    parser.add_argument(
        "--save-metrics",
        action="store_true",
        help="Save metrics to JSON file",
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
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
    
    if args.output_dir is None:
        args.output_dir = os.path.join(SCRIPT_DIR, f"results/{args.dataset}/eval")
    
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
    print(f"With background:   {not args.no_background}")
    print(f"Use cache:         {not args.no_cache}")
    print(f"Median scale:      {args.median_scale}")
    print(f"Average tokens:    {args.average_tokens}")
    if args.average_tokens:
        print(f"Weighting type:    {args.weighting_type}")
    print(f"Refine poses:      {args.refine_poses}")
    if args.refine_poses:
        print(f"  Iterations:      {args.refine_iterations}")
        print(f"  LR rotation:     {args.refine_lr_rotation}")
        print(f"  LR translation:  {args.refine_lr_translation}")
        print(f"  LR scale:        {args.refine_lr_scale}")
        print(f"  Refine scale:    {args.refine_scale}")
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
    
    # Handle --average-tokens mode
    if args.average_tokens:
        
        print("\n" + "=" * 60)
        print("AVERAGE TOKENS MODE: Creating canonical objects from averaged tokens")
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
        
        # Average tokens and decode canonical Gaussians for each object
        canonical_gaussians = {}
        pipeline = inference._pipeline
        
        for obj_idx in sorted(tokens_by_object.keys()):
            tokens_list = tokens_by_object[obj_idx]
            num_token_frames = len(tokens_list)
            
            print(f"\n  Object {obj_idx}: {num_token_frames} frames with tokens")
            
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
            
            print(f"    Canonical Gaussian: {canonical_gaussians[obj_idx].get_xyz.shape[0]} points")
        
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
            
            suffix_before = f"_averaged_{args.weighting_type}"
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
            
            # Create refinement config from command line arguments
            refinement_config = RefinementConfig(
                num_iterations=args.refine_iterations,
                lr_rotation=args.refine_lr_rotation,
                lr_translation=args.refine_lr_translation,
                lr_scale=args.refine_lr_scale,
                refine_scale=args.refine_scale,
            )
            
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
            
            suffix_after = f"_averaged_{args.weighting_type}_refined"
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
                    'num_iterations': args.refine_iterations,
                    'lr_rotation': args.refine_lr_rotation,
                    'lr_translation': args.refine_lr_translation,
                    'lr_scale': args.refine_lr_scale,
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
                    f"{args.scene_name}_averaged_{args.weighting_type}_refinement_history.json"
                )
                with open(refinement_path, 'w') as f:
                    json.dump(refinement_data, f, indent=2)
                print(f"\nSaved refinement loss history to {refinement_path}")
                
                # Plot refinement loss curves
                plot_path = os.path.join(
                    args.output_dir,
                    f"{args.scene_name}_averaged_{args.weighting_type}_refinement_history.png"
                )
                plot_refinement_history(refinement_data, plot_path)
        
        else:
            # No refinement requested - just evaluate once
            print("\n" + "=" * 60)
            print("Rendering frames with canonical objects + per-frame poses")
            print("=" * 60)
            
            suffix = f"_averaged_{args.weighting_type}"
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
            
            # Create refinement config
            refine_config = RefinementConfig(
                num_iterations=args.refine_iterations,
                lr_rotation=args.refine_lr_rotation,
                lr_translation=args.refine_lr_translation,
                lr_scale=args.refine_lr_scale,
                refine_scale=args.refine_scale,
                verbose=True,
            )
            
            print("\nPose refinement enabled:")
            print(f"  Iterations:       {refine_config.num_iterations}")
            print(f"  LR (rotation):    {refine_config.lr_rotation}")
            print(f"  LR (translation): {refine_config.lr_translation}")
            print(f"  LR (scale):       {refine_config.lr_scale}")
            print(f"  Refine scale:     {refine_config.refine_scale}")
            
            # Load all frame tokens
            print("\nLoading cached tokens for all frames...")
            tokens_by_object = load_all_frame_tokens(
                tokens_dir, args.scene_name, args.object_index, not args.no_background
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
                    'num_iterations': args.refine_iterations,
                    'lr_rotation': args.refine_lr_rotation,
                    'lr_translation': args.refine_lr_translation,
                    'lr_scale': args.refine_lr_scale,
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
                    tokens_dir, args.scene_name, args.object_index, not args.no_background
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
    main()

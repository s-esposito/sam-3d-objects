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
    
    # Evaluate with background
    python custom/evaluate_sequence.py --dataset kubric4d --scene-name scn02719 --with-background
    
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
    
    # Refine per-frame poses using differentiable rendering
    python custom/evaluate_sequence.py --dataset kubric4d --scene-name scn02719 --average-tokens --refine-poses
    
    # Refine poses with custom settings
    python custom/evaluate_sequence.py --dataset kubric4d --scene-name scn02719 --average-tokens --refine-poses --refine-iterations 200 --refine-lr-rotation 0.005 --refine-lr-translation 0.0005 --refine-lr-scale 0.0005

Weighting Types (for --average-tokens mode):
    - uniform:    Simple average of all frame tokens (default)
    - mask-area:  Weight by mask visibility - frames with larger masks contribute more
    - mask-error: Weight by inverse rendering error - frames with lower error contribute more

Pose Refinement (--refine-poses, only with --average-tokens):
    Optimizes per-frame poses (rotation, translation, scale) using differentiable
    Gaussian rendering. The canonical Gaussian object is frozen, and only the
    pose parameters are optimized to minimize MSE in the masked region.

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

from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp

from utils import (
    load_image,
    load_masks,
    render_gaussians_to_image,
    join_gaussians,
    setup_paths,
    load_and_process_depth,
    transform_scene_to_r3_convention,
    create_background_gaussians,
    save_comparison_image,
    save_tokens,
    redecode_slat,
    average_slat_tokens,
    compute_frame_weights_from_error,
    compute_frame_weights_from_masks,
    ensure_all_frames_have_tokens,
    process_frame_from_cache,
    process_frame_full_inference,
    refine_poses_for_sequence,
)
from evaluator import Evaluator
from inference import Inference, make_scene

def process_frame_for_eval(args, paths, frame_index, inference, tokens_dir):
    """
    Process a single frame, using cache if available.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    paths : dict
        Dataset paths
    frame_index : int
        Frame index to process
    inference : Inference
        Inference pipeline
    tokens_dir : str
        Directory where cached tokens are stored
        
    Returns
    -------
    tuple
        (rendered_image, gt_image, K_matrix) where images are torch tensors (H, W, 3)
    """
    # First try to load from cache
    if not args.no_cache:
        result = process_frame_from_cache(args, paths, frame_index, inference, tokens_dir)
        if result is not None:
            # print info about loaded cache
            print(f"    Loaded cached tokens for frame {frame_index} from {tokens_dir}")
            return result
        print("    Cache not found, running full inference...")
    
    # Fall back to full inference
    return process_frame_full_inference(args, paths, frame_index, inference)


def process_frame_with_canonical_object(args, paths, frame_index, inference, tokens_dir,
                                         canonical_gaussians, tokens_by_object):
    """
    Process a single frame using pre-decoded canonical Gaussians warped with per-frame pose.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    paths : dict
        Dataset paths
    frame_index : int
        Frame index to process
    inference : Inference
        Inference pipeline
    tokens_dir : str
        Directory where cached tokens are stored
    canonical_gaussians : dict
        Dictionary mapping obj_idx -> decoded Gaussian object (canonical, no pose applied)
    tokens_by_object : dict
        Dictionary mapping obj_idx -> list of (frame_idx, decoder_input) with poses
        
    Returns
    -------
    tuple
        (rendered_image, gt_image, K_matrix) where images are torch tensors (H, W, 3)
    """
    # Load frame's image and masks
    image_path = os.path.join(paths['frames_path'], paths['image_names'][frame_index])
    mask_path = os.path.join(paths['masks_path'], paths['mask_names'][frame_index])
    
    image = load_image(image_path)
    image = image[..., :3]
    H, W, _ = image.shape
    
    masks = load_masks(mask_path)
    if args.first_object_only:
        masks = masks[:1]
    
    # Load depth and compute K_matrix (needed for rendering)
    depth_names_for_frame = []
    if paths['dataset_type'] == 'kubric4d' and paths['depth_names']:
        depth_names_for_frame = [paths['depth_names'][frame_index]]
    
    pointmap, K_matrix, valid_mask = load_and_process_depth(
        paths['frames_path'],
        depth_names_for_frame,
        W, H,
        use_moge=args.use_moge,
        inference=inference,
        image=image
    )
    
    pointmap_original = pointmap.copy() if args.with_background else None
    
    # Get per-frame poses from cached tokens
    # Build outputs list with canonical gaussian + per-frame pose
    outputs = []
    
    for obj_idx in sorted(canonical_gaussians.keys()):
        # Find the pose for this frame
        frame_pose = None
        for fid, decoder_input in tokens_by_object[obj_idx]:
            if fid == frame_index:
                frame_pose = decoder_input
                break
        
        if frame_pose is None:
            print(f"    Warning: No pose found for object {obj_idx} at frame {frame_index}, skipping")
            continue
        
        # Build output with canonical gaussian and per-frame pose
        output = {
            'gaussian': [canonical_gaussians[obj_idx]],  # Wrap in list for make_scene
            'rotation': frame_pose['rotation'],
            'translation': frame_pose['translation'],
            'scale': frame_pose['scale'],
        }
        outputs.append(output)
    
    if len(outputs) == 0:
        raise ValueError(f"No objects with poses found for frame {frame_index}")
    
    # Create combined scene from all outputs (in PyTorch3D convention)
    scene_gs = make_scene(*outputs)
    
    # Transform scene from PyTorch3D to R3 convention
    new_scene_gs = transform_scene_to_r3_convention(scene_gs)
    
    # Add background Gaussians if requested
    if args.with_background and pointmap_original is not None:
        background_gs = create_background_gaussians(
            image, pointmap_original, masks, K_matrix
        )
        new_scene_gs = join_gaussians(background_gs, new_scene_gs)
    
    # Render Gaussians to image
    rendered = render_gaussians_to_image(new_scene_gs, K_matrix, W, H)
    
    # Convert ground truth to tensor
    gt_image = torch.from_numpy(image).float() / 255.0
    
    # Clamp rendered to [0, 1]
    rendered = torch.clamp(rendered.cpu(), 0.0, 1.0)
    
    return rendered, gt_image, K_matrix


def evaluate_with_canonical_objects(
    args, paths, frame_indices, inference, tokens_dir,
    canonical_gaussians, tokens_by_object, evaluator, device,
    suffix="", save_renders=True, save_metrics=True
):
    """
    Evaluate frames using canonical objects with per-frame poses.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    paths : dict
        Dataset paths
    frame_indices : list
        List of frame indices to process
    inference : Inference
        Inference pipeline
    tokens_dir : str
        Directory where cached tokens are stored
    canonical_gaussians : dict
        Dictionary mapping obj_idx -> canonical Gaussian
    tokens_by_object : dict
        Dictionary mapping obj_idx -> list of (frame_idx, decoder_input)
    evaluator : Evaluator
        Evaluator instance
    device : torch.device
        Device to use
    suffix : str
        Suffix for output filenames
    save_renders : bool
        Whether to save rendered images
    save_metrics : bool
        Whether to save metrics to JSON
        
    Returns
    -------
    dict
        Evaluation summary with metrics
    """
    rendered_frames = []
    gt_frames = []
    frame_indices_processed = []
    
    # Filter frame_indices to only those that have poses for ALL objects
    # (We need poses for every object to render a complete frame)
    frame_sets_per_object = []
    for obj_idx, tokens_list in tokens_by_object.items():
        obj_frames = set(fid for fid, _ in tokens_list)
        frame_sets_per_object.append(obj_frames)
    
    if frame_sets_per_object:
        # Intersection: frames that have poses for ALL objects
        available_frame_indices = frame_sets_per_object[0]
        for obj_frames in frame_sets_per_object[1:]:
            available_frame_indices = available_frame_indices & obj_frames
    else:
        available_frame_indices = set()
    
    frame_indices_to_process = [f for f in frame_indices if f in available_frame_indices]
    
    if len(frame_indices_to_process) < len(frame_indices):
        print(f"\nNote: Only {len(frame_indices_to_process)} of {len(frame_indices)} requested frames have poses for all objects")
        print(f"  Frames with complete poses: {sorted(available_frame_indices)}")
        # Show which frames are missing for which objects
        for obj_idx, tokens_list in tokens_by_object.items():
            obj_frames = set(fid for fid, _ in tokens_list)
            missing = set(frame_indices) - obj_frames
            if missing:
                print(f"  Object {obj_idx} missing frames: {sorted(missing)}")
    
    for frame_idx, frame_index in enumerate(frame_indices_to_process):
        print(f"\n  Processing frame {frame_index} ({frame_idx + 1}/{len(frame_indices_to_process)})")
        
        try:
            rendered, gt_image, K_matrix = process_frame_with_canonical_object(
                args, paths, frame_index, inference, tokens_dir,
                canonical_gaussians, tokens_by_object
            )
            
            # Convert to format expected by evaluator: (B, C, H, W)
            rendered_eval = rendered.permute(2, 0, 1).unsqueeze(0).to(device)
            gt_eval = gt_image.permute(2, 0, 1).unsqueeze(0).to(device)
            
            # Save comparison image if requested
            if save_renders and args.save_renders:
                render_dir = os.path.join(args.output_dir, "renders")
                os.makedirs(render_dir, exist_ok=True)
                output_path = os.path.join(render_dir, f"{args.scene_name}_frame_{frame_index:04d}{suffix}_comparison.png")
                save_comparison_image(rendered, gt_image, output_path, frame_index)
                print(f"    Saved comparison to {output_path}")
            
            # Store for sequence evaluation
            rendered_frames.append(rendered_eval)
            gt_frames.append(gt_eval)
            frame_indices_processed.append(frame_index)
            
        except Exception as e:
            print(f"    Error processing frame {frame_index}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Evaluate
    if len(rendered_frames) == 0:
        print("  No frames were successfully processed!")
        return None
    
    seq_metrics = evaluator.evaluate_sequence(gt_frames, rendered_frames)
    
    # Build summary
    frame_metrics = [
        {
            'frame_index': frame_indices_processed[i],
            'psnr': seq_metrics['psnr_values'][i],
            'ssim': seq_metrics['ssim_values'][i],
            'lpip': seq_metrics['lpip_values'][i],
        }
        for i in range(len(frame_indices_processed))
    ]
    
    summary = {
        'dataset': args.dataset,
        'scene_name': args.scene_name,
        'num_frames_evaluated': len(frame_indices_processed),
        'frame_stride': args.frame_stride,
        'with_background': args.with_background,
        'first_object_only': args.first_object_only,
        'average_tokens': args.average_tokens,
        'weighting_type': args.weighting_type if args.average_tokens else None,
        'suffix': suffix,
        'psnr_mean': float(seq_metrics['psnr_mean']),
        'psnr_std': float(seq_metrics['psnr_std']),
        'psnr_min': float(seq_metrics['psnr_min']),
        'psnr_max': float(seq_metrics['psnr_max']),
        'ssim_mean': float(seq_metrics['ssim_mean']),
        'ssim_std': float(seq_metrics['ssim_std']),
        'ssim_min': float(seq_metrics['ssim_min']),
        'ssim_max': float(seq_metrics['ssim_max']),
        'lpip_mean': float(seq_metrics['lpip_mean']),
        'lpip_std': float(seq_metrics['lpip_std']),
        'lpip_min': float(seq_metrics['lpip_min']),
        'lpip_max': float(seq_metrics['lpip_max']),
        'frame_metrics': frame_metrics,
    }
    
    # Save metrics to JSON if requested
    if save_metrics and args.save_metrics:
        metrics_path = os.path.join(args.output_dir, f"{args.scene_name}{suffix}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Saved metrics to {metrics_path}")
    
    return summary


def print_evaluation_summary(summary, title="Evaluation Summary"):
    """Print evaluation metrics in a formatted way."""
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")
    print(f"Frames evaluated:  {summary['num_frames_evaluated']}")
    print("\nPSNR (dB):")
    print(f"  Mean: {summary['psnr_mean']:.2f} ± {summary['psnr_std']:.2f}")
    print(f"  Range: [{summary['psnr_min']:.2f}, {summary['psnr_max']:.2f}]")
    print("\nSSIM:")
    print(f"  Mean: {summary['ssim_mean']:.4f} ± {summary['ssim_std']:.4f}")
    print(f"  Range: [{summary['ssim_min']:.4f}, {summary['ssim_max']:.4f}]")
    print("\nLPIPS (lower is better):")
    print(f"  Mean: {summary['lpip_mean']:.4f} ± {summary['lpip_std']:.4f}")
    print(f"  Range: [{summary['lpip_min']:.4f}, {summary['lpip_max']:.4f}]")


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
        "--first-object-only",
        action="store_true",
        help="Only process the first object/mask",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for inference",
    )
    parser.add_argument(
        "--with-background",
        action="store_true",
        help="Add background Gaussians from non-masked regions",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cached tokens and run full inference",
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
        help="Refine per-frame poses using differentiable Gaussian rendering (only with --average-tokens)",
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
        help="Learning rate for scale refinement",
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
    print(f"First object only: {args.first_object_only}")
    print(f"With background:   {args.with_background}")
    print(f"Use cache:         {not args.no_cache}")
    print(f"Average tokens:    {args.average_tokens}")
    if args.average_tokens:
        print(f"Weighting type:    {args.weighting_type}")
        print(f"Refine poses:      {args.refine_poses}")
        if args.refine_poses:
            print(f"  Iterations:      {args.refine_iterations}")
            print(f"  LR rotation:     {args.refine_lr_rotation}")
            print(f"  LR translation:  {args.refine_lr_translation}")
            print(f"  LR scale:        {args.refine_lr_scale}")
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
    
    # Collect all rendered and ground truth frames
    rendered_frames = []
    gt_frames = []
    frame_indices_processed = []
    
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
                suffix=suffix_before, save_renders=True, save_metrics=True
            )
            
            if summary_before:
                print_evaluation_summary(summary_before, "Results BEFORE Pose Refinement")
            
            # Now refine poses
            print("\n" + "=" * 60)
            print("REFINING POSES")
            print("=" * 60)
            
            tokens_by_object = refine_poses_for_sequence(
                canonical_gaussians,
                tokens_by_object,
                args, paths, inference,
                num_iterations=args.refine_iterations,
                lr_rotation=args.refine_lr_rotation,
                lr_translation=args.refine_lr_translation,
                lr_scale=args.refine_lr_scale,
            )
            
            # Evaluate AFTER refinement
            print("\n" + "=" * 60)
            print("EVALUATION AFTER POSE REFINEMENT")
            print("=" * 60)
            
            suffix_after = f"_averaged_{args.weighting_type}_refined"
            summary_after = evaluate_with_canonical_objects(
                args, paths, frame_indices, inference, tokens_dir,
                canonical_gaussians, tokens_by_object, evaluator, device,
                suffix=suffix_after, save_renders=True, save_metrics=True
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
        
        else:
            # No refinement requested - just evaluate once
            print("\n" + "=" * 60)
            print("Rendering frames with canonical objects + per-frame poses")
            print("=" * 60)
            
            suffix = f"_averaged_{args.weighting_type}"
            summary = evaluate_with_canonical_objects(
                args, paths, frame_indices, inference, tokens_dir,
                canonical_gaussians, tokens_by_object, evaluator, device,
                suffix=suffix, save_renders=True, save_metrics=True
            )
            
            if summary:
                print_evaluation_summary(summary, "Sequence Evaluation Summary")
                rendered_frames = []  # Mark as handled
                gt_frames = []
                frame_indices_processed = []
    
    else:
        # Standard mode: process each frame independently
        for frame_idx, frame_index in enumerate(frame_indices):
            print(f"\n{'='*60}")
            print(f"Processing frame {frame_index} ({frame_idx + 1}/{len(frame_indices)})")
            print(f"{'='*60}")
            
            try:
                rendered, gt_image, K_matrix = process_frame_for_eval(
                    args, paths, frame_index, inference, tokens_dir
                )
                
                # Convert to format expected by evaluator: (B, C, H, W)
                rendered_eval = rendered.permute(2, 0, 1).unsqueeze(0).to(device)
                gt_eval = gt_image.permute(2, 0, 1).unsqueeze(0).to(device)
                
                # Save comparison image if requested
                if args.save_renders:
                    render_dir = os.path.join(args.output_dir, "renders")
                    os.makedirs(render_dir, exist_ok=True)
                    output_path = os.path.join(render_dir, f"{args.scene_name}_frame_{frame_index:04d}_comparison.png")
                    save_comparison_image(rendered, gt_image, output_path, frame_index)
                    print(f"  Saved comparison to {output_path}")
                
                # Store for sequence evaluation
                rendered_frames.append(rendered_eval)
                gt_frames.append(gt_eval)
                frame_indices_processed.append(frame_index)
                
            except Exception as e:
                print(f"  Error processing frame {frame_index}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Evaluate all frames together using Evaluator.evaluate_sequence
    if len(rendered_frames) > 0:
        print(f"\n{'='*60}")
        print("Evaluating sequence...")
        print(f"{'='*60}")
        
        # Use Evaluator's evaluate_sequence method
        seq_metrics = evaluator.evaluate_sequence(gt_frames, rendered_frames)
        
        # Print per-frame metrics
        print("\nPer-frame metrics:")
        for i, frame_index in enumerate(frame_indices_processed):
            print(f"  Frame {frame_index}: PSNR={seq_metrics['psnr_values'][i]:.2f} dB, "
                  f"SSIM={seq_metrics['ssim_values'][i]:.4f}, "
                  f"LPIPS={seq_metrics['lpip_values'][i]:.4f}")
        
        print(f"\n{'='*60}")
        print("Sequence Evaluation Summary")
        print(f"{'='*60}")
        
        # Build frame_metrics for JSON output
        frame_metrics = [
            {
                'frame_index': frame_indices_processed[i],
                'psnr': seq_metrics['psnr_values'][i],
                'ssim': seq_metrics['ssim_values'][i],
                'lpip': seq_metrics['lpip_values'][i],
            }
            for i in range(len(frame_indices_processed))
        ]
        
        summary = {
            'dataset': args.dataset,
            'scene_name': args.scene_name,
            'num_frames_evaluated': len(frame_indices_processed),
            'frame_stride': args.frame_stride,
            'with_background': args.with_background,
            'first_object_only': args.first_object_only,
            'average_tokens': args.average_tokens,
            'weighting_type': args.weighting_type if args.average_tokens else None,
            'refine_poses': args.refine_poses if args.average_tokens else False,
            'refine_iterations': args.refine_iterations if (args.average_tokens and args.refine_poses) else None,
            'psnr_mean': float(seq_metrics['psnr_mean']),
            'psnr_std': float(seq_metrics['psnr_std']),
            'psnr_min': float(seq_metrics['psnr_min']),
            'psnr_max': float(seq_metrics['psnr_max']),
            'ssim_mean': float(seq_metrics['ssim_mean']),
            'ssim_std': float(seq_metrics['ssim_std']),
            'ssim_min': float(seq_metrics['ssim_min']),
            'ssim_max': float(seq_metrics['ssim_max']),
            'lpip_mean': float(seq_metrics['lpip_mean']),
            'lpip_std': float(seq_metrics['lpip_std']),
            'lpip_min': float(seq_metrics['lpip_min']),
            'lpip_max': float(seq_metrics['lpip_max']),
            'frame_metrics': frame_metrics,
        }
        
        print(f"Frames evaluated:  {summary['num_frames_evaluated']}")
        print("\nPSNR (dB):")
        print(f"  Mean: {summary['psnr_mean']:.2f} ± {summary['psnr_std']:.2f}")
        print(f"  Range: [{summary['psnr_min']:.2f}, {summary['psnr_max']:.2f}]")
        print("\nSSIM:")
        print(f"  Mean: {summary['ssim_mean']:.4f} ± {summary['ssim_std']:.4f}")
        print(f"  Range: [{summary['ssim_min']:.4f}, {summary['ssim_max']:.4f}]")
        print("\nLPIPS (lower is better):")
        print(f"  Mean: {summary['lpip_mean']:.4f} ± {summary['lpip_std']:.4f}")
        print(f"  Range: [{summary['lpip_min']:.4f}, {summary['lpip_max']:.4f}]")
        
        # Save metrics to JSON if requested
        if args.save_metrics:
            # Add suffix based on evaluation mode
            if args.average_tokens:
                refined_suffix = "_refined" if args.refine_poses else ""
                metrics_suffix = f"_averaged_{args.weighting_type}{refined_suffix}_metrics.json"
            else:
                metrics_suffix = "_metrics.json"
            metrics_path = os.path.join(args.output_dir, f"{args.scene_name}{metrics_suffix}")
            with open(metrics_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\nSaved metrics to {metrics_path}")
    else:
        print("\nNo frames were successfully processed!")
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

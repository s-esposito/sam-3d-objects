"""
Re-decode script for SAM3D.

This script loads cached decoder inputs (SLAT tokens) from disk and re-runs
just the decoder forward pass to regenerate Gaussians and/or Meshes.

Both the Gaussian and Mesh decoders share the same SLAT tokens as input,
so this script can regenerate either or both outputs from the cached tokens.

Features:
- Re-decode single frame tokens
- Average tokens across all frames to get a "canonical" object representation

Input:
    - SLAT tokens from: custom/results/{dataset}/tokens/

Output:
    - Gaussians saved to: custom/results/{dataset}/{redecoded|averaged}/gaussians/ (with _gaussians.ply suffix)
    - Meshes saved to: custom/results/{dataset}/{redecoded|averaged}/meshes/ (with _mesh.obj suffix)
    - Renders saved to: custom/results/{dataset}/{redecoded|averaged}/renders/ (with .png suffix)

Usage:
    # Single frame re-decode
    python notebook/redecode.py --dataset kubric4d --scene-name scn02719 --frame-index 0
    
    # Average tokens across all frames for canonical object
    python notebook/redecode.py --dataset kubric4d --scene-name scn02719 --average-frames
"""
import os
import sys
import argparse
import glob
import numpy as np
import torch

# Skip sam3d_objects initialization for lightweight usage
os.environ['LIDRA_SKIP_INIT'] = '1'

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp
from inference import Inference
from utils import (
    save_mesh_to_obj,
    load_masks,
    redecode_slat,
    average_slat_tokens,
    compute_frame_weights_from_error,
    compute_frame_weights_from_masks,
    setup_paths,
    load_decoder_inputs_from_cache,
    load_all_frame_tokens,
    get_cache_filename
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Re-decode SLAT tokens to Gaussians and/or Meshes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["kubric4d", "davis"],
        default="kubric4d",
        help="Dataset type",
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
        help="Name of the scene. Defaults: kubric4d='scn02719', davis='car-turn'",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="Frame index (0-based) for single-frame mode. If not specified, processes all available frames.",
    )
    
    # Averaging mode
    parser.add_argument(
        "--average-frames",
        action="store_true",
        help="Average SLAT tokens across all available frames to get canonical object representation",
    )
    parser.add_argument(
        "--weighted-average",
        action="store_true",
        help="Use weighted averaging based on mask visibility (only with --average-frames)",
    )
    
    # Cache configuration (must match how demo.py saved the cache)
    parser.add_argument(
        "--first-object-only",
        action="store_true",
        help="Load cache from --first-object-only run",
    )
    parser.add_argument(
        "--with-background",
        action="store_true",
        help="Load cache from --with-background run",
    )
    
    # Decoding options
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=["gaussian", "mesh"],
        choices=["gaussian", "mesh", "gaussian_4"],
        help="Output formats to decode",
    )
    parser.add_argument(
        "--object-index",
        type=int,
        default=None,
        help="Index of specific object to process (0-based). If not specified, processes all.",
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save outputs. Defaults to gaussians/<dataset>/redecoded/",
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Get paths
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
    
    if args.output_dir is None:
        suffix = "averaged" if args.average_frames else "redecoded"
        custom_dir = os.path.join(PROJECT_ROOT, "custom")
        args.output_dir = os.path.join(custom_dir, f"results/{args.dataset}/{suffix}")
    
    # Setup paths - now looks in custom/results/{dataset}/tokens/
    cached_results_path = setup_paths(dataset_type=args.dataset)
    
    # Create subdirectories for organized output
    gaussians_dir = os.path.join(args.output_dir, "gaussians")
    meshes_dir = os.path.join(args.output_dir, "meshes")
    
    os.makedirs(gaussians_dir, exist_ok=True)
    os.makedirs(meshes_dir, exist_ok=True)
    
    # Print configuration
    print("=" * 60)
    print("SAM3D Re-decode Configuration")
    print("=" * 60)
    print(f"Dataset:           {args.dataset}")
    print(f"Scene name:        {args.scene_name}")
    print(f"Average frames:    {args.average_frames}")
    if not args.average_frames and args.frame_index is not None:
        print(f"Frame index:       {args.frame_index}")
    elif not args.average_frames:
        print(f"Processing:        All available frames")
    print(f"Decode formats:    {args.formats}")
    print(f"Object index:      {args.object_index if args.object_index is not None else 'all'}")
    print(f"Output directory:  {args.output_dir}")
    print(f"  Gaussians:       {gaussians_dir}")
    print(f"  Meshes:          {meshes_dir}")
    print("=" * 60)
    
    # Initialize inference pipeline (needed to access the decoders)
    print("\nInitializing inference pipeline...")
    config_path = os.path.join(PROJECT_ROOT, "checkpoints", "hf", "pipeline.yaml")
    inference = Inference(config_path, compile=False)
    pipeline = inference._pipeline
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.average_frames:
        # Load tokens from all frames and average
        print("\nLoading tokens from all frames...")
        tokens_by_object = load_all_frame_tokens(
            cached_results_path, args.scene_name,
            args.first_object_only, args.with_background
        )
        
        if not tokens_by_object:
            print("\nNo cache files found!")
            print("Make sure you have run demo.py for multiple frames:")
            print(f"  python notebook/demo.py --dataset {args.dataset} --scene-name {args.scene_name} --frame-stride 10")
            return
        
        print(f"\nFound tokens for {len(tokens_by_object)} objects")
        
        # Determine which objects to process
        if args.object_index is not None:
            if args.object_index not in tokens_by_object:
                print(f"Error: Object index {args.object_index} not found")
                return
            object_indices = [args.object_index]
        else:
            object_indices = sorted(tokens_by_object.keys())
        
        # Process each object
        for obj_idx in object_indices:
            tokens_list = tokens_by_object[obj_idx]
            num_frames = len(tokens_list)
            
            print(f"\n{'='*60}")
            print(f"Processing object {obj_idx + 1} ({num_frames} frames)")
            print(f"{'='*60}")
            
            # Compute weights if weighted averaging is enabled
            weights = None
            if args.weighted_average:
                print("Computing frame weights from mask visibility...")
                weights = compute_frame_weights_from_masks(
                    tokens_list, 
                    args.dataset_path, 
                    args.scene_name, 
                    args.dataset,
                    obj_idx
                )
            
            # Average the SLAT tokens (with or without weights)
            avg_slat = average_slat_tokens(tokens_list, weights=weights)
            
            # Decode the averaged tokens
            decoded_outputs = redecode_slat(pipeline, avg_slat, formats=args.formats)
            
            # Save outputs
            suffix = "weighted" if weights is not None else "averaged"
            base_name = f"{args.scene_name}_object_{obj_idx+1}_{suffix}_{num_frames}frames"
            
            if "gaussian" in decoded_outputs:
                gaussians = decoded_outputs["gaussian"][0]
                ply_path = os.path.join(gaussians_dir, f"{base_name}_gaussians.ply")
                gaussians.save_ply(ply_path)
                print(f"Saved Gaussians to: {ply_path}")
            
            if "mesh" in decoded_outputs:
                mesh = decoded_outputs["mesh"][0]
                obj_path = os.path.join(meshes_dir, f"{base_name}_mesh.obj")
                save_mesh_to_obj(mesh, obj_path)
                print(f"Saved Mesh to: {obj_path}")
    
    else:
        # Multi-frame or single-frame mode
        # If frame_index is specified, process only that frame
        # Otherwise, process all available frames
        
        if args.frame_index is not None:
            # Single frame mode
            frame_indices = [args.frame_index]
        else:
            # Multi-frame mode - find all available cache files
            cache_pattern = f"{args.scene_name}_f*"
            if args.first_object_only:
                cache_pattern += "_first"
            if args.with_background:
                cache_pattern += "_bg"
            cache_pattern += "_sam3d_results.npz"
            
            cache_files = sorted(glob.glob(os.path.join(cached_results_path, cache_pattern)))
            
            if not cache_files:
                # Try without suffixes
                cache_pattern = f"{args.scene_name}_f*_sam3d_results.npz"
                cache_files = sorted(glob.glob(os.path.join(cached_results_path, cache_pattern)))
            
            if not cache_files:
                print(f"\nError: No cache files found for scene '{args.scene_name}'")
                print(f"Looked in: {cached_results_path}")
                print("\nMake sure you have run demo.py first to save decoder inputs.")
                print(f"Example: python custom/demo.py --dataset {args.dataset} --scene-name {args.scene_name} --frame-stride 10")
                return
            
            # Extract frame indices from filenames
            frame_indices = []
            for cache_file in cache_files:
                basename = os.path.basename(cache_file)
                try:
                    parts = basename.split('_')
                    frame_part = [p for p in parts if p.startswith('f') and p[1:].isdigit()][0]
                    frame_idx = int(frame_part[1:])
                    frame_indices.append(frame_idx)
                except (IndexError, ValueError):
                    continue
            
            frame_indices = sorted(frame_indices)
            print(f"\nFound {len(frame_indices)} frames to process: {frame_indices}")
        
        # Process each frame
        for frame_idx in frame_indices:
            cache_filename, cache_scene_name = get_cache_filename(
                args.scene_name, frame_idx, args.first_object_only, args.with_background
            )
            cache_file = os.path.join(cached_results_path, cache_filename)
            
            print(f"\n{'='*60}")
            print(f"Processing frame {frame_idx}")
            print(f"{'='*60}")
            print(f"Cache file: {cache_file}")
            
            # Check if cache file exists
            if not os.path.exists(cache_file):
                print(f"Warning: Cache file not found, skipping frame {frame_idx}")
                continue
            
            # Load decoder inputs from cache
            print("Loading decoder inputs...")
            decoder_inputs = load_decoder_inputs_from_cache(cache_file)
            print(f"Loaded {len(decoder_inputs)} objects")
            
            if len(decoder_inputs) == 0:
                print("No decoder inputs found in cache file, skipping.")
                continue
            
            # Determine which objects to process
            if args.object_index is not None:
                if args.object_index >= len(decoder_inputs):
                    print(f"Warning: Object index {args.object_index} out of range (0-{len(decoder_inputs)-1}), skipping frame")
                    continue
                object_indices = [args.object_index]
            else:
                object_indices = range(len(decoder_inputs))
            
            # Re-decode for each object
            for i in object_indices:
                print(f"\n  Re-decoding object {i + 1}/{len(decoder_inputs)}")
                
                decoder_input = decoder_inputs[i]
                slat = decoder_input['decoder_input_slat']
                
                # Re-run the decoder
                decoded_outputs = redecode_slat(pipeline, slat, formats=args.formats)
                
                # Save outputs
                base_name = f"{cache_scene_name}_object_{i+1}_redecoded"
                
                if "gaussian" in decoded_outputs:
                    gaussians = decoded_outputs["gaussian"][0]
                    ply_path = os.path.join(gaussians_dir, f"{base_name}_gaussians.ply")
                    gaussians.save_ply(ply_path)
                    print(f"  Saved Gaussians to: {ply_path}")
                
                if "mesh" in decoded_outputs:
                    mesh = decoded_outputs["mesh"][0]
                    obj_path = os.path.join(meshes_dir, f"{base_name}_mesh.obj")
                    save_mesh_to_obj(mesh, obj_path)
                    print(f"  Saved Mesh to: {obj_path}")
                
                # Print layout info (from original inference)
                if len(frame_indices) == 1:  # Only print details for single frame
                    print("\n  Original layout parameters:")
                    print(f"    Rotation:    {decoder_input['rotation'].cpu().numpy()}")
                    print(f"    Translation: {decoder_input['translation'].cpu().numpy()}")
                    print(f"    Scale:       {decoder_input['scale'].cpu().numpy()}")
    
    print(f"\n{'='*60}")
    print("Re-decoding complete!")
    print(f"Outputs saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

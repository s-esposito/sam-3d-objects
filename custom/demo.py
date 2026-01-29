"""
Demo script for Kubric4D and DAVIS dataset processing with SAM3D.
Processes multi-object scenes with depth maps and generates 3D Gaussian representations.

Supports:
- Kubric4D dataset with ground truth depth
- DAVIS dataset with MoGe depth estimation
- Background rendering (creating Gaussians for non-masked regions)
- Caching SLAT tokens for fast re-decoding

Usage Examples:

    # Single frame inference (Kubric4D with GT depth)
    python custom/demo.py --dataset kubric4d --scene-name scn02719 --frame-index 0
    
    # Multiple frames with stride (saves SLAT tokens for averaging)
    python custom/demo.py --dataset kubric4d --scene-name scn02719 --frame-stride 10
    
    # DAVIS dataset with MoGe depth estimation
    python custom/demo.py --dataset davis --scene-name car-turn --frame-index 0
    
    # Include background in reconstruction
    python custom/demo.py --dataset kubric4d --scene-name scn02719 --frame-index 0 --with-background
    
    # Process only first object
    python custom/demo.py --dataset kubric4d --scene-name scn02719 --frame-index 0 --first-object-only
    
    # Custom output formats
    python custom/demo.py --dataset kubric4d --scene-name scn02719 --frame-index 0 --formats gaussian mesh
    
    # Specify custom dataset path
    python custom/demo.py --dataset kubric4d --dataset-path /path/to/kubric4d --scene-name scn02719

Output:
    - Renders saved to: custom/results/{dataset}/renders (with .png suffix)
    - Gaussians saved to: custom/results/{dataset}/gaussians (with _gaussians.ply suffix)
    - Meshes saved to: custom/results/{dataset}/meshes (with _mesh.obj suffix)
    - SLAT tokens: custom/results/{dataset}/tokens/
"""
import os
import time
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils import (
    setup_paths,
    load_and_process_depth,
    transform_to_pytorch3d_convention,
    run_inference_on_masks,
    transform_scene_to_r3_convention,
    create_background_gaussians,
    load_image, 
    load_masks
)

import torch
from inference import Inference, make_scene
from utils import (
    join_gaussians,
    save_mesh_to_obj,
)



def visualize_image_and_masks(image, masks, output_path="image_and_masks.png"):
    """Create visualization of image with masks overlay."""
    def imshow(img, ax):
        ax.axis("off")
        ax.imshow(img)

    grid = (1, 1) if masks is None else (2, 2)
    fig, axes = plt.subplots(*grid, figsize=(12, 12))
    
    if masks is not None:
        mask_colors = sns.color_palette("husl", len(masks))
        black_image = np.zeros_like(image[..., :3], dtype=float)
        mask_display = np.copy(black_image)
        mask_union = np.zeros_like(image[..., :3])
        
        for i, mask in enumerate(masks):
            mask_display[mask] = mask_colors[i]
            mask_union |= mask[..., None] if mask.ndim == 2 else mask
        
        imshow(black_image, axes[0, 1])
        imshow(mask_display, axes[1, 0])
        imshow(image * mask_union, axes[1, 1])
        image_axe = axes[0, 0]
    else:
        image_axe = axes

    imshow(image, image_axe)
    fig.tight_layout(pad=0)
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved visualization to {output_path}")


def visualize_pointmap(pointmap, output_path="pointmap_visualization.png"):
    """Visualize the 3D pointmap with color coding and depth map."""
    # Map position to RGB colors for visualization
    normed_x = (pointmap[..., 0] - pointmap[..., 0].min()) / (pointmap[..., 0].max() - pointmap[..., 0].min() + 1e-8)
    normed_y = (pointmap[..., 1] - pointmap[..., 1].min()) / (pointmap[..., 1].max() - pointmap[..., 1].min() + 1e-8)
    normed_z = (pointmap[..., 2] - pointmap[..., 2].min()) / (pointmap[..., 2].max() - pointmap[..., 2].min() + 1e-8)
    color_map = np.stack([normed_x, normed_y, normed_z], axis=-1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Color encoding of pointmap
    ax1.imshow(color_map)
    ax1.set_title('Pointmap Color Visualization (RGB=XYZ)', fontsize=14)
    ax1.axis('off')

    # Depth visualization
    im = ax2.imshow(pointmap[..., 2], cmap='plasma')
    ax2.set_title('Pointmap Depth Visualization', fontsize=14)
    ax2.axis('off')

    # Add colorbar
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label='Depth (Z-coordinate)')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved pointmap visualization to {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Demo script for Kubric4D and DAVIS dataset processing with SAM3D",
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
        help="Frame index to process (0-based). If not specified, processes all frames with --frame-stride",
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
        help="Only process the first object/mask (useful for debugging)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for inference",
    )
    
    # Background and joint predictions
    parser.add_argument(
        "--with-background",
        action="store_true",
        help="Add background Gaussians from non-masked regions",
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save outputs. Defaults to gaussians/<dataset>/ in script directory",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cached results and rerun inference",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip visualization outputs",
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Get the project root directory (parent of custom directory)
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
        args.output_dir = os.path.join(SCRIPT_DIR, f"results/{args.dataset}")
    
    # Create subdirectories for organized output
    renders_dir = os.path.join(args.output_dir, "renders")
    gaussians_dir = os.path.join(args.output_dir, "gaussians")
    meshes_dir = os.path.join(args.output_dir, "meshes")
    tokens_dir = os.path.join(args.output_dir, "tokens")
    
    os.makedirs(renders_dir, exist_ok=True)
    os.makedirs(gaussians_dir, exist_ok=True)
    os.makedirs(meshes_dir, exist_ok=True)
    os.makedirs(tokens_dir, exist_ok=True)
    
    # Setup paths (pass tokens_dir to save SLAT tokens)
    paths = setup_paths(args.dataset_path, args.scene_name, dataset_type=args.dataset)
    
    # Determine which frames to process
    num_frames = len(paths['image_names'])
    if args.frame_index is not None:
        # Process single frame
        frame_indices = [args.frame_index]
    else:
        # Process all frames with stride
        frame_indices = list(range(0, num_frames, args.frame_stride))
    
    # Print configuration
    print("=" * 60)
    print("SAM3D Demo Configuration")
    print("=" * 60)
    print(f"Dataset:           {args.dataset}")
    print(f"Dataset path:      {args.dataset_path}")
    print(f"Scene name:        {args.scene_name}")
    print(f"Total frames:      {num_frames}")
    if args.frame_index is not None:
        print(f"Frame index:       {args.frame_index}")
    else:
        print(f"Frame stride:      {args.frame_stride}")
        print(f"Frames to process: {len(frame_indices)} frames")
    print(f"Use MoGe depth:    {args.use_moge}")
    print(f"First object only: {args.first_object_only}")
    print(f"With background:   {args.with_background}")
    print(f"Seed:              {args.seed}")
    print(f"Output directory:  {args.output_dir}")
    print(f"  Renders:         {renders_dir}")
    print(f"  Gaussians:       {gaussians_dir}")
    print(f"  Meshes:          {meshes_dir}")
    print(f"  SLAT tokens:     {tokens_dir}")
    print("=" * 60)
    
    # Initialize inference pipeline once (shared across all frames)
    inference = None
    TAG = "hf"
    config_path = os.path.join(PROJECT_ROOT, "checkpoints", TAG, "pipeline.yaml")
    print(f"Initializing inference pipeline from {config_path}")
    inference = Inference(config_path, compile=False)
    
    # Process each frame
    for frame_idx, frame_index in enumerate(frame_indices):
        print(f"\n{'='*60}")
        print(f"Processing frame {frame_index} ({frame_idx + 1}/{len(frame_indices)})")
        print(f"{'='*60}")
        
        process_frame(
            args=args,
            paths=paths,
            frame_index=frame_index,
            inference=inference,
            renders_dir=renders_dir,
            gaussians_dir=gaussians_dir,
            meshes_dir=meshes_dir,
            tokens_dir=tokens_dir,
        )
    
    print(f"\n{'='*60}")
    print(f"=== All {len(frame_indices)} frames processed! ===")
    print(f"{'='*60}")


def process_frame(args, paths, frame_index, inference, renders_dir, gaussians_dir, meshes_dir, tokens_dir):
    """Process a single frame."""
    # Load frame's image and masks
    image_path = os.path.join(paths['frames_path'], paths['image_names'][frame_index])
    mask_path = os.path.join(paths['masks_path'], paths['mask_names'][frame_index])
    
    image = load_image(image_path)
    image = image[..., :3]  # Drop alpha channel
    H, W, _ = image.shape
    
    masks = load_masks(mask_path)
    
    print(f"\nLoaded image: shape={image.shape}, dtype={image.dtype}, "
          f"min={image.min()}, max={image.max()}")
    print(f"Loaded {len(masks)} masks")
    
    # Filter masks if only processing first object
    if args.first_object_only:
        masks = masks[:1]
        print("--first-object-only: Processing only 1 mask")
    
    # Modify cache name based on configuration to avoid conflicts
    cache_parts = [args.scene_name, f"f{frame_index}"]
    if args.first_object_only:
        cache_parts.append("first")
    if args.with_background:
        cache_parts.append("bg")
    cache_scene_name = "_".join(cache_parts)
    
    # Visualize input data
    if not args.no_visualize:
        viz_path = os.path.join(renders_dir, f"{cache_scene_name}_input.png")
        visualize_image_and_masks(image, masks, output_path=viz_path)

    # Always run inference
    need_inference = True
    
    # Process depth and generate pointmap
    pointmap_original = None  # Keep original pointmap for background (before P3D transform)
    
    pointmap, K_matrix, valid_mask = load_and_process_depth(
        paths['frames_path'], 
        paths['depth_names'], 
        W, H, 
        use_moge=args.use_moge,
        inference=inference,
        image=image
    )
    
    # Keep copy of original pointmap for background rendering
    if args.with_background:
        pointmap_original = pointmap.copy()
    
    # Visualize pointmap
    if not args.no_visualize:
        pointmap_viz_path = os.path.join(renders_dir, f"{cache_scene_name}_pointmap.png")
        visualize_pointmap(pointmap, output_path=pointmap_viz_path)
    
    # Transform to PyTorch3D convention for inference
    pointmap = transform_to_pytorch3d_convention(pointmap)
    print("Transformed pointmap to PyTorch3D convention")

    # Visualize pointmap after transformation
    if not args.no_visualize:
        pointmap_p3d_viz_path = os.path.join(renders_dir, f"{cache_scene_name}_pointmap_p3d.png")
        visualize_pointmap(pointmap, output_path=pointmap_p3d_viz_path)
    
    if need_inference:
        # Run inference on all masks
        outputs = run_inference_on_masks(inference, image, masks, pointmap, seed=args.seed)
        
        # Cache results
        save_tokens(tokens_dir, cache_scene_name, outputs)
    
    # Save each raw object (before layout transform) for debugging
    from copy import deepcopy
    for i, output in enumerate(outputs):
        # Save raw Gaussian (canonical frame, before layout transform)
        raw_gs = deepcopy(output["gaussian"][0])
        raw_ply_path = os.path.join(gaussians_dir, f"{cache_scene_name}_object_{i+1}_raw_gaussians.ply")
        raw_gs.save_ply(raw_ply_path)
        print(f"Saved raw object {i+1} Gaussian (before layout) to {raw_ply_path}")
        
        # Save raw mesh if available (canonical frame, before layout transform)
        if "mesh" in output and output["mesh"] is not None:
            raw_mesh = output["mesh"][0]
            raw_mesh_path = os.path.join(meshes_dir, f"{cache_scene_name}_object_{i+1}_raw_mesh.obj")
            save_mesh_to_obj(raw_mesh, raw_mesh_path)
            print(f"Saved raw object {i+1} mesh (before layout) to {raw_mesh_path}")
        
        # Save GLB if available (includes texture)
        if "glb" in output and output["glb"] is not None:
            glb_path = os.path.join(meshes_dir, f"{cache_scene_name}_object_{i+1}_raw.glb")
            output["glb"].export(glb_path)
            print(f"Saved raw object {i+1} GLB to {glb_path}")
    
    # Save each object with layout applied (in PyTorch3D convention)
    for i, output in enumerate(outputs):
        obj_ply_path = os.path.join(gaussians_dir, f"{cache_scene_name}_object_{i+1}_gaussians.ply")
        obj_gs = make_scene(output)
        obj_gs.save_ply(obj_ply_path)
        print(f"Saved object {i+1} Gaussian (with layout) to {obj_ply_path}")
    
    # Create combined scene from all outputs (in PyTorch3D convention)
    print("\nCreating combined Gaussian scene...")
    scene_gs = make_scene(*outputs)
    
    # # Save as ply for debugging
    # debug_ply_path = os.path.join(args.output_dir, f"{cache_scene_name}_pytorch3d_convention.ply")
    # os.makedirs(args.output_dir, exist_ok=True)
    # scene_gs.save_ply(debug_ply_path)
    # print(f"Saved PyTorch3D convention Gaussian scene to {debug_ply_path}")
    
    # Transform scene from PyTorch3D to R3 convention (positions only, not rotations)
    print("Transforming scene to R3 convention...")
    new_scene_gs = transform_scene_to_r3_convention(scene_gs)
    
    # # Save as ply for debugging
    # debug_ply_path_r3 = os.path.join(args.output_dir, f"{cache_scene_name}_r3_convention.ply")
    # os.makedirs(args.output_dir, exist_ok=True)
    # new_scene_gs.save_ply(debug_ply_path_r3)
    # print(f"Saved R3 convention Gaussian scene to {debug_ply_path_r3}")
    
    # Add background Gaussians if requested
    if args.with_background and pointmap_original is not None:
        print("Creating background Gaussians...")
        background_gs = create_background_gaussians(
            image, pointmap_original, masks, K_matrix
        )
        print(f"Background Gaussians: {background_gs.get_xyz.shape[0]} points")
        
        # Join background with scene
        new_scene_gs = join_gaussians(background_gs, new_scene_gs)
        print(f"Combined scene: {new_scene_gs.get_xyz.shape[0]} total Gaussians")
    
    # Save gaussian splatting as PLY
    ply_path = os.path.join(gaussians_dir, f"{cache_scene_name}_gaussians.ply")
    new_scene_gs.save_ply(ply_path)
    print(f"Saved Gaussian scene to {ply_path}")
    
    # Render and compare with original
    if not args.no_visualize:
        output_render_path = os.path.join(renders_dir, f"{cache_scene_name}_render.png")
        render_and_compare(new_scene_gs, image, K_matrix, W, H, output_path=output_render_path)
    
    print(f"Frame {frame_index} complete!")


if __name__ == "__main__":
    main()

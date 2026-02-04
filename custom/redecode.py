"""
Re-decode script for SAM3D.

This script loads cached decoder inputs (SLAT tokens) from disk and re-runs
just the decoder forward pass to regenerate Gaussians and/or Meshes.

Both the Gaussian and Mesh decoders share the same SLAT tokens as input,
so this script can regenerate either or both outputs from the cached tokens.

Features:
- Re-decode single frame tokens
- Average tokens across all frames to get a "canonical" object representation
- Render outputs from multiple viewpoints for visualization (enabled by default)

Input:
    - SLAT tokens from: custom/results/{dataset}/tokens/

Output:
    - Gaussians saved to: custom/results/{dataset}/{perframe|averaged}/gaussians/ (with _gaussians.ply suffix)
    - Meshes saved to: custom/results/{dataset}/{perframe|averaged}/meshes/ (with _mesh.obj suffix)
    - Renders saved to: custom/results/{dataset}/{perframe|averaged}/renders/ (with .png suffix)

Usage:
    # Single frame re-decode (with rendering)
    python custom/redecode.py --dataset kubric4d --scene-name scn02719 --frame-index 0
    
    # Average tokens across all frames for canonical object
    python custom/redecode.py --dataset kubric4d --scene-name scn02719 --average-frames
    
    # Re-decode without rendering
    python custom/redecode.py --dataset kubric4d --scene-name scn02719 --average-frames --no-render
"""
import os
import sys
import argparse
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Skip sam3d_objects initialization for lightweight usage
os.environ['LIDRA_SKIP_INIT'] = '1'

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardFlatShader,
    BlendParams,
    TexturesVertex,
)
from pytorch3d.io import load_objs_as_meshes

from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp  # noqa: F401
from inference import Inference
from custom.utils import (
    save_mesh_to_obj,
    redecode_slat,
    average_slat_tokens,
    compute_frame_weights_from_masks,
    load_decoder_inputs_from_cache,
    load_all_frame_tokens,
    get_cache_filename,
    render_gaussians_scene,
)


# ============================================================================
# Rendering utilities
# ============================================================================

def get_camera_positions(distance=2.0):
    """
    Get camera positions for 6 standard viewpoints.
    
    Returns
    -------
    dict
        Maps view name -> (R, T) tuple for camera transform
    """
    views = {}
    
    # Top view (looking down)
    R, T = look_at_view_transform(dist=distance, elev=90, azim=0)
    views['top'] = (R, T)
    
    # Bottom view (looking up)
    R, T = look_at_view_transform(dist=distance, elev=-90, azim=0)
    views['bottom'] = (R, T)
    
    # Front view
    R, T = look_at_view_transform(dist=distance, elev=0, azim=0)
    views['front'] = (R, T)
    
    # Back view
    R, T = look_at_view_transform(dist=distance, elev=0, azim=180)
    views['back'] = (R, T)
    
    # Left view
    R, T = look_at_view_transform(dist=distance, elev=0, azim=-90)
    views['left'] = (R, T)
    
    # Right view
    R, T = look_at_view_transform(dist=distance, elev=0, azim=90)
    views['right'] = (R, T)
    
    return views


def render_gaussian_from_view(gaussian, R, T, image_size=512, fov=60.0):
    """
    Render Gaussian from a specific viewpoint.
    
    Parameters
    ----------
    gaussian : GaussianModel
        Gaussian splatting model
    R : torch.Tensor
        Rotation matrix (1, 3, 3)
    T : torch.Tensor
        Translation vector (1, 3)
    image_size : int
        Output image size
    fov : float
        Field of view in degrees
        
    Returns
    -------
    np.ndarray
        Rendered image (H, W, 3)
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create camera-to-world matrix
    # PyTorch3D uses row-major convention
    w2c = torch.eye(4, device=device)
    w2c[:3, :3] = R[0].T  # Transpose because PyTorch3D convention
    w2c[:3, 3] = T[0]
    c2w = torch.inverse(w2c)
    
    # Create intrinsics (simple perspective)
    focal_length = image_size / (2 * np.tan(np.radians(fov) / 2))
    K = torch.eye(3, device=device)
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[0, 2] = image_size / 2
    K[1, 2] = image_size / 2
    
    # Render using the utility function with white background
    white_bg = torch.ones(3, device=device)
    rendered, alpha = render_gaussians_scene(
        gaussian,
        c2w=c2w,
        K=K,
        w=image_size,
        h=image_size,
        bg_color=white_bg
    )
    
    return rendered.cpu().numpy()


def render_mesh_from_view(mesh, R, T, image_size=512, fov=60.0, device=None):
    """
    Render mesh from a specific viewpoint using PyTorch3D.
    
    Parameters
    ----------
    mesh : Meshes
        PyTorch3D mesh object
    R : torch.Tensor
        Rotation matrix (1, 3, 3)
    T : torch.Tensor
        Translation vector (1, 3)
    image_size : int
        Output image size
    fov : float
        Field of view in degrees
    device : torch.device
        Device to render on
        
    Returns
    -------
    np.ndarray
        Rendered image (H, W, 3)
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Move mesh to device
    mesh = mesh.to(device)
    
    # Create cameras
    cameras = FoVPerspectiveCameras(
        device=device,
        R=R.to(device),
        T=T.to(device),
        fov=fov
    )
    
    # Create rasterizer
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    
    # Create blend params that don't attenuate colors
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(1.0, 1.0, 1.0))
    
    # Create renderer with flat shader (no lighting, just albedo/vertex colors)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardFlatShader(
            device=device,
            cameras=cameras,
            blend_params=blend_params
        )
    )
    
    # Render
    with torch.no_grad():
        images = renderer(mesh)
    
    # Extract RGB (drop alpha channel)
    image = images[0, ..., :3]
    
    # HardFlatShader seems to darken colors even without lighting
    # Apply a brightness boost to compensate (empirical adjustment)
    image = torch.clamp(image * 1.8, 0.0, 1.0)
    
    # Flip vertically and horizontally to match coordinate system
    image = torch.flip(image, dims=[0, 1])
    
    return image.cpu().numpy()


def create_comparison_grid(gaussian_renders, mesh_renders, view_names):
    """
    Create a grid showing Gaussian and Mesh renders side by side.
    
    Parameters
    ----------
    gaussian_renders : dict
        Maps view_name -> rendered image
    mesh_renders : dict
        Maps view_name -> rendered image
    view_names : list
        List of view names to display
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with the comparison grid
    """
    n_views = len(view_names)
    
    # Create figure with 2 rows (Gaussians, Meshes) and n_views columns
    fig = plt.figure(figsize=(4 * n_views, 8))
    gs = gridspec.GridSpec(2, n_views, figure=fig, hspace=0.05, wspace=0.05)
    
    for col, view_name in enumerate(view_names):
        # Gaussian render
        ax_gaussian = fig.add_subplot(gs[0, col])
        ax_gaussian.imshow(gaussian_renders[view_name])
        ax_gaussian.axis('off')
        if col == 0:
            ax_gaussian.set_ylabel('Gaussians', fontsize=16, rotation=0, labelpad=60, va='center')
        ax_gaussian.set_title(view_name.capitalize(), fontsize=14)
        
        # Mesh render
        ax_mesh = fig.add_subplot(gs[1, col])
        ax_mesh.imshow(mesh_renders[view_name])
        ax_mesh.axis('off')
        if col == 0:
            ax_mesh.set_ylabel('Mesh', fontsize=16, rotation=0, labelpad=60, va='center')
    
    return fig


def render_outputs(gaussian, mesh_path, output_path, image_size=512, distance=2.0, fov=60.0):
    """
    Render Gaussian and Mesh from multiple viewpoints and create comparison grid.
    
    Parameters
    ----------
    gaussian : Gaussian
        Decoded Gaussian model
    mesh_path : str
        Path to saved mesh .obj file
    output_path : str
        Path to save the comparison image
    image_size : int
        Size of rendered images
    distance : float
        Camera distance from object
    fov : float
        Field of view in degrees
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Get camera viewpoints
    views = get_camera_positions(distance=distance)
    view_names = ['front', 'back', 'left', 'right', 'top', 'bottom']
    
    # Render Gaussian from all views
    print("    Rendering Gaussian from multiple viewpoints...")
    gaussian_renders = {}
    for view_name in view_names:
        R, T = views[view_name]
        rendered = render_gaussian_from_view(
            gaussian, R, T,
            image_size=image_size,
            fov=fov
        )
        gaussian_renders[view_name] = rendered
    
    # Render mesh from all views if available
    mesh_renders = {}
    if mesh_path and os.path.exists(mesh_path):
        print("    Rendering Mesh from multiple viewpoints...")
        try:
            # Load vertex colors from OBJ file manually
            verts_rgb = None
            with open(mesh_path, 'r') as f:
                vertex_colors = []
                for line in f:
                    if line.startswith('v '):
                        parts = line.strip().split()
                        # Check if vertex has color (format: v x y z r g b)
                        if len(parts) >= 7:
                            r, g, b = float(parts[4]), float(parts[5]), float(parts[6])
                            vertex_colors.append([r, g, b])
                        else:
                            # No color, use white
                            vertex_colors.append([1.0, 1.0, 1.0])
                
                if vertex_colors:
                    verts_rgb = torch.tensor(vertex_colors, dtype=torch.float32, device=device)[None]
            
            # Load mesh using PyTorch3D
            mesh = load_objs_as_meshes([mesh_path], device=device)
            
            # Apply vertex colors as texture
            if verts_rgb is not None:
                mesh.textures = TexturesVertex(verts_features=verts_rgb)
            elif mesh.textures is None:
                # Fallback: Add simple white texture if none exists
                verts_rgb = torch.ones_like(mesh.verts_packed())[None]
                mesh.textures = TexturesVertex(verts_features=verts_rgb.to(device))
            
            for view_name in view_names:
                R, T = views[view_name]
                rendered = render_mesh_from_view(
                    mesh, R, T,
                    image_size=image_size,
                    fov=fov,
                    device=device
                )
                mesh_renders[view_name] = rendered
        except Exception as e:
            print(f"    Error rendering mesh: {e}")
            # Create blank renders
            for view_name in view_names:
                mesh_renders[view_name] = np.ones((image_size, image_size, 3))
    else:
        # No mesh, create blank renders
        for view_name in view_names:
            mesh_renders[view_name] = np.ones((image_size, image_size, 3))
    
    # Create comparison grid and save
    print("    Creating comparison visualization...")
    fig = create_comparison_grid(gaussian_renders, mesh_renders, view_names)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved render: {output_path}")


# ============================================================================
# Argument parsing
# ============================================================================

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
        "--cache-object-index",
        type=int,
        default=None,
        help="Load cache from run with this --object-index (for cache filename matching)",
    )
    parser.add_argument(
        "--no-background",
        action="store_true",
        help="Load cache from run without background (cache files without 'bg' suffix)",
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
        help="Directory to save outputs. Defaults to gaussians/<dataset>/perframe/",
    )
    
    # Rendering options
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering outputs from multiple viewpoints",
    )
    parser.add_argument(
        "--render-size",
        type=int,
        default=512,
        help="Size of rendered images",
    )
    parser.add_argument(
        "--render-distance",
        type=float,
        default=2.0,
        help="Camera distance from object for rendering",
    )
    parser.add_argument(
        "--render-fov",
        type=float,
        default=40.0,
        help="Field of view in degrees for rendering",
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
        suffix = "averaged" if args.average_frames else "perframe"
        custom_dir = os.path.join(PROJECT_ROOT, "custom")
        args.output_dir = os.path.join(custom_dir, f"results/{args.dataset}/{suffix}")
    
    # Cached results path - where demo.py saves SLAT tokens
    custom_dir = os.path.join(PROJECT_ROOT, "custom")
    cached_results_path = os.path.join(custom_dir, f"results/{args.dataset}/tokens")
    
    # Create subdirectories for organized output
    gaussians_dir = os.path.join(args.output_dir, "gaussians")
    meshes_dir = os.path.join(args.output_dir, "meshes")
    renders_dir = os.path.join(args.output_dir, "renders")
    
    os.makedirs(gaussians_dir, exist_ok=True)
    os.makedirs(meshes_dir, exist_ok=True)
    if not args.no_render:
        os.makedirs(renders_dir, exist_ok=True)
    
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
        print("Processing:        All available frames")
    print(f"Decode formats:    {args.formats}")
    print(f"Object index:      {args.object_index if args.object_index is not None else 'all'}")
    print(f"Render:            {not args.no_render}")
    print(f"Output directory:  {args.output_dir}")
    print(f"  Gaussians:       {gaussians_dir}")
    print(f"  Meshes:          {meshes_dir}")
    if not args.no_render:
        print(f"  Renders:         {renders_dir}")
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
            args.object_index, not args.no_background
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
            
            gaussians = None
            mesh_path = None
            
            if "gaussian" in decoded_outputs:
                gaussians = decoded_outputs["gaussian"][0]
                ply_path = os.path.join(gaussians_dir, f"{base_name}_gaussians.ply")
                gaussians.save_ply(ply_path)
                print(f"Saved Gaussians to: {ply_path}")
            
            if "mesh" in decoded_outputs:
                mesh = decoded_outputs["mesh"][0]
                mesh_path = os.path.join(meshes_dir, f"{base_name}_mesh.obj")
                save_mesh_to_obj(mesh, mesh_path)
                print(f"Saved Mesh to: {mesh_path}")
            
            # Render if enabled (default)
            if not args.no_render and gaussians is not None:
                render_path = os.path.join(renders_dir, f"{base_name}_comparison.png")
                render_outputs(
                    gaussians, mesh_path, render_path,
                    image_size=args.render_size,
                    distance=args.render_distance,
                    fov=args.render_fov
                )
    
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
            if args.object_index is not None:
                cache_pattern += f"_obj{args.object_index}"
            if not args.no_background:
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
                args.scene_name, frame_idx, args.object_index, not args.no_background
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
            
            available_object_indices = [decoder_input['object_index'] for decoder_input in decoder_inputs]
            print(f"Available object indices in cache: {available_object_indices}")
            
            # build a map object_index -> decoder_input index
            index_map = {decoder_input['object_index']: idx for idx, decoder_input in enumerate(decoder_inputs)}
            
            if len(decoder_inputs) == 0:
                raise ValueError("No decoder inputs found in cache file")
            
            # Determine which objects to process
            if args.object_index is not None:
                object_indices = [args.object_index]
            else:
                object_indices = available_object_indices
            
            # Re-decode for each object
            for i in object_indices:
                print(f"\n  Re-decoding object {i + 1}/{len(decoder_inputs)}")
                
                if i not in available_object_indices:
                    print(f"  Warning: Object index {i} not found in cache for frame {frame_idx}, skipping")
                    continue
                
                decoder_input = decoder_inputs[index_map[i]]
                slat = decoder_input['decoder_input_slat']
                
                # Re-run the decoder
                decoded_outputs = redecode_slat(pipeline, slat, formats=args.formats)
                
                # Save outputs
                base_name = f"{cache_scene_name}_object_{i+1}_perframe"
                
                gaussians = None
                mesh_path = None
                
                if "gaussian" in decoded_outputs:
                    gaussians = decoded_outputs["gaussian"][0]
                    ply_path = os.path.join(gaussians_dir, f"{base_name}_gaussians.ply")
                    gaussians.save_ply(ply_path)
                    print(f"  Saved Gaussians to: {ply_path}")
                
                if "mesh" in decoded_outputs:
                    mesh = decoded_outputs["mesh"][0]
                    mesh_path = os.path.join(meshes_dir, f"{base_name}_mesh.obj")
                    save_mesh_to_obj(mesh, mesh_path)
                    print(f"  Saved Mesh to: {mesh_path}")
                
                # Render if enabled (default)
                if not args.no_render and gaussians is not None:
                    render_path = os.path.join(renders_dir, f"{base_name}_comparison.png")
                    render_outputs(
                        gaussians, mesh_path, render_path,
                        image_size=args.render_size,
                        distance=args.render_distance,
                        fov=args.render_fov
                    )
                
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

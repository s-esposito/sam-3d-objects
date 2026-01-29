"""
Render script for visualizing reconstructed Gaussians and Meshes.

Renders both Gaussians and meshes from multiple camera viewpoints for comparison.
Creates a grid visualization showing each representation from different angles.

Usage Examples:

    # Render redecoded single-frame results
    python custom/render_scene.py --dataset davis --scene-name car-turn --frame-index 0
    
    # Render averaged/weighted results
    python custom/render_scene.py --dataset davis --scene-name car-turn --mode averaged
    python custom/render_scene.py --dataset davis --scene-name car-turn --mode weighted
    
    # Specify custom paths
    python custom/render_scene.py --gaussians-dir custom/results/davis/redecoded/gaussians \
                                   --meshes-dir custom/results/davis/redecoded/meshes \
                                   --output-dir custom/results/davis/redecoded/renders
    
    # Render specific object only
    python custom/render_scene.py --dataset davis --scene-name car-turn --object-index 0

Output:
    Saves grid visualization showing:
    - Gaussians from 6 viewpoints (top, bottom, left, right, front, back)
    - Meshes from same 6 viewpoints
    - Side-by-side comparison
"""
import os
import sys

# Set environment variable before any other imports
os.environ['LIDRA_SKIP_INIT'] = '1'

import argparse
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pytorch3d.structures import Meshes
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
from pytorch3d.io import load_objs_as_meshes, load_ply
from utils import render_gaussians_scene


def load_gaussian_ply(ply_path):
    """
    Load Gaussian splatting PLY file.
    
    Returns a simple object with the PLY data for rendering.
    """
    # Import the Gaussian model
    from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model import Gaussian
    
    # Create Gaussian with default AABB (object-centric normalized space)
    aabb = [-1, -1, -1, 1, 1, 1]
    gaussian = Gaussian(aabb=aabb, sh_degree=0, device='cuda')
    gaussian.load_ply(ply_path)
    
    return gaussian


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


def render_gaussian_from_view(gaussian, R, T, image_size=512, fov=60):
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


def render_mesh_from_view(mesh, R, T, image_size=512, fov=60, device=None):
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
    # Use sigma=0 and gamma=1 for hard blending without color attenuation
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
    # Typical darkening factor is around 0.5-0.6, so multiply by ~1.8
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


def find_files(directory, pattern):
    """Find files matching pattern in directory."""
    if not os.path.exists(directory):
        return []
    
    files = glob.glob(os.path.join(directory, pattern))
    return sorted(files)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Render Gaussians and Meshes for comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["kubric4d", "davis"],
        default="davis",
        help="Dataset type",
    )
    parser.add_argument(
        "--scene-name",
        type=str,
        default=None,
        help="Scene name. Defaults: kubric4d='scn02719', davis='car-turn'",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="Frame index for redecoded results. If not specified with --mode=redecoded, renders all frames.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["redecoded", "averaged", "weighted"],
        default="redecoded",
        help="Type of results to render: redecoded (per-frame), averaged (uniform), or weighted (mask-based)",
    )
    parser.add_argument(
        "--object-index",
        type=int,
        default=None,
        help="Specific object to render (0-based). If not specified, renders all objects.",
    )
    
    # Custom paths (optional)
    parser.add_argument(
        "--gaussians-dir",
        type=str,
        default=None,
        help="Custom path to gaussians directory",
    )
    parser.add_argument(
        "--meshes-dir",
        type=str,
        default=None,
        help="Custom path to meshes directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for renders",
    )
    
    # Rendering options
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Size of rendered images",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=2.0,
        help="Camera distance from object",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=60.0,
        help="Field of view in degrees",
    )
    parser.add_argument(
        "--gamma-correction",
        action="store_true",
        help="Apply gamma correction (1/2.2) to mesh vertex colors",
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Get script directory
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Set defaults
    if args.scene_name is None:
        if args.dataset == "kubric4d":
            args.scene_name = "scn02719"
        else:
            args.scene_name = "car-turn"
    
    # Setup paths
    if args.gaussians_dir is None:
        args.gaussians_dir = os.path.join(SCRIPT_DIR, f"results/{args.dataset}/{args.mode}/gaussians")
    
    if args.meshes_dir is None:
        args.meshes_dir = os.path.join(SCRIPT_DIR, f"results/{args.dataset}/{args.mode}/meshes")
    
    if args.output_dir is None:
        args.output_dir = os.path.join(SCRIPT_DIR, f"results/{args.dataset}/{args.mode}/renders")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration
    print("=" * 60)
    print("Render Configuration")
    print("=" * 60)
    print(f"Dataset:        {args.dataset}")
    print(f"Scene:          {args.scene_name}")
    print(f"Mode:           {args.mode}")
    print(f"Frame index:    {args.frame_index if args.frame_index is not None else 'all'}")
    print(f"Object index:   {args.object_index if args.object_index is not None else 'all'}")
    print(f"Gaussians dir:  {args.gaussians_dir}")
    print(f"Meshes dir:     {args.meshes_dir}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Image size:     {args.image_size}")
    print("=" * 60)
    
    # Find Gaussian and Mesh files
    if args.mode == "redecoded" and args.frame_index is not None:
        # Single frame pattern
        gaussian_pattern = f"{args.scene_name}_f{args.frame_index}_object_*_redecoded_gaussians.ply"
        mesh_pattern = f"{args.scene_name}_f{args.frame_index}_object_*_redecoded_mesh.obj"
    elif args.mode == "averaged":
        gaussian_pattern = f"{args.scene_name}_object_*_averaged_*frames_gaussians.ply"
        mesh_pattern = f"{args.scene_name}_object_*_averaged_*frames_mesh.obj"
    elif args.mode == "weighted":
        gaussian_pattern = f"{args.scene_name}_object_*_weighted_*frames_gaussians.ply"
        mesh_pattern = f"{args.scene_name}_object_*_weighted_*frames_mesh.obj"
    else:
        # All redecoded frames
        gaussian_pattern = f"{args.scene_name}_f*_object_*_redecoded_gaussians.ply"
        mesh_pattern = f"{args.scene_name}_f*_object_*_redecoded_mesh.obj"
    
    gaussian_files = find_files(args.gaussians_dir, gaussian_pattern)
    mesh_files = find_files(args.meshes_dir, mesh_pattern)
    
    print(f"\nFound {len(gaussian_files)} Gaussian files")
    print(f"Found {len(mesh_files)} Mesh files")
    
    if len(gaussian_files) == 0:
        print(f"\nError: No Gaussian files found matching pattern: {gaussian_pattern}")
        print(f"In directory: {args.gaussians_dir}")
        return
    
    if len(mesh_files) == 0:
        print(f"\nWarning: No Mesh files found matching pattern: {mesh_pattern}")
        print(f"In directory: {args.meshes_dir}")
        print("Will render Gaussians only.")
    
    # Get camera viewpoints
    views = get_camera_positions(distance=args.distance)
    view_names = ['front', 'back', 'left', 'right', 'top', 'bottom']
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Process each Gaussian file
    for gaussian_file in gaussian_files:
        basename = os.path.basename(gaussian_file)
        base_name = basename.replace("_gaussians.ply", "")
        
        # Check if we should skip this object
        if args.object_index is not None:
            # Extract object index from filename
            try:
                obj_idx_str = basename.split("object_")[1].split("_")[0]
                obj_idx = int(obj_idx_str) - 1  # Convert to 0-based
                if obj_idx != args.object_index:
                    continue
            except:
                continue
        
        print(f"\n{'='*60}")
        print(f"Rendering: {base_name}")
        print(f"{'='*60}")
        
        # Load Gaussian
        print("Loading Gaussian...")
        gaussian = load_gaussian_ply(gaussian_file)
        
        # Render Gaussian from all views
        print("Rendering Gaussian from multiple viewpoints...")
        gaussian_renders = {}
        for view_name in view_names:
            R, T = views[view_name]
            rendered = render_gaussian_from_view(
                gaussian, R, T,
                image_size=args.image_size,
                fov=args.fov
            )
            gaussian_renders[view_name] = rendered
            print(f"  {view_name}: done")
        
        # Find corresponding mesh file
        mesh_file = gaussian_file.replace("gaussians.ply", "mesh.obj")
        mesh_file = mesh_file.replace(args.gaussians_dir, args.meshes_dir)
        
        mesh_renders = {}
        if os.path.exists(mesh_file):
            print("Loading Mesh...")
            try:
                # Load vertex colors from OBJ file manually
                verts_rgb = None
                with open(mesh_file, 'r') as f:
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
                        verts_rgb = torch.tensor(vertex_colors, dtype=torch.float32, device=device)[None]  # (1, V, 3)
                        # Apply gamma correction if requested
                        if args.gamma_correction:
                            verts_rgb = torch.pow(verts_rgb, 1.0 / 2.2)
                
                # Load mesh using PyTorch3D
                mesh = load_objs_as_meshes([mesh_file], device=device)
                
                # Apply vertex colors as texture
                if verts_rgb is not None:
                    mesh.textures = TexturesVertex(verts_features=verts_rgb)
                elif mesh.textures is None:
                    # Fallback: Add simple white texture if none exists
                    verts_rgb = torch.ones_like(mesh.verts_packed())[None]  # (1, V, 3)
                    mesh.textures = TexturesVertex(verts_features=verts_rgb.to(device))
                
                # Render mesh from all views
                print("Rendering Mesh from multiple viewpoints...")
                for view_name in view_names:
                    R, T = views[view_name]
                    rendered = render_mesh_from_view(
                        mesh, R, T,
                        image_size=args.image_size,
                        fov=args.fov,
                        device=device
                    )
                    mesh_renders[view_name] = rendered
                    print(f"  {view_name}: done")
            except Exception as e:
                print(f"Error loading/rendering mesh: {e}")
                # Create blank renders
                for view_name in view_names:
                    mesh_renders[view_name] = np.ones((args.image_size, args.image_size, 3))
        else:
            print(f"Mesh file not found: {mesh_file}")
            # Create blank renders
            for view_name in view_names:
                mesh_renders[view_name] = np.ones((args.image_size, args.image_size, 3))
        
        # Create comparison grid
        print("Creating visualization...")
        fig = create_comparison_grid(gaussian_renders, mesh_renders, view_names)
        
        # Save figure
        output_path = os.path.join(args.output_dir, f"{base_name}_comparison.png")
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved: {output_path}")
    
    print(f"\n{'='*60}")
    print("Rendering complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

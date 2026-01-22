"""
Demo script for Kubric4D and DAVIS dataset processing with SAM3D.
Processes multi-object scenes with depth maps and generates 3D Gaussian representations.

Supports:
- Kubric4D dataset with ground truth depth
- DAVIS dataset with MoGe depth estimation
- Background rendering (creating Gaussians for non-masked regions)
"""
import os
import time
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils import (
    load_image, 
    load_masks, 
    depth_to_pointmap, 
    radial_to_z_depth
)

import torch
from pytorch3d.transforms import Transform3d
from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply
from pytorch3d.renderer import look_at_view_transform
from inference import Inference, make_scene
from utils import (
    render_frame, 
    create_gaussians_object,
    create_gaussians_from_pointmap,
    join_gaussians,
    save_mesh_to_obj,
)

def setup_paths(dataset_path, scene_name, dataset_type="kubric4d"):
    """
    Setup and validate all necessary paths.
    
    Parameters
    ----------
    dataset_path : str
        Root path to the dataset
    scene_name : str
        Name of the scene to process
    dataset_type : str
        Either "kubric4d" or "davis"
        
    Returns
    -------
    dict
        Dictionary containing all paths and file lists
    """
    if dataset_type == "kubric4d":
        data_path = os.path.join(dataset_path, scene_name)
        frames_path = os.path.join(data_path, "frames_p0_v0")  # viewpoint 0
        cached_results_path = os.path.join(data_path, "cached_results")
        
        # Create cached results directory if it doesn't exist
        os.makedirs(cached_results_path, exist_ok=True)
        
        # Get sorted file lists
        image_names = sorted([f for f in os.listdir(frames_path) if f.startswith("rgba_") and f.endswith(".png")])
        mask_names = sorted([f for f in os.listdir(frames_path) if f.startswith("segmentation_") and f.endswith(".png")])
        depth_names = sorted([f for f in os.listdir(frames_path) if f.startswith("depth_") and f.endswith(".tiff")])
        
        return {
            'data_path': data_path,
            'frames_path': frames_path,
            'masks_path': frames_path,  # Same as frames path for Kubric4D
            'cached_results_path': cached_results_path,
            'image_names': image_names,
            'mask_names': mask_names,
            'depth_names': depth_names,
            'dataset_type': 'kubric4d',
        }
    
    elif dataset_type == "davis":
        frames_path = os.path.join(dataset_path, "JPEGImages", "Full-Resolution", scene_name)
        masks_path = os.path.join(dataset_path, "Annotations", "Full-Resolution", scene_name)
        cached_results_path = os.path.join(dataset_path, "cached_results", scene_name)
        
        # Create cached results directory if it doesn't exist
        os.makedirs(cached_results_path, exist_ok=True)
        
        # Get sorted file lists
        image_names = sorted([f for f in os.listdir(frames_path) if f.endswith(".jpg")])
        mask_names = sorted([f for f in os.listdir(masks_path) if f.endswith(".png")])
        
        return {
            'data_path': dataset_path,
            'frames_path': frames_path,
            'masks_path': masks_path,
            'cached_results_path': cached_results_path,
            'image_names': image_names,
            'mask_names': mask_names,
            'depth_names': [],  # DAVIS doesn't have depth files, uses MoGe
            'dataset_type': 'davis',
        }
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Use 'kubric4d' or 'davis'")


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


def load_and_process_depth(frames_path, depth_names, W, H, use_moge=False, inference=None, image=None):
    """
    Load and process depth maps to generate pointmap.
    
    Parameters
    ----------
    frames_path : str
        Path to the frames directory (for Kubric4D GT depth)
    depth_names : list
        List of depth file names (empty for DAVIS/MoGe mode)
    W, H : int
        Image dimensions
    use_moge : bool
        Whether to use MoGe depth model instead of GT depth
    inference : Inference
        Inference pipeline (required for MoGe mode)
    image : np.ndarray
        Input image (required for MoGe mode)
        
    Returns
    -------
    tuple
        (pointmap, K_matrix, valid_mask) where valid_mask is None for GT depth
    """
    valid_mask = None
    
    if not use_moge:
        # Load depth map from file (Kubric4D GT depth)
        depth_path = os.path.join(frames_path, depth_names[0])
        depth_map = load_image(depth_path, to_uint8=False)
        
        # Camera intrinsics for Kubric4D
        # Kubric uses horizontal FOV of ~53.13 degrees (0.927 rad)
        # For square pixels: fx = fy = W / (2 * tan(FOV/2))
        # With FOV=0.927 rad and W=576: fx = fy = 576
        fov_rad = 0.9272952180016122  # from Kubric4D metadata
        fx = W / (2 * np.tan(fov_rad / 2))
        fy = fx  # Square pixels
        cx = W / 2.0
        cy = H / 2.0
        
        print(f"Using camera intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        print(f"Radial depth map shape: {depth_map.shape}, dtype: {depth_map.dtype}, "
              f"min: {depth_map.min():.4f}, max: {depth_map.max():.4f}")
        
        # Convert radial depth to z-depth
        depth_map_z = radial_to_z_depth(depth_map, fx, fy, cx, cy)
        print(f"Z-depth map min: {depth_map_z.min():.4f}, max: {depth_map_z.max():.4f}")
        
    else:
        # Use MoGe depth model (for DAVIS or when GT depth not available)
        if inference is None or image is None:
            raise ValueError("MoGe mode requires inference pipeline and image")
        
        depth_model = inference._pipeline.depth_model
        loaded_image = inference._pipeline.image_to_float(image)
        loaded_image = torch.from_numpy(loaded_image)
        loaded_image_rgb = loaded_image.permute(2, 0, 1).contiguous()[:3]
        
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                depth_output = depth_model(loaded_image_rgb)
        
        depth_map_z = depth_output["depth"].cpu().numpy()
        valid_mask = depth_output["mask"].cpu().numpy()
        depth_map_z[~valid_mask] = 0.0
        
        intrinsics = depth_output["intrinsics"].cpu().numpy()
        
        fx = intrinsics[0, 0] * 1000.0
        fy = fx # Square pixels  # intrinsics[1, 1] * 1000.0
        cx = intrinsics[0, 2] * W
        cy = intrinsics[1, 2] * H
        
        print(f"MoGe intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    
    # Create intrinsics matrix
    K_matrix = np.eye(3)
    K_matrix[0, 0] = fx
    K_matrix[1, 1] = fy
    K_matrix[0, 2] = cx
    K_matrix[1, 2] = cy
    
    # Generate pointmap from depth
    pointmap = depth_to_pointmap(depth_map_z, K_matrix, valid_mask=valid_mask)
    print(f"Generated pointmap with shape: {pointmap.shape}, "
          f"min: {pointmap.min():.4f}, max: {pointmap.max():.4f}")
    
    return pointmap, K_matrix, valid_mask


def transform_to_pytorch3d_convention(pointmap):
    """Transform pointmap from R3 to PyTorch3D camera convention."""
    
    # Camera convention transformation (R3 -> PyTorch3D)
    r3_to_p3d_R, r3_to_p3d_T = look_at_view_transform(
        eye=np.array([[0, 0, -1]]),
        at=np.array([[0, 0, 0]]),
        up=np.array([[0, -1, 0]]),
    )

    # Convert rotation matrix to numpy
    r3_to_p3d_R_np = r3_to_p3d_R.cpu().numpy()[0]  # (3, 3)

    # Apply rotation using numpy matrix multiplication
    pointmap_transformed = pointmap @ r3_to_p3d_R_np.T
    
    return pointmap_transformed


def run_inference_on_masks(inference, image, masks, pointmap, seed=42):
    """
    Run SAM3D inference on all masks.
    
    Returns a list of outputs, each containing:
    - gaussian: List with raw Gaussian object (before layout transform)
    - rotation: Layout decoder rotation quaternion (local-to-camera)
    - translation: Layout decoder translation vector
    - scale: Layout decoder scale factor
    - gs: Shortcut to gaussian[0]
    
    The raw Gaussians are in canonical/local frame. Use make_scene() to apply
    the layout transformation and combine multiple objects.
    """
    pointmap_tensor = torch.from_numpy(pointmap).float().cuda()
    outputs = []
    
    print(f"\nRunning inference on {len(masks)} masks...")
    for i, mask in enumerate(masks):
        start_time = time.time()
        output = inference(image, mask, seed=seed, pointmap=pointmap_tensor)
        end_time = time.time()
        
        # Print layout decoder output for debugging
        print(f"  Mask {i+1}/{len(masks)}: {end_time - start_time:.2f}s")
        print(f"    Layout - rotation: {output['rotation'].cpu().numpy()}")
        print(f"    Layout - translation: {output['translation'].cpu().numpy()}")
        print(f"    Layout - scale: {output['scale'].cpu().numpy()}")
        print(f"    Raw Gaussian xyz range: [{output['gaussian'][0].get_xyz.min().item():.3f}, {output['gaussian'][0].get_xyz.max().item():.3f}]")
        
        # Print mesh info if available
        if "mesh" in output and output["mesh"] is not None:
            mesh = output["mesh"][0]
            print(f"    Mesh - vertices: {mesh.vertices.shape[0]}, faces: {mesh.faces.shape[0]}")
        
        outputs.append(output)
    
    return outputs


def save_cached_results(cached_results_path, scene_name, outputs):
    """
    Save inference results to cache.
    
    Stores the essential data needed to reconstruct the Gaussian scene:
    - gaussian: The Gaussian object for each mask
    - rotation: Object rotation quaternion
    - translation: Object translation vector
    - scale: Object scale factor
    - decoder_input_coords: Sparse 3D coordinates for decoder (optional)
    - decoder_input_slat: SLAT latent features for decoder (optional)
    """
    cache_file = os.path.join(cached_results_path, f"{scene_name}_sam3d_results.npz")
    
    # Extract and serialize the necessary data for each output
    cached_data = []
    for output in outputs:
        # Extract Gaussian model state
        gs = output["gaussian"][0]
        gs_data = {
            'xyz': gs.get_xyz.cpu().numpy(),
            'features_dc': gs.get_features.cpu().numpy(),
            'scaling': gs.get_scaling.cpu().numpy(),
            'rotation': gs.get_rotation.cpu().numpy(),
            'opacity': gs.get_opacity.cpu().numpy(),
            'aabb': gs.aabb.cpu().numpy() if hasattr(gs, 'aabb') and gs.aabb is not None else None,
            'mininum_kernel_size': gs.mininum_kernel_size if hasattr(gs, 'mininum_kernel_size') else None,
        }
        
        # Extract pose data
        output_data = {
            'gaussian_data': gs_data,
            'rotation': output["rotation"].cpu().numpy(),
            'translation': output["translation"].cpu().numpy(),
            'scale': output["scale"].cpu().numpy(),
        }
        
        # Save decoder inputs if available (for re-running decoder)
        if "decoder_input_coords" in output and "decoder_input_slat" in output:
            output_data['decoder_input_coords'] = output["decoder_input_coords"].cpu().numpy()
            # For SparseTensor slat, we need to save its features and coords
            slat = output["decoder_input_slat"]
            output_data['decoder_input_slat_feats'] = slat.feats.cpu().numpy()
            output_data['decoder_input_slat_coords'] = slat.coords.cpu().numpy()
        
        cached_data.append(output_data)
    
    # Save as numpy archive
    np.savez(
        cache_file,
        cached_data=np.array(cached_data, dtype=object),
        num_objects=len(outputs),
    )
    print(f"Cached results saved to {cache_file}")


def load_cached_results(cached_results_path, scene_name):
    """
    Load cached inference results and reconstruct the output format.
    
    Reconstructs outputs with the same structure expected by make_scene():
    - gaussian: List containing the Gaussian object
    - rotation: Object rotation quaternion tensor
    - translation: Object translation vector tensor
    - scale: Object scale factor tensor
    - decoder_input_coords: Sparse 3D coordinates (if available)
    - decoder_input_slat: SLAT latent features (if available)
    """
    from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model import Gaussian
    import spconv.pytorch as sp
    
    cache_file = os.path.join(cached_results_path, f"{scene_name}_sam3d_results.npz")
    
    if not os.path.exists(cache_file):
        return None
    
    print(f"Loading cached results from {cache_file}...")
    cached_archive = np.load(cache_file, allow_pickle=True)
    cached_data = cached_archive["cached_data"]
    
    # Reconstruct outputs format expected by make_scene
    outputs = []
    for data in cached_data:
        data = data.item() if hasattr(data, 'item') else data
        gs_data = data['gaussian_data']
        
        # Reconstruct Gaussian object with proper aabb
        aabb = gs_data['aabb'].tolist() if gs_data['aabb'] is not None else [0, 0, 0, 1, 1, 1]
        min_kernel = gs_data['mininum_kernel_size'] if gs_data['mininum_kernel_size'] is not None else 0.0
        
        gs = Gaussian(aabb=aabb, sh_degree=0, mininum_kernel_size=min_kernel)
        gs._xyz = torch.from_numpy(gs_data['xyz']).cuda()
        gs._features_dc = torch.from_numpy(gs_data['features_dc']).cuda()
        gs._scaling = torch.from_numpy(gs_data['scaling']).cuda()
        gs._rotation = torch.from_numpy(gs_data['rotation']).cuda()
        gs._opacity = torch.from_numpy(gs_data['opacity']).cuda()
        
        # Reconstruct output dict
        output = {
            'gaussian': [gs],
            'rotation': torch.from_numpy(data['rotation']).cuda(),
            'translation': torch.from_numpy(data['translation']).cuda(),
            'scale': torch.from_numpy(data['scale']).cuda(),
        }
        
        # Reconstruct decoder inputs if available
        if 'decoder_input_slat_feats' in data and 'decoder_input_slat_coords' in data:
            # Reconstruct SparseTensor for slat
            slat_feats = torch.from_numpy(data['decoder_input_slat_feats']).cuda()
            slat_coords = torch.from_numpy(data['decoder_input_slat_coords']).cuda()
            slat = sp.SparseConvTensor(
                features=slat_feats,
                indices=slat_coords,
                spatial_shape=[64, 64, 64],
                batch_size=1,
            )
            output['decoder_input_slat'] = slat
            
        if 'decoder_input_coords' in data:
            output['decoder_input_coords'] = torch.from_numpy(data['decoder_input_coords']).cuda()
        
        outputs.append(output)
    
    print(f"Loaded {len(outputs)} cached Gaussian objects")
    return outputs


def render_and_compare(scene_gs, image, K_matrix, W, H, output_path="rendered_vs_original.png"):
    """Render Gaussian scene and compare with original image."""
    # Use identity camera matrix (camera at origin)
    c2w = torch.eye(4)
    K = torch.from_numpy(K_matrix).float()
    
    rendered_frame, _ = render_frame(scene_gs, c2w=c2w, K=K, w=W, h=H)
    
    # Create comparison visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=14)
    ax1.axis('off')
    
    ax2.imshow(rendered_frame.cpu().numpy())
    ax2.set_title('Rendered from Gaussian Splats', fontsize=14)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved rendering comparison to {output_path}")


def create_background_gaussians(image, pointmap, masks, K_matrix):
    """
    Create Gaussian splats for the background (non-masked regions).
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    pointmap : np.ndarray
        3D pointmap (NOT in PyTorch3D convention, but in original R3 convention)
    masks : list
        List of object masks
    K_matrix : np.ndarray
        Camera intrinsics matrix
        
    Returns
    -------
    Gaussian
        Background Gaussian object
    """
    
    # Create combined mask of all objects
    background_mask = ~np.any(np.stack(masks, axis=0), axis=0)
    print(f"Background mask: {background_mask.sum()} pixels ({100*background_mask.mean():.1f}% of image)")
    
    # Create background Gaussians from the non-masked region
    gaussians_bg = create_gaussians_from_pointmap(
        image=image[background_mask],
        pointmap=pointmap[background_mask],
        K=K_matrix,
    )
    
    return gaussians_bg


def transform_scene_to_r3_convention(scene_gs):
    """
    Transform combined scene from PyTorch3D convention back to R3 convention.
    This should be done AFTER make_scene() on the combined scene.
    
    Parameters
    ----------
    scene_gs : Gaussian
        Scene Gaussian object in PyTorch3D convention
        
    Returns
    -------
    Gaussian
        Scene Gaussian object in R3 convention
    """
    
    # Get the denormalized xyz coordinates
    xyz_unnormalized = scene_gs.get_xyz  # This applies: xyz * aabb[3:] + aabb[:3]

    # Camera convention transformation (R3 -> PyTorch3D)
    r3_to_p3d_R, r3_to_p3d_T = look_at_view_transform(
        eye=np.array([[0, 0, -1]]),
        at=np.array([[0, 0, 0]]),
        up=np.array([[0, -1, 0]]),
        device=scene_gs.get_xyz.device,
    )

    # inverse transform (PyTorch3D -> R3)
    p3d_to_r3_R = r3_to_p3d_R.transpose(1, 2)

    # Transform positions
    camera_convention_transform = Transform3d(device=scene_gs.get_xyz.device).rotate(p3d_to_r3_R)
    xyz = camera_convention_transform.transform_points(xyz_unnormalized)

    # Transform rotations (quaternions)
    # Convert rotation matrix to quaternion (PyTorch3D uses wxyz format)
    p3d_to_r3_quat = matrix_to_quaternion(p3d_to_r3_R)  # (1, 4) in wxyz format
    
    # Get original rotations and apply the coordinate transform
    original_rots = scene_gs.get_rotation  # (N, 4) in wxyz format
    # Multiply quaternions: q_new = q_transform * q_original
    transformed_rots = quaternion_multiply(
        p3d_to_r3_quat.expand(original_rots.shape[0], -1),
        original_rots
    )

    # Create new Gaussians object
    new_scene_gs = create_gaussians_object(
        xyz=xyz,
        features=scene_gs.get_features,
        scales=scene_gs.get_scaling,
        rots=transformed_rots,
        opacities=scene_gs.get_opacity,
    )

    return new_scene_gs


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
    
    # Get the project root directory (parent of notebook directory)
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
        args.output_dir = os.path.join(SCRIPT_DIR, f"gaussians/{args.dataset}")
    
    # Setup paths
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
    print(f"Skip cache:        {args.no_cache}")
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
        )
    
    print(f"\n{'='*60}")
    print(f"=== All {len(frame_indices)} frames processed! ===")
    print(f"{'='*60}")


def process_frame(args, paths, frame_index, inference):
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
    
    # Visualize input data
    if not args.no_visualize:
        visualize_image_and_masks(image, masks)
    
    # Modify cache name based on configuration to avoid conflicts
    cache_parts = [args.scene_name, f"f{frame_index}"]
    if args.first_object_only:
        cache_parts.append("first")
    if args.with_background:
        cache_parts.append("bg")
    cache_scene_name = "_".join(cache_parts)
    
    # Check for cached results
    cached_outputs = None
    if not args.no_cache:
        cached_outputs = load_cached_results(paths['cached_results_path'], cache_scene_name)
    
    if cached_outputs is not None:
        print("Using cached inference results")
        outputs = cached_outputs
        need_inference = False
    else:
        print("No cache found, running inference pipeline")
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
        visualize_pointmap(pointmap)
    
    # Transform to PyTorch3D convention for inference
    pointmap = transform_to_pytorch3d_convention(pointmap)
    print("Transformed pointmap to PyTorch3D convention")

    # Visualize pointmap after transformation
    if not args.no_visualize:
        visualize_pointmap(pointmap, output_path="pointmap_visualization_to_pytorch3d.png")
    
    if need_inference:
        # Run inference on all masks
        outputs = run_inference_on_masks(inference, image, masks, pointmap, seed=args.seed)
        
        # Cache results
        save_cached_results(paths['cached_results_path'], cache_scene_name, outputs)
    
    # Save each raw object (before layout transform) for debugging
    from copy import deepcopy
    for i, output in enumerate(outputs):
        # Save raw Gaussian (canonical frame, before layout transform)
        raw_gs = deepcopy(output["gaussian"][0])
        raw_ply_path = os.path.join(args.output_dir, f"{cache_scene_name}_object_{i+1}_raw.ply")
        os.makedirs(args.output_dir, exist_ok=True)
        raw_gs.save_ply(raw_ply_path)
        print(f"Saved raw object {i+1} Gaussian (before layout) to {raw_ply_path}")
        
        # Save raw mesh if available (canonical frame, before layout transform)
        if "mesh" in output and output["mesh"] is not None:
            raw_mesh = output["mesh"][0]
            raw_mesh_path = os.path.join(args.output_dir, f"{cache_scene_name}_object_{i+1}_raw_mesh.obj")
            save_mesh_to_obj(raw_mesh, raw_mesh_path)
            print(f"Saved raw object {i+1} mesh (before layout) to {raw_mesh_path}")
        
        # Save GLB if available (includes texture)
        if "glb" in output and output["glb"] is not None:
            glb_path = os.path.join(args.output_dir, f"{cache_scene_name}_object_{i+1}_raw.glb")
            output["glb"].export(glb_path)
            print(f"Saved raw object {i+1} GLB to {glb_path}")
    
    # Save each object with layout applied (in PyTorch3D convention)
    for i, output in enumerate(outputs):
        obj_ply_path = os.path.join(args.output_dir, f"{cache_scene_name}_object_{i+1}.ply")
        os.makedirs(args.output_dir, exist_ok=True)
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
    os.makedirs(args.output_dir, exist_ok=True)
    ply_path = os.path.join(args.output_dir, f"{cache_scene_name}.ply")
    new_scene_gs.save_ply(ply_path)
    print(f"Saved Gaussian scene to {ply_path}")
    
    # Render and compare with original
    if not args.no_visualize:
        output_render_path = os.path.join(args.output_dir, f"{cache_scene_name}_render.png")
        render_and_compare(new_scene_gs, image, K_matrix, W, H, output_path=output_render_path)
    
    print(f"Frame {frame_index} complete!")


if __name__ == "__main__":
    main()

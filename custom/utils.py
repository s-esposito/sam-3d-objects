import os
import sys
import math
import numpy as np
import time
import glob
import torch
import matplotlib.pyplot as plt
from PIL import Image
from gsplat.rendering import rasterization
from pytorch3d.transforms import Transform3d
from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.transforms import quaternion_to_matrix, quaternion_invert


# Skip sam3d_objects initialization for lightweight tools
os.environ['LIDRA_SKIP_INIT'] = '1'

# Add parent directory to path to import sam3d_objects
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from inference import make_scene
from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model import Gaussian
from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp


def get_cache_filename(scene_name, frame_index, first_object_only=False, with_background=False):
    """Build the cache filename based on configuration."""
    cache_parts = [scene_name, f"f{frame_index}"]
    if first_object_only:
        cache_parts.append("first")
    if with_background:
        cache_parts.append("bg")
    cache_scene_name = "_".join(cache_parts)
    return f"{cache_scene_name}_sam3d_results.npz", cache_scene_name


def setup_paths(dataset_path, scene_name, dataset_type):
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
    output_dir : str
        Directory for outputs (Gaussians, meshes, cached SLAT tokens)
        
    Returns
    -------
    dict
        Dictionary containing all paths and file lists
    """
    if dataset_type == "kubric4d":
        data_path = os.path.join(dataset_path, scene_name)
        frames_path = os.path.join(data_path, "frames_p0_v0")  # viewpoint 0
        
        # Get sorted file lists
        image_names = sorted([f for f in os.listdir(frames_path) if f.startswith("rgba_") and f.endswith(".png")])
        mask_names = sorted([f for f in os.listdir(frames_path) if f.startswith("segmentation_") and f.endswith(".png")])
        depth_names = sorted([f for f in os.listdir(frames_path) if f.startswith("depth_") and f.endswith(".tiff")])
        
        return {
            'data_path': data_path,
            'frames_path': frames_path,
            'masks_path': frames_path,  # Same as frames path for Kubric4D
            'image_names': image_names,
            'mask_names': mask_names,
            'depth_names': depth_names,
            'dataset_type': 'kubric4d',
        }
    
    elif dataset_type == "davis":
        frames_path = os.path.join(dataset_path, "JPEGImages", "Full-Resolution", scene_name)
        masks_path = os.path.join(dataset_path, "Annotations", "Full-Resolution", scene_name)
        
        # Get sorted file lists
        image_names = sorted([f for f in os.listdir(frames_path) if f.endswith(".jpg")])
        mask_names = sorted([f for f in os.listdir(masks_path) if f.endswith(".png")])
        
        return {
            'data_path': dataset_path,
            'frames_path': frames_path,
            'masks_path': masks_path,
            'image_names': image_names,
            'mask_names': mask_names,
            'depth_names': [],  # DAVIS doesn't have depth files, uses MoGe
            'dataset_type': 'davis',
        }
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Use 'kubric4d' or 'davis'")

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


def save_tokens(tokens_path, scene_name, outputs):
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
    cache_file = os.path.join(tokens_path, f"{scene_name}_sam3d_results.npz")
    
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
    
    
def save_comparison_image(rendered, gt_image, output_path, frame_index):
    """Save side-by-side comparison of rendered and ground truth images."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Ground truth
    axes[0].imshow(gt_image.numpy())
    axes[0].set_title(f'Ground Truth (Frame {frame_index})')
    axes[0].axis('off')
    
    # Rendered
    axes[1].imshow(rendered.numpy())
    axes[1].set_title('Rendered from Gaussians')
    axes[1].axis('off')
    
    # Difference (amplified for visibility)
    diff = torch.abs(rendered - gt_image)
    diff_amplified = torch.clamp(diff * 5, 0, 1)  # Amplify differences
    axes[2].imshow(diff_amplified.numpy())
    axes[2].set_title('Absolute Difference (5x amplified)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


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


def save_mesh_to_obj(mesh, output_path):
    """
    Save a mesh object to an OBJ file.
    
    Parameters
    ----------
    mesh : MeshExtractResult or similar
        Mesh object with vertices, faces, and optionally vertex_attrs attributes.
        vertices should be (N, 3) tensor
        faces should be (M, 3) tensor
        vertex_attrs can be:
          - A tensor of shape (N, C) where C >= 3 (first 3 channels are RGB color)
          - A dict with 'color' key
          - None
    output_path : str
        Path to save the OBJ file
    """
    # Handle both 'vertices' and 'verts' attribute names
    if hasattr(mesh, 'vertices'):
        verts = mesh.vertices.cpu().numpy() if hasattr(mesh.vertices, 'cpu') else mesh.vertices
    elif hasattr(mesh, 'verts'):
        verts = mesh.verts.cpu().numpy() if hasattr(mesh.verts, 'cpu') else mesh.verts
    else:
        raise AttributeError("Mesh object has no 'vertices' or 'verts' attribute")
    
    faces = mesh.faces.cpu().numpy() if hasattr(mesh.faces, 'cpu') else mesh.faces
    
    # Check for vertex colors
    vertex_colors = None
    if hasattr(mesh, 'vertex_attrs') and mesh.vertex_attrs is not None:
        va = mesh.vertex_attrs
        # vertex_attrs can be a tensor directly or a dict
        if isinstance(va, dict):
            if 'color' in va:
                vc = va['color']
                vertex_colors = vc.cpu().numpy() if hasattr(vc, 'cpu') else vc
        elif hasattr(va, 'cpu'):
            # It's a tensor - assume first 3 channels are RGB
            va_np = va.cpu().numpy()
            if va_np.shape[-1] >= 3:
                vertex_colors = va_np[..., :3]
        elif isinstance(va, np.ndarray):
            if va.shape[-1] >= 3:
                vertex_colors = va[..., :3]
    elif hasattr(mesh, 'vertex_colors') and mesh.vertex_colors is not None:
        vertex_colors = mesh.vertex_colors.cpu().numpy() if hasattr(mesh.vertex_colors, 'cpu') else mesh.vertex_colors
    
    with open(output_path, 'w') as f:
        f.write(f"# OBJ file with {len(verts)} vertices and {len(faces)} faces\n")
        
        # Write vertices (with colors if available)
        for i, v in enumerate(verts):
            if vertex_colors is not None:
                c = vertex_colors[i]
                # Clamp colors to [0, 1]
                c = np.clip(c, 0, 1)
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
            else:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write faces (OBJ uses 1-indexed vertices)
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    print(f"Saved mesh to {output_path} ({len(verts)} vertices, {len(faces)} faces)")

C0 = 0.28209479177387814

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5

def render_gaussians_to_image(scene_gs, K_matrix, W, H, bg_color=None):
    """
    Render Gaussian scene to an image using identity camera (camera at origin).
    
    Parameters
    ----------
    scene_gs : Gaussian
        Gaussian scene object
    K_matrix : np.ndarray
        Camera intrinsics matrix (3, 3)
    W, H : int
        Image dimensions
    bg_color : torch.Tensor or None
        Background color (3,), defaults to white
        
    Returns
    -------
    torch.Tensor
        Rendered image (H, W, 3) in [0, 1] range
    """
    # Use identity camera matrix (camera at origin)
    c2w = torch.eye(4)
    K = torch.from_numpy(K_matrix).float()
    
    # Default to white background for evaluation
    if bg_color is None:
        bg_color = torch.ones(3)
    
    rendered_frame, alpha = render_frame(scene_gs, c2w=c2w, K=K, w=W, h=H, bg_color=bg_color)
    
    return rendered_frame

def render_frame(
    scene_gs,
    c2w,  # Camera-to-world transformation (4, 4)
    K,    # Camera intrinsics (3, 3)
    w, h, # Width and height
    bg_color=None,  # Background color as tensor (3,) or None for black
):
    """
    Render a single frame from the Gaussian scene using given camera parameters.
    
    Args:
        scene_gs: Gaussian scene object
        c2w: Camera-to-world transformation matrix (4, 4)
        K: Camera intrinsics matrix (3, 3)
        w: Image width
        h: Image height
        bg_color: Background color as tensor (3,) or None for black background
        
    Returns:
        Rendered image as numpy array (H, W, 3) in uint8 format
    """
    
    # Convert c2w to extrinsics (world-to-camera)
    # Extrinsics = inverse(c2w)
    w2c = torch.inverse(c2w.float())
    
    # Ensure tensors are on CUDA
    w2c = w2c.cuda() if not w2c.is_cuda else w2c
    Ks = K.cuda() if not K.is_cuda else K
    w2c = w2c.unsqueeze(0)  # [1, 4, 4]
    Ks = Ks.unsqueeze(0)    # [1, 3, 3]
    
    means = scene_gs.get_xyz  # [N, 3]
    rotations = scene_gs.get_rotation  # [N, 4]
    scales = scene_gs.get_scaling  # [N, 3]
    opacity = scene_gs.get_opacity  # [N, 1]
    features = scene_gs.get_features  # [N, 1, 3]
    width = w
    height = h
    near_plane = 0.1
    far_plane = 100000.0
    
    # Set background color (default to black if not provided)
    if bg_color is None:
        bg_color = torch.zeros(3, device=w2c.device)
    else:
        bg_color = bg_color.to(w2c.device)
    
    # Render
    with torch.no_grad():
        rgbd, alpha, info = rasterization(
            means=means,  # [N, 3]
            quats=rotations,  # [N, 4]
            scales=scales,  # [N, 3]
            opacities=opacity.squeeze(-1),  # [N]
            colors=features,  # [N, 3]
            viewmats=w2c,  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            near_plane=near_plane,
            far_plane=far_plane,
            render_mode="RGB",
            sh_degree=0,
            rasterize_mode="classic",
            distributed=False,
            camera_model="pinhole",
            packed=False,
            backgrounds=bg_color[None, ...],  # [1, 3]
        )
    
    # Convert to numpy and scale to uint8
    #  = res["color"].permute(1, 2, 0)  # (3, H, W) -> (H, W, 3)
    color = rgbd[0, ..., :3]  # (H, W, 3)
    alpha = alpha[0]    # (H, W)
    
    return color, alpha

def load_image(path, to_uint8=True):
    image = Image.open(path)
    image = np.array(image)
    if to_uint8:
        image = image.astype(np.uint8)
    return image


def load_masks(mask_path, indices_list=None):
    """Load segmentation masks from a file."""
    masks = []
    mask = load_image(mask_path)
    print(f"Loaded mask of shape: {mask.shape}, dtype: {mask.dtype}, min: {mask.min()}, max: {mask.max()}, unique values: {np.unique(mask)}")
    # get unique object ids
    object_ids = np.unique(mask)
    for object_id in object_ids:
        if object_id == 0:
            continue  # skip background
        if indices_list is not None and object_id.item() not in indices_list:
            continue
        object_mask = (mask == object_id)
        masks.append(object_mask)
    return masks


def radial_to_z_depth(radial_depth_map, fx, fy, cx, cy):
    """
    Convert a radial depth map r(u,v) to a z-depth map z(u,v)
    under a simple pinhole model with intrinsics (fx, fy, cx, cy).
    
    Mathematical derivation:
    - For a point at (x, y, z) in camera coordinates:
      x = (u - cx) * z / fx
      y = (v - cy) * z / fy
    - Radial distance: r = sqrt(x² + y² + z²)
    - Substituting: r = z * sqrt((u-cx)²/fx² + (v-cy)²/fy² + 1)
    - Therefore: z = r / sqrt((u-cx)²/fx² + (v-cy)²/fy² + 1)

    Parameters
    ----------
    radial_depth_map : (H, W) np.ndarray
        Array of radial depths (Euclidean distance from camera center).
    fx, fy : float
        Focal lengths of the camera in pixels.
    cx, cy : float
        Principal point (image center) in pixel coordinates.

    Returns
    -------
    z_depth_map : (H, W) np.ndarray
        The z-depth map (distance along optical axis).
    """
    assert fx is not None, "Focal length fx is not specified"
    assert fy is not None, "Focal length fy is not specified"
    assert cx is not None, "Principal point cx is not specified"
    assert cy is not None, "Principal point cy is not specified"

    H, W = radial_depth_map.shape[:2]

    # Create a grid of pixel coordinates
    # v corresponds to rows (height), u corresponds to cols (width)
    v_coords, u_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    # Convert from pixel coords to normalized camera-plane coords
    x_norm = (u_coords - cx) / fx
    y_norm = (v_coords - cy) / fy

    # Compute scaling factor: sqrt(x_norm² + y_norm² + 1)
    scale_factor = np.sqrt(x_norm**2 + y_norm**2 + 1)

    # Convert radial depth to z-depth: z = r / scale_factor
    z_depth_map = radial_depth_map / scale_factor

    # Preserve input dtype
    z_depth_map = z_depth_map.astype(radial_depth_map.dtype)

    return z_depth_map


def verify_reprojection(
    points_3d: np.ndarray,
    K_matrix: np.ndarray,
    H: int,
    W: int,
    num_samples: int = 100,
) -> dict:
    """
    Verify that 3D points correctly reproject back to their 2D pixel coordinates.
    
    Parameters
    ----------
    points_3d : np.ndarray
        (H, W, 3) array of 3D points in camera coordinates
    K_matrix : np.ndarray
        (3, 3) camera intrinsics matrix
    H, W : int
        Image height and width
    num_samples : int
        Number of random points to sample for verification
        
    Returns
    -------
    dict
        Dictionary containing reprojection statistics
    """
    fx = K_matrix[0, 0]
    fy = K_matrix[1, 1]
    cx = K_matrix[0, 2]
    cy = K_matrix[1, 2]
    
    # Sample random pixels
    np.random.seed(42)
    sample_v = np.random.randint(0, H, num_samples)
    sample_u = np.random.randint(0, W, num_samples)
    
    errors_u = []
    errors_v = []
    
    for v, u in zip(sample_v, sample_u):
        # Get 3D point
        point_3d = points_3d[v, u]
        x, y, z = point_3d
        
        # Skip if depth is invalid
        if z <= 0:
            continue
        
        # Reproject to 2D using pinhole camera model
        # u' = fx * (x/z) + cx
        # v' = fy * (y/z) + cy
        u_reproj = fx * (x / z) + cx
        v_reproj = fy * (y / z) + cy
        
        # Compute error
        error_u = abs(u_reproj - u)
        error_v = abs(v_reproj - v)
        
        errors_u.append(error_u)
        errors_v.append(error_v)
    
    errors_u = np.array(errors_u)
    errors_v = np.array(errors_v)
    
    stats = {
        'mean_error_u': errors_u.mean(),
        'max_error_u': errors_u.max(),
        'mean_error_v': errors_v.mean(),
        'max_error_v': errors_v.max(),
        'mean_error_total': np.sqrt(errors_u**2 + errors_v**2).mean(),
        'max_error_total': np.sqrt(errors_u**2 + errors_v**2).max(),
        'num_samples': len(errors_u),
    }
    
    return stats


def compute_conegs_scaling(
    points_3d_camera: torch.Tensor,
    points_depth: torch.Tensor,
    K_inv: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Gaussian scaling based on pixel footprint.
    
    Parameters
    ----------
    points_3d_camera : torch.Tensor
        (N, 3) camera-space 3D points for each pixel
    points_depth : torch.Tensor
        (N,) z-depth for each pixel
    K_inv : torch.Tensor
        (3, 3) inverse intrinsics
        
    Returns
    -------
    torch.Tensor
        (N, 1) isotropic Gaussian stddev per pixel
    """
    eps = 1e-6

    # Unnormalized ray direction for each pixel:
    # p_cam = z * d  =>  d = p_cam / z
    z = points_3d_camera[:, 2].clamp_min(eps)  # (N,)
    d = points_3d_camera / z[:, None]  # (N,3)
    d_norm = torch.linalg.norm(d, dim=1).clamp_min(eps)  # (N,)

    # Metric distance from camera origin to the 3D point (along the ray)
    s = points_depth  # (N,)

    # Constant pixel footprint (no distortion)
    col0 = K_inv[:, 0]
    col1 = K_inv[:, 1]
    pixel_width = 0.5 * (torch.linalg.norm(col0) + torch.linalg.norm(col1))

    pixel_width = pixel_width * (2.0 / math.sqrt(12.0))

    sigma = pixel_width * (s / d_norm)  # (N,)
    return sigma[:, None]


def create_gaussians_object(
    xyz: torch.Tensor,
    features: torch.Tensor,
    scales: torch.Tensor,
    rots: torch.Tensor,
    opacities: torch.Tensor,
) -> Gaussian:
    # Compute AABB (axis-aligned bounding box) from the pointmap
    # Format: [min_x, min_y, min_z, size_x, size_y, size_z]
    xyz_min = xyz.min(dim=0)[0]
    xyz_max = xyz.max(dim=0)[0]
    xyz_size = xyz_max - xyz_min
    aabb = torch.cat([xyz_min, xyz_size]).tolist()
    print(f"Computed AABB: {aabb}")
    print(f"  Min: [{xyz_min[0]:.4f}, {xyz_min[1]:.4f}, {xyz_min[2]:.4f}]")
    print(f"  Max: [{xyz_max[0]:.4f}, {xyz_max[1]:.4f}, {xyz_max[2]:.4f}]")
    print(f"  Size: [{xyz_size[0]:.4f}, {xyz_size[1]:.4f}, {xyz_size[2]:.4f}]")
    
    # Normalize xyz to [0, 1] range for internal storage
    # The Gaussian model expects normalized coordinates and will denormalize using AABB
    xyz_normalized = (xyz - xyz_min) / xyz_size
    print(f"Normalized xyz: min={xyz_normalized.min():.6f}, max={xyz_normalized.max():.6f}")
    
    
    print(f"Converted RGB to SH features: min={features.min():.6f}, max={features.max():.6f}")

    # Create Gaussian model with computed AABB
    gaussians = Gaussian(aabb=aabb, scaling_bias=0.0, opacity_bias=0.0)
    
    # Move all tensors to CUDA
    device = 'cuda'
    xyz_normalized = xyz_normalized.to(device)
    features = features.to(device)
    scales = scales.to(device)
    rots = rots.to(device)
    opacities = opacities.to(device)
    
    # Initialize gaussians with the computed values
    gaussians._xyz = xyz_normalized  # Use normalized coordinates!
    gaussians._features_dc = features
    
    # Disable scale_bias and opacity_bias, move to correct device
    gaussians.scale_bias = torch.tensor(0.0, device=gaussians._xyz.device)
    gaussians.opacity_bias = torch.tensor(0.0, device=gaussians._xyz.device)
    
    # Debug: check scaling before and after inverse activation
    scales_internal = gaussians.inverse_scaling_activation(scales)
    
    gaussians._scaling = scales_internal
    gaussians._rotation = rots - gaussians.rots_bias[None, :]
    
    # Clamp opacities to avoid numerical issues with inverse_sigmoid at exactly 0 or 1
    opacities_clamped = torch.clamp(opacities, 1e-6, 1.0 - 1e-6)
    opacities_internal = gaussians.inverse_opacity_activation(opacities_clamped)
    
    gaussians._opacity = opacities_internal
    
    print(f"\nGaussians initialized on device: {gaussians._xyz.device}")
    print(f"AABB device: {gaussians.aabb.device}")
    print(f"\nfeatures shape: {gaussians.get_features.shape}, min: {gaussians.get_features.min().item():.3f}, max: {gaussians.get_features.max().item():.3f}")
    print(f"opacities shape: {gaussians.get_opacity.shape}, min: {gaussians.get_opacity.min().item():.3f}, max: {gaussians.get_opacity.max().item():.3f}")
    print(f"scaling shape: {gaussians.get_scaling.shape}, min: {gaussians.get_scaling.min().item():.6f}, max: {gaussians.get_scaling.max().item():.6f}")
    print(f"rotation shape: {gaussians.get_rotation.shape}, min: {gaussians.get_rotation.min().item():.3f}, max: {gaussians.get_rotation.max().item():.3f}")
    
    return gaussians


# def create_gaussians_object(
#     xyz: torch.Tensor,
#     features: torch.Tensor,
#     scales: torch.Tensor,
#     rots: torch.Tensor,
#     opacities: torch.Tensor,
# ) -> Gaussian:
#     # Compute AABB (axis-aligned bounding box) from the pointmap
#     # Format: [min_x, min_y, min_z, size_x, size_y, size_z]
#     xyz_min = xyz.min(dim=0)[0]
#     xyz_max = xyz.max(dim=0)[0]
#     xyz_size = xyz_max - xyz_min
#     aabb = torch.cat([xyz_min, xyz_size]).tolist()
#     print(f"Computed AABB: {aabb}")
#     print(f"  Min: [{xyz_min[0]:.4f}, {xyz_min[1]:.4f}, {xyz_min[2]:.4f}]")
#     print(f"  Max: [{xyz_max[0]:.4f}, {xyz_max[1]:.4f}, {xyz_max[2]:.4f}]")
#     print(f"  Size: [{xyz_size[0]:.4f}, {xyz_size[1]:.4f}, {xyz_size[2]:.4f}]")
    
#     # Normalize xyz to [0, 1] range for internal storage
#     # The Gaussian model expects normalized coordinates and will denormalize using AABB
#     xyz_normalized = (xyz - xyz_min) / xyz_size
#     print(f"Normalized xyz: min={xyz_normalized.min():.6f}, max={xyz_normalized.max():.6f}")
    
    
#     print(f"Converted RGB to SH features: min={features.min():.6f}, max={features.max():.6f}")

#     # Create Gaussian model with computed AABB
#     gaussians = Gaussian(aabb=aabb, scaling_bias=0.0, opacity_bias=0.0)
    
#     # Move all tensors to CUDA
#     device = 'cuda'
#     xyz_normalized = xyz_normalized.to(device)
#     features = features.to(device)
#     scales = scales.to(device)
#     rots = rots.to(device)
#     opacities = opacities.to(device)
    
#     # Initialize gaussians with the computed values
#     gaussians._xyz = xyz_normalized  # Use normalized coordinates!
#     gaussians._features_dc = features
    
#     # Disable scale_bias and opacity_bias, move to correct device
#     gaussians.scale_bias = torch.tensor(0.0, device=gaussians._xyz.device)
#     gaussians.opacity_bias = torch.tensor(0.0, device=gaussians._xyz.device)
    
#     # Debug: check scaling before and after inverse activation
#     scales_internal = gaussians.inverse_scaling_activation(scales)
    
#     gaussians._scaling = scales_internal
#     gaussians._rotation = rots
    
#     # Clamp opacities to avoid numerical issues with inverse_sigmoid at exactly 0 or 1
#     opacities_clamped = torch.clamp(opacities, 1e-6, 1.0 - 1e-6)
#     opacities_internal = gaussians.inverse_opacity_activation(opacities_clamped)
    
#     gaussians._opacity = opacities_internal
    
#     print(f"\nGaussians initialized on device: {gaussians._xyz.device}")
#     print(f"AABB device: {gaussians.aabb.device}")
#     print(f"\nfeatures shape: {gaussians.get_features.shape}, min: {gaussians.get_features.min().item():.3f}, max: {gaussians.get_features.max().item():.3f}")
#     print(f"opacities shape: {gaussians.get_opacity.shape}, min: {gaussians.get_opacity.min().item():.3f}, max: {gaussians.get_opacity.max().item():.3f}")
#     print(f"scaling shape: {gaussians.get_scaling.shape}, min: {gaussians.get_scaling.min().item():.6f}, max: {gaussians.get_scaling.max().item():.6f}")
#     print(f"rotation shape: {gaussians.get_rotation.shape}, min: {gaussians.get_rotation.min().item():.3f}, max: {gaussians.get_rotation.max().item():.3f}")
    
#     return gaussians


def join_gaussians(*gaussian_objects: Gaussian) -> Gaussian:
    """
    Join multiple Gaussian objects into a single combined Gaussian object.
    
    Args:
        *gaussian_objects: Variable number of Gaussian objects to combine
        
    Returns:
        Combined Gaussian object containing all gaussians from input objects
    """
    if len(gaussian_objects) == 0:
        raise ValueError("At least one Gaussian object must be provided")
    
    if len(gaussian_objects) == 1:
        return gaussian_objects[0]
    
    # Collect all properties from each Gaussian object
    all_xyz = []
    all_features = []
    all_scales = []
    all_rots = []
    all_opacities = []
    
    for gs in gaussian_objects:
        all_xyz.append(gs.get_xyz)
        all_features.append(gs.get_features)
        all_scales.append(gs.get_scaling)
        all_rots.append(gs.get_rotation)
        all_opacities.append(gs.get_opacity)
    
    # Concatenate all properties
    combined_xyz = torch.cat(all_xyz, dim=0)
    combined_features = torch.cat(all_features, dim=0)
    combined_scales = torch.cat(all_scales, dim=0)
    combined_rots = torch.cat(all_rots, dim=0)
    combined_opacities = torch.cat(all_opacities, dim=0)
    
    # Create new combined Gaussian object
    combined_gs = create_gaussians_object(
        xyz=combined_xyz,
        features=combined_features,
        scales=combined_scales,
        rots=combined_rots,
        opacities=combined_opacities,
    )
    
    return combined_gs


def depth_to_pointmap(
    depth_map: np.ndarray,
    K: np.ndarray,
    normalize_depth: bool = False,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Convert depth map to 3D pointmap using camera intrinsics.
    
    Parameters
    ----------
    depth_map : np.ndarray
        Depth map as a NumPy array
    fx, fy : float
        Focal lengths in pixels
    cx, cy : float
        Principal point coordinates
        
    Returns
    -------
    np.ndarray
        Pointmap as a np.ndarray of shape (H, W, 3)
    """
    H, W = depth_map.shape[:2]
    
    if valid_mask is not None:
        # Set 2 * max_depth where ~valid_mask
        depth_map[~valid_mask] = 2 * np.max(depth_map)
    
    # Normalize depth if requested
    if normalize_depth:
        depth_map = depth_map / depth_map.max()
    
    print(f"Using camera intrinsics: K={K}")
    print(f"Depth map with shape: {depth_map.shape}, dtype: {depth_map.dtype}, min: {depth_map.min()}, max: {depth_map.max()}")
    
    # Generate 3D point cloud from z-depth
    # Create pixel coordinate grids (u, v)
    v_coords, u_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Convert to 3D coordinates using pinhole camera model
    z = depth_map
    x = (u_coords - K[0, 2]) * z / K[0, 0]
    y = (v_coords - K[1, 2]) * z / K[1, 1]
    
    pointmap = np.stack((x, y, z), axis=-1)  # (H, W, 3)
    
    print(f"Generated pointmap with shape: {pointmap.shape}, min: {pointmap.min():.3f}, max: {pointmap.max():.3f}")
    
    return pointmap

def create_gaussians_from_pointmap(
    image: np.ndarray,
    pointmap: np.ndarray,
    K: np.ndarray,
    output_path: str | None = None,
    valid_mask: np.ndarray | None = None,
) -> Gaussian:
    """
    Create Gaussian splats from pointmap and RGB image.
    
    Parameters
    ----------
    image : np.ndarray
        RGB image as a NumPy array
    pointmap : np.ndarray
        Pointmap as a NumPy array of shape (H, W, 3)
    K : np.ndarray
        Camera intrinsics matrix
    output_path : str | None, optional
        Path to save the Gaussian PLY file
        
    Returns
    -------
    Gaussian
        The created Gaussian model
    """
    # Load image
    # H, W, _ = image.shape
    
    # Load depth map
    depth_map = pointmap[..., 2]
    
    if valid_mask is not None:
        # Set 2 * max_depth where ~valid_mask
        depth_map[~valid_mask] = 2 * np.max(depth_map)
    
    # Create Gaussians from pointmap
    # Reshape pointmap to (N, 3)
    xyz = pointmap.reshape(-1, 3)
    xyz = torch.from_numpy(xyz).float() # (N, 3)
    
    # Convert RGB to SH degree 0
    # SH0 = (RGB - 0.5) / C0, where C0 = 0.28209479177387814
    rgb = image.reshape(-1, 3).astype(np.float32) / 255.0  # Normalize to [0, 1]
    # rgb = torch.from_numpy(rgb).float()
    features = RGB2SH(rgb)
    features = torch.from_numpy(features).float().unsqueeze(1)  # (N, 1, 3) for SH degree 0
    
    # Compute scales using compute_conegs_scaling
    K_torch = torch.from_numpy(K).float()
    K_inv = torch.inverse(K_torch)
    
    # Get depth values (z-coordinate) from pointmap (in world coordinates)
    points_depth = xyz[:, 2]  # (N,)
    
    # Compute scaling using the function (this gives us scales in world coordinates)
    scales_sigma_world = compute_conegs_scaling(xyz, points_depth, K_inv)  # (N, 1)
    
    # IMPORTANT: Scales should remain in world coordinates!
    # The get_scaling() function does NOT denormalize - it just applies activation
    # Only get_xyz() denormalizes coordinates using AABB
    # So scales must be in the same coordinate system as the denormalized xyz
    
    # Apply multiplier
    scales_sigma_world = scales_sigma_world
    
    # Make it isotropic (same scale in all 3 dimensions)
    scales = scales_sigma_world.repeat(1, 3)  # (N, 3)
    
    # All rotations should be identity quaternion [0, 0, 0, 1]
    rots = torch.zeros((xyz.shape[0], 4), dtype=torch.float32)
    rots[:, -1] = 1
    
    # All opacities should be 1.0
    opacities = torch.ones((xyz.shape[0], 1), dtype=torch.float32)
    
    # Create Gaussian model
    gaussians = create_gaussians_object(
        xyz=xyz,
        features=features,
        scales=scales,
        rots=rots,
        opacities=opacities,
    )
    
    # Save gaussians to ply if output path provided
    if output_path is not None:
        gaussians.save_ply(output_path)
        print(f"\nSaved Gaussians to: {output_path}")
    
    return gaussians


def redecode_slat(pipeline, slat, formats=["gaussian", "mesh"]):
    """
    Re-run the decoder forward pass using saved SLAT tokens.
    """
    print(f"Re-decoding SLAT tokens to formats: {formats}")
    print(f"  SLAT features shape: {slat.feats.shape}")
    print(f"  SLAT coords shape: {slat.coords.shape}")
    
    with torch.no_grad():
        decoded_outputs = pipeline.decode_slat(slat, formats=formats)
        
    # Print info about decoded outputs
    if "gaussian" in decoded_outputs:
        gs = decoded_outputs["gaussian"][0]
        print(f"  Decoded Gaussians: {gs.get_xyz.shape[0]} points")
        print(f"    xyz range: [{gs.get_xyz.min().item():.3f}, {gs.get_xyz.max().item():.3f}]")
    
    if "mesh" in decoded_outputs:
        mesh = decoded_outputs["mesh"][0]
        print(f"  Decoded Mesh: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")
        
    return decoded_outputs


def compute_frame_weights_from_masks(tokens_list, dataset_path, scene_name, dataset_type, obj_idx):
    """
    Compute frame weights based on object mask visibility.
    
    Frames with larger mask areas (more visible object) get higher weights.
    """
    weights = []
    
    for frame_idx, decoder_input in tokens_list:
        try:
            if dataset_type == "kubric4d":
                frames_path = os.path.join(dataset_path, scene_name, "frames_p0_v0")
                mask_files = sorted([f for f in os.listdir(frames_path) if f.startswith("segmentation_") and f.endswith(".png")])
                
                if frame_idx < len(mask_files):
                    mask_path = os.path.join(frames_path, mask_files[frame_idx])
                    masks = load_masks(mask_path)
                    
                    if obj_idx < len(masks):
                        mask = masks[obj_idx]
                        mask_area = np.count_nonzero(mask)
                        weights.append(mask_area)
                    else:
                        weights.append(1.0)
                else:
                    weights.append(1.0)
                    
            elif dataset_type == "davis":
                masks_path = os.path.join(dataset_path, "Annotations", "Full-Resolution", scene_name)
                mask_files = sorted([f for f in os.listdir(masks_path) if f.endswith(".png")])
                
                if frame_idx < len(mask_files):
                    mask_path = os.path.join(masks_path, mask_files[frame_idx])
                    masks = load_masks(mask_path)
                    
                    if obj_idx < len(masks):
                        mask = masks[obj_idx]
                        mask_area = np.count_nonzero(mask)
                        weights.append(mask_area)
                    else:
                        weights.append(1.0)
                else:
                    weights.append(1.0)
            else:
                weights.append(1.0)
                
        except Exception as e:
            print(f"    Warning: Could not load mask for frame {frame_idx}, object {obj_idx}: {e}")
            weights.append(1.0)
    
    weights = torch.tensor(weights, dtype=torch.float32)
    
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = torch.ones_like(weights) / len(weights)
    
    return weights


def compute_frame_weights_from_error(tokens_by_object, obj_idx, args, paths, inference, tokens_dir):
    """
    Compute frame weights based on per-frame rendering error in the masked region.
    
    Frames with lower reconstruction error get higher weights.
    Error is computed as masked MSE between rendered and ground truth image.
    
    Parameters
    ----------
    tokens_by_object : dict
        Dictionary mapping obj_idx -> list of (frame_idx, decoder_input) tuples
    obj_idx : int
        Object index to compute weights for
    args : argparse.Namespace
        Command line arguments
    paths : dict
        Dataset paths
    inference : Inference
        Inference pipeline
    tokens_dir : str
        Directory where cached tokens are stored
        
    Returns
    -------
    torch.Tensor
        Weights for each frame, shape (num_frames,), normalized to sum to 1.0
        Higher weight = lower error = better reconstruction
    """
    tokens_list = tokens_by_object[obj_idx]
    errors = []
    pipeline = inference._pipeline
    
    print(f"    Computing error-based weights for object {obj_idx}...")
    
    for frame_idx, decoder_input in tokens_list:
        try:
            # Load frame's image and mask
            image_path = os.path.join(paths['frames_path'], paths['image_names'][frame_idx])
            mask_path = os.path.join(paths['masks_path'], paths['mask_names'][frame_idx])
            
            image = load_image(image_path)
            image = image[..., :3]
            H, W, _ = image.shape
            
            masks = load_masks(mask_path)
            if obj_idx < len(masks):
                mask = masks[obj_idx]
            else:
                print(f"      Frame {frame_idx}: Object {obj_idx} not in masks, using uniform weight")
                errors.append(1.0)
                continue
            
            # Load depth for K_matrix
            depth_names_for_frame = []
            if paths['dataset_type'] == 'kubric4d' and paths['depth_names']:
                depth_names_for_frame = [paths['depth_names'][frame_idx]]
            
            pointmap, K_matrix, valid_mask = load_and_process_depth(
                paths['frames_path'],
                depth_names_for_frame,
                W, H,
                use_moge=args.use_moge,
                inference=inference,
                image=image
            )
            
            # Decode this frame's tokens to get Gaussian
            slat = decoder_input['decoder_input_slat']
            decoded = redecode_slat(pipeline, slat, formats=["gaussian"])
            
            # Build output and render
            output = {
                'gaussian': decoded['gaussian'],
                'rotation': decoder_input['rotation'],
                'translation': decoder_input['translation'],
                'scale': decoder_input['scale'],
            }
            
            scene_gs = make_scene(output)
            new_scene_gs = transform_scene_to_r3_convention(scene_gs)
            
            # Render
            rendered = render_gaussians_to_image(new_scene_gs, K_matrix, W, H)
            rendered = torch.clamp(rendered.cpu(), 0.0, 1.0)
            
            # Ground truth
            gt_image = torch.from_numpy(image).float() / 255.0
            
            # Compute masked MSE
            mask_tensor = torch.from_numpy(mask).float()
            if mask_tensor.dim() == 2:
                mask_tensor = mask_tensor.unsqueeze(-1)  # (H, W, 1)
            
            # Compute error only in masked region
            diff = (rendered - gt_image) ** 2
            masked_diff = diff * mask_tensor
            
            # Mean error in masked region
            num_masked_pixels = mask_tensor.sum()
            if num_masked_pixels > 0:
                masked_mse = masked_diff.sum() / (num_masked_pixels * 3)  # 3 for RGB channels
                errors.append(masked_mse.item())
            else:
                errors.append(1.0)  # Default error if no masked pixels
            
            print(f"      Frame {frame_idx}: masked MSE = {errors[-1]:.6f}")
            
        except Exception as e:
            print(f"      Frame {frame_idx}: Error computing - {e}")
            errors.append(1.0)
    
    # Convert errors to weights: lower error = higher weight
    # Use inverse error with softmax-like normalization
    errors = torch.tensor(errors, dtype=torch.float32)
    
    # Avoid division by zero and use inverse
    epsilon = 1e-6
    inverse_errors = 1.0 / (errors + epsilon)
    
    # Normalize to sum to 1
    weights = inverse_errors / inverse_errors.sum()
    
    print("    Error-based weights computed:")
    for i, (frame_idx, _) in enumerate(tokens_list):
        print(f"      Frame {frame_idx}: error={errors[i]:.6f}, weight={weights[i]:.4f}")
    
    return weights


def average_slat_tokens(tokens_list, weights=None):
    """
    Average SLAT tokens across multiple frames with optional weighting.
    
    Parameters
    ----------
    tokens_list : list of tuples
        List of (frame_index, decoder_input) tuples
    weights : torch.Tensor, optional
        Per-frame weights, shape (num_frames,). If None, uses uniform weights.
        
    Returns
    -------
    sp.SparseTensor
        Averaged SLAT tokens
    """
    if len(tokens_list) == 1:
        return tokens_list[0][1]['decoder_input_slat']
    
    if weights is None:
        weights = torch.ones(len(tokens_list), dtype=torch.float32) / len(tokens_list)
        weights = weights.cuda()
        print(f"  Averaging tokens from {len(tokens_list)} frames (uniform weights)...")
    else:
        weights = weights.cuda()
        print(f"  Averaging tokens from {len(tokens_list)} frames (weighted)...")
    
    # Collect all coordinates and features
    all_coords = []
    all_feats = []
    
    for frame_idx, decoder_input in tokens_list:
        coords = decoder_input['decoder_input_slat'].coords
        feats = decoder_input['decoder_input_slat'].feats
        all_coords.append(coords)
        all_feats.append(feats)
    
    def coords_to_dict(coords, feats):
        return {tuple(c.tolist()): f for c, f in zip(coords, feats)}
    
    def coords_to_set(coords):
        return {tuple(c.tolist()) for c in coords}
    
    coord_feat_maps = []
    coord_sets = []
    for coords, feats in zip(all_coords, all_feats):
        coord_feat_maps.append(coords_to_dict(coords, feats))
        coord_sets.append(coords_to_set(coords))
    
    # Find union of all coordinates
    all_coord_set = set()
    for cs in coord_sets:
        all_coord_set.update(cs)
    
    print(f"    Total unique coordinates: {len(all_coord_set)}")
    
    # Compute weighted average for each coordinate
    averaged_coords = []
    averaged_feats = []
    
    for coord in all_coord_set:
        feats_at_coord = []
        weights_at_coord = []
        
        for i, cfm in enumerate(coord_feat_maps):
            if coord in cfm:
                feats_at_coord.append(cfm[coord])
                weights_at_coord.append(weights[i])
        
        stacked_feats = torch.stack(feats_at_coord, dim=0)
        stacked_weights = torch.stack(weights_at_coord).unsqueeze(1)
        stacked_weights = stacked_weights / stacked_weights.sum()
        
        avg_feat = (stacked_feats * stacked_weights).sum(dim=0)
        
        averaged_coords.append(list(coord))
        averaged_feats.append(avg_feat)
    
    avg_coords = torch.tensor(averaged_coords, dtype=torch.int32, device='cuda')
    avg_feats = torch.stack(averaged_feats, dim=0)
    
    print(f"    Averaged result: {avg_coords.shape[0]} tokens, features shape {avg_feats.shape}")
    
    avg_slat = sp.SparseTensor(
        coords=avg_coords,
        feats=avg_feats,
    ).cuda()
    
    return avg_slat


def load_decoder_inputs_from_cache(cache_file):
    """
    Load decoder inputs (SLAT tokens) and pose from a cached results file.
    
    Parameters
    ----------
    cache_file : str
        Path to the .npz cache file created by demo.py
        
    Returns
    -------
    list of dict
        List of decoder inputs, one per object. Each dict contains:
        - decoder_input_slat: SparseTensor with SLAT latent features
        - rotation, translation, scale: Layout parameters (pose)
        
    Raises
    ------
    FileNotFoundError
        If the cache file does not exist
    ValueError
        If decoder inputs or poses are missing from the cache
    """
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Cache file not found: {cache_file}")
    
    cached_archive = np.load(cache_file, allow_pickle=True)
    cached_data = cached_archive["cached_data"]
    
    decoder_inputs = []
    for i, data in enumerate(cached_data):
        data = data.item() if hasattr(data, 'item') else data
        
        # Check if decoder inputs are available
        if 'decoder_input_slat_feats' not in data or 'decoder_input_slat_coords' not in data:
            raise ValueError(
                f"Object {i} in {cache_file} does not have decoder inputs saved. "
                f"Re-run demo.py to regenerate the cache with decoder inputs."
            )
        
        # Check if poses are available
        required_pose_keys = ['rotation', 'translation', 'scale']
        missing_keys = [k for k in required_pose_keys if k not in data]
        if missing_keys:
            raise ValueError(
                f"Object {i} in {cache_file} is missing pose parameters: {missing_keys}. "
                f"Re-run demo.py to regenerate the cache with pose data."
            )
        
        slat_feats = torch.from_numpy(data['decoder_input_slat_feats']).cuda()
        slat_coords = torch.from_numpy(data['decoder_input_slat_coords']).cuda()
        
        # Reconstruct SparseTensor for slat
        slat = sp.SparseTensor(
            coords=slat_coords,
            feats=slat_feats,
        ).cuda()
        
        decoder_input = {
            'decoder_input_slat': slat,
            'rotation': torch.from_numpy(data['rotation']).cuda(),
            'translation': torch.from_numpy(data['translation']).cuda(),
            'scale': torch.from_numpy(data['scale']).cuda(),
        }
        
        decoder_inputs.append(decoder_input)
    
    return decoder_inputs


def load_all_frame_tokens(tokens_dir, scene_name, first_object_only=False, with_background=False):
    """
    Load SLAT tokens from all available frames for a scene.
    
    Returns
    -------
    dict
        Dictionary mapping object_index -> list of (frame_index, decoder_input) tuples
    """
    # Find all cache files for this scene
    cache_pattern = f"{scene_name}_f*"
    if first_object_only:
        cache_pattern += "_first"
    if with_background:
        cache_pattern += "_bg"
    cache_pattern += "_sam3d_results.npz"
    
    cache_files = sorted(glob.glob(os.path.join(tokens_dir, cache_pattern)))
    
    if not cache_files:
        # Try without suffixes
        cache_pattern = f"{scene_name}_f*_sam3d_results.npz"
        cache_files = sorted(glob.glob(os.path.join(tokens_dir, cache_pattern)))
    
    print(f"Found {len(cache_files)} cache files for scene '{scene_name}'")
    
    # Organize tokens by object index
    # tokens_by_object[obj_idx] = [(frame_idx, decoder_input), ...]
    tokens_by_object = {}
    
    for cache_file in cache_files:
        # Extract frame index from filename
        basename = os.path.basename(cache_file)
        # Pattern: {scene_name}_f{frame_idx}_...
        try:
            parts = basename.split('_')
            frame_part = [p for p in parts if p.startswith('f') and p[1:].isdigit()][0]
            frame_idx = int(frame_part[1:])
        except (IndexError, ValueError):
            print(f"Warning: Could not parse frame index from {basename}, skipping")
            continue
        
        print(f"  Loading frame {frame_idx} from {basename}")
        
        try:
            decoder_inputs = load_decoder_inputs_from_cache(cache_file)
        except Exception as e:
            print(f"    Error loading: {e}")
            continue
        
        for obj_idx, decoder_input in enumerate(decoder_inputs):
            if obj_idx not in tokens_by_object:
                tokens_by_object[obj_idx] = []
            tokens_by_object[obj_idx].append((frame_idx, decoder_input))
    
    # Sort by frame index
    for obj_idx in tokens_by_object:
        tokens_by_object[obj_idx].sort(key=lambda x: x[0])
    
    return tokens_by_object


def compute_and_cache_frame_tokens(
    args, paths, frame_index, inference, tokens_dir
):
    """
    Compute tokens for a single frame and cache them.
    
    This runs full SAM3D inference on the frame and saves the results
    in the same format as demo.py.
    
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
        Directory where cached tokens will be stored
        
    Returns
    -------
    str
        Path to the cached file
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
    
    print(f"    Loaded image {image.shape}, {len(masks)} masks")
    
    # Process depth
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
    
    # Transform to PyTorch3D convention for inference
    pointmap = transform_to_pytorch3d_convention(pointmap)
    
    # Run inference
    outputs = run_inference_on_masks(inference, image, masks, pointmap, seed=args.seed)
    
    # Build cache filename (matching demo.py format)
    cache_parts = [args.scene_name, f"f{frame_index}"]
    if args.first_object_only:
        cache_parts.append("first")
    if args.with_background:
        cache_parts.append("bg")
    cache_scene_name = "_".join(cache_parts)
    
    # Cache results
    os.makedirs(tokens_dir, exist_ok=True)
    save_tokens(tokens_dir, cache_scene_name, outputs)
    
    cache_file = os.path.join(tokens_dir, f"{cache_scene_name}_sam3d_results.npz")
    return cache_file


def ensure_all_frames_have_tokens(
    args, paths, frame_indices, inference, tokens_dir
):
    """
    Ensure all requested frames have cached tokens.
    
    For frames without cached tokens, compute and cache them.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    paths : dict
        Dataset paths
    frame_indices : list
        List of frame indices that should have tokens
    inference : Inference
        Inference pipeline
    tokens_dir : str
        Directory where cached tokens are stored
        
    Returns
    -------
    dict
        tokens_by_object dictionary with all requested frames
    """
    # First, load existing tokens
    tokens_by_object = load_all_frame_tokens(
        tokens_dir, args.scene_name,
        args.first_object_only, args.with_background
    )
    
    # Find which frames already have tokens
    existing_frame_indices = set()
    for obj_idx, tokens_list in tokens_by_object.items():
        for fid, _ in tokens_list:
            existing_frame_indices.add(fid)
    
    # Find missing frames
    missing_frames = [f for f in frame_indices if f not in existing_frame_indices]
    
    if not missing_frames:
        print(f"All {len(frame_indices)} requested frames have cached tokens")
        return tokens_by_object
    
    print(f"\n{len(missing_frames)} frames need inference: {missing_frames}")
    print("Computing and caching missing frames...")
    
    for i, frame_index in enumerate(missing_frames):
        print(f"\n  Frame {frame_index} ({i + 1}/{len(missing_frames)})")
        
        try:
            cache_file = compute_and_cache_frame_tokens(
                args, paths, frame_index, inference, tokens_dir
            )
            
            # Load the newly cached tokens and add to tokens_by_object
            decoder_inputs = load_decoder_inputs_from_cache(cache_file)
            for obj_idx, decoder_input in enumerate(decoder_inputs):
                if obj_idx not in tokens_by_object:
                    tokens_by_object[obj_idx] = []
                tokens_by_object[obj_idx].append((frame_index, decoder_input))
            
            print(f"    Cached to {os.path.basename(cache_file)}")
            
        except Exception as e:
            print(f"    Error computing frame {frame_index}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Re-sort by frame index
    for obj_idx in tokens_by_object:
        tokens_by_object[obj_idx].sort(key=lambda x: x[0])
    
    print(f"\nNow have tokens for {len(existing_frame_indices | set(missing_frames))} frames")
    
    return tokens_by_object


def render_gaussian_differentiable(
    means, quats, scales, opacities, features,
    K_matrix, W, H, bg_color=None
):
    """
    Differentiable Gaussian rendering using gsplat.
    
    Parameters
    ----------
    means : torch.Tensor
        Gaussian positions (N, 3)
    quats : torch.Tensor
        Gaussian rotations as quaternions (N, 4)
    scales : torch.Tensor
        Gaussian scales (N, 3)
    opacities : torch.Tensor
        Gaussian opacities (N,) or (N, 1)
    features : torch.Tensor
        Gaussian colors/features (N, 3) or (N, 1, 3)
    K_matrix : np.ndarray or torch.Tensor
        Camera intrinsics (3, 3)
    W, H : int
        Image dimensions
    bg_color : torch.Tensor or None
        Background color (3,), defaults to white
        
    Returns
    -------
    torch.Tensor
        Rendered image (H, W, 3)
    """
    device = means.device
    
    # Identity camera (camera at origin looking along +Z)
    w2c = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)  # [1, 4, 4]
    
    # Intrinsics
    if isinstance(K_matrix, np.ndarray):
        K = torch.from_numpy(K_matrix).float().to(device)
    else:
        K = K_matrix.float().to(device)
    K = K.unsqueeze(0)  # [1, 3, 3]
    
    # Handle opacity shape
    if opacities.dim() == 2:
        opacities = opacities.squeeze(-1)  # (N,)
    
    # Handle features shape - gsplat with sh_degree=0 expects (N, K, 3) where K=1
    # So features should be (N, 1, 3)
    if features.dim() == 2:
        features = features.unsqueeze(1)  # (N, 3) -> (N, 1, 3)
    
    # Default to white background
    if bg_color is None:
        bg_color = torch.ones(3, device=device)
    else:
        bg_color = bg_color.to(device)
    
    # Render using gsplat
    rgbd, alpha, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=features,
        viewmats=w2c,
        Ks=K,
        width=W,
        height=H,
        near_plane=0.1,
        far_plane=100000.0,
        render_mode="RGB",
        sh_degree=0,
        rasterize_mode="classic",
        distributed=False,
        camera_model="pinhole",
        packed=False,
        backgrounds=bg_color[None, ...],
    )
    
    return rgbd[0, ..., :3]  # (H, W, 3)


def apply_pose_to_gaussian(
    canonical_gs,
    rotation,
    translation,
    scale
):
    """
    Apply a pose transformation to Gaussian positions and rotations.
    
    Parameters
    ----------
    canonical_gs : Gaussian
        The canonical Gaussian object (frozen, not modified)
    rotation : torch.Tensor
        Quaternion rotation (1, 4) or (4,)
    translation : torch.Tensor
        Translation (1, 3) or (3,)
    scale : torch.Tensor
        Scale factor (1, 3) or (3,) or (1,) or scalar
        
    Returns
    -------
    tuple
        (transformed_means, transformed_quats, scales, opacities, features)
    """
    # Get canonical Gaussian attributes
    xyz_local = canonical_gs.get_xyz  # (N, 3)
    rot_local = canonical_gs.get_rotation  # (N, 4)
    scales_local = canonical_gs.get_scaling  # (N, 3)
    opacities = canonical_gs.get_opacity  # (N, 1)
    features = canonical_gs.get_features  # (N, 1, 3) or (N, K, 3)
    
    # Ensure rotation is (4,)
    if rotation.dim() == 2:
        rotation = rotation.squeeze(0)
    
    # Ensure translation is (3,)
    if translation.dim() == 2:
        translation = translation.squeeze(0)
    
    # Ensure scale is (3,)
    if scale.dim() == 0:
        scale = scale.expand(3)
    elif scale.dim() == 1 and scale.shape[0] == 1:
        scale = scale.expand(3)
    elif scale.dim() == 2:
        scale = scale.squeeze(0)
        if scale.shape[0] == 1:
            scale = scale.expand(3)
    
    # Normalize quaternion
    rotation = rotation / rotation.norm()
    
    # Convert quaternion to rotation matrix
    R = quaternion_to_matrix(rotation.unsqueeze(0)).squeeze(0)  # (3, 3)
    
    # Transform positions: xyz_world = R @ (xyz_local * scale) + translation
    scaled_xyz = xyz_local * scale
    transformed_xyz = torch.mm(scaled_xyz, R.T) + translation
    
    # Transform rotations: rot_world = quaternion_multiply(rotation_inv, rot_local)
    # Note: Using inverse because of the convention in make_scene
    rotation_inv = quaternion_invert(rotation.unsqueeze(0)).squeeze(0)
    transformed_rot = quaternion_multiply(
        rotation_inv.unsqueeze(0).expand(rot_local.shape[0], -1),
        rot_local
    )
    
    # Transform scales
    transformed_scales = scales_local * scale
    
    return transformed_xyz, transformed_rot, transformed_scales, opacities, features


def refine_pose_for_frame(
    canonical_gs,
    initial_rotation,
    initial_translation,
    initial_scale,
    gt_image,
    mask,
    K_matrix,
    num_iterations=100,
    lr_rotation=0.01,
    lr_translation=0.001,
    lr_scale=0.001,
    verbose=True
):
    """
    Refine pose parameters using differentiable Gaussian rendering.
    
    Parameters
    ----------
    canonical_gs : Gaussian
        The canonical Gaussian object (frozen)
    initial_rotation : torch.Tensor
        Initial quaternion rotation (1, 4)
    initial_translation : torch.Tensor
        Initial translation (1, 3)
    initial_scale : torch.Tensor
        Initial scale (1, 3) or (1,)
    gt_image : torch.Tensor
        Ground truth image (H, W, 3) in [0, 1]
    mask : np.ndarray or torch.Tensor
        Object mask (H, W), boolean
    K_matrix : np.ndarray
        Camera intrinsics (3, 3)
    num_iterations : int
        Number of optimization iterations
    lr_rotation : float
        Learning rate for rotation
    lr_translation : float
        Learning rate for translation
    lr_scale : float
        Learning rate for scale
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        Refined pose parameters with keys:
        - rotation: refined quaternion (1, 4)
        - translation: refined translation (1, 3)
        - scale: refined uniform scale (1, 3)
        - loss_history: list of loss values at each iteration
        - best_iteration: iteration index with lowest loss
    """
    device = initial_rotation.device
    H, W = gt_image.shape[:2]
    
    # Prepare ground truth
    if isinstance(gt_image, np.ndarray):
        gt_image = torch.from_numpy(gt_image).float().to(device)
    else:
        gt_image = gt_image.float().to(device)
    
    # Prepare mask
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask).bool().to(device)
    else:
        mask = mask.bool().to(device)
    
    # Initialize optimizable parameters
    # Use a 6D rotation representation for better optimization, or just optimize quaternion directly
    opt_rotation = initial_rotation.clone().detach().requires_grad_(True)
    opt_translation = initial_translation.clone().detach().requires_grad_(True)
    
    # For scale, we optimize a single scalar to ensure uniform scaling
    # (make_scene requires scale[0] == scale[1] == scale[2])
    initial_scale_flat = initial_scale.view(-1)
    if initial_scale_flat.shape[0] == 3:
        # Use the mean of the 3 components as the initial scalar scale
        initial_scale_scalar = initial_scale_flat.mean().view(1)
    else:
        initial_scale_scalar = initial_scale_flat[:1]
    opt_scale_scalar = initial_scale_scalar.clone().detach().requires_grad_(True)
    
    # Create optimizer with different learning rates
    optimizer = torch.optim.Adam([
        {'params': [opt_rotation], 'lr': lr_rotation},
        {'params': [opt_translation], 'lr': lr_translation},
        {'params': [opt_scale_scalar], 'lr': lr_scale},
    ])
    
    # Background color (white for evaluation)
    bg_color = torch.ones(3, device=device)
    
    # Get Gaussian attributes (frozen)
    with torch.no_grad():
        xyz_local = canonical_gs.get_xyz.clone()
        rot_local = canonical_gs.get_rotation.clone()
        scales_local = canonical_gs.get_scaling.clone()
        opacities = canonical_gs.get_opacity.clone()
        features = canonical_gs.get_features.clone()
        # Note: features shape is (N, 1, 3) or (N, K, 3) - render_gaussian_differentiable handles this
    
    best_loss = float('inf')
    best_params = None
    best_iteration = 0
    loss_history = []
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        # Normalize quaternion
        rotation_normalized = opt_rotation / opt_rotation.norm()
        
        # Ensure proper shapes
        rotation = rotation_normalized.view(-1)
        if rotation.shape[0] != 4:
            rotation = rotation[:4]
        
        translation = opt_translation.view(-1)
        if translation.shape[0] != 3:
            translation = translation[:3]
        
        # Uniform scale: expand scalar to 3D
        scale = opt_scale_scalar.expand(3)
        
        # Convert quaternion to rotation matrix
        R = quaternion_to_matrix(rotation.unsqueeze(0)).squeeze(0)  # (3, 3)
        
        # Transform positions
        scaled_xyz = xyz_local * scale
        transformed_xyz = torch.mm(scaled_xyz, R.T) + translation
        
        # Transform rotations
        rotation_inv = quaternion_invert(rotation.unsqueeze(0)).squeeze(0)
        transformed_rot = quaternion_multiply(
            rotation_inv.unsqueeze(0).expand(rot_local.shape[0], -1),
            rot_local
        )
        
        # Transform scales
        transformed_scales = scales_local * scale
        
        # Handle opacities
        opacities_flat = opacities.squeeze(-1) if opacities.dim() == 2 else opacities
        
        # Render
        rendered = render_gaussian_differentiable(
            transformed_xyz,
            transformed_rot,
            transformed_scales,
            opacities_flat,
            features,
            K_matrix, W, H,
            bg_color=bg_color
        )
        
        # Compute loss in masked region only
        # MSE loss
        diff = (rendered - gt_image) ** 2
        masked_diff = diff[mask]
        loss = masked_diff.mean()
        
        # Optional: add regularization to prevent large deviations from initial pose
        reg_weight = 0.001
        reg_rot = ((rotation_normalized - initial_rotation / initial_rotation.norm()) ** 2).sum()
        reg_trans = ((opt_translation - initial_translation) ** 2).sum()
        reg_scale = ((opt_scale_scalar - initial_scale_scalar) ** 2).sum()
        loss = loss + reg_weight * (reg_rot + reg_trans + reg_scale)
        
        # Record loss for this iteration
        loss_history.append(loss.item())
        
        # Backprop
        loss.backward()
        optimizer.step()
        
        # Track best
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_iteration = iteration
            # Store scale as uniform 3D tensor (required by make_scene)
            scale_3d = opt_scale_scalar.clone().detach().expand(3).reshape(1, 3)
            best_params = {
                'rotation': opt_rotation.clone().detach(),
                'translation': opt_translation.clone().detach(),
                'scale': scale_3d,
            }
        
        if verbose and (iteration % 20 == 0 or iteration == num_iterations - 1):
            print(f"      Iteration {iteration}: loss = {loss.item():.6f}")
    
    if verbose:
        print(f"      Best loss: {best_loss:.6f} (iteration {best_iteration})")
    
    # Normalize the rotation in best_params
    best_params['rotation'] = best_params['rotation'] / best_params['rotation'].norm()
    
    # Add loss history and best iteration to the result
    best_params['loss_history'] = loss_history
    best_params['best_iteration'] = best_iteration
    
    return best_params


def refine_poses_for_sequence(
    canonical_gaussians,
    tokens_by_object,
    args, paths, inference,
    num_iterations=100,
    lr_rotation=0.01,
    lr_translation=0.001,
    lr_scale=0.001,
):
    """
    Refine per-frame poses for all objects using differentiable rendering.
    
    Parameters
    ----------
    canonical_gaussians : dict
        Dictionary mapping object_index -> canonical Gaussian
    tokens_by_object : dict
        Dictionary mapping object_index -> list of (frame_index, decoder_input)
    args : argparse.Namespace
        Command line arguments
    paths : dict
        Dataset paths
    inference : Inference
        Inference pipeline (for depth estimation if needed)
    num_iterations : int
        Number of optimization iterations per frame
        
    Returns
    -------
    dict
        Refined tokens_by_object with updated poses
    """
    print("\n  Refining per-frame poses with differentiable rendering...")
    
    refined_tokens = {}
    
    for obj_idx in sorted(canonical_gaussians.keys()):
        print(f"\n    Object {obj_idx}:")
        refined_tokens[obj_idx] = []
        canonical_gs = canonical_gaussians[obj_idx]
        
        for frame_idx, decoder_input in tokens_by_object[obj_idx]:
            print(f"      Frame {frame_idx}:")
            
            # Load frame data
            image_path = os.path.join(paths['frames_path'], paths['image_names'][frame_idx])
            mask_path = os.path.join(paths['masks_path'], paths['mask_names'][frame_idx])
            
            image = load_image(image_path)
            image = image[..., :3]
            H, W, _ = image.shape
            
            masks = load_masks(mask_path)
            if args.first_object_only:
                masks = masks[:1]
            
            # Get the mask for this object
            if obj_idx < len(masks):
                mask = masks[obj_idx]
            else:
                print(f"        Warning: No mask for object {obj_idx}, skipping refinement")
                refined_tokens[obj_idx].append((frame_idx, decoder_input))
                continue
            
            # Load depth and compute K_matrix
            depth_names_for_frame = []
            if paths['dataset_type'] == 'kubric4d' and paths['depth_names']:
                depth_names_for_frame = [paths['depth_names'][frame_idx]]
            
            pointmap, K_matrix, valid_mask = load_and_process_depth(
                paths['frames_path'],
                depth_names_for_frame,
                W, H,
                use_moge=args.use_moge,
                inference=inference,
                image=image
            )
            
            # Ground truth image
            gt_image = torch.from_numpy(image).float().cuda() / 255.0
            
            # Initial pose
            initial_rotation = decoder_input['rotation']
            initial_translation = decoder_input['translation']
            initial_scale = decoder_input['scale']
            
            # Refine pose
            refined_pose = refine_pose_for_frame(
                canonical_gs,
                initial_rotation,
                initial_translation,
                initial_scale,
                gt_image,
                mask,
                K_matrix,
                num_iterations=num_iterations,
                lr_rotation=lr_rotation,
                lr_translation=lr_translation,
                lr_scale=lr_scale,
                verbose=True
            )
            
            # Create refined decoder input (including loss history for plotting)
            refined_decoder_input = {
                'decoder_input_slat': decoder_input['decoder_input_slat'],
                'rotation': refined_pose['rotation'],
                'translation': refined_pose['translation'],
                'scale': refined_pose['scale'],
                'refinement_loss_history': refined_pose['loss_history'],
                'refinement_best_iteration': refined_pose['best_iteration'],
            }
            
            refined_tokens[obj_idx].append((frame_idx, refined_decoder_input))
    
    return refined_tokens


def process_frame_from_cache(args, paths, frame_index, inference, tokens_dir):
    """
    Process a single frame by loading cached tokens and re-decoding.
    
    Returns
    -------
    tuple
        (rendered_image, gt_image, K_matrix) or None if cache not found
    """
    # Build cache filename
    cache_filename, cache_scene_name = get_cache_filename(
        args.scene_name, frame_index, args.first_object_only, args.with_background
    )
    cache_file = os.path.join(tokens_dir, cache_filename)
    
    # Check if cache exists
    if not os.path.exists(cache_file):
        return None
    
    print(f"    Loading cached tokens from {cache_filename}")
    
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
    
    # Keep original pointmap for background
    pointmap_original = pointmap.copy() if args.with_background else None
    
    # Load decoder inputs from cache
    decoder_inputs = load_decoder_inputs_from_cache(cache_file)
    
    if len(decoder_inputs) == 0:
        print("    No decoder inputs found in cache")
        return None
    
    # Re-decode each object and build outputs compatible with make_scene
    pipeline = inference._pipeline
    outputs = []
    
    for i, decoder_input in enumerate(decoder_inputs):
        slat = decoder_input['decoder_input_slat']
        
        # Re-decode to get Gaussian
        decoded = redecode_slat(pipeline, slat, formats=["gaussian"])
        
        # Build output dict compatible with make_scene
        output = {
            'gaussian': decoded['gaussian'],
            'rotation': decoder_input['rotation'],
            'translation': decoder_input['translation'],
            'scale': decoder_input['scale'],
        }
        outputs.append(output)
    
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


def process_frame_full_inference(args, paths, frame_index, inference):
    """
    Process a single frame using full inference (no cache).
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
    
    print(f"\n  Frame {frame_index}: Loaded image {image.shape}, {len(masks)} masks")
    
    # Process depth
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
    
    # Transform to PyTorch3D convention
    pointmap = transform_to_pytorch3d_convention(pointmap)
    
    # Run full inference
    outputs = run_inference_on_masks(inference, image, masks, pointmap, seed=args.seed)
    
    # Create combined scene
    scene_gs = make_scene(*outputs)
    new_scene_gs = transform_scene_to_r3_convention(scene_gs)
    
    # Add background if requested
    if args.with_background and pointmap_original is not None:
        background_gs = create_background_gaussians(
            image, pointmap_original, masks, K_matrix
        )
        new_scene_gs = join_gaussians(background_gs, new_scene_gs)
    
    # Render
    rendered = render_gaussians_to_image(new_scene_gs, K_matrix, W, H)
    gt_image = torch.from_numpy(image).float() / 255.0
    rendered = torch.clamp(rendered.cpu(), 0.0, 1.0)
    
    return rendered, gt_image, K_matrix
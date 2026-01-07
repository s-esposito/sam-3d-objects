import os
import sys
import math
import numpy as np
import torch
from PIL import Image
from gsplat.rendering import rasterization

# Skip sam3d_objects initialization for lightweight tools
os.environ['LIDRA_SKIP_INIT'] = '1'

# Add parent directory to path to import sam3d_objects
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model import Gaussian

C0 = 0.28209479177387814

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5

def render_frame(
    scene_gs,
    c2w,  # Camera-to-world transformation (4, 4)
    K,    # Camera intrinsics (3, 3)
    w, h, # Width and height
):
    """
    Render a single frame from the Gaussian scene using given camera parameters.
    
    Args:
        scene_gs: Gaussian scene object
        c2w: Camera-to-world transformation matrix (4, 4)
        K: Camera intrinsics matrix (3, 3)
        w: Image width
        h: Image height
        bg_color: Background color (R, G, B) tuple
        
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
            # **kwargs,
            # backgrounds=bg_color[None, ...],  # [1, 3]
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


def load_masks(mask_path, indices_list=None):
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


def _compute_conegs_scaling(
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
    gaussians._rotation = rots
    
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

def create_gaussians_from_depth(
    image: np.ndarray,
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    normalize_depth: bool = False,
    output_path: str = None,
) -> Gaussian:
    """
    Create Gaussian splats from depth map and RGB image.
    
    Parameters
    ----------
    image : np.ndarray
        RGB image as a NumPy array
    depth : np.ndarray
        Depth map as a NumPy array
    fx, fy : float
        Focal lengths in pixels
    cx, cy : float
        Principal point coordinates
    normalize_depth : bool
        Whether to normalize depth to [0, 1]
    output_path : str, optional
        Path to save the Gaussian PLY file
        
    Returns
    -------
    Gaussian
        The created Gaussian model
    """
    # Load image
    H, W, _ = image.shape
    
    # Create intrinsics matrix
    K_matrix = np.eye(3)
    K_matrix[0, 0] = fx
    K_matrix[1, 1] = fy
    K_matrix[0, 2] = cx
    K_matrix[1, 2] = cy
    
    # Load depth map
    depth_map = depth
    
    # Normalize depth if requested
    if normalize_depth:
        depth_map = depth_map / depth_map.max()
    
    print(f"Using camera intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    print(f"Depth map with shape: {depth_map.shape}, dtype: {depth_map.dtype}, min: {depth_map.min()}, max: {depth_map.max()}")
    
    # Generate 3D point cloud from z-depth
    # Create pixel coordinate grids (u, v)
    v_coords, u_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Convert to 3D coordinates using pinhole camera model
    z = depth_map
    x = (u_coords - cx) * z / fx
    y = (v_coords - cy) * z / fy
    
    points = np.stack((x, y, z), axis=-1)  # (H, W, 3)
    pointmap = torch.from_numpy(points).float()  # (H, W, 3)
    
    print(f"Generated pointmap with shape: {pointmap.shape}, min: {pointmap.min():.3f}, max: {pointmap.max():.3f}")
    
    # Create Gaussians from pointmap
    # Reshape pointmap to (N, 3)
    xyz = pointmap.reshape(-1, 3)
    
    # Convert RGB to SH degree 0
    # SH0 = (RGB - 0.5) / C0, where C0 = 0.28209479177387814
    rgb = image.reshape(-1, 3).astype(np.float32) / 255.0  # Normalize to [0, 1]
    # rgb = torch.from_numpy(rgb).float()
    features = RGB2SH(rgb)
    features = torch.from_numpy(features).float().unsqueeze(1)  # (N, 1, 3) for SH degree 0
    
    # Compute scales using _compute_conegs_scaling
    K_torch = torch.from_numpy(K_matrix).float()
    K_inv = torch.inverse(K_torch)
    
    # Get depth values (z-coordinate) from pointmap (in world coordinates)
    points_depth = xyz[:, 2]  # (N,)
    
    # Compute scaling using the function (this gives us scales in world coordinates)
    scales_sigma_world = _compute_conegs_scaling(xyz, points_depth, K_inv)  # (N, 1)
    
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
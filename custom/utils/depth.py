"""
Depth processing utilities for the SAM3D-Objects pipeline.

This module provides functions for depth map processing, pointmap generation,
and coordinate system transformations between R3 and PyTorch3D conventions.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import torch
from pytorch3d.renderer import look_at_view_transform

if TYPE_CHECKING:
    from inference import Inference


def radial_to_z_depth(
    radial_depth_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """
    Convert a radial depth map to a z-depth map using pinhole camera model.

    Mathematical derivation:
    - For a point at (x, y, z) in camera coordinates:
      x = (u - cx) * z / fx
      y = (v - cy) * z / fy
    - Radial distance: r = sqrt(x² + y² + z²)
    - Substituting: r = z * sqrt((u-cx)²/fx² + (v-cy)²/fy² + 1)
    - Therefore: z = r / sqrt((u-cx)²/fx² + (v-cy)²/fy² + 1)

    Parameters
    ----------
    radial_depth_map : np.ndarray
        Array of radial depths (Euclidean distance from camera center), shape (H, W).
    fx : float
        Horizontal focal length of the camera in pixels.
    fy : float
        Vertical focal length of the camera in pixels.
    cx : float
        Principal point x-coordinate in pixel coordinates.
    cy : float
        Principal point y-coordinate in pixel coordinates.

    Returns
    -------
    np.ndarray
        The z-depth map (distance along optical axis), shape (H, W).

    Raises
    ------
    AssertionError
        If any camera intrinsic parameter is None.

    Examples
    --------
    >>> radial_depth = np.ones((480, 640)) * 5.0  # 5 meters radial
    >>> z_depth = radial_to_z_depth(radial_depth, fx=500, fy=500, cx=320, cy=240)
    >>> z_depth.shape
    (480, 640)
    """
    assert fx is not None, "Focal length fx is not specified"
    assert fy is not None, "Focal length fy is not specified"
    assert cx is not None, "Principal point cx is not specified"
    assert cy is not None, "Principal point cy is not specified"

    H, W = radial_depth_map.shape[:2]

    # Create a grid of pixel coordinates
    # v corresponds to rows (height), u corresponds to cols (width)
    v_coords, u_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

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


def depth_to_pointmap(
    depth_map: np.ndarray,
    K: np.ndarray,
    normalize_depth: bool = False,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convert depth map to 3D pointmap using camera intrinsics.

    Parameters
    ----------
    depth_map : np.ndarray
        Depth map as a NumPy array of shape (H, W).
    K : np.ndarray
        Camera intrinsics matrix of shape (3, 3).
    normalize_depth : bool, optional
        Whether to normalize depth values to [0, 1] range. Default: False.
    valid_mask : np.ndarray, optional
        Boolean mask indicating valid depth values, shape (H, W).
        Invalid pixels are set to 2 * max_depth.

    Returns
    -------
    np.ndarray
        Pointmap as a NumPy array of shape (H, W, 3) where each pixel
        contains its 3D (x, y, z) coordinates in camera space.

    Examples
    --------
    >>> depth = np.random.rand(480, 640) * 10  # Random depth 0-10m
    >>> K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
    >>> pointmap = depth_to_pointmap(depth, K)
    >>> pointmap.shape
    (480, 640, 3)
    """
    H, W = depth_map.shape[:2]

    if valid_mask is not None:
        # Set 2 * max_depth where ~valid_mask
        depth_map = depth_map.copy()
        depth_map[~valid_mask] = 2 * np.max(depth_map)

    # Normalize depth if requested
    if normalize_depth:
        depth_map = depth_map / depth_map.max()

    print(f"Using camera intrinsics: K={K}")
    print(
        f"Depth map with shape: {depth_map.shape}, dtype: {depth_map.dtype}, "
        f"min: {depth_map.min()}, max: {depth_map.max()}"
    )

    # Generate 3D point cloud from z-depth
    # Create pixel coordinate grids (u, v)
    v_coords, u_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    # Convert to 3D coordinates using pinhole camera model
    z = depth_map
    x = (u_coords - K[0, 2]) * z / K[0, 0]
    y = (v_coords - K[1, 2]) * z / K[1, 1]

    pointmap = np.stack((x, y, z), axis=-1)  # (H, W, 3)

    print(
        f"Generated pointmap with shape: {pointmap.shape}, "
        f"min: {pointmap.min():.3f}, max: {pointmap.max():.3f}"
    )

    return pointmap


def transform_to_pytorch3d_convention(pointmap: np.ndarray) -> np.ndarray:
    """
    Transform pointmap from R3 to PyTorch3D camera convention.

    R3 convention: X-right, Y-down, Z-forward
    PyTorch3D convention: X-left, Y-up, Z-forward

    Parameters
    ----------
    pointmap : np.ndarray
        Pointmap in R3 convention, shape (H, W, 3) or (N, 3).

    Returns
    -------
    np.ndarray
        Pointmap in PyTorch3D convention, same shape as input.

    Notes
    -----
    This transformation is applied before running SAM3D inference,
    as the model internally uses PyTorch3D conventions.
    """
    # Camera convention transformation (R3 -> PyTorch3D)
    r3_to_p3d_R, _ = look_at_view_transform(
        eye=np.array([[0, 0, -1]]),
        at=np.array([[0, 0, 0]]),
        up=np.array([[0, -1, 0]]),
    )

    # Convert rotation matrix to numpy
    r3_to_p3d_R_np = r3_to_p3d_R.cpu().numpy()[0]  # (3, 3)

    # Apply rotation using numpy matrix multiplication
    pointmap_transformed = pointmap @ r3_to_p3d_R_np.T

    return pointmap_transformed


def load_and_process_depth(
    frames_path: str,
    depth_names: list[str],
    W: int,
    H: int,
    use_moge: bool = False,
    inference: Optional["Inference"] = None,
    image: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load and process depth maps to generate pointmap.

    Supports two modes:
    1. Ground truth depth (Kubric4D): Load from TIFF files
    2. MoGe estimated depth: Run monocular depth estimation model

    Parameters
    ----------
    frames_path : str
        Path to the frames directory (for Kubric4D GT depth).
    depth_names : list[str]
        List of depth file names (empty for DAVIS/MoGe mode).
    W : int
        Image width.
    H : int
        Image height.
    use_moge : bool, optional
        Whether to use MoGe depth model instead of GT depth. Default: False.
    inference : Inference, optional
        Inference pipeline (required for MoGe mode).
    image : np.ndarray, optional
        Input image (required for MoGe mode).

    Returns
    -------
    tuple
        (pointmap, K_matrix, valid_mask) where:
        - pointmap: 3D coordinates array of shape (H, W, 3)
        - K_matrix: Camera intrinsics matrix of shape (3, 3)
        - valid_mask: Boolean mask of valid depth values, or None for GT depth

    Raises
    ------
    ValueError
        If MoGe mode is enabled but inference or image is not provided.
    """
    import os

    from .io_utils import load_image

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
        print(
            f"Radial depth map shape: {depth_map.shape}, dtype: {depth_map.dtype}, "
            f"min: {depth_map.min():.4f}, max: {depth_map.max():.4f}"
        )

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
        fy = fx  # Square pixels
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
    print(
        f"Generated pointmap with shape: {pointmap.shape}, "
        f"min: {pointmap.min():.4f}, max: {pointmap.max():.4f}"
    )

    return pointmap, K_matrix, valid_mask


def verify_reprojection(
    points_3d: np.ndarray,
    K_matrix: np.ndarray,
    H: int,
    W: int,
    num_samples: int = 100,
) -> dict:
    """
    Verify that 3D points correctly reproject back to their 2D pixel coordinates.

    This function samples random pixels, reprojects their 3D coordinates to 2D,
    and computes the reprojection error statistics.

    Parameters
    ----------
    points_3d : np.ndarray
        Array of 3D points in camera coordinates, shape (H, W, 3).
    K_matrix : np.ndarray
        Camera intrinsics matrix, shape (3, 3).
    H : int
        Image height.
    W : int
        Image width.
    num_samples : int, optional
        Number of random points to sample for verification. Default: 100.

    Returns
    -------
    dict
        Dictionary containing reprojection statistics:
        - mean_error_u: Mean error in x direction
        - max_error_u: Maximum error in x direction
        - mean_error_v: Mean error in y direction
        - max_error_v: Maximum error in y direction
        - mean_error_total: Mean total reprojection error
        - max_error_total: Maximum total reprojection error
        - num_samples: Number of valid samples used
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
        "mean_error_u": errors_u.mean(),
        "max_error_u": errors_u.max(),
        "mean_error_v": errors_v.mean(),
        "max_error_v": errors_v.max(),
        "mean_error_total": np.sqrt(errors_u**2 + errors_v**2).mean(),
        "max_error_total": np.sqrt(errors_u**2 + errors_v**2).max(),
        "num_samples": len(errors_u),
    }

    return stats


def compute_conegs_scaling(
    points_3d_camera: torch.Tensor,
    points_depth: torch.Tensor,
    K_inv: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Gaussian scaling based on pixel footprint.

    This function calculates the appropriate Gaussian standard deviation
    for each 3D point based on its depth and the camera intrinsics.

    Parameters
    ----------
    points_3d_camera : torch.Tensor
        Camera-space 3D points for each pixel, shape (N, 3).
    points_depth : torch.Tensor
        Z-depth for each pixel, shape (N,).
    K_inv : torch.Tensor
        Inverse intrinsics matrix, shape (3, 3).

    Returns
    -------
    torch.Tensor
        Isotropic Gaussian standard deviation per pixel, shape (N, 1).

    Notes
    -----
    The scaling is based on the pixel footprint at each depth,
    ensuring consistent Gaussian sizes relative to the projected area.
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


__all__ = [
    "radial_to_z_depth",
    "depth_to_pointmap",
    "transform_to_pytorch3d_convention",
    "load_and_process_depth",
    "verify_reprojection",
    "compute_conegs_scaling",
]

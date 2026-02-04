"""
Gaussian splatting operations and utilities.

This module provides functions for creating, manipulating, and transforming
3D Gaussian splat representations, including coordinate system conversions.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.transforms import (
    Transform3d,
    matrix_to_quaternion,
    quaternion_multiply,
)

# Add parent directory to path to import sam3d_objects
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model import Gaussian

if TYPE_CHECKING:
    pass

# Spherical harmonics constant (degree 0)
C0: float = 0.28209479177387814


def RGB2SH(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB colors to spherical harmonics coefficients (degree 0).

    Parameters
    ----------
    rgb : np.ndarray
        RGB colors in [0, 1] range, shape (..., 3).

    Returns
    -------
    np.ndarray
        SH coefficients, same shape as input.

    Examples
    --------
    >>> rgb = np.array([0.5, 0.5, 0.5])
    >>> sh = RGB2SH(rgb)
    >>> sh
    array([0., 0., 0.])
    """
    return (rgb - 0.5) / C0


def SH2RGB(sh: np.ndarray) -> np.ndarray:
    """
    Convert spherical harmonics coefficients (degree 0) to RGB colors.

    Parameters
    ----------
    sh : np.ndarray
        SH coefficients, shape (..., 3).

    Returns
    -------
    np.ndarray
        RGB colors in [0, 1] range, same shape as input.

    Examples
    --------
    >>> sh = np.array([0., 0., 0.])
    >>> rgb = SH2RGB(sh)
    >>> rgb
    array([0.5, 0.5, 0.5])
    """
    return sh * C0 + 0.5


def create_gaussians_object(
    xyz: torch.Tensor,
    features: torch.Tensor,
    scales: torch.Tensor,
    rots: torch.Tensor,
    opacities: torch.Tensor,
) -> Gaussian:
    """
    Create a Gaussian model from raw parameters.

    This function handles the normalization and internal representation
    required by the Gaussian model, including computing the axis-aligned
    bounding box (AABB) for coordinate normalization.

    Parameters
    ----------
    xyz : torch.Tensor
        3D positions, shape (N, 3).
    features : torch.Tensor
        SH features (color), shape (N, 1, 3) or (N, K, 3).
    scales : torch.Tensor
        Gaussian scales, shape (N, 3).
    rots : torch.Tensor
        Quaternion rotations (wxyz format), shape (N, 4).
    opacities : torch.Tensor
        Opacities in [0, 1], shape (N, 1).

    Returns
    -------
    Gaussian
        Initialized Gaussian model with all parameters set.

    Notes
    -----
    The Gaussian model internally stores normalized coordinates and applies
    activation functions to scales and opacities. This function handles
    the necessary transformations.
    """
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
    device = "cuda"
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

    # Apply inverse activation to convert external scales to internal representation
    scales_internal = gaussians.inverse_scaling_activation(scales)

    gaussians._scaling = scales_internal
    gaussians._rotation = rots - gaussians.rots_bias[None, :]

    # Clamp opacities to avoid numerical issues with inverse_sigmoid at exactly 0 or 1
    opacities_clamped = torch.clamp(opacities, 1e-6, 1.0 - 1e-6)
    opacities_internal = gaussians.inverse_opacity_activation(opacities_clamped)

    gaussians._opacity = opacities_internal

    print(f"\nGaussians initialized on device: {gaussians._xyz.device}")
    print(f"AABB device: {gaussians.aabb.device}")
    print(
        f"\nfeatures shape: {gaussians.get_features.shape}, "
        f"min: {gaussians.get_features.min().item():.3f}, "
        f"max: {gaussians.get_features.max().item():.3f}"
    )
    print(
        f"opacities shape: {gaussians.get_opacity.shape}, "
        f"min: {gaussians.get_opacity.min().item():.3f}, "
        f"max: {gaussians.get_opacity.max().item():.3f}"
    )
    print(
        f"scaling shape: {gaussians.get_scaling.shape}, "
        f"min: {gaussians.get_scaling.min().item():.6f}, "
        f"max: {gaussians.get_scaling.max().item():.6f}"
    )
    print(
        f"rotation shape: {gaussians.get_rotation.shape}, "
        f"min: {gaussians.get_rotation.min().item():.3f}, "
        f"max: {gaussians.get_rotation.max().item():.3f}"
    )

    return gaussians


def join_gaussians(*gaussian_objects: Gaussian) -> Gaussian:
    """
    Join multiple Gaussian objects into a single combined Gaussian object.

    Parameters
    ----------
    *gaussian_objects : Gaussian
        Variable number of Gaussian objects to combine.

    Returns
    -------
    Gaussian
        Combined Gaussian object containing all gaussians from input objects.

    Raises
    ------
    ValueError
        If no Gaussian objects are provided.

    Examples
    --------
    >>> combined = join_gaussians(gs1, gs2, gs3)
    >>> combined.get_xyz.shape[0]
    30000  # Sum of points from gs1, gs2, gs3
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


def create_gaussians_from_pointmap(
    image: np.ndarray,
    pointmap: np.ndarray,
    K: np.ndarray,
    output_path: Optional[str] = None,
    valid_mask: Optional[np.ndarray] = None,
) -> Gaussian:
    """
    Create Gaussian splats from pointmap and RGB image.

    Parameters
    ----------
    image : np.ndarray
        RGB image as a NumPy array, shape (N, 3) for flattened or (H, W, 3).
    pointmap : np.ndarray
        Pointmap as a NumPy array of shape (N, 3) for flattened or (H, W, 3).
    K : np.ndarray
        Camera intrinsics matrix, shape (3, 3).
    output_path : str, optional
        Path to save the Gaussian PLY file.
    valid_mask : np.ndarray, optional
        Boolean mask indicating valid depth values.

    Returns
    -------
    Gaussian
        The created Gaussian model.
    """
    from .depth import compute_conegs_scaling

    # Load depth map
    depth_map = pointmap[..., 2]

    if valid_mask is not None:
        # Set 2 * max_depth where ~valid_mask
        depth_map = depth_map.copy()
        depth_map[~valid_mask] = 2 * np.max(depth_map)

    # Create Gaussians from pointmap
    # Reshape pointmap to (N, 3)
    xyz = pointmap.reshape(-1, 3)
    xyz = torch.from_numpy(xyz).float()  # (N, 3)

    # Convert RGB to SH degree 0
    # SH0 = (RGB - 0.5) / C0, where C0 = 0.28209479177387814
    rgb = image.reshape(-1, 3).astype(np.float32) / 255.0  # Normalize to [0, 1]
    features = RGB2SH(rgb)
    features = torch.from_numpy(features).float().unsqueeze(1)  # (N, 1, 3) for SH degree 0

    # Compute scales using compute_conegs_scaling
    K_torch = torch.from_numpy(K).float()
    K_inv = torch.inverse(K_torch)

    # Get depth values (z-coordinate) from pointmap (in world coordinates)
    points_depth = xyz[:, 2]  # (N,)

    # Compute scaling using the function (this gives us scales in world coordinates)
    scales_sigma_world = compute_conegs_scaling(xyz, points_depth, K_inv)  # (N, 1)

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


def create_background_gaussians(
    image: np.ndarray,
    pointmap: np.ndarray,
    masks: List[np.ndarray],
    K_matrix: np.ndarray,
) -> Gaussian:
    """
    Create Gaussian splats for the background (non-masked regions).

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W, 3).
    pointmap : np.ndarray
        3D pointmap (NOT in PyTorch3D convention, but in original R3 convention),
        shape (H, W, 3).
    masks : list of np.ndarray
        List of object masks, each shape (H, W).
    K_matrix : np.ndarray
        Camera intrinsics matrix, shape (3, 3).

    Returns
    -------
    Gaussian
        Background Gaussian object.

    Notes
    -----
    The background is defined as all pixels that are NOT covered by any
    of the provided object masks.
    """
    # Create combined mask of all objects
    background_mask = ~np.any(np.stack(masks, axis=0), axis=0)
    print(
        f"Background mask: {background_mask.sum()} pixels "
        f"({100*background_mask.mean():.1f}% of image)"
    )

    # Create background Gaussians from the non-masked region
    gaussians_bg = create_gaussians_from_pointmap(
        image=image[background_mask],
        pointmap=pointmap[background_mask],
        K=K_matrix,
    )

    return gaussians_bg


def transform_scene_to_r3_convention(scene_gs: Gaussian) -> Gaussian:
    """
    Transform combined scene from PyTorch3D convention back to R3 convention.

    This should be done AFTER make_scene() on the combined scene.

    PyTorch3D convention: X-left, Y-up, Z-forward
    R3 convention: X-right, Y-down, Z-forward

    Parameters
    ----------
    scene_gs : Gaussian
        Scene Gaussian object in PyTorch3D convention.

    Returns
    -------
    Gaussian
        Scene Gaussian object in R3 convention.

    Notes
    -----
    This transformation is the inverse of transform_to_pytorch3d_convention
    and should be applied to the output of make_scene() before rendering.
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
        p3d_to_r3_quat.expand(original_rots.shape[0], -1), original_rots
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


__all__ = [
    "C0",
    "RGB2SH",
    "SH2RGB",
    "create_gaussians_object",
    "join_gaussians",
    "create_gaussians_from_pointmap",
    "create_background_gaussians",
    "transform_scene_to_r3_convention",
]

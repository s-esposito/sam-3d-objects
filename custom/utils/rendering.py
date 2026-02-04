"""
Gaussian splatting rendering utilities.

This module provides functions for rendering 3D Gaussian scenes to images
using differentiable rendering via gsplat.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from gsplat.rendering import rasterization

if TYPE_CHECKING:
    from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model import (
        Gaussian,
    )


def render_gaussian_params(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    features: torch.Tensor,
    c2w: torch.Tensor,
    K_matrix: torch.Tensor | np.ndarray,
    W: int,
    H: int,
    bg_color: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Differentiable Gaussian rendering using gsplat.

    Parameters
    ----------
    means : torch.Tensor
        Gaussian positions, shape (N, 3).
    quats : torch.Tensor
        Gaussian rotations as quaternions (wxyz format), shape (N, 4).
    scales : torch.Tensor
        Gaussian scales, shape (N, 3).
    opacities : torch.Tensor
        Gaussian opacities, shape (N,) or (N, 1).
    features : torch.Tensor
        Gaussian colors/features, shape (N, 3) or (N, 1, 3).
    c2w : torch.Tensor
        Camera-to-world transformation matrix, shape (4, 4) or (1, 4, 4).
    K_matrix : torch.Tensor or np.ndarray
        Camera intrinsics, shape (3, 3) or (1, 3, 3).
    W : int
        Image width.
    H : int
        Image height.
    bg_color : torch.Tensor, optional
        Background color, shape (3,). Defaults to black [0, 0, 0].

    Returns
    -------
    tuple
        (rgb, alpha, depth) where:
        - rgb: Rendered image, shape (H, W, 3)
        - alpha: Alpha/opacity map, shape (H, W)
        - depth: Depth map, shape (H, W)

    Examples
    --------
    >>> rgb, alpha, depth = render_gaussian_params(
    ...     means, quats, scales, opacities, features,
    ...     c2w=torch.eye(4), K_matrix=K, W=640, H=480
    ... )
    >>> rgb.shape
    torch.Size([480, 640, 3])
    """
    device = means.device

    # View matrix: from camera to world
    if isinstance(c2w, np.ndarray):
        c2w_torch = torch.from_numpy(c2w).float().to(device)
    else:
        c2w_torch = c2w.float().to(device)
    w2c = torch.inverse(c2w_torch)

    if w2c.dim() == 2:
        w2c = w2c.unsqueeze(0)  # [1, 4, 4]

    # Intrinsics
    if isinstance(K_matrix, np.ndarray):
        K = torch.from_numpy(K_matrix).float().to(device)
    else:
        K = K_matrix.float().to(device)

    if K.dim() == 2:
        K = K.unsqueeze(0)  # [1, 3, 3]

    # Handle opacity shape
    if opacities.dim() == 2:
        opacities = opacities.squeeze(-1)  # (N,)

    # Handle features shape - gsplat with sh_degree=0 expects (N, K, 3) where K=1
    # So features should be (N, 1, 3)
    if features.dim() == 2:
        features = features.unsqueeze(1)  # (N, 3) -> (N, 1, 3)

    # Default to black background
    if bg_color is None:
        bg_color = torch.zeros(3, device=device)
    else:
        bg_color = bg_color.to(device)

    # Render using gsplat
    rgbd, alpha, info = rasterization(
        means=means,  # (N, 3)
        quats=quats,  # (N, 4)
        scales=scales,  # (N, 3)
        opacities=opacities,  # (N,)
        colors=features,  # (N, 1, 3)
        viewmats=w2c,  # (1, 4, 4)
        Ks=K,  # (1, 3, 3)
        width=W,
        height=H,
        near_plane=0.1,
        far_plane=100000.0,
        render_mode="RGB+ED",
        sh_degree=0,
        rasterize_mode="classic",
        distributed=False,
        camera_model="pinhole",
        packed=False,
        backgrounds=bg_color[None, ...],
    )

    rgb = rgbd[0, ..., :3]  # (H, W, 3)
    depth = rgbd[0, ..., 3]  # (H, W)
    alpha = alpha[0, ..., 0]  # (H, W)

    return rgb, alpha, depth


def render_gaussians_scene(
    scene_gs: "Gaussian",
    c2w: torch.Tensor,
    K: torch.Tensor,
    w: int,
    h: int,
    bg_color: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Render a single frame from the Gaussian scene using given camera parameters.

    Parameters
    ----------
    scene_gs : Gaussian
        Gaussian scene object.
    c2w : torch.Tensor
        Camera-to-world transformation matrix, shape (4, 4).
    K : torch.Tensor
        Camera intrinsics matrix, shape (3, 3).
    w : int
        Image width.
    h : int
        Image height.
    bg_color : torch.Tensor, optional
        Background color, shape (3,). Defaults to black.

    Returns
    -------
    tuple
        (rgb, alpha) where:
        - rgb: Rendered image, shape (H, W, 3), in [0, 1] range
        - alpha: Alpha/opacity map, shape (H, W)
    """
    # Ensure tensors are on CUDA
    c2w = c2w.cuda() if not c2w.is_cuda else c2w
    Ks = K.cuda() if not K.is_cuda else K

    if c2w.dim() == 2:
        c2w = c2w.unsqueeze(0)  # [1, 4, 4]

    if Ks.dim() == 2:
        Ks = Ks.unsqueeze(0)  # [1, 3, 3]

    means = scene_gs.get_xyz  # [N, 3]
    rotations = scene_gs.get_rotation  # [N, 4]
    scales = scene_gs.get_scaling  # [N, 3]
    opacity = scene_gs.get_opacity  # [N, 1]
    features = scene_gs.get_features  # [N, 1, 3]
    width = w
    height = h

    # Set background color (default to black if not provided)
    if bg_color is None:
        bg_color = torch.zeros(3, device=c2w.device)
    else:
        bg_color = bg_color.to(c2w.device)

    rgb, alpha, depth = render_gaussian_params(
        means=means,
        quats=rotations,
        scales=scales,
        opacities=opacity,
        features=features,
        c2w=c2w,
        K_matrix=Ks,
        W=width,
        H=height,
        bg_color=bg_color,
    )

    return rgb, alpha


def render_gaussians_to_image(
    scene_gs: "Gaussian",
    K_matrix: np.ndarray,
    W: int,
    H: int,
    bg_color: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Render Gaussian scene to an image using identity camera (camera at origin).

    Parameters
    ----------
    scene_gs : Gaussian
        Gaussian scene object.
    K_matrix : np.ndarray
        Camera intrinsics matrix, shape (3, 3).
    W : int
        Image width.
    H : int
        Image height.
    bg_color : torch.Tensor, optional
        Background color, shape (3,). Defaults to black.

    Returns
    -------
    torch.Tensor
        Rendered image, shape (H, W, 3), in [0, 1] range.

    Examples
    --------
    >>> rendered = render_gaussians_to_image(scene_gs, K_matrix, 640, 480)
    >>> rendered.shape
    torch.Size([480, 640, 3])
    """
    # Use identity camera matrix (camera at origin)
    c2w = torch.eye(4)
    K = torch.from_numpy(K_matrix).float()

    # Default to black background for evaluation
    if bg_color is None:
        bg_color = torch.zeros(3)

    rendered_frame, alpha = render_gaussians_scene(
        scene_gs, c2w=c2w, K=K, w=W, h=H, bg_color=bg_color
    )

    return rendered_frame


def render_and_compare(
    scene_gs: "Gaussian",
    image: np.ndarray,
    K_matrix: np.ndarray,
    W: int,
    H: int,
    output_path: str = "rendered_vs_original.png",
) -> None:
    """
    Render Gaussian scene and save side-by-side comparison with original image.

    Parameters
    ----------
    scene_gs : Gaussian
        Gaussian scene object.
    image : np.ndarray
        Original image for comparison, shape (H, W, 3).
    K_matrix : np.ndarray
        Camera intrinsics matrix, shape (3, 3).
    W : int
        Image width.
    H : int
        Image height.
    output_path : str, optional
        Path to save the comparison image. Default: "rendered_vs_original.png".
    """
    # Use identity camera matrix (camera at origin)
    c2w = torch.eye(4)
    K = torch.from_numpy(K_matrix).float()

    rendered_frame, _ = render_gaussians_scene(scene_gs, c2w=c2w, K=K, w=W, h=H)

    # Create comparison visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    ax1.imshow(image)
    ax1.set_title("Original Image", fontsize=14)
    ax1.axis("off")

    ax2.imshow(rendered_frame.cpu().numpy())
    ax2.set_title("Rendered from Gaussian Splats", fontsize=14)
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved rendering comparison to {output_path}")


def save_comparison_image(
    rendered: torch.Tensor,
    gt_image: torch.Tensor,
    output_path: str,
    frame_index: int,
) -> None:
    """
    Save side-by-side comparison of rendered and ground truth images.

    Creates a figure with three panels: ground truth, rendered, and
    amplified difference.

    Parameters
    ----------
    rendered : torch.Tensor
        Rendered image, shape (H, W, 3), in [0, 1] range.
    gt_image : torch.Tensor
        Ground truth image, shape (H, W, 3), in [0, 1] range.
    output_path : str
        Path to save the comparison image.
    frame_index : int
        Frame index for labeling in the figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Ground truth
    axes[0].imshow(gt_image.numpy())
    axes[0].set_title(f"Ground Truth (Frame {frame_index})")
    axes[0].axis("off")

    # Rendered
    axes[1].imshow(rendered.numpy())
    axes[1].set_title("Rendered from Gaussians")
    axes[1].axis("off")

    # Difference (amplified for visibility)
    diff = torch.abs(rendered - gt_image)
    diff_amplified = torch.clamp(diff * 5, 0, 1)  # Amplify differences
    axes[2].imshow(diff_amplified.numpy())
    axes[2].set_title("Absolute Difference (5x amplified)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "render_gaussian_params",
    "render_gaussians_scene",
    "render_gaussians_to_image",
    "render_and_compare",
    "save_comparison_image",
]

"""
Pose refinement utilities using differentiable rendering.

This module provides functions for refining object poses (rotation, translation,
scale) using gradient-based optimization with differentiable Gaussian rendering.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.transforms import (
    Transform3d,
    matrix_to_quaternion,
    quaternion_invert,
    quaternion_multiply,
    quaternion_to_matrix,
)

from .config import RefinementConfig
from .rendering import render_gaussian_params

if TYPE_CHECKING:
    from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model import (
        Gaussian,
    )


def _get_flow_model() -> Any:
    """
    Lazy-load the optical flow model for correspondence loss.

    Returns
    -------
    RAFT
        Pre-trained optical flow model.

    Notes
    -----
    The flow model is cached in the RefinementConfig to avoid
    reloading on every call.
    """
    # Add sea_raft_core to path
    custom_dir = Path(__file__).resolve().parents[1]
    sea_raft_path = custom_dir / "submodules" / "sea_raft_core"
    if str(sea_raft_path) not in sys.path:
        sys.path.insert(0, str(sea_raft_path))

    from raft import RAFT
    from utils.utils import json_to_args

    ckpt_path = str(sea_raft_path / "checkpoints" / "Tartan-C-T-TSKH-spring540x960-M.pth")
    json_path = str(sea_raft_path / "configs" / "spring-L.json")

    args = json_to_args(json_path)
    model = RAFT(args)
    model.load_ckpt(ckpt_path)
    model.cuda()
    model.eval()

    return model


def _compute_flow_loss(
    rendered_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    mask: torch.Tensor,
    flow_model: Any,
) -> torch.Tensor:
    """
    Compute optical flow magnitude between rendered and GT images (monitoring only).

    The flow from rendered to GT tells us where each rendered pixel should move
    to match the GT. We want this flow to be zero (pixels already aligned).

    NOTE: This is a monitoring metric only - gradients do NOT flow back through
    the flow model due to internal .detach() calls in RAFT's iterative refinement.
    The flow magnitude tracks alignment quality but does not contribute to optimization.

    Parameters
    ----------
    rendered_rgb : torch.Tensor
        Rendered image (H, W, 3) in [0, 1].
    gt_rgb : torch.Tensor
        Ground truth image (H, W, 3) in [0, 1].
    mask : torch.Tensor
        Object mask (H, W), boolean.
    flow_model : RAFT
        Pre-loaded optical flow model.

    Returns
    -------
    torch.Tensor
        Flow magnitude metric (no gradients).
    """
    H, W = rendered_rgb.shape[:2]
    device = rendered_rgb.device

    # Convert to RAFT input format: [B, C, H, W] in [0, 255]
    rendered_t = (rendered_rgb.detach().permute(2, 0, 1).unsqueeze(0) * 255.0).contiguous()
    gt_t = (gt_rgb.detach().permute(2, 0, 1).unsqueeze(0) * 255.0).contiguous()

    # Compute flow from rendered to GT (no gradients - monitoring only)
    with torch.no_grad():
        flow, _ = flow_model.calc_flow(rendered_t, gt_t)

    # Flow shape: [1, 2, H, W] - (u, v) displacement
    flow = flow.squeeze(0).permute(1, 2, 0)  # [H, W, 2]

    # Compute magnitude of flow in masked region
    # Ideally, if render matches GT, flow should be zero
    flow_magnitude = torch.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2 + 1e-8)

    # Mean flow magnitude in masked region (monitoring metric)
    masked_flow_mag = flow_magnitude[mask]
    if masked_flow_mag.numel() > 0:
        flow_metric = masked_flow_mag.mean()
    else:
        flow_metric = torch.tensor(0.0, device=device)

    return flow_metric


def _compute_center_of_mass_loss(
    alpha: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute center-of-mass alignment loss between rendered alpha and GT mask.

    This loss directly guides translation optimization by penalizing the
    distance between the centroid of the rendered silhouette and the GT mask.
    It provides useful gradients even when there is zero overlap between
    the rendered object and the target mask.

    Parameters
    ----------
    alpha : torch.Tensor
        Rendered alpha/opacity map, shape (H, W), values in [0, 1].
    mask : torch.Tensor
        Ground truth binary mask, shape (H, W), boolean or float.

    Returns
    -------
    torch.Tensor
        Scalar loss representing squared distance between centroids.

    Notes
    -----
    The loss is normalized by image diagonal to be resolution-independent.
    """
    H, W = alpha.shape
    device = alpha.device

    # Create coordinate grids
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )

    # Compute rendered center of mass
    alpha_sum = alpha.sum() + 1e-6
    render_cx = (alpha * x_coords).sum() / alpha_sum
    render_cy = (alpha * y_coords).sum() / alpha_sum

    # Compute GT mask center of mass
    mask_float = mask.float()
    mask_sum = mask_float.sum() + 1e-6
    gt_cx = (mask_float * x_coords).sum() / mask_sum
    gt_cy = (mask_float * y_coords).sum() / mask_sum

    # Squared distance between centers, normalized by image diagonal
    diagonal = (H**2 + W**2) ** 0.5
    com_loss = ((render_cx - gt_cx) ** 2 + (render_cy - gt_cy) ** 2) / (diagonal**2)

    return com_loss


def _compute_signed_distance_transform(
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute signed distance transform of a binary mask.

    The SDT is positive outside the mask (distance to nearest mask pixel)
    and negative inside the mask (distance to nearest non-mask pixel).
    This creates a smooth field that guides optimization toward the mask.

    Parameters
    ----------
    mask : torch.Tensor
        Binary mask, shape (H, W), boolean or float.

    Returns
    -------
    torch.Tensor
        Signed distance transform, shape (H, W). Positive outside mask,
        negative inside mask.

    Notes
    -----
    This function uses scipy.ndimage.distance_transform_edt and is not
    differentiable. The SDT should be precomputed once per frame.
    """
    from scipy.ndimage import distance_transform_edt

    mask_np = mask.detach().cpu().numpy().astype(bool)

    # Distance from outside points to nearest inside point
    dist_outside = distance_transform_edt(~mask_np)

    # Distance from inside points to nearest outside point
    dist_inside = distance_transform_edt(mask_np)

    # Signed: positive outside, negative inside
    sdt = dist_outside - dist_inside

    return torch.from_numpy(sdt).float().to(mask.device)


def _compute_sdt_loss(
    alpha: torch.Tensor,
    sdt: torch.Tensor,
) -> torch.Tensor:
    """
    Compute signed distance transform loss for silhouette alignment.

    This loss penalizes rendered alpha weighted by the signed distance to
    the GT mask boundary. Pixels rendered outside the mask (positive SDT)
    contribute positive loss, while pixels inside (negative SDT) contribute
    negative loss (reward). This guides the optimization to move rendered
    content toward and inside the mask.

    Parameters
    ----------
    alpha : torch.Tensor
        Rendered alpha/opacity map, shape (H, W), values in [0, 1].
    sdt : torch.Tensor
        Precomputed signed distance transform of GT mask, shape (H, W).

    Returns
    -------
    torch.Tensor
        Scalar loss. Lower values indicate better alignment.

    Notes
    -----
    The loss is normalized by total alpha and image diagonal for stability.
    """
    H, W = alpha.shape

    # Normalize SDT by image diagonal for resolution independence
    diagonal = (H**2 + W**2) ** 0.5
    sdt_normalized = sdt / diagonal

    # Weighted mean: alpha-weighted average of SDT values
    # Positive outside mask, negative inside -> want to minimize
    alpha_sum = alpha.sum() + 1e-6
    sdt_loss = (alpha * sdt_normalized).sum() / alpha_sum

    return sdt_loss


def _compute_soft_iou_loss(
    alpha: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute soft intersection-over-union loss for silhouette matching.

    Soft IoU provides a differentiable approximation to the IoU metric,
    treating alpha values as soft membership. This loss is effective for
    fine-grained shape matching once the rendered object is roughly
    aligned with the target mask.

    Parameters
    ----------
    alpha : torch.Tensor
        Rendered alpha/opacity map, shape (H, W), values in [0, 1].
    mask : torch.Tensor
        Ground truth binary mask, shape (H, W), boolean or float.

    Returns
    -------
    torch.Tensor
        Scalar loss in [0, 1]. 0 = perfect overlap, 1 = no overlap.

    Notes
    -----
    Unlike BCE, IoU considers global shape overlap rather than per-pixel
    agreement, making it more robust to small misalignments.
    """
    mask_float = mask.float()

    # Soft intersection: element-wise minimum (or product for soft version)
    intersection = (alpha * mask_float).sum()

    # Soft union: sum of both minus intersection
    union = alpha.sum() + mask_float.sum() - intersection + 1e-6

    # IoU loss: 1 - IoU
    iou = intersection / union
    iou_loss = 1.0 - iou

    return iou_loss


def _compute_ssim_loss(
    rgb: torch.Tensor,
    gt_masked: torch.Tensor,
) -> torch.Tensor:
    """
    Compute SSIM (Structural Similarity) loss between rendered and GT images.

    SSIM captures perceptual similarity by comparing luminance, contrast, and
    structure. It's more robust to small misalignments than pixel-wise losses
    and provides better gradients for pose optimization.

    Parameters
    ----------
    rgb : torch.Tensor
        Rendered RGB image, shape (H, W, 3), values in [0, 1].
    gt_masked : torch.Tensor
        Ground truth RGB image with background masked to black,
        shape (H, W, 3), values in [0, 1].

    Returns
    -------
    torch.Tensor
        Scalar SSIM loss in [0, 1]. 0 = identical, 1 = completely different.

    Notes
    -----
    Uses pytorch_msssim for efficient GPU computation.
    """
    from pytorch_msssim import ssim

    # Convert from (H, W, C) to (B, C, H, W)
    rgb_bchw = rgb.permute(2, 0, 1).unsqueeze(0)
    gt_bchw = gt_masked.permute(2, 0, 1).unsqueeze(0)

    # Compute SSIM (returns similarity, we want loss)
    ssim_value = ssim(rgb_bchw, gt_bchw, data_range=1.0, size_average=True)
    ssim_loss = 1.0 - ssim_value

    return ssim_loss


def _compute_multiscale_rgb_loss(
    rgb: torch.Tensor,
    gt_image: torch.Tensor,
    mask: torch.Tensor,
    config: "RefinementConfig",
) -> torch.Tensor:
    """
    Compute multi-scale RGB loss for robust pose optimization.

    Multi-scale loss helps escape local minima by computing the loss at
    multiple resolutions. Lower resolutions capture global alignment errors
    while higher resolutions refine fine details.

    The GT image background is masked to black while the loss is computed on
    the full image. This penalizes false positives (rendering outside mask)
    while not penalizing false negatives in the background region.

    Parameters
    ----------
    rgb : torch.Tensor
        Rendered RGB image, shape (H, W, 3), values in [0, 1].
    gt_image : torch.Tensor
        Ground truth RGB image, shape (H, W, 3), values in [0, 1].
    mask : torch.Tensor
        Object mask, shape (H, W), boolean.
    config : RefinementConfig
        Configuration with rgb_loss_type, rgb_multiscale_scales,
        and rgb_multiscale_weights.

    Returns
    -------
    torch.Tensor
        Scalar RGB loss averaged across all scales.

    Notes
    -----
    Uses L1 or L2 loss based on config.rgb_loss_type.
    Downsampling uses area interpolation for anti-aliasing.
    """
    import torch.nn.functional as F

    device = rgb.device
    total_loss = torch.tensor(0.0, device=device)

    # Prepare tensors for interpolation: (H, W, C) -> (1, C, H, W)
    rgb_nchw = rgb.permute(2, 0, 1).unsqueeze(0)
    gt_nchw = gt_image.permute(2, 0, 1).unsqueeze(0)
    mask_nchw = mask.float().unsqueeze(0).unsqueeze(0)

    for scale, weight in zip(config.rgb_multiscale_scales, config.rgb_multiscale_weights):
        if scale == 1.0:
            # Full resolution - no interpolation needed
            rgb_scaled = rgb
            gt_scaled = gt_image
            mask_scaled = mask.float()
        else:
            # Downsample using area interpolation (anti-aliased)
            rgb_scaled = F.interpolate(
                rgb_nchw, scale_factor=scale, mode="area"
            ).squeeze(0).permute(1, 2, 0)
            gt_scaled = F.interpolate(
                gt_nchw, scale_factor=scale, mode="area"
            ).squeeze(0).permute(1, 2, 0)
            # Use area for mask then threshold
            mask_scaled = F.interpolate(
                mask_nchw, scale_factor=scale, mode="area"
            ).squeeze(0).squeeze(0)
            mask_scaled = (mask_scaled > 0.5).float()

        # Mask GT background to black, compare full image
        # This penalizes false positives (rendering outside mask)
        gt_masked = gt_scaled * mask_scaled.unsqueeze(-1)

        if config.rgb_loss_type == "l1":
            diff = (rgb_scaled - gt_masked).abs()
        else:  # l2
            diff = (rgb_scaled - gt_masked) ** 2

        # Mean over all pixels (full image comparison)
        n_pixels = rgb_scaled.shape[0] * rgb_scaled.shape[1]
        scale_loss = diff.sum() / (n_pixels * 3)

        total_loss = total_loss + weight * scale_loss

    return total_loss


def apply_pose_to_gaussian(
    canonical_gs: "Gaussian",
    rotation: torch.Tensor,
    translation: torch.Tensor,
    scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply a pose transformation to Gaussian positions and rotations.

    Parameters
    ----------
    canonical_gs : Gaussian
        The canonical Gaussian object (frozen, not modified).
    rotation : torch.Tensor
        Quaternion rotation (1, 4) or (4,).
    translation : torch.Tensor
        Translation (1, 3) or (3,).
    scale : torch.Tensor
        Scale factor (1, 3) or (3,) or (1,) or scalar.

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
    # PyTorch3D convention: quaternion_to_matrix gives R such that points are transformed as p @ R
    R = quaternion_to_matrix(rotation.unsqueeze(0)).squeeze(0)  # (3, 3)

    # Transform positions following compose_transform convention:
    # tfm = Scale(scale).compose(Rotate(R)).compose(Translate(trans))
    # transformed = points * scale @ R + trans
    scaled_xyz = xyz_local * scale
    transformed_xyz = torch.mm(scaled_xyz, R) + translation  # Note: @ R, not @ R.T

    # Transform rotations: rot_world = quaternion_multiply(rotation_inv, rot_local)
    # Note: Using inverse because of the convention in make_scene
    rotation_inv = quaternion_invert(rotation.unsqueeze(0)).squeeze(0)
    transformed_rot = quaternion_multiply(
        rotation_inv.unsqueeze(0).expand(rot_local.shape[0], -1), rot_local
    )

    # Transform scales
    transformed_scales = scales_local * scale

    return transformed_xyz, transformed_rot, transformed_scales, opacities, features


def _render_frame_with_pose(
    canonical_gs: "Gaussian",
    rotation: torch.Tensor,
    translation: torch.Tensor,
    scale: torch.Tensor,
    K_matrix: np.ndarray,
    W: int,
    H: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Render a single frame with given pose parameters.

    Parameters
    ----------
    canonical_gs : Gaussian
        The canonical Gaussian object.
    rotation : torch.Tensor
        Quaternion rotation.
    translation : torch.Tensor
        Translation vector.
    scale : torch.Tensor
        Scale factor.
    K_matrix : np.ndarray
        Camera intrinsics.
    W : int
        Image width.
    H : int
        Image height.
    device : torch.device
        Device to use.

    Returns
    -------
    tuple
        (rgb, alpha, depth) where rgb is (H, W, 3) and alpha is (H, W), depth is (H, W).
    """
    # Apply pose to canonical Gaussian
    transformed_xyz, transformed_rot, transformed_scales, opacities, features = (
        apply_pose_to_gaussian(
            canonical_gs=canonical_gs, rotation=rotation, translation=translation, scale=scale
        )
    )

    # Transform from PyTorch3D convention to R3 convention
    r3_to_p3d_R, r3_to_p3d_T = look_at_view_transform(
        eye=np.array([[0, 0, -1]]),
        at=np.array([[0, 0, 0]]),
        up=np.array([[0, -1, 0]]),
        device=device,
    )

    # Inverse transform (PyTorch3D -> R3)
    p3d_to_r3_R = r3_to_p3d_R.transpose(1, 2)

    # Transform positions
    camera_convention_transform = Transform3d(device=device).rotate(p3d_to_r3_R)
    transformed_xyz = camera_convention_transform.transform_points(transformed_xyz)

    # Transform rotations (quaternions)
    p3d_to_r3_quat = matrix_to_quaternion(p3d_to_r3_R)
    transformed_rot = quaternion_multiply(
        p3d_to_r3_quat.expand(transformed_rot.shape[0], -1), transformed_rot
    )

    # Handle opacities
    opacities_flat = opacities.squeeze(-1) if opacities.dim() == 2 else opacities

    # C2W
    c2w = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)

    # Background color
    bg_color = torch.zeros(3, device=device)

    # Render
    rgb, alpha, depth = render_gaussian_params(
        transformed_xyz,
        transformed_rot,
        transformed_scales,
        opacities_flat,
        features,
        c2w,
        K_matrix,
        W,
        H,
        bg_color=bg_color,
    )

    return rgb, alpha, depth


def _compute_frame_loss(
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    gt_image: torch.Tensor,
    mask: torch.Tensor,
    config: RefinementConfig,
    flow_model: Optional[Any] = None,
    sdt: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute loss for a single frame.

    The loss consists of:
    - RGB loss: L1 or L2 between rendered and GT image, optionally multi-scale
    - SSIM loss: Structural similarity loss (optional, weight > 0 to enable)
    - Silhouette loss: Combined loss for pose-aware silhouette alignment
      - Center-of-mass: Aligns centroids (guides translation)
      - SDT: Signed distance transform (guides all params, escapes local minima)
      - IoU: Soft intersection-over-union (fine shape matching)

    GT background is always masked to black for RGB/SSIM loss computation.
    The loss is computed on the full image to penalize false positives.

    Parameters
    ----------
    rgb : torch.Tensor
        Rendered RGB image (H, W, 3).
    alpha : torch.Tensor
        Rendered alpha/opacity (H, W).
    gt_image : torch.Tensor
        Ground truth image (H, W, 3).
    mask : torch.Tensor
        Object mask (H, W), boolean.
    config : RefinementConfig
        Refinement configuration with loss weights and RGB loss settings.
    flow_model : optional
        Pre-loaded flow model for correspondence loss (currently unused).
    sdt : torch.Tensor, optional
        Precomputed signed distance transform of the mask, shape (H, W).
        If None and silhouette_sdt_weight > 0, it will be computed here.

    Returns
    -------
    dict
        Dictionary with loss tensors:
        - 'rgb_loss': Scalar RGB loss (multi-scale if enabled)
        - 'ssim_loss': Scalar SSIM loss
        - 'silhouette_loss': Combined silhouette loss scalar
        - 'silhouette_com': Center-of-mass loss component
        - 'silhouette_sdt': SDT loss component
        - 'silhouette_iou': IoU loss component
    """
    device = rgb.device
    H, W = rgb.shape[:2]

    # Mask GT background to black (always applied)
    gt_masked = gt_image * mask.float().unsqueeze(-1)

    # RGB loss: L1 or L2, optionally multi-scale
    if config.rgb_multiscale:
        # Multi-scale loss for robust optimization
        rgb_loss_value = _compute_multiscale_rgb_loss(rgb, gt_image, mask, config)
    else:
        # Single-scale loss with masked GT
        if config.rgb_loss_type == "l1":
            diff = (rgb - gt_masked).abs()
        else:  # l2
            diff = (rgb - gt_masked) ** 2
        # Mean over all pixels (full image comparison)
        rgb_loss_value = diff.sum() / (H * W * 3)

    # SSIM loss (optional, enabled when weight > 0)
    ssim_loss_value = torch.tensor(0.0, device=device)
    if config.rgb_ssim_weight > 0:
        ssim_loss_value = _compute_ssim_loss(rgb, gt_masked)

    # Silhouette loss components
    alpha_squeezed = alpha.squeeze(0) if alpha.dim() == 3 else alpha

    # Initialize loss components
    com_loss_value = torch.tensor(0.0, device=device)
    sdt_loss_value = torch.tensor(0.0, device=device)
    iou_loss_value = torch.tensor(0.0, device=device)
    
    # TODO: placeholder for flow loss (currently unused in optimization)
    flow_loss_value = torch.tensor(0.0, device=device)

    if config.silhouette_weight > 0:
        # Center-of-mass loss: directly guides translation
        if config.silhouette_com_weight > 0:
            com_loss_value = _compute_center_of_mass_loss(alpha_squeezed, mask)

        # Signed distance transform loss: guides all params, escapes local minima
        if config.silhouette_sdt_weight > 0:
            if sdt is None:
                sdt = _compute_signed_distance_transform(mask)
            sdt_loss_value = _compute_sdt_loss(alpha_squeezed, sdt)

        # Soft IoU loss: fine-grained shape matching
        if config.silhouette_iou_weight > 0:
            iou_loss_value = _compute_soft_iou_loss(alpha_squeezed, mask)
    
    return {
        "rgb_loss": rgb_loss_value,  # Scalar RGB loss
        "ssim_loss": ssim_loss_value,  # Scalar SSIM loss (0 if disabled)
        "silhouette_com": com_loss_value,
        "silhouette_sdt": sdt_loss_value,
        "silhouette_iou": iou_loss_value,
        "flow_loss": flow_loss_value,
    }


def _prepare_frame_data_for_refinement(
    args: Any,
    paths: Dict[str, Any],
    frame_idx: int,
    obj_idx: int,
    inference: Any,
) -> Optional[Dict[str, Any]]:
    """
    Load and prepare frame data for refinement.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    paths : dict
        Dataset paths.
    frame_idx : int
        Frame index.
    obj_idx : int
        Object index.
    inference : Inference
        Inference pipeline.

    Returns
    -------
    dict or None
        Dictionary with 'image', 'mask', 'K_matrix', 'H', 'W' or None if data unavailable.
    """
    from .depth import load_and_process_depth
    from .io_utils import load_image, load_masks

    image_path = os.path.join(paths["frames_path"], paths["image_names"][frame_idx])
    mask_path = os.path.join(paths["masks_path"], paths["mask_names"][frame_idx])

    image = load_image(image_path)
    image = image[..., :3]
    H, W, _ = image.shape

    masks = load_masks(mask_path)
    if args.object_index is not None:
        masks = [masks[args.object_index]]

    # Get the mask for this object
    if obj_idx >= len(masks):
        return None
    mask = masks[obj_idx]

    # Load depth and compute K_matrix
    depth_names_for_frame = []
    if paths["dataset_type"] == "kubric4d" and paths["depth_names"]:
        depth_names_for_frame = [paths["depth_names"][frame_idx]]

    pointmap, K_matrix, valid_mask = load_and_process_depth(
        paths["frames_path"],
        depth_names_for_frame,
        W,
        H,
        use_moge=args.use_moge,
        inference=inference,
        image=image,
    )

    return {
        "image": image,
        "mask": mask,
        "K_matrix": K_matrix,
        "H": H,
        "W": W,
    }


def refine_pose_for_frame(
    canonical_gs: "Gaussian",
    initial_rotation: torch.Tensor,
    initial_translation: torch.Tensor,
    initial_scale: torch.Tensor,
    gt_image: torch.Tensor,
    mask: torch.Tensor | np.ndarray,
    K_matrix: np.ndarray,
    config: RefinementConfig,
) -> Dict[str, Any]:
    """
    Refine pose parameters using differentiable Gaussian rendering.

    Parameters
    ----------
    canonical_gs : Gaussian
        The canonical Gaussian object (frozen).
    initial_rotation : torch.Tensor
        Initial quaternion rotation (1, 4).
    initial_translation : torch.Tensor
        Initial translation (1, 3).
    initial_scale : torch.Tensor
        Initial scale (1, 3) or (1,).
    gt_image : torch.Tensor
        Ground truth image (H, W, 3) in [0, 1].
    mask : np.ndarray or torch.Tensor
        Object mask (H, W), boolean.
    K_matrix : np.ndarray
        Camera intrinsics (3, 3).
    config : RefinementConfig
        Configuration for refinement hyperparameters.

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

    # Precompute signed distance transform for silhouette loss (expensive, do once)
    sdt = None
    if config.silhouette_weight > 0 and config.silhouette_sdt_weight > 0:
        sdt = _compute_signed_distance_transform(mask)

    # Load flow model if flow loss is enabled
    flow_model = None
    if config.use_flow:
        if config.flow_model is not None:
            flow_model = config.flow_model
        else:
            print("      Loading flow model for correspondence loss...")
            flow_model = _get_flow_model()
            config.flow_model = flow_model  # Cache for future use

    # Initialize optimizable parameters
    opt_rotation = initial_rotation.clone().detach().requires_grad_(True)
    opt_translation = initial_translation.clone().detach().requires_grad_(True)

    # For scale, we optimize a single scalar to ensure uniform scaling
    initial_scale_flat = initial_scale.view(-1)
    if initial_scale_flat.shape[0] == 3:
        initial_scale_scalar = initial_scale_flat.mean().view(1)
    else:
        initial_scale_scalar = initial_scale_flat[:1]

    # Set up scale parameter based on refinement mode
    if config.refine_scale == "perframe":
        opt_scale_scalar = initial_scale_scalar.clone().detach().requires_grad_(True)
    else:
        opt_scale_scalar = initial_scale_scalar.clone().detach().requires_grad_(False)

    # Create optimizer with different learning rates
    param_groups = [
        {"params": [opt_rotation], "lr": config.lr_rotation},
        {"params": [opt_translation], "lr": config.lr_translation},
    ]
    if config.refine_scale == "perframe":
        param_groups.append({"params": [opt_scale_scalar], "lr": config.lr_scale})

    optimizer = torch.optim.Adam(param_groups)

    # Background color (black for evaluation)
    bg_color = torch.zeros(3, device=device)

    best_loss = float("inf")
    best_params = {}
    best_iteration = 0
    loss_history = []

    for iteration in range(config.num_iterations):
        optimizer.zero_grad()

        # Render frame with current pose
        rgb, alpha, depth = _render_frame_with_pose(
            canonical_gs, opt_rotation, opt_translation, opt_scale_scalar, K_matrix, W, H, device
        )

        # Compute RGB, SSIM, and silhouette loss
        losses_dict = _compute_frame_loss(rgb, alpha, gt_image, mask, config, flow_model, sdt)
        
        # RGB and SSIM losses are separate scalars for logging
        rgb_loss_value = losses_dict["rgb_loss"]  # Excludes SSIM
        ssim_loss_value = losses_dict["ssim_loss"]  #  (0 if disabled)
        flow_loss_value = losses_dict["flow_loss"]  #  (0 if disabled)
        
        # Extract individual silhouette components for logging
        silh_com = losses_dict["silhouette_com"]
        silh_sdt = losses_dict["silhouette_sdt"]
        silh_iou = losses_dict["silhouette_iou"]
        
        # Combined silhouette loss
        silhouette_loss_value = (
            config.silhouette_com_weight * silh_com
            + config.silhouette_sdt_weight * silh_sdt
            + config.silhouette_iou_weight * silh_iou
        )

        # Combine RGB + weighted SSIM + weighted silhouette + weighted flow losses
        weighted_ssim_loss_value = config.rgb_ssim_weight * ssim_loss_value
        weighted_silhouette_loss_value = config.silhouette_weight * silhouette_loss_value
        weighted_flow_loss = config.flow_weight * flow_loss_value
        base_loss = rgb_loss_value + weighted_ssim_loss_value + weighted_silhouette_loss_value + weighted_flow_loss

        # Debug: Check if rendered object overlaps with mask
        if iteration == 0 and config.verbose:
            with torch.no_grad():
                rendered_in_mask = rgb[mask]
                bg_in_mask = bg_color.expand(rendered_in_mask.shape[0], -1)
                non_bg_mask = (rendered_in_mask - bg_in_mask).abs().sum(dim=-1) > 0.01
                print(
                    f"      [Overlap check] Mask pixels: {mask.sum().item()}, "
                    f"Non-background in mask: {non_bg_mask.sum().item()} "
                    f"({100*non_bg_mask.float().mean().item():.1f}%)"
                )

        # DEBUG: save visuals of the losses (optional, disabled by default)
        # Uncomment to enable debug visualization
        # if iteration % 10 == 0:
        #     alpha_np = alpha.squeeze().detach().cpu().numpy()
        #     mask_np = mask.detach().cpu().numpy().astype(float)
        #     fig = plt.figure(figsize=(16, 4))
        #     ax1 = fig.add_subplot(1, 4, 1)
        #     ax1.imshow(rgb_loss_pixelwise.sum(dim=-1).detach().cpu().numpy(), cmap="hot")
        #     ax1.set_title(f"RGB Loss ({rgb_loss_value.item():.4f})")
        #     ax2 = fig.add_subplot(1, 4, 2)
        #     ax2.imshow(alpha_np, cmap="gray", vmin=0, vmax=1)
        #     ax2.set_title("Rendered Alpha")
        #     ax3 = fig.add_subplot(1, 4, 3)
        #     ax3.imshow(mask_np, cmap="gray")
        #     ax3.set_title("GT Mask")
        #     ax4 = fig.add_subplot(1, 4, 4)
        #     # Overlay: green=mask, red=alpha, yellow=overlap
        #     overlay = np.stack([alpha_np, mask_np, np.zeros_like(alpha_np)], axis=-1)
        #     ax4.imshow(overlay)
        #     ax4.set_title(f"IoU={1-silh_iou.item():.3f}")
        #     plt.suptitle(f"Iter {iteration}: CoM={silh_com.item():.4f}, SDT={silh_sdt.item():.4f}")
        #     plt.tight_layout()
        #     plt.savefig(f"debug_frame_refinement_iter{iteration}.png")
        #     plt.close(fig)

        # Optional: add regularization to prevent large deviations from initial pose
        if config.use_regularization:
            opt_rotation_normalized = opt_rotation / opt_rotation.norm()
            reg_rot = (
                (opt_rotation_normalized - initial_rotation / initial_rotation.norm()) ** 2
            ).sum()
            reg_trans = ((opt_translation - initial_translation) ** 2).sum()
            if config.refine_scale == "perframe":
                reg_scale = ((opt_scale_scalar - initial_scale_scalar) ** 2).sum()
                weighted_reg_loss = config.regularization_weight * (reg_rot + reg_trans + reg_scale)
            else:
                weighted_reg_loss = config.regularization_weight * (reg_rot + reg_trans)
        else:
            weighted_reg_loss = torch.tensor(0.0, device=device)

        # Total loss
        loss = base_loss + weighted_reg_loss

        # Record loss for this iteration
        loss_history.append(
            {
                "total": loss.item(),
                "rgb": rgb_loss_value.item(),
                "ssim": ssim_loss_value.item(),
                "silhouette": silhouette_loss_value.item(),
                "silhouette_com": silh_com.item(),
                "silhouette_sdt": silh_sdt.item(),
                "silhouette_iou": silh_iou.item(),
                "flow": flow_loss_value.item(),
                "regularization": weighted_reg_loss.item(),
            }
        )

        # Backprop
        loss.backward()

        # Debug: Check gradients on first iteration
        if iteration == 0 and config.verbose:
            rot_grad = opt_rotation.grad
            trans_grad = opt_translation.grad
            print(
                f"      [Gradient check] rotation grad: "
                f"{rot_grad.abs().max().item() if rot_grad is not None else 'None':.6e}"
            )
            print(
                f"      [Gradient check] translation grad: "
                f"{trans_grad.abs().max().item() if trans_grad is not None else 'None':.6e}"
            )
            if config.refine_scale == "perframe":
                scale_grad = opt_scale_scalar.grad
                print(
                    f"      [Gradient check] scale grad: "
                    f"{scale_grad.abs().max().item() if scale_grad is not None else 'None':.6e}"
                )
            else:
                print("      [Gradient check] scale grad: N/A (scale refinement disabled)")
            print(f"      [Gradient check] rgb requires_grad: {rgb.requires_grad}")
            print(f"      [Gradient check] loss requires_grad: {loss.requires_grad}")

        optimizer.step()

        # Track best
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_iteration = iteration
            # Store scale as uniform 3D tensor (required by make_scene)
            scale_3d = opt_scale_scalar.clone().detach().expand(3).reshape(1, 3)
            best_params = {
                "rotation": opt_rotation.clone().detach(),
                "translation": opt_translation.clone().detach(),
                "scale": scale_3d,
            }

        if config.verbose and (
            iteration % config.log_interval == 0 or iteration == config.num_iterations - 1
        ):
            silh_details = ""
            if config.silhouette_weight > 0:
                silh_details = (
                    f" (com={silh_com.item():.4f}, sdt={silh_sdt.item():.4f}, "
                    f"iou={silh_iou.item():.4f})"
                )
            print(
                f"      Iteration {iteration}: total={loss.item():.6f} "
                f"rgb={rgb_loss_value.item():.6f} "
                f"ssim={ssim_loss_value.item():.6f} "
                f"silh={silhouette_loss_value.item():.6f}{silh_details} "
                f"reg={weighted_reg_loss.item():.8f}"
            )

    if config.verbose:
        print(f"      Best loss: {best_loss:.6f} (iteration {best_iteration})")

    # Normalize the rotation in best_params
    best_params["rotation"] = best_params["rotation"] / best_params["rotation"].norm()

    # Add loss history and best iteration to the result
    best_params["loss_history"] = loss_history
    best_params["best_iteration"] = best_iteration

    return best_params


def refine_poses_global_scale(
    canonical_gs: "Gaussian",
    tokens_list: List[Tuple[int, Dict[str, Any]]],
    obj_idx: int,
    args: Any,
    paths: Dict[str, Any],
    inference: Any,
    config: RefinementConfig,
) -> List[Tuple[int, Dict[str, Any]]]:
    """
    Refine poses for a single object with global scale optimization.

    This function optimizes:
    - Per-frame rotation and translation
    - A single global scale shared across all frames

    Parameters
    ----------
    canonical_gs : Gaussian
        The canonical Gaussian for this object.
    tokens_list : list
        List of (frame_index, decoder_input) tuples for this object.
    obj_idx : int
        Object index (for logging).
    args : argparse.Namespace
        Command line arguments.
    paths : dict
        Dataset paths.
    inference : Inference
        Inference pipeline.
    config : RefinementConfig
        Configuration for refinement hyperparameters.

    Returns
    -------
    list
        Refined list of (frame_index, decoder_input) tuples with updated poses.
    """
    device = torch.device("cuda")
    num_frames = len(tokens_list)

    if num_frames == 0:
        return []

    print(f"\n    Refining object {obj_idx} with global scale ({num_frames} frames)")

    # Load all frame data upfront
    frame_data = {}
    valid_frame_indices = []

    for frame_idx, decoder_input in tokens_list:
        data = _prepare_frame_data_for_refinement(args, paths, frame_idx, obj_idx, inference)
        if data is not None:
            mask_tensor = torch.from_numpy(data["mask"]).bool().cuda()
            # Precompute SDT for silhouette loss (expensive, do once per frame)
            sdt = None
            if config.silhouette_weight > 0 and config.silhouette_sdt_weight > 0:
                sdt = _compute_signed_distance_transform(mask_tensor)
            frame_data[frame_idx] = {
                "gt_image": torch.from_numpy(data["image"]).float().cuda() / 255.0,
                "mask": mask_tensor,
                "sdt": sdt,
                "K_matrix": data["K_matrix"],
                "H": data["H"],
                "W": data["W"],
                "decoder_input": decoder_input,
            }
            valid_frame_indices.append(frame_idx)
        else:
            print(f"      Warning: Could not load data for frame {frame_idx}, skipping")

    if len(valid_frame_indices) == 0:
        print(f"      No valid frames for object {obj_idx}")
        return tokens_list

    # Initialize optimizable parameters
    opt_rotations = {}
    opt_translations = {}
    initial_rotations = {}
    initial_translations = {}

    # Use first frame's scale as the reference for global scale
    first_frame_idx = valid_frame_indices[0]
    first_decoder_input = frame_data[first_frame_idx]["decoder_input"]
    initial_scale_flat = first_decoder_input["scale"].view(-1)
    if initial_scale_flat.shape[0] == 3:
        initial_scale_scalar = initial_scale_flat.mean().view(1)
    else:
        initial_scale_scalar = initial_scale_flat[:1]

    # Global scale parameter
    opt_global_scale = initial_scale_scalar.clone().detach().to(device).requires_grad_(True)

    print(f"      Initial global scale (from frame {first_frame_idx}): {opt_global_scale.item():.6f}")

    # Initialize per-frame parameters
    for frame_idx in valid_frame_indices:
        decoder_input = frame_data[frame_idx]["decoder_input"]

        initial_rotations[frame_idx] = decoder_input["rotation"].clone().detach().to(device)
        initial_translations[frame_idx] = decoder_input["translation"].clone().detach().to(device)

        opt_rotations[frame_idx] = (
            decoder_input["rotation"].clone().detach().to(device).requires_grad_(True)
        )
        opt_translations[frame_idx] = (
            decoder_input["translation"].clone().detach().to(device).requires_grad_(True)
        )

    # Create optimizer with all parameters
    param_groups = [
        {"params": [opt_global_scale], "lr": config.lr_scale},
    ]
    for frame_idx in valid_frame_indices:
        param_groups.append({"params": [opt_rotations[frame_idx]], "lr": config.lr_rotation})
        param_groups.append({"params": [opt_translations[frame_idx]], "lr": config.lr_translation})

    optimizer = torch.optim.Adam(param_groups)

    # Determine batch size
    batch_size = config.batch_size if config.batch_size > 0 else len(valid_frame_indices)
    batch_size = min(batch_size, len(valid_frame_indices))

    print(f"      Batch size: {batch_size} (out of {len(valid_frame_indices)} frames)")

    # Track best parameters
    best_loss = float("inf")
    best_params = {}
    best_iteration = 0
    loss_history = []

    for iteration in range(config.num_iterations):
        optimizer.zero_grad()

        # Sample batch of frames
        if batch_size >= len(valid_frame_indices):
            batch_frame_indices = valid_frame_indices
        else:
            batch_frame_indices = list(
                np.random.choice(valid_frame_indices, size=batch_size, replace=False)
            )

        # Accumulate loss over batch
        total_rgb_loss = 0.0
        total_ssim_loss = 0.0
        total_silhouette_loss = 0.0
        total_reg_loss = 0.0

        for frame_idx in batch_frame_indices:
            data = frame_data[frame_idx]

            # Render frame with current parameters
            rgb, alpha, depth = _render_frame_with_pose(
                canonical_gs,
                opt_rotations[frame_idx],
                opt_translations[frame_idx],
                opt_global_scale,
                data["K_matrix"],
                data["W"],
                data["H"],
                device,
            )

            # Compute frame loss (pass precomputed SDT)
            losses_dict = _compute_frame_loss(
                rgb, alpha, data["gt_image"], data["mask"], config,
                flow_model=None, sdt=data["sdt"]
            )

            # RGB and SSIM losses are separate
            rgb_loss_value = losses_dict["rgb_loss"]
            ssim_loss_value = losses_dict["ssim_loss"]
            
            # Extract individual silhouette components for logging
            silh_com = losses_dict["silhouette_com"]
            silh_sdt = losses_dict["silhouette_sdt"]
            silh_iou = losses_dict["silhouette_iou"]
            
            # Combined silhouette loss
            silhouette_loss_value = (
                config.silhouette_com_weight * silh_com
                + config.silhouette_sdt_weight * silh_sdt
                + config.silhouette_iou_weight * silh_iou
            )
            
            # Combine RGB + weighted SSIM + weighted silhouette + weighted flow losses
            weighted_ssim_loss_value = config.rgb_ssim_weight * ssim_loss_value
            weighted_silhouette_loss_value = config.silhouette_weight * silhouette_loss_value

            # Track separately for logging, but combine for optimization
            total_rgb_loss += rgb_loss_value
            total_ssim_loss += weighted_ssim_loss_value
            # Silhouette loss is also a scalar
            total_silhouette_loss += weighted_silhouette_loss_value

            # Per-frame regularization (optional)
            if config.use_regularization:
                opt_rot_normalized = opt_rotations[frame_idx] / opt_rotations[frame_idx].norm()
                init_rot_normalized = (
                    initial_rotations[frame_idx] / initial_rotations[frame_idx].norm()
                )
                reg_rot = ((opt_rot_normalized - init_rot_normalized) ** 2).sum()
                reg_trans = ((opt_translations[frame_idx] - initial_translations[frame_idx]) ** 2).sum()
                total_reg_loss += config.regularization_weight * (reg_rot + reg_trans)

        # Global scale regularization (optional)
        if config.use_regularization:
            scale_reg = ((opt_global_scale - initial_scale_scalar.to(device)) ** 2).sum()
            total_reg_loss += config.regularization_weight * scale_reg

        # Average losses over batch
        batch_rgb_loss = total_rgb_loss / len(batch_frame_indices)
        batch_ssim_loss = total_ssim_loss / len(batch_frame_indices)
        batch_silhouette_loss = total_silhouette_loss / len(batch_frame_indices)
        batch_reg_loss = (
            total_reg_loss / len(batch_frame_indices)
            if config.use_regularization
            else torch.tensor(0.0, device=device)
        )

        # Total loss
        total_loss = (
            batch_rgb_loss
            + batch_ssim_loss
            + batch_silhouette_loss
            + batch_reg_loss
        )

        # Record loss (ssim logged separately, unweighted)
        loss_history.append(
            {
                "total": total_loss.item(),
                "rgb": batch_rgb_loss.item(),
                "ssim": batch_ssim_loss.item(),
                "silhouette": batch_silhouette_loss.item(),
                "regularization": batch_reg_loss.item() if config.use_regularization else 0.0,
            }
        )

        # Backprop
        total_loss.backward()

        # Debug on first iteration
        if iteration == 0 and config.verbose:
            print(
                f"      [Gradient check] global_scale grad: "
                f"{opt_global_scale.grad.abs().max().item() if opt_global_scale.grad is not None else 'None':.6e}"
            )

        optimizer.step()

        # Track best
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_iteration = iteration
            best_params = {
                "global_scale": opt_global_scale.clone().detach(),
                "rotations": {fi: opt_rotations[fi].clone().detach() for fi in valid_frame_indices},
                "translations": {
                    fi: opt_translations[fi].clone().detach() for fi in valid_frame_indices
                },
            }

        if config.verbose and (
            iteration % config.log_interval == 0 or iteration == config.num_iterations - 1
        ):
            print(
                f"      Iteration {iteration}: total={total_loss.item():.6f} "
                f"rgb={batch_rgb_loss.item():.6f} "
                f"silh={batch_silhouette_loss.item():.6f} "
                f"reg={batch_reg_loss.item():.8f} "
                f"scale={opt_global_scale.item():.6f}"
            )

    if config.verbose:
        print(f"      Best loss: {best_loss:.6f} (iteration {best_iteration})")
        print(f"      Final global scale: {best_params['global_scale'].item():.6f}")

    # Build refined tokens list
    refined_tokens_list = []
    global_scale_3d = best_params["global_scale"].expand(3).reshape(1, 3)

    for frame_idx, decoder_input in tokens_list:
        if frame_idx in valid_frame_indices:
            refined_rotation = best_params["rotations"][frame_idx]
            refined_rotation = refined_rotation / refined_rotation.norm()

            refined_decoder_input = {
                "decoder_input_slat": decoder_input["decoder_input_slat"],
                "rotation": refined_rotation,
                "translation": best_params["translations"][frame_idx],
                "scale": global_scale_3d.clone(),
                "refinement_loss_history": loss_history,
                "refinement_best_iteration": best_iteration,
            }
            refined_tokens_list.append((frame_idx, refined_decoder_input))
        else:
            refined_tokens_list.append((frame_idx, decoder_input))

    return refined_tokens_list


def refine_poses_for_sequence(
    canonical_gaussians: Dict[int, Any],
    tokens_by_object: Dict[int, List[Tuple[int, Dict[str, Any]]]],
    args: Any,
    paths: Dict[str, Any],
    inference: Any,
    config: RefinementConfig,
    per_frame_canonical: bool = False,
) -> Dict[int, List[Tuple[int, Dict[str, Any]]]]:
    """
    Refine per-frame poses for all objects using differentiable rendering.

    Parameters
    ----------
    canonical_gaussians : dict
        Dictionary mapping object_index -> canonical Gaussian (if per_frame_canonical=False)
        OR Dictionary mapping object_index -> dict[frame_index -> canonical Gaussian]
        (if per_frame_canonical=True).
    tokens_by_object : dict
        Dictionary mapping object_index -> list of (frame_index, decoder_input).
    args : argparse.Namespace
        Command line arguments.
    paths : dict
        Dataset paths.
    inference : Inference
        Inference pipeline.
    config : RefinementConfig
        Configuration for refinement hyperparameters.
    per_frame_canonical : bool, optional
        If True, use per-frame canonical Gaussians (standard mode).
        If False, use shared canonical Gaussians across frames (averaged-tokens mode).

    Returns
    -------
    dict
        Refined tokens_by_object with updated poses.
    """
    from .depth import load_and_process_depth
    from .io_utils import load_image, load_masks

    # Dispatch to global scale refinement if requested
    if config.refine_scale == "global":
        print("\n  Refining poses with GLOBAL scale optimization (batch-based)...")

        refined_tokens = {}

        for obj_idx in sorted(tokens_by_object.keys()):
            if per_frame_canonical:
                if obj_idx not in canonical_gaussians or len(canonical_gaussians[obj_idx]) == 0:
                    print(f"    Warning: No canonical Gaussian for object {obj_idx}, skipping")
                    refined_tokens[obj_idx] = tokens_by_object[obj_idx]
                    continue
                first_frame = min(canonical_gaussians[obj_idx].keys())
                canonical_gs = canonical_gaussians[obj_idx][first_frame]
                print(f"    Object {obj_idx}: Using canonical Gaussian from frame {first_frame}")
            else:
                if obj_idx not in canonical_gaussians:
                    print(f"    Warning: No canonical Gaussian for object {obj_idx}, skipping")
                    refined_tokens[obj_idx] = tokens_by_object[obj_idx]
                    continue
                canonical_gs = canonical_gaussians[obj_idx]

            refined_tokens[obj_idx] = refine_poses_global_scale(
                canonical_gs,
                tokens_by_object[obj_idx],
                obj_idx,
                args,
                paths,
                inference,
                config,
            )

        return refined_tokens

    # Original per-frame refinement logic
    print("\n  Refining per-frame poses with differentiable rendering...")

    refined_tokens = {}

    for obj_idx in sorted(tokens_by_object.keys()):
        print(f"\n    Object {obj_idx}:")
        refined_tokens[obj_idx] = []

        for frame_idx, decoder_input in tokens_by_object[obj_idx]:
            print(f"      Frame {frame_idx}:")

            # Get the canonical Gaussian
            if per_frame_canonical:
                if (
                    obj_idx not in canonical_gaussians
                    or frame_idx not in canonical_gaussians[obj_idx]
                ):
                    print(
                        f"        Warning: No canonical Gaussian for object {obj_idx} "
                        f"frame {frame_idx}, skipping"
                    )
                    refined_tokens[obj_idx].append((frame_idx, decoder_input))
                    continue
                canonical_gs = canonical_gaussians[obj_idx][frame_idx]
            else:
                if obj_idx not in canonical_gaussians:
                    print(f"        Warning: No canonical Gaussian for object {obj_idx}, skipping")
                    refined_tokens[obj_idx].append((frame_idx, decoder_input))
                    continue
                canonical_gs = canonical_gaussians[obj_idx]

            # Load frame data
            image_path = os.path.join(paths["frames_path"], paths["image_names"][frame_idx])
            mask_path = os.path.join(paths["masks_path"], paths["mask_names"][frame_idx])

            image = load_image(image_path)
            image = image[..., :3]
            H, W, _ = image.shape

            masks = load_masks(mask_path)
            if args.object_index is not None:
                masks = [masks[args.object_index]]

            if obj_idx < len(masks):
                mask = masks[obj_idx]
            else:
                print(f"        Warning: No mask for object {obj_idx}, skipping refinement")
                refined_tokens[obj_idx].append((frame_idx, decoder_input))
                continue

            # Load depth and compute K_matrix
            depth_names_for_frame = []
            if paths["dataset_type"] == "kubric4d" and paths["depth_names"]:
                depth_names_for_frame = [paths["depth_names"][frame_idx]]

            pointmap, K_matrix, valid_mask = load_and_process_depth(
                paths["frames_path"],
                depth_names_for_frame,
                W,
                H,
                use_moge=args.use_moge,
                inference=inference,
                image=image,
            )

            # Ground truth image
            gt_image = torch.from_numpy(image).float().cuda() / 255.0

            # Initial pose
            initial_rotation = decoder_input["rotation"]
            initial_translation = decoder_input["translation"]
            initial_scale = decoder_input["scale"]

            # Refine pose
            refined_pose = refine_pose_for_frame(
                canonical_gs,
                initial_rotation,
                initial_translation,
                initial_scale,
                gt_image,
                mask,
                K_matrix,
                config=config,
            )

            # Create refined decoder input
            refined_decoder_input = {
                "decoder_input_slat": decoder_input["decoder_input_slat"],
                "rotation": refined_pose["rotation"],
                "translation": refined_pose["translation"],
                "scale": refined_pose["scale"],
                "refinement_loss_history": refined_pose["loss_history"],
                "refinement_best_iteration": refined_pose["best_iteration"],
            }

            refined_tokens[obj_idx].append((frame_idx, refined_decoder_input))

    return refined_tokens


def decode_per_frame_gaussians(
    tokens_by_object: Dict[int, List[Tuple[int, Dict[str, Any]]]],
    pipeline: Any,
) -> Dict[int, Dict[int, "Gaussian"]]:
    """
    Decode Gaussians for each (frame, object) pair from cached tokens.

    Parameters
    ----------
    tokens_by_object : dict
        Dictionary mapping object_index -> list of (frame_index, decoder_input).
    pipeline : Pipeline
        The inference pipeline for decoding.

    Returns
    -------
    dict
        Dictionary mapping object_index -> dict[frame_index -> canonical Gaussian].
    """
    from .tokens import redecode_slat

    canonical_gaussians_per_frame = {}

    for obj_idx in sorted(tokens_by_object.keys()):
        canonical_gaussians_per_frame[obj_idx] = {}

        for frame_idx, decoder_input in tokens_by_object[obj_idx]:
            slat = decoder_input["decoder_input_slat"]
            decoded = redecode_slat(pipeline, slat, formats=["gaussian"])
            canonical_gaussians_per_frame[obj_idx][frame_idx] = decoded["gaussian"][0]

    return canonical_gaussians_per_frame


__all__ = [
    "RefinementConfig",
    "apply_pose_to_gaussian",
    "refine_pose_for_frame",
    "refine_poses_global_scale",
    "refine_poses_for_sequence",
    "decode_per_frame_gaussians",
]

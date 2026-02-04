"""
Evaluation utilities for the SAM3D-Objects pipeline.

This module provides functions for evaluating reconstruction quality
including frame processing, metrics computation, and summary generation.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import torch

if TYPE_CHECKING:
    from evaluator import Evaluator
    from inference import Inference
    from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model import (
        Gaussian,
    )


def process_frame_from_cache(
    args: Any,
    paths: Dict[str, Any],
    frame_index: int,
    inference: "Inference",
    tokens_dir: str,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, np.ndarray, List[np.ndarray]]]:
    """
    Process a single frame by loading cached tokens and re-decoding.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    paths : dict
        Dataset paths.
    frame_index : int
        Frame index to process.
    inference : Inference
        Inference pipeline.
    tokens_dir : str
        Directory where cached tokens are stored.

    Returns
    -------
    tuple or None
        (rendered_image, gt_image, K_matrix, masks) or None if cache not found.
        - rendered_image: torch.Tensor (H, W, 3)
        - gt_image: torch.Tensor (H, W, 3)
        - K_matrix: np.ndarray (3, 3)
        - masks: list of np.ndarray (H, W) boolean masks per object
    """
    from inference import make_scene

    from .depth import load_and_process_depth
    from .gaussian import create_background_gaussians, join_gaussians, transform_scene_to_r3_convention
    from .io_utils import get_cache_filename, load_image, load_masks
    from .rendering import render_gaussians_to_image
    from .tokens import load_decoder_inputs_from_cache, redecode_slat

    # Build cache filename
    cache_filename, cache_scene_name = get_cache_filename(
        args.scene_name, frame_index, args.object_index, args.background
    )
    cache_file = os.path.join(tokens_dir, cache_filename)

    # Check if cache exists
    if not os.path.exists(cache_file):
        return None

    print(f"    Loading cached tokens from {cache_filename}")

    # Load frame's image and masks
    image_path = os.path.join(paths["frames_path"], paths["image_names"][frame_index])
    mask_path = os.path.join(paths["masks_path"], paths["mask_names"][frame_index])

    image = load_image(image_path)
    image = image[..., :3]
    H, W, _ = image.shape

    masks = load_masks(mask_path)
    if args.object_index is not None:
        masks = [masks[args.object_index]]

    # Load depth and compute K_matrix (needed for rendering)
    depth_names_for_frame = []
    if paths["dataset_type"] == "kubric4d" and paths["depth_names"]:
        depth_names_for_frame = [paths["depth_names"][frame_index]]

    pointmap, K_matrix, valid_mask = load_and_process_depth(
        paths["frames_path"],
        depth_names_for_frame,
        W,
        H,
        use_moge=args.use_moge,
        inference=inference,
        image=image,
    )

    # Keep original pointmap for background
    pointmap_original = pointmap.copy() if args.background else None

    # Load decoder inputs from cache
    decoder_inputs = load_decoder_inputs_from_cache(cache_file)

    if len(decoder_inputs) == 0:
        print("    No decoder inputs found in cache")
        return None

    # Re-decode each object and build outputs compatible with make_scene
    pipeline = inference._pipeline
    outputs = []

    for i, decoder_input in enumerate(decoder_inputs):
        slat = decoder_input["decoder_input_slat"]

        # Re-decode to get Gaussian
        decoded = redecode_slat(pipeline, slat, formats=["gaussian"])

        # Build output dict compatible with make_scene
        output = {
            "gaussian": decoded["gaussian"],
            "rotation": decoder_input["rotation"],
            "translation": decoder_input["translation"],
            "scale": decoder_input["scale"],
        }
        outputs.append(output)

    # Create combined scene from all outputs (in PyTorch3D convention)
    scene_gs = make_scene(*outputs)

    # Transform scene from PyTorch3D to R3 convention
    new_scene_gs = transform_scene_to_r3_convention(scene_gs)

    # Add background Gaussians if requested
    if args.background and pointmap_original is not None:
        background_gs = create_background_gaussians(image, pointmap_original, masks, K_matrix)
        new_scene_gs = join_gaussians(background_gs, new_scene_gs)

    # Render Gaussians to image
    rendered = render_gaussians_to_image(new_scene_gs, K_matrix, W, H)

    # Convert ground truth to tensor
    gt_image = torch.from_numpy(image).float() / 255.0

    # Clamp rendered to [0, 1]
    rendered = torch.clamp(rendered.cpu(), 0.0, 1.0)

    return rendered, gt_image, K_matrix, masks


def process_frame_full_inference(
    args: Any,
    paths: Dict[str, Any],
    frame_index: int,
    inference: "Inference",
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, List[np.ndarray]]:
    """
    Process a single frame using full inference (no cache).

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    paths : dict
        Dataset paths.
    frame_index : int
        Frame index to process.
    inference : Inference
        Inference pipeline.

    Returns
    -------
    tuple
        (rendered_image, gt_image, K_matrix, masks) where images are torch tensors (H, W, 3)
        and masks is a list of numpy arrays (H, W) boolean masks per object.
    """
    from inference import make_scene

    from .depth import load_and_process_depth, transform_to_pytorch3d_convention
    from .gaussian import create_background_gaussians, join_gaussians, transform_scene_to_r3_convention
    from .inference_utils import run_inference_on_masks
    from .io_utils import load_image, load_masks
    from .rendering import render_gaussians_to_image

    # Load frame's image and masks
    image_path = os.path.join(paths["frames_path"], paths["image_names"][frame_index])
    mask_path = os.path.join(paths["masks_path"], paths["mask_names"][frame_index])

    image = load_image(image_path)
    image = image[..., :3]
    H, W, _ = image.shape

    masks = load_masks(mask_path)
    if args.object_index is not None:
        masks = [masks[args.object_index]]

    print(f"\n  Frame {frame_index}: Loaded image {image.shape}, {len(masks)} masks")

    # Process depth
    depth_names_for_frame = []
    if paths["dataset_type"] == "kubric4d" and paths["depth_names"]:
        depth_names_for_frame = [paths["depth_names"][frame_index]]

    pointmap, K_matrix, valid_mask = load_and_process_depth(
        paths["frames_path"],
        depth_names_for_frame,
        W,
        H,
        use_moge=args.use_moge,
        inference=inference,
        image=image,
    )

    pointmap_original = pointmap.copy() if args.background else None

    # Transform to PyTorch3D convention
    pointmap = transform_to_pytorch3d_convention(pointmap)

    # Run full inference
    outputs = run_inference_on_masks(inference, image, masks, pointmap, seed=args.seed)

    # Create combined scene
    scene_gs = make_scene(*outputs)
    new_scene_gs = transform_scene_to_r3_convention(scene_gs)

    # Add background if requested
    if args.background and pointmap_original is not None:
        background_gs = create_background_gaussians(image, pointmap_original, masks, K_matrix)
        new_scene_gs = join_gaussians(background_gs, new_scene_gs)

    # Render
    rendered = render_gaussians_to_image(new_scene_gs, K_matrix, W, H)
    gt_image = torch.from_numpy(image).float() / 255.0
    rendered = torch.clamp(rendered.cpu(), 0.0, 1.0)

    return rendered, gt_image, K_matrix, masks


def process_frame_for_eval(
    args: Any,
    paths: Dict[str, Any],
    frame_index: int,
    inference: "Inference",
    tokens_dir: str,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, List[np.ndarray]]:
    """
    Process a single frame, using cache if available.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    paths : dict
        Dataset paths.
    frame_index : int
        Frame index to process.
    inference : Inference
        Inference pipeline.
    tokens_dir : str
        Directory where cached tokens are stored.

    Returns
    -------
    tuple
        (rendered_image, gt_image, K_matrix, masks) where images are torch tensors (H, W, 3)
        and masks is a list of numpy arrays (H, W) boolean masks per object.
    """
    # First try to load from cache
    if args.use_cache:
        result = process_frame_from_cache(args, paths, frame_index, inference, tokens_dir)
        if result is not None:
            print(f"    Loaded cached tokens for frame {frame_index} from {tokens_dir}")
            return result
        print("    Cache not found, running full inference...")

    # Fall back to full inference
    return process_frame_full_inference(args, paths, frame_index, inference)


def process_frame_with_canonical_object(
    args: Any,
    paths: Dict[str, Any],
    frame_index: int,
    inference: "Inference",
    tokens_dir: str,
    canonical_gaussians: Dict[int, Any],
    tokens_by_object: Dict[int, List[Tuple[int, Dict[str, Any]]]],
    per_frame_canonical: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, List[np.ndarray], "Gaussian"]:
    """
    Process a single frame using pre-decoded canonical Gaussians warped with per-frame pose.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    paths : dict
        Dataset paths.
    frame_index : int
        Frame index to process.
    inference : Inference
        Inference pipeline.
    tokens_dir : str
        Directory where cached tokens are stored.
    canonical_gaussians : dict
        Dictionary mapping obj_idx -> decoded Gaussian object (if per_frame_canonical=False)
        OR dict[obj_idx][frame_idx] -> decoded Gaussian object (if per_frame_canonical=True).
    tokens_by_object : dict
        Dictionary mapping obj_idx -> list of (frame_idx, decoder_input) with poses.
    per_frame_canonical : bool, optional
        If True, use per-frame canonical Gaussians (standard mode).
        If False, use shared canonical Gaussians across frames (averaged-tokens mode).

    Returns
    -------
    tuple
        (rendered_image, gt_image, K_matrix, masks, scene_gs) where images are torch tensors
        (H, W, 3), masks is a list of np arrays, and scene_gs is the Gaussian scene.
    """
    from inference import make_scene

    from .depth import load_and_process_depth
    from .gaussian import create_background_gaussians, join_gaussians, transform_scene_to_r3_convention
    from .io_utils import load_image, load_masks
    from .rendering import render_gaussians_to_image

    # Load frame's image and masks
    image_path = os.path.join(paths["frames_path"], paths["image_names"][frame_index])
    mask_path = os.path.join(paths["masks_path"], paths["mask_names"][frame_index])

    image = load_image(image_path)
    image = image[..., :3]
    H, W, _ = image.shape

    masks = load_masks(mask_path)
    if args.object_index is not None:
        if args.object_index < len(masks):
            masks = [masks[args.object_index]]
        else:
            print(f"Warning: --object-index {args.object_index} out of range (only {len(masks)} masks)")
            return None

    # Load depth and compute K_matrix (needed for rendering)
    depth_names_for_frame = []
    if paths["dataset_type"] == "kubric4d" and paths["depth_names"]:
        depth_names_for_frame = [paths["depth_names"][frame_index]]

    pointmap, K_matrix, valid_mask = load_and_process_depth(
        paths["frames_path"],
        depth_names_for_frame,
        W,
        H,
        use_moge=args.use_moge,
        inference=inference,
        image=image,
    )

    pointmap_original = pointmap.copy() if args.background else None

    # Get per-frame poses from cached tokens
    # Build outputs list with canonical gaussian + per-frame pose
    outputs = []

    for obj_idx in sorted(tokens_by_object.keys()):
        # Get the canonical Gaussian for this object (and frame, if per_frame_canonical)
        if per_frame_canonical:
            if obj_idx not in canonical_gaussians or frame_index not in canonical_gaussians[obj_idx]:
                print(
                    f"    Warning: No canonical Gaussian for object {obj_idx} frame {frame_index}, skipping"
                )
                continue
            canonical_gs = canonical_gaussians[obj_idx][frame_index]
        else:
            if obj_idx not in canonical_gaussians:
                print(f"    Warning: No canonical Gaussian for object {obj_idx}, skipping")
                continue
            canonical_gs = canonical_gaussians[obj_idx]

        # Find the pose for this frame
        frame_pose = None
        for fid, decoder_input in tokens_by_object[obj_idx]:
            if fid == frame_index:
                frame_pose = decoder_input
                break

        if frame_pose is None:
            print(f"    Warning: No pose found for object {obj_idx} at frame {frame_index}, skipping")
            continue

        # Build output with canonical gaussian and per-frame pose
        output = {
            "gaussian": [canonical_gs],  # Wrap in list for make_scene
            "rotation": frame_pose["rotation"],
            "translation": frame_pose["translation"],
            "scale": frame_pose["scale"],
        }
        outputs.append(output)

    # Create combined scene from all outputs (in PyTorch3D convention)
    scene_gs = make_scene(*outputs)

    # Transform scene from PyTorch3D to R3 convention
    new_scene_gs = transform_scene_to_r3_convention(scene_gs)

    # Store scene without background for point cloud export
    scene_gs_no_bg = new_scene_gs

    # Add background Gaussians if requested
    if args.background and pointmap_original is not None:
        background_gs = create_background_gaussians(image, pointmap_original, masks, K_matrix)
        new_scene_gs = join_gaussians(background_gs, new_scene_gs)

    # Render Gaussians to image
    rendered = render_gaussians_to_image(new_scene_gs, K_matrix, W, H)

    # Convert ground truth to tensor
    gt_image = torch.from_numpy(image).float() / 255.0

    # Clamp rendered to [0, 1]
    rendered = torch.clamp(rendered.cpu(), 0.0, 1.0)

    return rendered, gt_image, K_matrix, masks, scene_gs_no_bg


def _build_evaluation_summary(
    evaluator: "Evaluator",
    rendered_frames: List[torch.Tensor],
    gt_frames: List[torch.Tensor],
    frame_indices_processed: List[int],
    per_object_data: Dict[int, Dict[str, List[torch.Tensor]]],
    args: Any,
    suffix: str,
) -> Dict[str, Any]:
    """
    Build evaluation summary from collected frames (shared by both evaluation modes).

    Parameters
    ----------
    evaluator : Evaluator
        Evaluator instance.
    rendered_frames : list
        List of rendered frame tensors (B, C, H, W).
    gt_frames : list
        List of ground truth frame tensors (B, C, H, W).
    frame_indices_processed : list
        List of frame indices that were processed.
    per_object_data : dict
        Dictionary mapping obj_idx -> {'rendered': [], 'gt': []}.
    args : argparse.Namespace
        Command line arguments.
    suffix : str
        Suffix for output filenames.

    Returns
    -------
    dict
        Evaluation summary with metrics.
    """
    print(f"\n{'='*60}")
    print("Evaluating sequence...")
    print(f"{'='*60}")

    # Use Evaluator's evaluate_sequence method for full-frame metrics
    seq_metrics = evaluator.evaluate_sequence(gt_frames, rendered_frames)

    # Print per-frame metrics (full frame)
    print("\nPer-frame metrics (full frame):")
    for i, frame_index in enumerate(frame_indices_processed):
        print(
            f"  Frame {frame_index}: PSNR={seq_metrics['psnr_values'][i]:.2f} dB, "
            f"SSIM={seq_metrics['ssim_values'][i]:.4f}, "
            f"LPIPS={seq_metrics['lpip_values'][i]:.4f}"
        )

    # Build frame_metrics with nested structure: full_frame, obj_0, obj_1, ...
    frame_metrics = {
        "full_frame": [
            {
                "frame_index": frame_indices_processed[i],
                "psnr": seq_metrics["psnr_values"][i],
                "ssim": seq_metrics["ssim_values"][i],
                "lpip": seq_metrics["lpip_values"][i],
            }
            for i in range(len(frame_indices_processed))
        ]
    }

    # Evaluate per-object metrics
    per_object_summary = {}
    for obj_idx in sorted(per_object_data.keys()):
        obj_key = f"obj_{obj_idx}"
        if per_object_data[obj_idx]["rendered"] and per_object_data[obj_idx]["gt"]:
            obj_metrics = evaluator.evaluate_sequence(
                per_object_data[obj_idx]["gt"], per_object_data[obj_idx]["rendered"]
            )
            frame_metrics[obj_key] = [
                {
                    "frame_index": frame_indices_processed[i],
                    "psnr": obj_metrics["psnr_values"][i],
                    "ssim": obj_metrics["ssim_values"][i],
                    "lpip": obj_metrics["lpip_values"][i],
                }
                for i in range(len(frame_indices_processed))
            ]
            per_object_summary[obj_key] = {
                "psnr_mean": float(obj_metrics["psnr_mean"]),
                "psnr_std": float(obj_metrics["psnr_std"]),
                "ssim_mean": float(obj_metrics["ssim_mean"]),
                "ssim_std": float(obj_metrics["ssim_std"]),
                "lpip_mean": float(obj_metrics["lpip_mean"]),
                "lpip_std": float(obj_metrics["lpip_std"]),
            }

    # Determine if using canonical mode
    canonicalization = getattr(args, "canonicalization", "none")
    use_canonical = canonicalization in ["average", "pickone"]

    summary = {
        "dataset": args.dataset,
        "scene_name": args.scene_name,
        "num_frames_evaluated": len(frame_indices_processed),
        "frame_stride": args.frame_stride,
        "with_background": args.background,
        "object_index": args.object_index,
        "canonicalization": canonicalization,
        "weighting_type": args.weighting_type if canonicalization == "average" else None,
        "canon_frame": getattr(args, "canon_frame", None) if canonicalization == "pickone" else None,
        "refine_poses": getattr(args, "refine_poses", False) if use_canonical else False,
        "refine_iterations": (
            getattr(args, "refine_iterations", None)
            if (use_canonical and getattr(args, "refine_poses", False))
            else None
        ),
        "suffix": suffix,
        # Full-frame summary metrics
        "psnr_mean": float(seq_metrics["psnr_mean"]),
        "psnr_std": float(seq_metrics["psnr_std"]),
        "psnr_min": float(seq_metrics["psnr_min"]),
        "psnr_max": float(seq_metrics["psnr_max"]),
        "ssim_mean": float(seq_metrics["ssim_mean"]),
        "ssim_std": float(seq_metrics["ssim_std"]),
        "ssim_min": float(seq_metrics["ssim_min"]),
        "ssim_max": float(seq_metrics["ssim_max"]),
        "lpip_mean": float(seq_metrics["lpip_mean"]),
        "lpip_std": float(seq_metrics["lpip_std"]),
        "lpip_min": float(seq_metrics["lpip_min"]),
        "lpip_max": float(seq_metrics["lpip_max"]),
        # Per-object summary metrics
        "per_object_summary": per_object_summary,
        # Frame-level metrics (full_frame, obj_0, obj_1, ...)
        "frame_metrics": frame_metrics,
    }

    # Save metrics to JSON if requested
    if args.save_metrics:
        metrics_path = os.path.join(args.output_dir, f"{args.scene_name}{suffix}_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Saved metrics to {metrics_path}")

    return summary


def evaluate_standard_mode(
    args: Any,
    paths: Dict[str, Any],
    frame_indices: List[int],
    inference: "Inference",
    tokens_dir: str,
    evaluator: "Evaluator",
    device: torch.device,
    suffix: str = "",
    save_renders: bool = True,
    median_scales: Optional[Dict[int, torch.Tensor]] = None,
) -> Dict[str, Any]:
    """
    Evaluate frames using standard per-frame inference (no token averaging).

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    paths : dict
        Dataset paths.
    frame_indices : list
        List of frame indices to process.
    inference : Inference
        Inference pipeline.
    tokens_dir : str
        Directory where cached tokens are stored.
    evaluator : Evaluator
        Evaluator instance.
    device : torch.device
        Device to use.
    suffix : str, optional
        Suffix for output filenames.
    save_renders : bool, optional
        Whether to save rendered images.
    median_scales : dict, optional
        Dictionary mapping obj_idx -> median scale tensor.

    Returns
    -------
    dict
        Evaluation summary with metrics.
    """
    from .rendering import save_comparison_image

    rendered_frames = []
    gt_frames = []
    frame_indices_processed = []
    per_object_data: Dict[int, Dict[str, List[torch.Tensor]]] = {}

    for frame_idx, frame_index in enumerate(frame_indices):
        print(f"\n{'='*60}")
        print(f"Processing frame {frame_index} ({frame_idx + 1}/{len(frame_indices)})")
        print(f"{'='*60}")

        rendered, gt_image, K_matrix, masks = process_frame_for_eval(
            args, paths, frame_index, inference, tokens_dir
        )

        # Convert to format expected by evaluator: (B, C, H, W)
        rendered_eval = rendered.permute(2, 0, 1).unsqueeze(0).to(device)
        gt_eval = gt_image.permute(2, 0, 1).unsqueeze(0).to(device)

        # Save comparison image if requested
        if save_renders and args.save_renders:
            render_dir = os.path.join(args.output_dir, "renders")
            os.makedirs(render_dir, exist_ok=True)
            output_path = os.path.join(
                render_dir, f"{args.scene_name}_frame_{frame_index:04d}{suffix}_comparison.png"
            )
            save_comparison_image(rendered, gt_image, output_path, frame_index)
            print(f"  Saved comparison to {output_path}")

        # Store for sequence evaluation
        rendered_frames.append(rendered_eval)
        gt_frames.append(gt_eval)
        frame_indices_processed.append(frame_index)

        # Initialize per-object data storage on first frame
        if not per_object_data:
            for obj_idx in range(len(masks)):
                per_object_data[obj_idx] = {"rendered": [], "gt": []}

        # Store per-object masked data
        for obj_idx, mask in enumerate(masks):
            if obj_idx not in per_object_data:
                per_object_data[obj_idx] = {"rendered": [], "gt": []}
            mask_tensor = torch.from_numpy(mask).float().to(device)
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)

            rendered_masked = rendered_eval * mask_tensor
            gt_masked = gt_eval * mask_tensor

            per_object_data[obj_idx]["rendered"].append(rendered_masked)
            per_object_data[obj_idx]["gt"].append(gt_masked)

    return _build_evaluation_summary(
        evaluator,
        rendered_frames,
        gt_frames,
        frame_indices_processed,
        per_object_data,
        args,
        suffix,
    )


def evaluate_with_canonical_objects(
    args: Any,
    paths: Dict[str, Any],
    frame_indices: List[int],
    inference: "Inference",
    tokens_dir: str,
    canonical_gaussians: Dict[int, Any],
    tokens_by_object: Dict[int, List[Tuple[int, Dict[str, Any]]]],
    evaluator: "Evaluator",
    device: torch.device,
    suffix: str = "",
    save_renders: bool = True,
    per_frame_canonical: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate frames using canonical objects with per-frame poses.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    paths : dict
        Dataset paths.
    frame_indices : list
        List of frame indices to process.
    inference : Inference
        Inference pipeline.
    tokens_dir : str
        Directory where cached tokens are stored.
    canonical_gaussians : dict
        Dictionary mapping obj_idx -> canonical Gaussian (if per_frame_canonical=False)
        OR dict[obj_idx][frame_idx] -> canonical Gaussian (if per_frame_canonical=True).
    tokens_by_object : dict
        Dictionary mapping obj_idx -> list of (frame_idx, decoder_input).
    evaluator : Evaluator
        Evaluator instance.
    device : torch.device
        Device to use.
    suffix : str, optional
        Suffix for output filenames.
    save_renders : bool, optional
        Whether to save rendered images.
    per_frame_canonical : bool, optional
        If True, use per-frame canonical Gaussians (standard mode).

    Returns
    -------
    dict
        Evaluation summary with metrics.
    """
    from .rendering import save_comparison_image
    from .temporal import (
        add_frame_to_temporal_point_cloud,
        finalize_temporal_point_cloud,
        save_temporal_point_cloud,
    )

    rendered_frames = []
    gt_frames = []
    frame_indices_processed = []

    # Filter frame_indices to only those that have poses for ALL objects
    frame_sets_per_object = []
    for obj_idx, tokens_list in tokens_by_object.items():
        obj_frames = set(fid for fid, _ in tokens_list)
        frame_sets_per_object.append(obj_frames)

    if frame_sets_per_object:
        available_frame_indices = frame_sets_per_object[0]
        for obj_frames in frame_sets_per_object[1:]:
            available_frame_indices = available_frame_indices & obj_frames
    else:
        available_frame_indices = set()

    frame_indices_to_process = [f for f in frame_indices if f in available_frame_indices]

    if len(frame_indices_to_process) < len(frame_indices):
        print(
            f"\nNote: Only {len(frame_indices_to_process)} of {len(frame_indices)} "
            "requested frames have poses for all objects"
        )
        print(f"  Frames with complete poses: {sorted(available_frame_indices)}")
        for obj_idx, tokens_list in tokens_by_object.items():
            obj_frames = set(fid for fid, _ in tokens_list)
            missing = set(frame_indices) - obj_frames
            if missing:
                print(f"  Object {obj_idx} missing frames: {sorted(missing)}")

    per_object_data: Dict[int, Dict[str, List[torch.Tensor]]] = {}
    num_objects = len(tokens_by_object)

    for obj_idx in range(num_objects):
        per_object_data[obj_idx] = {"rendered": [], "gt": []}

    # Initialize temporal point cloud storage
    save_point_clouds = getattr(args, "save_point_clouds", False)
    pc_storage = None
    if save_point_clouds:
        pc_storage = save_temporal_point_cloud(args.output_dir, args.scene_name, suffix)

    # Process each frame
    for frame_idx, frame_index in enumerate(frame_indices_to_process):
        print(f"\n  Processing frame {frame_index} ({frame_idx + 1}/{len(frame_indices_to_process)})")

        result = process_frame_with_canonical_object(
            args,
            paths,
            frame_index,
            inference,
            tokens_dir,
            canonical_gaussians,
            tokens_by_object,
            per_frame_canonical=per_frame_canonical,
        )

        rendered, gt_image, K_matrix, masks, scene_gs = result

        # Save point cloud for this frame
        if save_point_clouds and pc_storage is not None and scene_gs is not None:
            add_frame_to_temporal_point_cloud(pc_storage, frame_index, scene_gs)

        # Convert to format expected by evaluator: (B, C, H, W)
        rendered_eval = rendered.permute(2, 0, 1).unsqueeze(0).to(device)
        gt_eval = gt_image.permute(2, 0, 1).unsqueeze(0).to(device)

        # Save comparison image if requested
        if save_renders and args.save_renders:
            render_dir = os.path.join(args.output_dir, "renders")
            os.makedirs(render_dir, exist_ok=True)
            output_path = os.path.join(
                render_dir, f"{args.scene_name}_frame_{frame_index:04d}{suffix}_comparison.png"
            )
            save_comparison_image(rendered, gt_image, output_path, frame_index)
            print(f"    Saved comparison to {output_path}")

        rendered_frames.append(rendered_eval)
        gt_frames.append(gt_eval)
        frame_indices_processed.append(frame_index)

        # Store per-object masked data for evaluation
        for obj_idx, mask in enumerate(masks):
            mask_tensor = torch.from_numpy(mask).float().to(device)
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)

            rendered_masked = rendered_eval * mask_tensor
            gt_masked = gt_eval * mask_tensor

            per_object_data[obj_idx]["rendered"].append(rendered_masked)
            per_object_data[obj_idx]["gt"].append(gt_masked)

    # Finalize and save point clouds
    if save_point_clouds and pc_storage is not None:
        finalize_temporal_point_cloud(pc_storage)

    return _build_evaluation_summary(
        evaluator,
        rendered_frames,
        gt_frames,
        frame_indices_processed,
        per_object_data,
        args,
        suffix,
    )


def print_evaluation_summary(summary: Dict[str, Any], title: str = "Evaluation Summary") -> None:
    """
    Print evaluation metrics in a formatted way.

    Parameters
    ----------
    summary : dict
        Evaluation summary dictionary.
    title : str, optional
        Title for the summary output.
    """
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")
    print(f"Frames evaluated:  {summary['num_frames_evaluated']}")

    # Full-frame metrics
    print("\n[Full Frame Metrics]")
    print("PSNR (dB):")
    print(f"  Mean: {summary['psnr_mean']:.2f} ± {summary['psnr_std']:.2f}")
    print(f"  Range: [{summary['psnr_min']:.2f}, {summary['psnr_max']:.2f}]")
    print("SSIM:")
    print(f"  Mean: {summary['ssim_mean']:.4f} ± {summary['ssim_std']:.4f}")
    print(f"  Range: [{summary['ssim_min']:.4f}, {summary['ssim_max']:.4f}]")
    print("LPIPS (lower is better):")
    print(f"  Mean: {summary['lpip_mean']:.4f} ± {summary['lpip_std']:.4f}")
    print(f"  Range: [{summary['lpip_min']:.4f}, {summary['lpip_max']:.4f}]")

    # Per-object metrics if available
    if "per_object_summary" in summary and summary["per_object_summary"]:
        for obj_key, obj_metrics in summary["per_object_summary"].items():
            print(f"\n[{obj_key} Metrics (masked)]")
            print("PSNR (dB):")
            print(f"  Mean: {obj_metrics['psnr_mean']:.2f} ± {obj_metrics['psnr_std']:.2f}")
            print("SSIM:")
            print(f"  Mean: {obj_metrics['ssim_mean']:.4f} ± {obj_metrics['ssim_std']:.4f}")
            print("LPIPS (lower is better):")
            print(f"  Mean: {obj_metrics['lpip_mean']:.4f} ± {obj_metrics['lpip_std']:.4f}")


__all__ = [
    "process_frame_from_cache",
    "process_frame_full_inference",
    "process_frame_for_eval",
    "process_frame_with_canonical_object",
    "evaluate_standard_mode",
    "evaluate_with_canonical_objects",
    "print_evaluation_summary",
]

"""
Inference utilities for the SAM3D-Objects pipeline.

This module provides helper functions for running inference on masks
and managing cached inference results.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

if TYPE_CHECKING:
    import numpy as np
    from inference import Inference


def run_inference_on_masks(
    inference: "Inference",
    image: "np.ndarray",
    masks: List["np.ndarray"],
    pointmap: "np.ndarray",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Run SAM3D inference on all masks.

    Parameters
    ----------
    inference : Inference
        The inference pipeline object.
    image : np.ndarray
        Input image, shape (H, W, 3).
    masks : list of np.ndarray
        List of boolean masks, each shape (H, W).
    pointmap : np.ndarray
        3D pointmap in PyTorch3D convention, shape (H, W, 3).
    seed : int, optional
        Random seed for reproducibility. Default: 42.

    Returns
    -------
    list of dict
        List of outputs, each containing:
        - gaussian: List with raw Gaussian object (before layout transform)
        - rotation: Layout decoder rotation quaternion (local-to-camera)
        - translation: Layout decoder translation vector
        - scale: Layout decoder scale factor
        - mesh: Extracted mesh (if available)
        - decoder_input_slat: SLAT tokens for caching

    Notes
    -----
    The raw Gaussians are in canonical/local frame. Use make_scene() to apply
    the layout transformation and combine multiple objects.

    Examples
    --------
    >>> outputs = run_inference_on_masks(inference, image, masks, pointmap)
    >>> len(outputs)
    3  # One output per mask
    >>> outputs[0]['rotation'].shape
    torch.Size([1, 4])
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
        print(
            f"    Raw Gaussian xyz range: "
            f"[{output['gaussian'][0].get_xyz.min().item():.3f}, "
            f"{output['gaussian'][0].get_xyz.max().item():.3f}]"
        )

        # Print mesh info if available
        if "mesh" in output and output["mesh"] is not None:
            mesh = output["mesh"][0]
            print(f"    Mesh - vertices: {mesh.vertices.shape[0]}, faces: {mesh.faces.shape[0]}")

        outputs.append(output)

    return outputs


def compute_and_cache_frame_tokens(
    args: Any,
    paths: Dict[str, Any],
    frame_index: int,
    inference: "Inference",
    tokens_dir: str,
) -> str:
    """
    Compute tokens for a single frame and cache them.

    This runs full SAM3D inference on the frame and saves the results
    in the same format as demo.py.

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
        Directory where cached tokens will be stored.

    Returns
    -------
    str
        Path to the cached file.
    """
    from .depth import load_and_process_depth, transform_to_pytorch3d_convention
    from .io_utils import load_image, load_masks
    from .tokens import save_tokens

    # Load frame's image and masks
    image_path = os.path.join(paths["frames_path"], paths["image_names"][frame_index])
    mask_path = os.path.join(paths["masks_path"], paths["mask_names"][frame_index])

    image = load_image(image_path)
    image = image[..., :3]
    H, W, _ = image.shape

    masks = load_masks(mask_path)
    if args.object_index is not None:
        masks = [masks[args.object_index]]

    print(f"    Loaded image {image.shape}, {len(masks)} masks")

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

    # Transform to PyTorch3D convention for inference
    pointmap = transform_to_pytorch3d_convention(pointmap)

    # Run inference
    outputs = run_inference_on_masks(inference, image, masks, pointmap, seed=args.seed)

    # Build cache filename (matching demo.py format)
    cache_parts = [args.scene_name, f"f{frame_index}"]
    if args.object_index is not None:
        cache_parts.append(f"obj{args.object_index}")
    if args.background:
        cache_parts.append("bg")
    cache_scene_name = "_".join(cache_parts)

    # Cache results with frame and object index information
    os.makedirs(tokens_dir, exist_ok=True)
    if args.object_index is not None:
        # Processing single object
        object_indices = [args.object_index]
    else:
        # Processing all objects
        object_indices = list(range(len(outputs)))
    save_tokens(tokens_dir, cache_scene_name, outputs, frame_index, object_indices)

    cache_file = os.path.join(tokens_dir, f"{cache_scene_name}_sam3d_results.npz")
    return cache_file


def ensure_all_frames_have_tokens(
    args: Any,
    paths: Dict[str, Any],
    frame_indices: List[int],
    inference: "Inference",
    tokens_dir: str,
) -> Dict[int, List[tuple]]:
    """
    Ensure all requested frames have cached tokens.

    For frames without cached tokens, compute and cache them.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    paths : dict
        Dataset paths.
    frame_indices : list of int
        List of frame indices that should have tokens.
    inference : Inference
        Inference pipeline.
    tokens_dir : str
        Directory where cached tokens are stored.

    Returns
    -------
    dict
        tokens_by_object dictionary with all requested frames.
    """
    from .tokens import load_all_frame_tokens, load_decoder_inputs_from_cache

    # First, load existing tokens
    tokens_by_object = load_all_frame_tokens(
        tokens_dir, args.scene_name, args.object_index, args.background
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

        cache_file = compute_and_cache_frame_tokens(args, paths, frame_index, inference, tokens_dir)

        # Load the newly cached tokens and add to tokens_by_object
        decoder_inputs = load_decoder_inputs_from_cache(cache_file)
        for obj_idx, decoder_input in enumerate(decoder_inputs):
            if obj_idx not in tokens_by_object:
                tokens_by_object[obj_idx] = []
            tokens_by_object[obj_idx].append((frame_index, decoder_input))

        print(f"    Cached to {os.path.basename(cache_file)}")

    # Re-sort by frame index
    for obj_idx in tokens_by_object:
        tokens_by_object[obj_idx].sort(key=lambda x: x[0])

    print(f"\nNow have tokens for {len(existing_frame_indices | set(missing_frames))} frames")

    return tokens_by_object


__all__ = [
    "run_inference_on_masks",
    "compute_and_cache_frame_tokens",
    "ensure_all_frames_have_tokens",
]

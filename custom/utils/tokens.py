"""
SLAT token caching and manipulation utilities.

This module provides functions for saving, loading, and manipulating
SLAT (Sparse Latent Transformer) tokens, enabling efficient caching
of inference results and token-space operations like averaging.
"""

from __future__ import annotations

import glob
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import torch

if TYPE_CHECKING:
    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp


def save_tokens(
    tokens_path: str,
    scene_name: str,
    outputs: List[Dict[str, Any]],
    frame_index: int,
    object_indices: List[int],
) -> None:
    """
    Save inference results to cache.

    Stores the essential data needed to reconstruct the Gaussian scene:
    - rotation: Object rotation quaternion
    - translation: Object translation vector
    - scale: Object scale factor
    - decoder_input_coords: Sparse 3D coordinates for decoder
    - decoder_input_slat: SLAT latent features for decoder
    - frame_index: Index of the frame this data comes from
    - object_index: Index of the object in the original mask list

    Note: Gaussians are NOT saved as they can be recomputed from the decoder inputs.

    Parameters
    ----------
    tokens_path : str
        Directory to save the cache file.
    scene_name : str
        Name for the cache file (without extension).
    outputs : list of dict
        List of output dictionaries from inference.
    frame_index : int
        Frame index this data comes from.
    object_indices : list of int
        List of object indices corresponding to each output. Must have the same
        length as outputs. Each index represents the object's position in the
        original mask list.

    Raises
    ------
    ValueError
        If outputs and object_indices have different lengths.

    Examples
    --------
    >>> save_tokens("/cache", "scene_f0", outputs, frame_index=0, object_indices=[0, 1])
    Cached results saved to /cache/scene_f0_sam3d_results.npz
    """
    if len(outputs) != len(object_indices):
        raise ValueError(
            f"outputs and object_indices must have same length, "
            f"got {len(outputs)} and {len(object_indices)}"
        )

    cache_file = os.path.join(tokens_path, f"{scene_name}_sam3d_results.npz")

    # Extract and serialize the necessary data for each output
    cached_data = []
    for i, output in enumerate(outputs):
        # Extract pose data (gaussians can be recomputed from decoder inputs)
        output_data = {
            "rotation": output["rotation"].cpu().numpy(),
            "translation": output["translation"].cpu().numpy(),
            "scale": output["scale"].cpu().numpy(),
            # Store frame and object indices for tracking
            "frame_index": frame_index,
            "object_index": object_indices[i],
        }

        # Save decoder inputs if available (for re-running decoder)
        if "decoder_input_coords" in output and "decoder_input_slat" in output:
            output_data["decoder_input_coords"] = output["decoder_input_coords"].cpu().numpy()
            # For SparseTensor slat, we need to save its features and coords
            slat = output["decoder_input_slat"]
            output_data["decoder_input_slat_feats"] = slat.feats.cpu().numpy()
            output_data["decoder_input_slat_coords"] = slat.coords.cpu().numpy()

        cached_data.append(output_data)

    # Save as numpy archive
    np.savez(
        cache_file,
        cached_data=np.array(cached_data, dtype=object),
        num_objects=len(outputs),
        frame_index=frame_index,  # Also save at top level for quick access
    )
    print(f"Cached results saved to {cache_file}")


def load_decoder_inputs_from_cache(cache_file: str) -> List[Dict[str, Any]]:
    """
    Load decoder inputs (SLAT tokens) and pose from a cached results file.

    Parameters
    ----------
    cache_file : str
        Path to the .npz cache file created by demo.py.

    Returns
    -------
    list of dict
        List of decoder inputs, one per object. Each dict contains:
        - object_index: Index of the object in the original mask list
        - frame_index: Index of the frame this data comes from
        - decoder_input_slat: SparseTensor with SLAT latent features
        - rotation: Layout rotation tensor
        - translation: Layout translation tensor
        - scale: Layout scale tensor

    Raises
    ------
    FileNotFoundError
        If the cache file does not exist.

    Examples
    --------
    >>> decoder_inputs = load_decoder_inputs_from_cache("cache/scene_f0_sam3d_results.npz")
    >>> len(decoder_inputs)
    2  # Two objects in the scene
    >>> decoder_inputs[0]['rotation'].shape
    torch.Size([1, 4])
    """
    # Import here to avoid circular imports
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp

    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Cache file not found: {cache_file}")

    cached_archive = np.load(cache_file, allow_pickle=True)
    cached_data = cached_archive["cached_data"]

    decoder_inputs = []
    for i, data in enumerate(cached_data):

        data = data.item() if hasattr(data, "item") else data

        slat_feats = torch.from_numpy(data["decoder_input_slat_feats"]).cuda()
        slat_coords = torch.from_numpy(data["decoder_input_slat_coords"]).cuda()

        # Reconstruct SparseTensor for slat
        slat = sp.SparseTensor(
            coords=slat_coords,
            feats=slat_feats,
        ).cuda()

        # Load object_index and frame_index from data if available
        object_index = data["object_index"]
        frame_index = data["frame_index"]

        decoder_input = {
            "object_index": object_index,
            "frame_index": frame_index,
            "decoder_input_slat": slat,
            "rotation": torch.from_numpy(data["rotation"]).cuda(),
            "translation": torch.from_numpy(data["translation"]).cuda(),
            "scale": torch.from_numpy(data["scale"]).cuda(),
        }

        decoder_inputs.append(decoder_input)

    return decoder_inputs


def load_all_frame_tokens(
    tokens_dir: str,
    scene_name: str,
    object_index: Optional[int] = None,
    with_background: bool = False,
) -> Dict[int, List[Tuple[int, Dict[str, Any]]]]:
    """
    Load SLAT tokens from all available frames for a scene.

    Parameters
    ----------
    tokens_dir : str
        Directory containing cached token files.
    scene_name : str
        Name of the scene.
    object_index : int, optional
        If provided, only load tokens for this specific object.
    with_background : bool, optional
        Whether to look for files with background suffix. Default: False.

    Returns
    -------
    dict
        Dictionary mapping object_index -> list of (frame_index, decoder_input) tuples.
        Each decoder_input is a dict with SLAT tokens and pose parameters.

    Examples
    --------
    >>> tokens = load_all_frame_tokens("/cache", "my_scene")
    >>> tokens.keys()
    dict_keys([0, 1])  # Two objects
    >>> len(tokens[0])
    24  # 24 frames for object 0
    """
    # Find all cache files for this scene
    cache_pattern = f"{scene_name}_f*"
    if object_index is not None:
        cache_pattern += f"_obj{object_index}"
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
    tokens_by_object: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}

    for cache_file in cache_files:
        # Extract frame index from filename
        basename = os.path.basename(cache_file)
        # Pattern: {scene_name}_f{frame_idx}_...
        parts = basename.split("_")
        frame_part = [p for p in parts if p.startswith("f") and p[1:].isdigit()][0]
        frame_idx = int(frame_part[1:])

        print(f"  Loading frame {frame_idx} from {basename}")

        decoder_inputs = load_decoder_inputs_from_cache(cache_file)

        for obj_idx, decoder_input in enumerate(decoder_inputs):
            if obj_idx not in tokens_by_object:
                tokens_by_object[obj_idx] = []
            tokens_by_object[obj_idx].append((frame_idx, decoder_input))

    # Sort by frame index
    for obj_idx in tokens_by_object:
        tokens_by_object[obj_idx].sort(key=lambda x: x[0])

    return tokens_by_object


def average_slat_tokens(
    tokens_list: List[Tuple[int, Dict[str, Any]]],
    weights: Optional[torch.Tensor] = None,
) -> "sp.SparseTensor":
    """
    Average SLAT tokens across multiple frames with optional weighting.

    This function computes a weighted average of SLAT features across frames.
    Coordinates that appear in multiple frames are averaged; coordinates that
    only appear in some frames use only the available features.

    Parameters
    ----------
    tokens_list : list of tuples
        List of (frame_index, decoder_input) tuples.
    weights : torch.Tensor, optional
        Per-frame weights, shape (num_frames,). If None, uses uniform weights.

    Returns
    -------
    sp.SparseTensor
        Averaged SLAT tokens as a SparseTensor.

    Examples
    --------
    >>> avg_slat = average_slat_tokens(tokens_list)
    >>> avg_slat.feats.shape
    torch.Size([10000, 256])  # 10000 averaged tokens, 256 feature dim
    """
    # Import here to avoid circular imports
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp

    if len(tokens_list) == 1:
        return tokens_list[0][1]["decoder_input_slat"]

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
        coords = decoder_input["decoder_input_slat"].coords
        feats = decoder_input["decoder_input_slat"].feats
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

    avg_coords = torch.tensor(averaged_coords, dtype=torch.int32, device="cuda")
    avg_feats = torch.stack(averaged_feats, dim=0)

    print(f"    Averaged result: {avg_coords.shape[0]} tokens, features shape {avg_feats.shape}")

    avg_slat = sp.SparseTensor(
        coords=avg_coords,
        feats=avg_feats,
    ).cuda()

    return avg_slat


def apply_median_scale_to_tokens(
    tokens_by_object: Dict[int, List[Tuple[int, Dict[str, Any]]]]
) -> Dict[int, List[Tuple[int, Dict[str, Any]]]]:
    """
    Apply median scale normalization to tokens.

    For each object, computes the median of all per-frame predicted scales
    and replaces each frame's scale with that median. This provides more
    consistent object sizing across the sequence.

    When changing scale, the translation (z-depth) is also adjusted to maintain
    constant projected size in the image. The relationship is:
        z_new = z_old * (s_new / s_old)

    Parameters
    ----------
    tokens_by_object : dict
        Dictionary mapping object_index -> list of (frame_index, decoder_input) tuples.

    Returns
    -------
    dict
        Modified tokens_by_object with median scales applied (modifies in place).

    Notes
    -----
    This function modifies the input dictionary in place and also returns it.
    """
    for obj_idx, tokens_list in tokens_by_object.items():
        if len(tokens_list) == 0:
            continue

        # Collect all scales for this object
        scales = []
        for frame_idx, decoder_input in tokens_list:
            scale = decoder_input["scale"]
            if isinstance(scale, torch.Tensor):
                scales.append(scale.cpu().numpy())
            else:
                scales.append(np.array(scale))

        scales = np.array(scales)  # Shape: (num_frames, 3) or (num_frames,)

        # Compute median scale
        median_scale = np.median(scales, axis=0)

        print(f"  Object {obj_idx}: median scale = {median_scale}")
        print(f"    Scale range: min={scales.min(axis=0)}, max={scales.max(axis=0)}")

        # Apply median scale to all frames and adjust translation accordingly
        for i, (frame_idx, decoder_input) in enumerate(tokens_list):
            # Get original scale (scalar value - flatten and use first component)
            old_scale = scales[i].flatten()
            old_scale_scalar = float(old_scale[0])  # Assume uniform scale

            new_scale_flat = (
                median_scale.flatten()
                if hasattr(median_scale, "flatten")
                else np.array([median_scale]).flatten()
            )
            new_scale_scalar = float(new_scale_flat[0])

            # Avoid division by zero
            if abs(old_scale_scalar) < 1e-8:
                print(f"    Warning: Frame {frame_idx} has near-zero scale, skipping depth adjustment")
                scale_ratio = 1.0
            else:
                scale_ratio = new_scale_scalar / old_scale_scalar

            # Adjust translation z-component to maintain constant projected size
            # z_new = z_old * (s_new / s_old)
            translation = decoder_input["translation"]
            if isinstance(translation, torch.Tensor):
                old_z = (
                    translation[..., 2].item() if translation.dim() > 1 else translation[2].item()
                )
                new_z = old_z * scale_ratio
                if translation.dim() > 1:
                    translation[..., 2] = new_z
                else:
                    translation[2] = new_z
            else:
                old_z = float(np.array(translation).flatten()[2])
                new_z = old_z * scale_ratio
                translation_arr = np.array(translation)
                translation_arr.flat[2] = new_z
                decoder_input["translation"] = translation_arr.reshape(np.array(translation).shape)

            # Apply median scale
            if isinstance(decoder_input["scale"], torch.Tensor):
                decoder_input["scale"] = torch.tensor(
                    median_scale,
                    dtype=decoder_input["scale"].dtype,
                    device=decoder_input["scale"].device,
                )
            else:
                decoder_input["scale"] = median_scale

            if i == 0:  # Print adjustment info for first frame only
                print(
                    f"    Example (frame {frame_idx}): scale_ratio={scale_ratio:.4f}, "
                    f"z: {old_z:.4f} -> {new_z:.4f}"
                )

    return tokens_by_object


def redecode_slat(
    pipeline: Any,
    slat: "sp.SparseTensor",
    formats: List[str] = ["gaussian", "mesh"],
) -> Dict[str, Any]:
    """
    Re-run the decoder forward pass using saved SLAT tokens.

    Parameters
    ----------
    pipeline : Pipeline
        The SAM3D pipeline with decode_slat method.
    slat : sp.SparseTensor
        SLAT tokens to decode.
    formats : list of str, optional
        Output formats to decode. Default: ["gaussian", "mesh"].

    Returns
    -------
    dict
        Decoded outputs with keys for each requested format.

    Examples
    --------
    >>> outputs = redecode_slat(pipeline, slat, formats=["gaussian"])
    >>> gs = outputs["gaussian"][0]
    >>> gs.get_xyz.shape
    torch.Size([10000, 3])
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


def find_best_canon_frame(
    tokens_list: List[Tuple[int, Dict[str, Any]]],
    dataset_path: str,
    scene_name: str,
    dataset_type: str,
    obj_idx: int,
) -> Tuple[int, int]:
    """
    Find the frame with the highest mask coverage for an object.

    Parameters
    ----------
    tokens_list : list
        List of (frame_index, decoder_input) tuples.
    dataset_path : str
        Root path to the dataset.
    scene_name : str
        Name of the scene.
    dataset_type : str
        Either "kubric4d" or "davis".
    obj_idx : int
        Object index to find the best frame for.

    Returns
    -------
    tuple
        (best_frame_index, best_mask_area) where best_frame_index is the
        frame with maximum mask area for this object.
    """
    from .io_utils import load_masks

    best_frame = tokens_list[0][0]  # Default to first available frame
    best_mask_area = 0

    for frame_idx, decoder_input in tokens_list:
        mask_area = 0

        if dataset_type == "kubric4d":
            frames_path = os.path.join(dataset_path, scene_name, "frames_p0_v0")
            mask_files = sorted(
                [
                    f
                    for f in os.listdir(frames_path)
                    if f.startswith("segmentation_") and f.endswith(".png")
                ]
            )

            if frame_idx < len(mask_files):
                mask_path = os.path.join(frames_path, mask_files[frame_idx])
                masks = load_masks(mask_path)

                if obj_idx < len(masks):
                    mask = masks[obj_idx]
                    mask_area = np.count_nonzero(mask)

        elif dataset_type == "davis":
            masks_path = os.path.join(dataset_path, "Annotations", "Full-Resolution", scene_name)
            mask_files = sorted([f for f in os.listdir(masks_path) if f.endswith(".png")])

            if frame_idx < len(mask_files):
                mask_path = os.path.join(masks_path, mask_files[frame_idx])
                masks = load_masks(mask_path)

                if obj_idx < len(masks):
                    mask = masks[obj_idx]
                    mask_area = np.count_nonzero(mask)

        if mask_area > best_mask_area:
            best_mask_area = mask_area
            best_frame = frame_idx

    return best_frame, best_mask_area


def compute_frame_weights_from_masks(
    tokens_list: List[Tuple[int, Dict[str, Any]]],
    dataset_path: str,
    scene_name: str,
    dataset_type: str,
    obj_idx: int,
) -> torch.Tensor:
    """
    Compute frame weights based on object mask visibility.

    Frames with larger mask areas (more visible object) get higher weights.

    Parameters
    ----------
    tokens_list : list
        List of (frame_index, decoder_input) tuples.
    dataset_path : str
        Root path to the dataset.
    scene_name : str
        Name of the scene.
    dataset_type : str
        Either "kubric4d" or "davis".
    obj_idx : int
        Object index to compute weights for.

    Returns
    -------
    torch.Tensor
        Weights for each frame, shape (num_frames,), normalized to sum to 1.0.
    """
    from .io_utils import load_masks

    weights = []

    for frame_idx, decoder_input in tokens_list:

        if dataset_type == "kubric4d":
            frames_path = os.path.join(dataset_path, scene_name, "frames_p0_v0")
            mask_files = sorted(
                [
                    f
                    for f in os.listdir(frames_path)
                    if f.startswith("segmentation_") and f.endswith(".png")
                ]
            )

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
            raise ValueError(f"Unsupported dataset type for mask-based weights: {dataset_type}")

    weights = torch.tensor(weights, dtype=torch.float32)

    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = torch.ones_like(weights) / len(weights)

    return weights


def compute_frame_weights_from_error(
    tokens_by_object: Dict[int, List[Tuple[int, Dict[str, Any]]]],
    obj_idx: int,
    args: Any,
    paths: Dict[str, Any],
    inference: Any,
    tokens_dir: str,
) -> torch.Tensor:
    """
    Compute frame weights based on per-frame rendering error in the masked region.

    Frames with lower reconstruction error get higher weights.
    Error is computed as masked MSE between rendered and ground truth image.

    Parameters
    ----------
    tokens_by_object : dict
        Dictionary mapping obj_idx -> list of (frame_idx, decoder_input) tuples.
    obj_idx : int
        Object index to compute weights for.
    args : argparse.Namespace
        Command line arguments.
    paths : dict
        Dataset paths.
    inference : Inference
        Inference pipeline.
    tokens_dir : str
        Directory where cached tokens are stored.

    Returns
    -------
    torch.Tensor
        Weights for each frame, shape (num_frames,), normalized to sum to 1.0.
        Higher weight = lower error = better reconstruction.
    """
    import os

    from inference import make_scene

    from .depth import load_and_process_depth
    from .gaussian import transform_scene_to_r3_convention
    from .io_utils import load_image, load_masks
    from .rendering import render_gaussians_to_image

    tokens_list = tokens_by_object[obj_idx]
    errors = []
    pipeline = inference._pipeline

    print(f"    Computing error-based weights for object {obj_idx}...")

    for frame_idx, decoder_input in tokens_list:

        # Load frame's image and mask
        image_path = os.path.join(paths["frames_path"], paths["image_names"][frame_idx])
        mask_path = os.path.join(paths["masks_path"], paths["mask_names"][frame_idx])

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

        # Decode this frame's tokens to get Gaussian
        slat = decoder_input["decoder_input_slat"]
        decoded = redecode_slat(pipeline, slat, formats=["gaussian"])

        # Build output and render
        output = {
            "gaussian": decoded["gaussian"],
            "rotation": decoder_input["rotation"],
            "translation": decoder_input["translation"],
            "scale": decoder_input["scale"],
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


__all__ = [
    "save_tokens",
    "load_decoder_inputs_from_cache",
    "load_all_frame_tokens",
    "average_slat_tokens",
    "apply_median_scale_to_tokens",
    "redecode_slat",
    "find_best_canon_frame",
    "compute_frame_weights_from_masks",
    "compute_frame_weights_from_error",
]

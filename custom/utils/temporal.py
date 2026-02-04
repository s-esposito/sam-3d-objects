"""
Temporal point cloud utilities.

This module provides functions for accumulating and saving temporal
point cloud sequences from Gaussian splatting scenes.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Dict, List

import numpy as np

if TYPE_CHECKING:
    from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model import (
        Gaussian,
    )


def save_temporal_point_cloud(
    output_dir: str,
    scene_name: str,
    suffix: str = "",
) -> Dict:
    """
    Initialize a temporal point cloud storage structure.

    Returns a dictionary that can accumulate per-frame point clouds,
    then be saved to disk.

    Parameters
    ----------
    output_dir : str
        Directory to save the point cloud.
    scene_name : str
        Name of the scene.
    suffix : str, optional
        Suffix for output filename. Default: "".

    Returns
    -------
    dict
        Storage structure with 'frames' list and metadata.

    Examples
    --------
    >>> storage = save_temporal_point_cloud("/output", "my_scene", "_refined")
    >>> storage['frames']
    []
    """
    return {
        "output_dir": output_dir,
        "scene_name": scene_name,
        "suffix": suffix,
        "frames": [],  # List of frame data dicts
    }


def add_frame_to_temporal_point_cloud(
    storage: Dict,
    frame_idx: int,
    scene_gs: "Gaussian",
) -> None:
    """
    Add a frame's Gaussian data to the temporal point cloud storage.

    Parameters
    ----------
    storage : dict
        Storage structure from save_temporal_point_cloud.
    frame_idx : int
        Frame index.
    scene_gs : Gaussian
        Gaussian scene object (already in R3/world convention).

    Notes
    -----
    Extracts xyz coordinates, colors (from SH features), scales, and opacities
    from the Gaussian scene and stores them in the accumulator.
    """
    # Extract point cloud data from Gaussian
    xyz = scene_gs.get_xyz.detach().cpu().numpy()  # (N, 3)

    # Get colors from SH features (assuming DC component only for visualization)
    features = scene_gs.get_features.detach().cpu().numpy()  # (N, K, 3) or (N, 3)
    if features.ndim == 3:
        # Use DC component (first SH band)
        sh_dc = features[:, 0, :]  # (N, 3)
    else:
        sh_dc = features
    # Convert SH to RGB
    rgb = sh_dc * 0.28209479177387814 + 0.5  # SH2RGB
    rgb = np.clip(rgb, 0, 1)

    # Get scales and opacities for optional filtering
    scales = scene_gs.get_scaling.detach().cpu().numpy()  # (N, 3)
    opacities = scene_gs.get_opacity.detach().cpu().numpy()  # (N, 1)

    storage["frames"].append(
        {
            "frame_idx": frame_idx,
            "xyz": xyz.astype(np.float32),
            "rgb": rgb.astype(np.float32),
            "scales": scales.astype(np.float32),
            "opacities": opacities.astype(np.float32),
        }
    )


def finalize_temporal_point_cloud(storage: Dict) -> str:
    """
    Save the temporal point cloud to disk.

    Saves:
    - A combined .npz file with all frames
    - A metadata JSON file

    Parameters
    ----------
    storage : dict
        Storage structure with accumulated frames.

    Returns
    -------
    str
        Path to the output directory.

    Examples
    --------
    >>> pc_dir = finalize_temporal_point_cloud(storage)
    Saved combined temporal point cloud to /output/point_clouds/my_scene_temporal.npz
    Saved 24 frames to /output/point_clouds
      Total points: 240000
    """
    output_dir = storage["output_dir"]
    scene_name = storage["scene_name"]
    suffix = storage["suffix"]

    # Create point cloud output directory
    pc_dir = os.path.join(output_dir, "point_clouds")
    os.makedirs(pc_dir, exist_ok=True)

    # Save combined file
    if len(storage["frames"]) > 0:
        combined_filename = f"{scene_name}_temporal{suffix}.npz"
        combined_filepath = os.path.join(pc_dir, combined_filename)

        # Combine all frames into arrays
        all_xyz = []
        all_rgb = []
        all_frame_ids = []
        all_scales = []
        all_opacities = []

        for frame_data in storage["frames"]:
            n_points = len(frame_data["xyz"])
            all_xyz.append(frame_data["xyz"])
            all_rgb.append(frame_data["rgb"])
            all_frame_ids.append(np.full(n_points, frame_data["frame_idx"], dtype=np.int32))
            all_scales.append(frame_data["scales"])
            all_opacities.append(frame_data["opacities"])

        np.savez_compressed(
            combined_filepath,
            xyz=np.concatenate(all_xyz, axis=0),
            rgb=np.concatenate(all_rgb, axis=0),
            frame_ids=np.concatenate(all_frame_ids, axis=0),
            scales=np.concatenate(all_scales, axis=0),
            opacities=np.concatenate(all_opacities, axis=0),
            num_frames=len(storage["frames"]),
            frame_indices=np.array([f["frame_idx"] for f in storage["frames"]], dtype=np.int32),
        )
        print(f"Saved combined temporal point cloud to {combined_filepath}")

    # Save metadata
    metadata = {
        "scene_name": scene_name,
        "suffix": suffix,
        "num_frames": len(storage["frames"]),
        "frame_indices": [f["frame_idx"] for f in storage["frames"]],
        "total_points": sum(len(f["xyz"]) for f in storage["frames"]),
    }

    metadata_path = os.path.join(pc_dir, f"{scene_name}_temporal{suffix}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {len(storage['frames'])} frames to {pc_dir}")
    print(f"  Total points: {metadata['total_points']}")

    return pc_dir


__all__ = [
    "save_temporal_point_cloud",
    "add_frame_to_temporal_point_cloud",
    "finalize_temporal_point_cloud",
]

"""
File I/O utilities for the SAM3D-Objects pipeline.

This module provides functions for loading images, masks, and setting up
dataset paths, as well as saving various output formats.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    import torch


def load_image(path: str, to_uint8: bool = True) -> np.ndarray:
    """
    Load an image from disk.

    Parameters
    ----------
    path : str
        Path to the image file. Supports PNG, JPG, TIFF, etc.
    to_uint8 : bool, optional
        Whether to convert the image to uint8 dtype. Default: True.
        Set to False for depth maps or floating-point images.

    Returns
    -------
    np.ndarray
        Loaded image as a NumPy array.

    Examples
    --------
    >>> img = load_image("image.png")
    >>> img.dtype
    dtype('uint8')
    >>> depth = load_image("depth.tiff", to_uint8=False)
    >>> depth.dtype
    dtype('float32')
    """
    image = Image.open(path)
    image = np.array(image)
    if to_uint8:
        image = image.astype(np.uint8)
    return image


def load_masks(
    mask_path: str,
    indices_list: Optional[List[int]] = None,
) -> List[np.ndarray]:
    """
    Load segmentation masks from a file.

    Parses a segmentation mask image where each unique pixel value
    represents a different object instance (0 = background).

    Parameters
    ----------
    mask_path : str
        Path to the segmentation mask image file.
    indices_list : list of int, optional
        If provided, only load masks for objects with these IDs.
        IDs correspond to pixel values in the mask image.

    Returns
    -------
    list of np.ndarray
        List of boolean mask arrays, one per detected object.
        Each mask has shape (H, W) with True where the object is present.

    Notes
    -----
    - Pixel value 0 is always treated as background and skipped
    - Objects are returned in order of their pixel value IDs

    Examples
    --------
    >>> masks = load_masks("segmentation.png")
    >>> len(masks)
    3  # If there are 3 objects in the scene
    >>> masks[0].shape
    (480, 640)
    >>> masks[0].dtype
    dtype('bool')
    """
    masks = []
    mask = load_image(mask_path)
    print(
        f"Loaded mask of shape: {mask.shape}, dtype: {mask.dtype}, "
        f"min: {mask.min()}, max: {mask.max()}, unique values: {np.unique(mask)}"
    )
    # Get unique object ids
    object_ids = np.unique(mask)
    for object_id in object_ids:
        if object_id == 0:
            continue  # skip background
        if indices_list is not None and object_id.item() not in indices_list:
            continue
        object_mask = mask == object_id
        masks.append(object_mask)
    return masks


def setup_paths(
    dataset_path: str,
    scene_name: str,
    dataset_type: str,
) -> dict:
    """
    Setup and validate all necessary paths for a dataset.

    Parameters
    ----------
    dataset_path : str
        Root path to the dataset.
    scene_name : str
        Name of the scene to process.
    dataset_type : str
        Either "kubric4d" or "davis".

    Returns
    -------
    dict
        Dictionary containing all paths and file lists:
        - data_path: Root path to scene data
        - frames_path: Path to frame images
        - masks_path: Path to segmentation masks
        - image_names: Sorted list of image filenames
        - mask_names: Sorted list of mask filenames
        - depth_names: Sorted list of depth filenames (empty for DAVIS)
        - dataset_type: The dataset type string

    Raises
    ------
    ValueError
        If dataset_type is not "kubric4d" or "davis".

    Examples
    --------
    >>> paths = setup_paths("/data/kubric4d", "scene_001", "kubric4d")
    >>> paths['frames_path']
    '/data/kubric4d/scene_001/frames_p0_v0'
    >>> len(paths['image_names'])
    24  # Number of frames in the scene
    """
    if dataset_type == "kubric4d":
        data_path = os.path.join(dataset_path, scene_name)
        frames_path = os.path.join(data_path, "frames_p0_v0")  # viewpoint 0

        # Get sorted file lists
        image_names = sorted(
            [f for f in os.listdir(frames_path) if f.startswith("rgba_") and f.endswith(".png")]
        )
        mask_names = sorted(
            [
                f
                for f in os.listdir(frames_path)
                if f.startswith("segmentation_") and f.endswith(".png")
            ]
        )
        depth_names = sorted(
            [f for f in os.listdir(frames_path) if f.startswith("depth_") and f.endswith(".tiff")]
        )

        return {
            "data_path": data_path,
            "frames_path": frames_path,
            "masks_path": frames_path,  # Same as frames path for Kubric4D
            "image_names": image_names,
            "mask_names": mask_names,
            "depth_names": depth_names,
            "dataset_type": "kubric4d",
        }

    elif dataset_type == "davis":
        frames_path = os.path.join(dataset_path, "JPEGImages", "Full-Resolution", scene_name)
        masks_path = os.path.join(dataset_path, "Annotations", "Full-Resolution", scene_name)

        # Get sorted file lists
        image_names = sorted([f for f in os.listdir(frames_path) if f.endswith(".jpg")])
        mask_names = sorted([f for f in os.listdir(masks_path) if f.endswith(".png")])

        return {
            "data_path": dataset_path,
            "frames_path": frames_path,
            "masks_path": masks_path,
            "image_names": image_names,
            "mask_names": mask_names,
            "depth_names": [],  # DAVIS doesn't have depth files, uses MoGe
            "dataset_type": "davis",
        }
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Use 'kubric4d' or 'davis'")


def get_cache_filename(
    scene_name: str,
    frame_index: int,
    object_index: Optional[int] = None,
    with_background: bool = False,
) -> Tuple[str, str]:
    """
    Build the cache filename based on configuration.

    Parameters
    ----------
    scene_name : str
        Name of the scene.
    frame_index : int
        Index of the frame.
    object_index : int, optional
        Index of specific object (None for all objects).
    with_background : bool, optional
        Whether background is included. Default: False.

    Returns
    -------
    tuple
        (filename, scene_name_with_suffix) where:
        - filename: Full cache filename (e.g., "scene_f0_obj1_bg_sam3d_results.npz")
        - scene_name_with_suffix: Scene name component without extension

    Examples
    --------
    >>> get_cache_filename("my_scene", 5, object_index=2)
    ('my_scene_f5_obj2_sam3d_results.npz', 'my_scene_f5_obj2')
    >>> get_cache_filename("scene", 0, with_background=True)
    ('scene_f0_bg_sam3d_results.npz', 'scene_f0_bg')
    """
    cache_parts = [scene_name, f"f{frame_index}"]
    if object_index is not None:
        cache_parts.append(f"obj{object_index}")
    if with_background:
        cache_parts.append("bg")
    cache_scene_name = "_".join(cache_parts)
    return f"{cache_scene_name}_sam3d_results.npz", cache_scene_name


def save_mesh_to_obj(mesh: "torch.Tensor", output_path: str) -> None:
    """
    Save a mesh object to an OBJ file.

    Parameters
    ----------
    mesh : MeshExtractResult or similar
        Mesh object with the following attributes:
        - vertices or verts: Tensor of shape (N, 3)
        - faces: Tensor of shape (M, 3)
        - vertex_attrs (optional): Can be:
          - A tensor of shape (N, C) where C >= 3 (first 3 channels are RGB color)
          - A dict with 'color' key
          - None
        - vertex_colors (optional): Alternative to vertex_attrs
    output_path : str
        Path to save the OBJ file.

    Notes
    -----
    - OBJ files use 1-indexed vertices
    - Vertex colors are clamped to [0, 1] range
    - Colors are written in the "v x y z r g b" format

    Examples
    --------
    >>> save_mesh_to_obj(mesh, "output/model.obj")
    Saved mesh to output/model.obj (10000 vertices, 20000 faces)
    """
    # Handle both 'vertices' and 'verts' attribute names
    if hasattr(mesh, "vertices"):
        verts = mesh.vertices.cpu().numpy() if hasattr(mesh.vertices, "cpu") else mesh.vertices
    elif hasattr(mesh, "verts"):
        verts = mesh.verts.cpu().numpy() if hasattr(mesh.verts, "cpu") else mesh.verts
    else:
        raise AttributeError("Mesh object has no 'vertices' or 'verts' attribute")

    faces = mesh.faces.cpu().numpy() if hasattr(mesh.faces, "cpu") else mesh.faces

    # Check for vertex colors
    vertex_colors = None
    if hasattr(mesh, "vertex_attrs") and mesh.vertex_attrs is not None:
        va = mesh.vertex_attrs
        # vertex_attrs can be a tensor directly or a dict
        if isinstance(va, dict):
            if "color" in va:
                vc = va["color"]
                vertex_colors = vc.cpu().numpy() if hasattr(vc, "cpu") else vc
        elif hasattr(va, "cpu"):
            # It's a tensor - assume first 3 channels are RGB
            va_np = va.cpu().numpy()
            if va_np.shape[-1] >= 3:
                vertex_colors = va_np[..., :3]
        elif isinstance(va, np.ndarray):
            if va.shape[-1] >= 3:
                vertex_colors = va[..., :3]
    elif hasattr(mesh, "vertex_colors") and mesh.vertex_colors is not None:
        vertex_colors = (
            mesh.vertex_colors.cpu().numpy()
            if hasattr(mesh.vertex_colors, "cpu")
            else mesh.vertex_colors
        )

    with open(output_path, "w") as f:
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


__all__ = [
    "load_image",
    "load_masks",
    "setup_paths",
    "get_cache_filename",
    "save_mesh_to_obj",
]

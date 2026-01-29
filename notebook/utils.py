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


def save_mesh_to_obj(mesh, output_path):
    """
    Save a mesh object to an OBJ file.
    
    Parameters
    ----------
    mesh : MeshExtractResult or similar
        Mesh object with vertices, faces, and optionally vertex_attrs attributes.
        vertices should be (N, 3) tensor
        faces should be (M, 3) tensor
        vertex_attrs can be:
          - A tensor of shape (N, C) where C >= 3 (first 3 channels are RGB color)
          - A dict with 'color' key
          - None
    output_path : str
        Path to save the OBJ file
    """
    # Handle both 'vertices' and 'verts' attribute names
    if hasattr(mesh, 'vertices'):
        verts = mesh.vertices.cpu().numpy() if hasattr(mesh.vertices, 'cpu') else mesh.vertices
    elif hasattr(mesh, 'verts'):
        verts = mesh.verts.cpu().numpy() if hasattr(mesh.verts, 'cpu') else mesh.verts
    else:
        raise AttributeError("Mesh object has no 'vertices' or 'verts' attribute")
    
    faces = mesh.faces.cpu().numpy() if hasattr(mesh.faces, 'cpu') else mesh.faces
    
    # Check for vertex colors
    vertex_colors = None
    if hasattr(mesh, 'vertex_attrs') and mesh.vertex_attrs is not None:
        va = mesh.vertex_attrs
        # vertex_attrs can be a tensor directly or a dict
        if isinstance(va, dict):
            if 'color' in va:
                vc = va['color']
                vertex_colors = vc.cpu().numpy() if hasattr(vc, 'cpu') else vc
        elif hasattr(va, 'cpu'):
            # It's a tensor - assume first 3 channels are RGB
            va_np = va.cpu().numpy()
            if va_np.shape[-1] >= 3:
                vertex_colors = va_np[..., :3]
        elif isinstance(va, np.ndarray):
            if va.shape[-1] >= 3:
                vertex_colors = va[..., :3]
    elif hasattr(mesh, 'vertex_colors') and mesh.vertex_colors is not None:
        vertex_colors = mesh.vertex_colors.cpu().numpy() if hasattr(mesh.vertex_colors, 'cpu') else mesh.vertex_colors
    
    with open(output_path, 'w') as f:
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
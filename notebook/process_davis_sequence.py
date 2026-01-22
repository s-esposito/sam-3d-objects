#!/usr/bin/env python3
"""
Process DAVIS video sequences frame by frame.
Runs inference on every Nth frame and saves Gaussian splat outputs.
"""

import os
import sys
import time
import numpy as np
import torch
import argparse
import imageio
from pathlib import Path

# Set up environment
os.environ['LIDRA_SKIP_INIT'] = '1'

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from inference import Inference, make_scene
from utils import load_image, load_masks, depth_to_pointmap, create_gaussians_from_pointmap, create_gaussians_object, render_frame, join_gaussians
from pytorch3d.transforms import Transform3d, matrix_to_quaternion, quaternion_multiply
from pytorch3d.renderer import look_at_view_transform

# Set torch inference mode globally
torch.set_grad_enabled(False)


def process_davis_sequence(
    dataset_path: str,
    scene_name: str,
    output_dir: str,
    frame_step: int = 10,
    checkpoint_tag: str = "hf",
):
    """
    Process a DAVIS video sequence frame by frame.
    
    Parameters
    ----------
    dataset_path : str
        Path to DAVIS dataset root
    scene_name : str
        Name of the scene to process
    output_dir : str
        Directory to save outputs
    frame_step : int
        Process every Nth frame (default: 10)
    checkpoint_tag : str
        Checkpoint tag to use (default: "hf")
    """
    
    # Set up paths
    frames_path = os.path.join(dataset_path, "JPEGImages", "Full-Resolution", scene_name)
    masks_path = os.path.join(dataset_path, "Annotations", "Full-Resolution", scene_name)
    
    # Create output directory
    output_scene_dir = os.path.join(output_dir, scene_name)
    os.makedirs(output_scene_dir, exist_ok=True)
    
    # Get frame and mask lists
    image_names = sorted([f for f in os.listdir(frames_path) if f.endswith(".jpg")])
    mask_names = sorted([f for f in os.listdir(masks_path) if f.endswith(".png")])
    
    print(f"Found {len(image_names)} frames and {len(mask_names)} masks")
    print(f"Processing every {frame_step} frames")
    
    # Initialize inference pipeline
    print("Loading inference pipeline...")
    path = os.path.dirname(__file__)
    config_path = f"{path}/../checkpoints/{checkpoint_tag}/pipeline.yaml"
    inference = Inference(config_path, compile=False)
    depth_model = inference._pipeline.depth_model
    
    # Process frames
    frames_to_process = list(range(0, len(image_names), frame_step))
    print(f"Will process {len(frames_to_process)} frames: {frames_to_process}")
    
    for fid in frames_to_process:
        frame_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Processing frame {fid}: {image_names[fid]}")
        print(f"{'='*60}")
        
        # Load image
        image_path = os.path.join(frames_path, image_names[fid])
        image = load_image(image_path)
        image = image[..., :3]  # Drop alpha channel if present
        H, W, _ = image.shape
        
        # Load masks
        mask_path = os.path.join(masks_path, mask_names[fid])
        masks = load_masks(mask_path)
        print(f"Image shape: {image.shape}, {len(masks)} objects detected")
        
        # Run depth inference
        print("Running depth inference...")
        depth_start = time.time()
        
        loaded_image = inference._pipeline.image_to_float(image)
        loaded_image = torch.from_numpy(loaded_image)
        loaded_image_rgb = loaded_image.permute(2, 0, 1).contiguous()[:3]
        
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                depth_output = depth_model(loaded_image_rgb)
        
        depth_map_z = depth_output["depth"].cpu().numpy()
        valid_mask = depth_output["mask"].cpu().numpy()
        depth_map_z[~valid_mask] = 0.0
        
        # Get camera intrinsics
        intrinsics = depth_output["intrinsics"].cpu().numpy()
        fx = intrinsics[0, 0] * 1000.0
        fy = intrinsics[1, 1] * 1000.0
        cx = intrinsics[0, 2] * W
        cy = intrinsics[1, 2] * H
        
        K_matrix = np.eye(3)
        K_matrix[0, 0] = fx
        K_matrix[1, 1] = fy
        K_matrix[0, 2] = cx
        K_matrix[1, 2] = cy
        
        print(f"Depth inference: {time.time() - depth_start:.2f}s")
        
        # Get pointmap from depth
        pointmap = depth_to_pointmap(depth_map_z, K_matrix, valid_mask=valid_mask)
        
        # Get background Guassians
        
        background_mask = ~np.any(np.stack(masks, axis=0), axis=0)
        
        gaussians_bg = create_gaussians_from_pointmap(
            image=image[background_mask],
            pointmap=pointmap[background_mask],
            K=K_matrix,
        )
        
        # # Render and save background only scene for debugging
        # c2w = torch.eye(4)
        # K_render = torch.from_numpy(K_matrix).float().cuda()
        
        # rendered_bg_frame, rendered_bg_alpha = render_frame(
        #     gaussians_bg,
        #     c2w=c2w,
        #     K=K_render,
        #     w=W,
        #     h=H
        # )
        
        # output_bg_render_path = os.path.join(output_scene_dir, f"frame_{fid:05d}_background_render.png")
        # rendered_bg_frame_np = rendered_bg_frame.cpu().numpy()
        # rendered_bg_frame_uint8 = (np.clip(rendered_bg_frame_np, 0, 1) * 255).astype(np.uint8)
        # imageio.imwrite(output_bg_render_path, rendered_bg_frame_uint8)
        # print(f"Saved background render: {output_bg_render_path}")
        # exit(0)
        
        # Transform pointmap to PyTorch3D convention
        # Camera convention transformation (R3 -> PyTorch3D)
        r3_to_p3d_R, r3_to_p3d_T = look_at_view_transform(
            eye=np.array([[0, 0, -1]]),
            at=np.array([[0, 0, 0]]),
            up=np.array([[0, -1, 0]]),
        )
        
        # Inverse transform (PyTorch3D -> R3)
        p3d_to_r3_R = r3_to_p3d_R.transpose(1, 2)
        
        # Convert rotation matrix to numpy
        r3_to_p3d_R_np = r3_to_p3d_R.cpu().numpy()[0]  # (3, 3)
        
        # Apply rotation using numpy matrix multiplication
        pointmap_transformed = pointmap[valid_mask] @ r3_to_p3d_R_np.T
        pointmap[valid_mask] = pointmap_transformed
        
        # Convert to torch for inference
        pointmap_torch = torch.from_numpy(pointmap).float().cuda()
        
        # Run inference for each mask
        print(f"Running inference on {len(masks)} objects...")
        outputs = []
        for i, mask in enumerate(masks):
            obj_start = time.time()
            output = inference(image, mask, seed=42, pointmap=pointmap_torch)
            print(f"  Object {i+1}/{len(masks)}: {time.time() - obj_start:.2f}s")
            outputs.append(output)
        
        # Create scene
        print("Creating Gaussian scene...")
        scene_gs = make_scene(*outputs)
        
        # # save original PyTorch3D convention version
        # output_ply_p3d_path = os.path.join(output_scene_dir, f"frame_{fid:05d}_p3d.ply")
        # scene_gs.save_ply(output_ply_p3d_path)
        # print(f"Saved PyTorch3D convention: {output_ply_p3d_path}")
        
        # Transform back to R3 convention for rendering (matching notebook)
        print("Transforming to R3 convention...")
        xyz_unnormalized = scene_gs.get_xyz  # This applies: xyz * aabb[3:] + aabb[:3]
        quats = scene_gs.get_rotation  # (N, 4) in wxyz format
        
        p3d_to_r3_R = p3d_to_r3_R.to(device=scene_gs.get_xyz.device)
        camera_convention_transform = Transform3d(device=scene_gs.get_xyz.device).rotate(p3d_to_r3_R)
        xyz = camera_convention_transform.transform_points(xyz_unnormalized)
        
        # rotate
        rotation_quat = matrix_to_quaternion(p3d_to_r3_R)  # (1, 4) wxyz
        rotation_quat_expanded = rotation_quat.expand(quats.shape[0], -1)  # (N, 4)
        quats_transformed = quaternion_multiply(rotation_quat_expanded, quats)  # (N, 4)
        quats = quats_transformed # - scene_gs.rots_bias[None, :]
   
        # create new Gaussians object
        new_scene_gs = create_gaussians_object(
            xyz=xyz,
            features=scene_gs.get_features,
            scales=scene_gs.get_scaling,
            rots=quats,
            opacities=scene_gs.get_opacity,
        )
        
        # Save R3 convention version
        output_ply_r3_path = os.path.join(output_scene_dir, f"frame_{fid:05d}.ply")
        new_scene_gs.save_ply(output_ply_r3_path)
        print(f"Saved R3 convention: {output_ply_r3_path}")
        
        # Join background gaussians with new_scene_gs
        new_scene_gs = join_gaussians(gaussians_bg, new_scene_gs)
        
        # Render from a modified camera viewpoint
        # Move camera back (positive z in camera space = negative z in world) and shift right (+x)
        print("Rendering from camera viewpoint...")
        c2w = torch.eye(4)
        c2w_side = c2w.clone()
        c2w_side[0, 3] = 0.2   # Shift right by 0.5 units (x-axis)
        c2w_side[1, 3] = -0.2   # Shift up (y-axis)
        c2w_side[2, 3] = -0.5  # Move back by 0.5 units (negative z in world space)
    
        K_render = torch.from_numpy(K_matrix).float()
        
        rendered_frame, rendered_alpha = render_frame(
            new_scene_gs,
            c2w=c2w,
            K=K_render,
            w=W,
            h=H
        )
        
        rendered_frame_side, rendered_alpha = render_frame(
            new_scene_gs,
            c2w=c2w_side,
            K=K_render,
            w=W,
            h=H
        )
        
        # Combine both renders side by side (vertically)
        rendered_frame = torch.cat([rendered_frame, rendered_frame_side], dim=0)
        
        # Save rendered frame
        rendered_frame_np = rendered_frame.cpu().numpy()
        rendered_frame_uint8 = (np.clip(rendered_frame_np, 0, 1) * 255).astype(np.uint8)
        
        output_render_path = os.path.join(output_scene_dir, f"frame_{fid:05d}_render.png")
        imageio.imwrite(output_render_path, rendered_frame_uint8)
        print(f"Saved render: {output_render_path}")
                
        # Clean up GPU memory between frames
        print("Cleaning up memory...")
        del scene_gs, new_scene_gs, xyz, xyz_unnormalized
        del camera_convention_transform, p3d_to_r3_R
        del quats, quats_transformed, rotation_quat, rotation_quat_expanded
        del outputs, rendered_frame, rendered_alpha, rendered_frame_np, rendered_frame_uint8
        del depth_output, depth_map_z, valid_mask, intrinsics
        del pointmap, pointmap_transformed, pointmap_torch, r3_to_p3d_R_np
        del loaded_image, loaded_image_rgb, K_render
        torch.cuda.empty_cache()
        
        frame_time = time.time() - frame_start_time
        print(f"\nFrame {fid} total time: {frame_time:.2f}s")
        print(f"Estimated time remaining: {frame_time * (len(frames_to_process) - frames_to_process.index(fid) - 1) / 60:.1f} minutes")
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Outputs saved to: {output_scene_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Process DAVIS video sequences with Gaussian splatting"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/mnt/lustre/work/geiger/gwb987/data/DAVIS",
        help="Path to DAVIS dataset root",
    )
    parser.add_argument(
        "--scene-name",
        type=str,
        default="train",
        help="Name of the scene to process",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/davis",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=10,
        help="Process every Nth frame (default: 10)",
    )
    parser.add_argument(
        "--checkpoint-tag",
        type=str,
        default="hf",
        help="Checkpoint tag to use (default: hf)",
    )
    
    args = parser.parse_args()
    
    # Make output directory absolute if relative
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(os.path.dirname(__file__), args.output_dir)
    
    print(f"DAVIS Video Sequence Processing")
    print(f"Dataset: {args.dataset_path}")
    print(f"Scene: {args.scene_name}")
    print(f"Output: {args.output_dir}")
    print(f"Frame step: {args.frame_step}")
    print(f"Checkpoint: {args.checkpoint_tag}")
    print()
    
    process_davis_sequence(
        dataset_path=args.dataset_path,
        scene_name=args.scene_name,
        output_dir=args.output_dir,
        frame_step=args.frame_step,
        checkpoint_tag=args.checkpoint_tag,
    )


if __name__ == "__main__":
    main()

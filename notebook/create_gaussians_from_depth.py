#!/usr/bin/env python3
"""
Create Gaussian Splats from depth maps and RGB images.
This script converts depth maps and RGB images into 3D Gaussian representations.
"""

import os
import sys
import numpy as np
import torch

from utils import (
    load_image,
    create_gaussians_from_depth,
    render_frame,
    radial_to_z_depth,
)

def main():
    """Main function for command-line usage."""
    # Example configuration for Kubric4D dataset
    DATASET_PATH = "/mnt/lustre/work/geiger/gwb987/data/kubric4d"
    SCENE_NAME = "scn02719"
    DATA_PATH = os.path.join(DATASET_PATH, SCENE_NAME)
    FRAMES_PATH = os.path.join(DATA_PATH, "frames_p0_v0")  # viewpoint 0
    
    # Get first frame
    IMAGE_NAMES = sorted([f for f in os.listdir(FRAMES_PATH) if f.startswith("rgba_") and f.endswith(".png")])
    IMAGE_PATH = os.path.join(FRAMES_PATH, IMAGE_NAMES[0])
    
    # Get first depth map
    DEPTH_NAMES = sorted([f for f in os.listdir(FRAMES_PATH) if f.startswith("depth_") and f.endswith(".tiff")])
    DEPTH_PATH = os.path.join(FRAMES_PATH, DEPTH_NAMES[0])
    
    # Load image to get dimensions
    image = load_image(IMAGE_PATH)
    H, W, _ = image.shape
    
    # Camera intrinsics
    fx = float(W)
    fy = float(H)
    cx = W / 2.0
    cy = H / 2.0
    K_matrix = np.eye(3)
    K_matrix[0, 0] = fx
    K_matrix[1, 1] = fy
    K_matrix[0, 2] = cx
    K_matrix[1, 2] = cy
    
    depth_map = load_image(DEPTH_PATH, to_uint8=False)
    
    # Convert radial depth to z-depth
    depth_map_z = radial_to_z_depth(depth_map, fx, fy, cx, cy)
    print(f"Converted radial depth map to z-depth map with shape: {depth_map_z.shape}, dtype: {depth_map_z.dtype}, min: {depth_map_z.min()}, max: {depth_map_z.max()}")
    
    # Create Gaussians
    output_path = "gaussians.ply"
    gaussians = create_gaussians_from_depth(
        image=image,
        depth=depth_map_z,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        normalize_depth=False,
        output_path=output_path,
    )
    
    # Render Gaussians
    c2w = torch.eye(4)

    # Create intrinsics matrix (3x3)
    K = torch.from_numpy(K_matrix).float()

    # Render the frame
    rendered_frame, rendered_alpha = render_frame(
        gaussians, 
        c2w=c2w, 
        K=K, 
        w=W, 
        h=H
    )

    # Display the rendered frame alongside the original image
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=14)
    # ax1.axis('off')

    ax2.imshow(rendered_frame.cpu().numpy())
    ax2.set_title('Rendered from Gaussian Splats', fontsize=14)
    #   ax2.axis('off')

    plt.tight_layout()
    # plt.show()
    plt.savefig("rendered_gaussians.png")
    
    # display alpha
    plt.figure(figsize=(8, 8))
    plt.imshow(rendered_alpha.cpu().numpy(), cmap='gray')
    plt.title('Rendered Alpha from Gaussian Splats', fontsize=14)
    plt.axis('off')
    plt.savefig("rendered_alpha.png")
    
    print(f"\nDone! Created {gaussians.get_xyz.shape[0]:,} Gaussian splats.")


if __name__ == "__main__":
    main()

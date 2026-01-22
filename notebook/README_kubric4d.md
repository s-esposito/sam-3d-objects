# Kubric4D Demo Script

This document describes the `demo_kubric4d.py` script for processing Kubric4D dataset scenes with SAM3D.

## Overview

The script processes multi-object scenes from the Kubric4D dataset, converting RGB images with depth maps and segmentation masks into 3D Gaussian representations. It includes intelligent caching to avoid reprocessing and provides comprehensive visualization outputs.

## Features

- **Automatic caching**: Results are cached to avoid redundant inference runs
- **Modular structure**: Clean separation of concerns with dedicated functions for each task
- **Comprehensive visualizations**: Generates multiple visualization outputs
- **Depth processing**: Supports both ground-truth depth maps and MoGe depth estimation
- **Multi-object support**: Handles scenes with multiple segmented objects

## Directory Structure

```
/mnt/lustre/work/geiger/gwb987/data/kubric4d/
└── <scene_name>/
    ├── frames_p0_v0/          # Frame data for viewpoint 0
    │   ├── rgba_*.png         # RGBA images
    │   ├── segmentation_*.png # Segmentation masks
    │   └── depth_*.tiff       # Ground-truth depth maps
    └── cached_results/        # Created automatically
        └── <scene_name>_sam3d_results.npz
```

## Configuration

Edit the `main()` function to configure:

```python
DATASET_PATH = "/mnt/lustre/work/geiger/gwb987/data/kubric4d"  # Path to dataset
SCENE_NAME = "scn02719"           # Scene identifier
USE_MOGE = False                  # Use MoGe depth estimation (True) or GT depth (False)
SEED = 42                         # Random seed for reproducibility
```

## Usage

### Basic Usage

```bash
cd /home/geiger/gwb987/work/codebase/sam-3d-objects/notebook
python demo_kubric4d.py
```

### First Run (No Cache)

On the first run, the script will:
1. Load RGB image and segmentation masks
2. Process depth maps and create 3D pointmap
3. Run SAM3D inference on each object mask
4. Cache results for future runs
5. Generate Gaussian splatting scene
6. Save PLY file and visualizations

### Subsequent Runs (With Cache)

On subsequent runs with existing cache:
1. Load RGB image and segmentation masks
2. Load cached inference results
3. Skip inference pipeline initialization
4. Generate Gaussian splatting scene
5. Save PLY file and visualizations

## Outputs

The script generates several output files:

### Visualization Files (Current Directory)
- `image_and_masks.png`: Grid showing original image and segmentation masks
- `pointmap_visualization.png`: Color-coded 3D pointmap and depth visualization
- `rendered_vs_original.png`: Comparison of original image and rendered Gaussians

### Gaussian Splatting Files
- `gaussians/kubric4d/<scene_name>.ply`: Gaussian splatting scene in PLY format

### Cache Files
- `<dataset_path>/<scene_name>/cached_results/<scene_name>_sam3d_results.npz`: Cached inference results

## Functions

### Path Management
- `setup_paths(dataset_path, scene_name)`: Initialize and validate directory structure

### Visualization
- `visualize_image_and_masks(image, masks, output_path)`: Create mask overlay visualization
- `visualize_pointmap(pointmap, output_path)`: Visualize 3D pointmap with depth map
- `render_and_compare(scene_gs, image, K_matrix, W, H, output_path)`: Compare rendered vs original

### Depth Processing
- `load_and_process_depth(frames_path, depth_names, W, H, use_moge, inference, image)`: Load and process depth maps
- `transform_to_pytorch3d_convention(pointmap)`: Transform coordinate systems

### Inference
- `run_inference_on_masks(inference, image, masks, pointmap, seed)`: Run SAM3D on all masks

### Caching
- `save_cached_results(cached_results_path, scene_name, outputs)`: Save results to cache
- `load_cached_results(cached_results_path, scene_name)`: Load cached results

## Cache Management

### Clear Cache
To force re-inference, delete the cache file:
```bash
rm /mnt/lustre/work/geiger/gwb987/data/kubric4d/<scene_name>/cached_results/<scene_name>_sam3d_results.npz
```

### Cache Format
The cache stores:
- `sam3d_gaussians`: List of Gaussian representations for each object
- `num_objects`: Number of objects in the scene

## Depth Options

### Ground Truth Depth (USE_MOGE=False)
- Uses Kubric4D ground-truth radial depth maps
- Converts radial depth to z-depth
- Faster processing
- More accurate depth values

### MoGe Depth Estimation (USE_MOGE=True)
- Uses MoGe depth model for estimation
- Requires inference pipeline initialization
- Slower but works without ground-truth depth
- Useful for real-world images

## Camera Intrinsics

Default Kubric4D camera intrinsics (for 512x512 images):
- Focal length: `fx = W`, `fy = H`
- Principal point: `cx = W/2`, `cy = H/2`

These can be adjusted in the `load_and_process_depth()` function if needed.

## Coordinate Systems

The script handles two coordinate systems:
1. **R3 Convention**: Standard right-handed coordinate system
2. **PyTorch3D Convention**: PyTorch3D camera convention

Transformations are applied automatically to ensure correct rendering.

## Troubleshooting

### Out of Memory Errors
- Reduce image resolution
- Process fewer masks at once
- Use ground-truth depth instead of MoGe

### Missing Files
- Ensure dataset path is correct
- Verify scene name exists in dataset
- Check that `frames_p0_v0` directory contains expected files

### Rendering Issues
- Verify camera intrinsics match your data
- Check coordinate system transformations
- Ensure Gaussian scene has valid parameters

## Dependencies

Required packages:
- numpy
- torch
- matplotlib
- seaborn
- pytorch3d
- PIL
- gsplat

See `requirements.txt` in the project root for complete dependencies.

## Notes

- The script processes only the first frame (frame 0) of each scene
- Multiple viewpoints can be processed by changing `frames_p0_v0` to other viewpoint directories
- Temporal processing (video) requires modifications to handle frame sequences
- Cache is scene-specific and frame-specific

## Future Enhancements

Potential improvements:
- [ ] Process multiple frames/viewpoints
- [ ] Command-line argument parsing
- [ ] Batch processing of multiple scenes
- [ ] Video rendering from temporal sequences
- [ ] Interactive visualization
- [ ] Quality metrics computation

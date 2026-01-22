# SAM-3D-Objects: Gaussian Decoder Architecture

This document describes the architecture of the Gaussian decoder used in SAM-3D-Objects for lifting 2D images to 3D Gaussian Splatting representations.

## Overview

The system uses a **Sparse Latent Transformer (SLAT)** architecture to decode structured latent tokens into 3D Gaussian parameters. The pipeline consists of:

1. **Image Encoder** (DINOv2) - extracts 2D features from input image
2. **Depth Model** (MoGe) - estimates 3D pointmap from image
3. **Structure Encoder** - converts 2D features + pointmap to sparse 3D latents
4. **SLAT Gaussian Decoder** - decodes sparse latents to 3D Gaussians
5. **Layout Decoder** - predicts object pose (rotation, translation, scale)

```
Input Image + Mask + Pointmap
         ↓
    Image Encoder (DINOv2)
         ↓
    2D Feature Maps (H/14 × W/14 × 1024)
         ↓
    Structure Encoder (Sparse Latent Transformer)
         ↓
    Sparse 3D Latent Tokens (N × 64)
         ↓
    SLAT Gaussian Decoder (DiT)
         ↓
    3D Gaussian Parameters (M × 59)
         ↓
    Layout Decoder
         ↓
    Posed Gaussians in Camera Space
```

## Input Specifications

### Image Input
- **Resolution**: Variable, but internally processed at DINOv2 patch size (14×14 pixels per patch)
- **Format**: RGB image, normalized to [0, 1]
- **Channels**: 3 (RGB)

### Mask Input
- **Resolution**: Same as input image
- **Format**: Binary mask indicating object region
- **Channels**: 1

### Pointmap Input
- **Resolution**: Same as input image (H × W × 3)
- **Format**: Per-pixel 3D coordinates in camera space
- **Coordinate System**: PyTorch3D convention (X-right, Y-up, Z-into-screen)
- **Source**: Either ground truth depth (Kubric4D) or MoGe depth estimation

## Architecture Components

### 1. Image Encoder (DINOv2)

Pre-trained DINOv2 ViT-Large model extracts semantic features:

| Parameter | Value |
|-----------|-------|
| Architecture | ViT-Large |
| Patch Size | 14 × 14 pixels |
| Feature Dimension | 1024 |
| Output Resolution | H/14 × W/14 |

### 2. Structure Encoder

Converts 2D features and 3D pointmap into sparse 3D latent tokens:

| Parameter | Value |
|-----------|-------|
| Input Channels | 1024 (DINOv2 features) |
| Output Channels | 64 (latent dimension) |
| Resolution | 64³ voxel grid |
| Sparsity | Only occupied voxels are processed |

The encoder:
1. Projects DINOv2 features to latent dimension
2. Voxelizes the pointmap into a 64³ grid
3. Assigns features to voxels based on 3D position
4. Creates sparse tensor with only non-empty voxels

### 3. SLAT Gaussian Decoder

A Diffusion Transformer (DiT) that processes sparse 3D tokens:

| Parameter | Value |
|-----------|-------|
| Model Channels | 1024 |
| Latent Channels | 64 |
| Number of Blocks | 12 |
| Number of Heads | 16 |
| MLP Ratio | 4 |
| Attention Mode | Shift Window |
| Window Size | 8 |
| Output Channels | ~59 (Gaussian parameters) |
| Precision | FP16 |

#### Transformer Block Structure

Each of the 12 transformer blocks contains:

```
Input Features
     ↓
Adaptive Layer Norm
     ↓
Shift-Window Self-Attention (16 heads)
     ↓
Residual Connection
     ↓
Adaptive Layer Norm
     ↓
MLP (4× expansion)
     ↓
Residual Connection
     ↓
Output Features
```

#### Shift-Window Attention

For efficiency with sparse 3D data, the decoder uses shift-window attention:
- Window size: 8×8×8 voxels
- Alternating shifted and non-shifted windows
- Enables global context while maintaining efficiency

### 4. Gaussian Representation

Each sparse token is decoded into Gaussian parameters:

| Parameter | Channels | Description |
|-----------|----------|-------------|
| `_xyz` | 3 | Position (x, y, z) |
| `_opacity` | 1 | Opacity (logit space) |
| `_scaling` | 3 | Scale (log space) |
| `_rotation` | 4 | Rotation (quaternion, normalized) |
| `_features_dc` | 3 | DC spherical harmonics (base color) |
| `_features_rest` | 45 | Higher-order SH (15 bands × 3 channels) |
| **Total** | **59** | Per-Gaussian parameters |

#### Gaussian Conversion

```python
def to_gaussians(self, x: torch.Tensor):
    """
    Convert decoder output to Gaussian parameters.
    
    x: (N, 59) raw decoder output
    """
    pos = x[:, 0:3]           # xyz position
    opacity = x[:, 3:4]       # opacity (1 channel)
    scale = x[:, 4:7]         # log scale (3 channels)  
    rotation = x[:, 7:11]     # quaternion (4 channels)
    sh = x[:, 11:]            # spherical harmonics (48 channels)
    
    # Normalize rotation quaternion
    rotation = rotation / rotation.norm(dim=-1, keepdim=True)
    
    # Split SH into DC and rest
    features_dc = sh[:, :3]
    features_rest = sh[:, 3:]
```

### 5. Layout Decoder

Predicts scene-level pose for the object:

| Output | Shape | Description |
|--------|-------|-------------|
| `rotation` | (4,) | Quaternion (local-to-camera rotation) |
| `translation` | (3,) | Position in camera space |
| `scale` | (1,) | Object scale factor |

The layout is applied to raw Gaussians via `make_scene()`:

```python
# Transform Gaussian positions
xyz_world = rotation @ (xyz_local * scale) + translation

# Transform Gaussian orientations  
rot_world = quaternion_multiply(quaternion_invert(rotation), rot_local)

# Scale Gaussian scales
scale_world = scale_local * scale
```

## Configuration

From `checkpoints/hf/pipeline.yaml`:

```yaml
# Sparse Structure VAE (encoder)
sparse_structure_vae:
  resolution: 64
  model_channels: 64
  latent_channels: 8
  num_blocks: 3
  num_heads: 8

# SLAT Decoder for Gaussians
slat_decoder_gs:
  resolution: 64
  model_channels: 1024
  latent_channels: 64
  num_blocks: 12
  num_heads: 16
  mlp_ratio: 4
  attn_mode: "shift_window"
  window_size: 8
  use_fp16: true

# Gaussian representation
gaussian:
  sh_degree: 2
  aabb: [-0.5, -0.5, -0.5, 1.0, 1.0, 1.0]  # Bounding box
```

## Typical Dimensions

For a typical inference with a 576×384 image:

| Stage | Shape | Description |
|-------|-------|-------------|
| Input Image | (384, 576, 3) | RGB image |
| Input Mask | (384, 576) | Binary object mask |
| Input Pointmap | (384, 576, 3) | 3D coordinates |
| DINOv2 Features | (27, 41, 1024) | Patch features |
| Sparse Latents | (N, 64) | N ≈ 1000-10000 tokens |
| Output Gaussians | (M, 59) | M ≈ 10000-500000 Gaussians |

## Output

The decoder produces a `Gaussian` object with:

```python
gaussian.get_xyz        # (N, 3) - positions
gaussian.get_rotation   # (N, 4) - quaternions
gaussian.get_scaling    # (N, 3) - scales
gaussian.get_opacity    # (N, 1) - opacities
gaussian.get_features   # (N, 1, 48) - spherical harmonics
```

These can be:
1. **Rendered** using Gaussian splatting renderer
2. **Exported** as `.ply` point cloud
3. **Combined** with other objects via `join_gaussians()`

## Coordinate Systems

### R3 Convention (Standard)
- X: Right
- Y: Down  
- Z: Forward (into screen)

### PyTorch3D Convention (Used internally)
- X: Left
- Y: Up
- Z: Forward (into screen)

The pipeline converts between these conventions:
1. Input pointmap is transformed to PyTorch3D convention before inference
2. Output Gaussians are transformed back to R3 convention for rendering

```python
# R3 to PyTorch3D
r3_to_p3d_R = look_at_view_transform(
    eye=[[0, 0, -1]], at=[[0, 0, 0]], up=[[0, -1, 0]]
)[0]
pointmap_p3d = pointmap_r3 @ r3_to_p3d_R.T

# PyTorch3D to R3 (inverse)
p3d_to_r3_R = r3_to_p3d_R.T
xyz_r3 = xyz_p3d @ p3d_to_r3_R.T
```

## Inference Pipeline Summary

1. **Load input**: Image, mask, and compute/load pointmap
2. **Transform pointmap**: R3 → PyTorch3D convention
3. **Encode structure**: Extract DINOv2 features → sparse 3D latents
4. **Decode Gaussians**: SLAT decoder → raw Gaussian parameters
5. **Decode layout**: Predict rotation, translation, scale
6. **Apply layout**: Transform raw Gaussians to camera space
7. **Transform back**: PyTorch3D → R3 convention for rendering
8. **Render**: Use Gaussian splatting to produce final image

## Re-running the Decoder

The inference pipeline returns decoder inputs that allow re-running the forward pass without recomputing the entire pipeline:

### Returned Decoder Inputs

- **`decoder_input_coords`**: Sparse 3D coordinates from the structure encoder (SparseTensor coordinates)
- **`decoder_input_slat`**: SLAT latent features (SparseTensor)

### Saving and Loading Decoder Inputs

The `demo.py` script automatically saves decoder inputs to the cache file. When you run inference with `--no-cache`, it will:

1. Run full inference pipeline
2. Save decoder inputs (`decoder_input_slat_feats`, `decoder_input_slat_coords`) to cache
3. These can be loaded later to re-run just the decoder

### Usage Example (Python API)

```python
# Run full inference once
output = pipeline.run(image, mask, pointmap_dict)

# Extract decoder inputs (returned by pipeline)
slat = output["decoder_input_slat"]

# Later, re-run just the decoder to get the same Gaussians
from utils import rerun_gaussian_decoder
decoded = rerun_gaussian_decoder(pipeline, slat, formats=["gaussian"])
gaussians = decoded["gaussian"][0]  # Same as output["gaussian"][0]
```

### Usage Example (Command Line)

```bash
# Run inference and save decoder inputs
python notebook/demo.py --dataset davis --scene-name car-turn --no-cache

# Later, re-run decoder from cached inputs
python notebook/rerun_decoder_example.py \
    --cache-file /path/to/cached_results/car-turn_f0_sam3d_results.npz \
    --output-dir ./redecoded_outputs
```

### What Gets Saved

The cache file (`.npz`) contains:
- **Gaussian parameters**: xyz, features, scaling, rotation, opacity
- **Layout parameters**: rotation, translation, scale
- **Decoder inputs**: SLAT features and coordinates (SparseTensor data)

### Use Cases

This is useful for:
- **Fast iteration**: Experiment with decoder configurations without expensive preprocessing
- **Decoder analysis**: Study decoder behavior with fixed inputs
- **Reproducibility**: Exactly reproduce Gaussians from saved decoder state
- **Debugging**: Compare decoder outputs with different settings

**Note**: The decoder inputs include both the sparse latent features (`slat`) and the spatial coordinates (`coords`). The `coords` are embedded in the SparseTensor but can be accessed separately if needed.

## Key Implementation Files

- `sam3d_objects/model/backbone/slat/slat_decoder_gs.py` - SLAT Gaussian decoder
- `sam3d_objects/model/backbone/tdfy_dit/` - DiT transformer backbone
- `sam3d_objects/model/backbone/tdfy_dit/representations/gaussian/` - Gaussian representation
- `sam3d_objects/pipeline/inference_pipeline_pointmap.py` - Main inference pipeline
- `notebook/inference.py` - High-level inference API
- `notebook/demo.py` - Demo script with complete pipeline

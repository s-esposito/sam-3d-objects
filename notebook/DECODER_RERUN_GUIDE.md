# Re-running the Gaussian Decoder

This guide explains how to save and re-use decoder inputs to reproduce Gaussian outputs without re-running the full inference pipeline.

## Overview

The SAM-3D-Objects inference pipeline has been enhanced to save the intermediate decoder inputs (SLAT latent features) that can be used to re-run just the decoder forward pass. This is much faster than re-running the entire pipeline (image encoding, structure encoding, SLAT sampling, etc.).

## What Gets Saved

When you run inference, the following decoder inputs are now saved to cache:

- **`decoder_input_slat_feats`**: The feature tensor from the SLAT (Sparse Latent Transformer)
- **`decoder_input_slat_coords`**: The sparse coordinate indices
- **`decoder_input_coords`**: Additional coordinate information (optional)

These are stored in the `.npz` cache files alongside the Gaussian outputs.

## Method 1: Using the Python API

### Step 1: Run Inference

```python
from inference import Inference
import torch

# Initialize pipeline
inference = Inference("checkpoints/hf/pipeline.yaml", compile=False)

# Run inference
image = load_image("path/to/image.jpg")
mask = load_mask("path/to/mask.png")
pointmap = torch.from_numpy(pointmap_array).float().cuda()

output = inference(image, mask, pointmap=pointmap)

# Decoder inputs are now in the output
slat = output["decoder_input_slat"]
coords = output["decoder_input_coords"]
```

### Step 2: Re-run Decoder

```python
from utils import rerun_gaussian_decoder

# Re-run decoder with saved inputs
decoded = rerun_gaussian_decoder(
    inference._pipeline,  # The pipeline object
    slat,                 # The saved SLAT features
    formats=["gaussian"]  # Output formats
)

# Get the Gaussians (identical to original)
gaussians = decoded["gaussian"][0]
gaussians.save_ply("redecoded_output.ply")
```

## Method 2: Using Command Line Tools

### Step 1: Run Inference with Demo Script

```bash
# Run inference on DAVIS dataset
python notebook/demo.py \
    --dataset davis \
    --scene-name car-turn \
    --frame-index 0 \
    --no-cache

# Or on Kubric4D dataset
python notebook/demo.py \
    --dataset kubric4d \
    --scene-name scn02719 \
    --frame-index 0 \
    --no-cache
```

This creates a cache file with decoder inputs:
```
/path/to/cached_results/car-turn_f0_sam3d_results.npz
```

### Step 2: Re-run Decoder from Cache

```bash
python notebook/rerun_decoder_example.py \
    --cache-file /path/to/cached_results/car-turn_f0_sam3d_results.npz \
    --output-dir ./redecoded_outputs
```

Options:
- `--cache-file`: Path to the cached results file
- `--output-dir`: Where to save re-decoded Gaussians
- `--object-index`: Process only specific object (0-based index)

### Process Specific Object

```bash
python notebook/rerun_decoder_example.py \
    --cache-file /path/to/cached_results/scene_sam3d_results.npz \
    --object-index 0 \
    --output-dir ./object_0_redecoded
```

## Files Modified

1. **`sam3d_objects/pipeline/inference_pipeline_pointmap.py`**
   - Returns `decoder_input_coords` and `decoder_input_slat` in output dict

2. **`notebook/demo.py`**
   - `save_cached_results()`: Saves decoder inputs to cache
   - `load_cached_results()`: Reconstructs SparseTensor from saved data

3. **`notebook/utils.py`**
   - `rerun_gaussian_decoder()`: Helper function to re-run decoder

4. **`notebook/rerun_decoder_example.py`** (NEW)
   - Standalone script for re-running decoder from cache

5. **`notebook/ARCHITECTURE.md`**
   - Documentation of decoder inputs and usage

## Use Cases

### Fast Experimentation
Re-run decoder with different settings without expensive image preprocessing:
```python
# Original inference
output = inference(image, mask, pointmap=pointmap)
slat = output["decoder_input_slat"]

# Try different decoder formats quickly
mesh_output = rerun_gaussian_decoder(pipeline, slat, formats=["mesh"])
gaussian_output = rerun_gaussian_decoder(pipeline, slat, formats=["gaussian"])
```

### Decoder Analysis
Study decoder behavior with fixed inputs:
```python
# Same inputs, different random seeds (if applicable)
decoded_1 = rerun_gaussian_decoder(pipeline, slat, formats=["gaussian"])
decoded_2 = rerun_gaussian_decoder(pipeline, slat, formats=["gaussian"])

# Compare outputs
diff = (decoded_1["gaussian"][0].get_xyz - decoded_2["gaussian"][0].get_xyz).abs().max()
print(f"Max difference: {diff}")  # Should be 0.0 for deterministic decoder
```

### Reproducibility
Exactly reproduce results from saved decoder state:
```python
# Load from cache weeks later
cached_outputs = load_cached_results(cache_path, scene_name)
slat = cached_outputs[0]["decoder_input_slat"]

# Get exact same Gaussians
decoded = rerun_gaussian_decoder(pipeline, slat, formats=["gaussian"])
```

## Performance

| Stage | Time (approx) | Memory |
|-------|---------------|--------|
| Full inference | ~2-5 seconds | ~4GB GPU |
| Re-run decoder only | ~0.2-0.5 seconds | ~2GB GPU |

**Speedup**: ~10x faster for decoder-only execution

## Technical Details

### SparseTensor Format

The decoder input is a SparseTensor (from spconv.pytorch):
```python
slat = sp.SparseConvTensor(
    features=slat_feats,  # (N, C) float32, C=64
    indices=slat_coords,   # (N, 4) int32, [batch_idx, z, y, x]
    spatial_shape=[64, 64, 64],
    batch_size=1,
)
```

### Cache File Structure

The `.npz` file contains:
```python
{
    'cached_data': [
        {
            'gaussian_data': {...},  # Gaussian parameters
            'rotation': array(...),  # Layout rotation
            'translation': array(...),  # Layout translation
            'scale': array(...),  # Layout scale
            'decoder_input_slat_feats': array(...),  # NEW
            'decoder_input_slat_coords': array(...),  # NEW
            'decoder_input_coords': array(...),  # NEW (optional)
        },
        # ... more objects
    ],
    'num_objects': int,
}
```

## Troubleshooting

### "Object does not have decoder inputs saved"

This means the cache was created before the decoder input saving feature. Re-run inference with `--no-cache`:

```bash
python notebook/demo.py --dataset davis --scene-name car-turn --no-cache
```

### SparseTensor reconstruction issues

Make sure `spconv.pytorch` is installed:
```bash
pip install spconv-cu118  # or appropriate CUDA version
```

### Different results than original

The re-decoded Gaussians should be **bit-for-bit identical** to the original. If not, check:
- Same PyTorch version
- Same random state (if decoder uses randomness)
- Decoder model weights unchanged

## See Also

- **ARCHITECTURE.md**: Full pipeline architecture documentation
- **demo.py**: Main inference script with caching
- **utils.py**: Helper functions including `rerun_gaussian_decoder()`

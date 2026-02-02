# Evaluate Sequence Documentation

This document describes how to use the `evaluate_sequence.py` script to evaluate the quality of SAM3D Gaussian reconstructions on video sequences.

## Overview

The script evaluates Gaussian reconstruction quality by:
1. Loading a video sequence (Kubric4D or DAVIS dataset)
2. Loading cached SLAT tokens if available, or running full inference
3. Re-decoding tokens and applying saved poses to get Gaussians
4. Rendering the Gaussians back to images
5. Comparing rendered images with ground truth using PSNR, SSIM, and LPIPS metrics

## Prerequisites

Before running evaluation, you typically need to have run `demo.py` to generate cached tokens:

```bash
# Generate tokens for a sequence
python custom/demo.py --dataset davis --scene-name car-turn --frame-stride 10 --save-tokens
```

## Basic Usage

```bash
# Evaluate a Kubric4D sequence
python custom/evaluate_sequence.py --dataset kubric4d --scene-name scn02719

# Evaluate a DAVIS sequence
python custom/evaluate_sequence.py --dataset davis --scene-name car-turn
```

## Command-Line Arguments

### Dataset Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | `kubric4d` | Dataset type. Choices: `kubric4d`, `davis` |
| `--dataset-path` | str | auto | Path to dataset root. Defaults to standard paths based on dataset type |
| `--scene-name` | str | auto | Name of the scene to process. Defaults: `scn02719` (kubric4d), `car-turn` (davis) |
| `--frame-index` | int | None | Specific frame index to process (0-based). If set, only this frame is evaluated |
| `--frame-stride` | int | `10` | Stride for iterating over frames when `--frame-index` is not specified |

### Processing Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use-moge` | flag | False | Use MoGe depth model instead of ground truth depth. **Required for DAVIS** (auto-enabled) |
| `--object-index 0` | flag | False | Only process the first object/mask in each frame |
| `--seed` | int | `42` | Random seed for inference |
| `--with-background` | flag | False | Add background Gaussians from non-masked regions |
| `--no-cache` | flag | False | Ignore cached tokens and run full inference |

### Token Averaging Options

These options enable creating a **canonical object** by averaging SLAT tokens across frames, then rendering with per-frame poses.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--average-tokens` | flag | False | Enable token averaging mode |
| `--weighting-type` | str | `uniform` | Weighting method for averaging. Choices: `uniform`, `mask-area`, `mask-error` |

#### Weighting Types Explained

- **`uniform`**: Simple average of all frame tokens (equal weight)
- **`mask-area`**: Weight by mask visibility - frames with larger visible masks contribute more to the canonical object
- **`mask-error`**: Weight by inverse rendering error - frames with lower reconstruction error (better quality) contribute more

### Pose Refinement Options

These options enable **differentiable pose optimization** to refine per-frame poses after creating the canonical object.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--refine-poses` | flag | False | Enable pose refinement (only works with `--average-tokens`) |
| `--refine-iterations` | int | `100` | Number of optimization iterations per frame |
| `--refine-lr-rotation` | float | `0.01` | Learning rate for rotation (quaternion) |
| `--refine-lr-translation` | float | `0.001` | Learning rate for translation |
| `--refine-lr-scale` | float | `0.001` | Learning rate for scale |

#### How Pose Refinement Works

1. A canonical Gaussian object is created from averaged tokens
2. For each frame, the initial pose (rotation, translation, scale) from cached tokens is used as starting point
3. Differentiable Gaussian rendering computes the loss between rendered and ground truth image in the masked region
4. Only pose parameters are optimized (Gaussians are frozen)
5. A small regularization term prevents excessive deviation from the initial pose

### Output Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output-dir` | str | auto | Directory to save outputs. Defaults to `custom/results/{dataset}/eval/` |
| `--save-renders` | flag | False | Save rendered images and side-by-side comparisons |
| `--save-metrics` | flag | False | Save metrics to JSON file |

## Usage Examples

### Basic Evaluation

```bash
# Evaluate Kubric4D with default settings
python custom/evaluate_sequence.py --dataset kubric4d --scene-name scn02719

# Evaluate DAVIS sequence every 5 frames
python custom/evaluate_sequence.py --dataset davis --scene-name car-turn --frame-stride 5

# Evaluate a single frame
python custom/evaluate_sequence.py --dataset kubric4d --scene-name scn02719 --frame-index 0
```

### With Background and Output

```bash
# Evaluate with background and save visual comparisons
python custom/evaluate_sequence.py --dataset davis --scene-name car-turn \
    --with-background --save-renders --save-metrics
```

### Token Averaging Mode

```bash
# Average tokens with uniform weights
python custom/evaluate_sequence.py --dataset kubric4d --scene-name scn02719 \
    --average-tokens

# Weight by mask visibility (larger masks = higher weight)
python custom/evaluate_sequence.py --dataset kubric4d --scene-name scn02719 \
    --average-tokens --weighting-type mask-area

# Weight by rendering error (lower error = higher weight)
python custom/evaluate_sequence.py --dataset davis --scene-name car-turn \
    --average-tokens --weighting-type mask-error
```

### Pose Refinement

```bash
# Basic pose refinement
python custom/evaluate_sequence.py --dataset davis --scene-name car-turn \
    --average-tokens --refine-poses

# Pose refinement with custom settings
python custom/evaluate_sequence.py --dataset kubric4d --scene-name scn02719 \
    --average-tokens --refine-poses \
    --refine-iterations 200 \
    --refine-lr-rotation 0.005 \
    --refine-lr-translation 0.0005

# Full pipeline: weighted averaging + pose refinement + save outputs
python custom/evaluate_sequence.py --dataset davis --scene-name car-turn \
    --average-tokens --weighting-type mask-area --refine-poses \
    --with-background --save-renders --save-metrics
```

## Output Files

### Rendered Images

When `--save-renders` is enabled, comparison images are saved to:
```
custom/results/{dataset}/eval/renders/{scene_name}_frame_{XXXX}_{suffix}_comparison.png
```

The suffix indicates the evaluation mode:
- Standard mode: (no suffix)
- Averaged tokens: `_averaged_{weighting_type}`
- With pose refinement: `_averaged_{weighting_type}_refined`

### Metrics JSON

When `--save-metrics` is enabled, metrics are saved to:
```
custom/results/{dataset}/eval/{scene_name}_{suffix}_metrics.json
```

The JSON file contains:
```json
{
  "dataset": "davis",
  "scene_name": "car-turn",
  "num_frames_evaluated": 9,
  "frame_stride": 10,
  "with_background": true,
  "first_object_only": false,
  "average_tokens": true,
  "weighting_type": "uniform",
  "refine_poses": false,
  "refine_iterations": null,
  "psnr_mean": 25.43,
  "psnr_std": 2.15,
  "psnr_min": 21.87,
  "psnr_max": 28.92,
  "ssim_mean": 0.8721,
  "ssim_std": 0.0312,
  "ssim_min": 0.8234,
  "ssim_max": 0.9156,
  "lpip_mean": 0.1523,
  "lpip_std": 0.0245,
  "lpip_min": 0.1102,
  "lpip_max": 0.1987,
  "frame_metrics": [
    {"frame_index": 0, "psnr": 25.12, "ssim": 0.8654, "lpip": 0.1432},
    ...
  ]
}
```

## Evaluation Metrics

| Metric | Range | Better | Description |
|--------|-------|--------|-------------|
| **PSNR** | 0 to ∞ dB | Higher | Peak Signal-to-Noise Ratio. Measures pixel-level similarity |
| **SSIM** | 0 to 1 | Higher | Structural Similarity Index. Measures perceptual similarity |
| **LPIPS** | 0 to 1 | Lower | Learned Perceptual Image Patch Similarity. Deep feature-based perceptual metric |

## Workflow Modes

### 1. Standard Mode (Per-Frame Independent)

Each frame is processed independently using its own cached tokens or full inference.

```
Frame 0: tokens_0 → decode → Gaussian_0 → render
Frame 1: tokens_1 → decode → Gaussian_1 → render
...
```

### 2. Token Averaging Mode (`--average-tokens`)

A canonical object is created by averaging tokens, then rendered with per-frame poses.

```
tokens_0, tokens_1, ... → average → canonical_tokens → decode → Canonical_Gaussian
                                                                      ↓
Frame 0: Canonical_Gaussian + pose_0 → render
Frame 1: Canonical_Gaussian + pose_1 → render
...
```

### 3. Pose Refinement Mode (`--average-tokens --refine-poses`)

Same as token averaging, but poses are optimized using differentiable rendering.

```
tokens_0, tokens_1, ... → average → canonical_tokens → decode → Canonical_Gaussian
                                                                      ↓
Frame 0: Canonical_Gaussian + optimize(pose_0) → render
Frame 1: Canonical_Gaussian + optimize(pose_1) → render
...
```

## Troubleshooting

### "No cached tokens found!"

Run `demo.py` first to generate tokens:
```bash
python custom/demo.py --dataset davis --scene-name car-turn --frame-stride 10 --save-tokens
```

### Memory issues with pose refinement

Reduce the number of frames or use a larger stride:
```bash
python custom/evaluate_sequence.py --dataset davis --scene-name car-turn \
    --average-tokens --refine-poses --frame-stride 20
```

### DAVIS requires MoGe depth

The script automatically enables `--use-moge` for DAVIS since it has no ground truth depth.

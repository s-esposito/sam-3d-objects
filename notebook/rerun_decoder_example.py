"""
Example script showing how to re-run the Gaussian decoder from saved decoder inputs.

This demonstrates loading cached inference results that include decoder inputs,
and then re-running just the decoder forward pass to reproduce the exact same
Gaussians without re-running the entire inference pipeline.

Usage:
    python rerun_decoder_example.py --cache-file /path/to/cached_results/scene_sam3d_results.npz
"""
import os
import sys
import argparse
import numpy as np
import torch
import spconv.pytorch as sp

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from inference import Inference
from utils import rerun_gaussian_decoder


def load_decoder_inputs_from_cache(cache_file):
    """
    Load decoder inputs from a cached results file.
    
    Parameters
    ----------
    cache_file : str
        Path to the .npz cache file
        
    Returns
    -------
    list of dict
        List of decoder inputs, one per object. Each dict contains:
        - decoder_input_slat: SparseTensor with SLAT latent features
        - decoder_input_coords: Sparse 3D coordinates (optional, embedded in slat)
    """
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Cache file not found: {cache_file}")
    
    print(f"Loading decoder inputs from: {cache_file}")
    cached_archive = np.load(cache_file, allow_pickle=True)
    cached_data = cached_archive["cached_data"]
    
    decoder_inputs = []
    for i, data in enumerate(cached_data):
        data = data.item() if hasattr(data, 'item') else data
        
        # Check if decoder inputs are available
        if 'decoder_input_slat_feats' not in data or 'decoder_input_slat_coords' not in data:
            print(f"Warning: Object {i} does not have decoder inputs saved.")
            print("Make sure you ran inference with the updated pipeline that saves decoder inputs.")
            continue
        
        # Reconstruct SparseTensor for slat
        slat_feats = torch.from_numpy(data['decoder_input_slat_feats']).cuda()
        slat_coords = torch.from_numpy(data['decoder_input_slat_coords']).cuda()
        slat = sp.SparseConvTensor(
            features=slat_feats,
            indices=slat_coords,
            spatial_shape=[64, 64, 64],
            batch_size=1,
        )
        
        decoder_input = {
            'decoder_input_slat': slat,
        }
        
        if 'decoder_input_coords' in data:
            decoder_input['decoder_input_coords'] = torch.from_numpy(data['decoder_input_coords']).cuda()
        
        decoder_inputs.append(decoder_input)
    
    print(f"Loaded decoder inputs for {len(decoder_inputs)} objects")
    return decoder_inputs


def main():
    parser = argparse.ArgumentParser(
        description="Re-run Gaussian decoder from cached inputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        required=True,
        help="Path to cached results .npz file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./decoder_rerun_outputs",
        help="Directory to save re-decoded Gaussians",
    )
    parser.add_argument(
        "--object-index",
        type=int,
        default=None,
        help="Index of object to process (0-based). If not specified, processes all objects.",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize inference pipeline (needed to access the decoder)
    print("Initializing inference pipeline...")
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    config_path = os.path.join(PROJECT_ROOT, "checkpoints", "hf", "pipeline.yaml")
    inference = Inference(config_path, compile=False)
    pipeline = inference._pipeline
    
    # Load decoder inputs from cache
    decoder_inputs = load_decoder_inputs_from_cache(args.cache_file)
    
    if len(decoder_inputs) == 0:
        print("No decoder inputs found in cache file. Exiting.")
        return
    
    # Determine which objects to process
    if args.object_index is not None:
        if args.object_index >= len(decoder_inputs):
            print(f"Error: Object index {args.object_index} out of range (0-{len(decoder_inputs)-1})")
            return
        object_indices = [args.object_index]
    else:
        object_indices = range(len(decoder_inputs))
    
    # Re-run decoder for each object
    for i in object_indices:
        print(f"\n{'='*60}")
        print(f"Re-running decoder for object {i}")
        print(f"{'='*60}")
        
        decoder_input = decoder_inputs[i]
        slat = decoder_input['decoder_input_slat']
        
        # Re-run the decoder
        print("Running decoder forward pass...")
        decoded_outputs = rerun_gaussian_decoder(
            pipeline,
            slat,
            formats=["gaussian"]
        )
        
        # Extract Gaussian object
        gaussians = decoded_outputs["gaussian"][0]
        print(f"Decoded Gaussians: {gaussians.get_xyz.shape[0]} points")
        print(f"  xyz range: [{gaussians.get_xyz.min().item():.3f}, {gaussians.get_xyz.max().item():.3f}]")
        print(f"  opacity range: [{gaussians.get_opacity.min().item():.3f}, {gaussians.get_opacity.max().item():.3f}]")
        
        # Save to PLY file
        cache_name = os.path.splitext(os.path.basename(args.cache_file))[0]
        output_path = os.path.join(args.output_dir, f"{cache_name}_object_{i}_redecoded.ply")
        gaussians.save_ply(output_path)
        print(f"Saved re-decoded Gaussians to: {output_path}")
    
    print(f"\n{'='*60}")
    print(f"All objects processed successfully!")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

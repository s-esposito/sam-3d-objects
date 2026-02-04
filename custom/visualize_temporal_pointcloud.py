#!/usr/bin/env python3
"""
Temporal Point Cloud Visualizer using Open3D.

This script visualizes temporal point clouds saved by evaluate_sequence.py.
It supports:
- Playback through time
- Manual frame selection
- Filtering by opacity
- Color by time

Usage:
    # Visualize combined temporal point cloud
    python custom/visualize_temporal_pointcloud.py path/to/scene_temporal.npz
    
    # Visualize with opacity filtering
    python custom/visualize_temporal_pointcloud.py path/to/scene_temporal.npz --opacity-threshold 0.5
    
    # Visualize individual frames
    python custom/visualize_temporal_pointcloud.py path/to/point_clouds/ --frame-by-frame

Controls:
    Space       - Play/Pause animation
    Left/Right  - Previous/Next frame
    R           - Reset view
    Q/Escape    - Quit
    +/-         - Increase/Decrease point size
    C           - Toggle color mode (RGB / time-based)
    O           - Toggle opacity filtering
"""

from dataclasses import dataclass
from typing import Optional
import os
import numpy as np
import time

import tyro

try:
    import open3d as o3d
except ImportError:
    print("Error: Open3D is required. Install with: pip install open3d")
    exit(1)


@dataclass
class VisualizationConfig:
    """Configuration for temporal point cloud visualization."""
    
    input: str
    """Path to temporal .npz file or directory with per-frame files."""
    
    opacity_threshold: float = 0.0
    """Filter points with opacity below this threshold (0-1)."""
    
    show_all: bool = False
    """Show all frames at once (with time-based coloring)."""
    
    frame: Optional[int] = None
    """Show only a specific frame index (default: show all frames interactively)."""
    
    playback_fps: float = 10.0
    """Frames per second for playback animation."""


def load_temporal_point_cloud(filepath):
    """Load a combined temporal point cloud from .npz file."""
    data = np.load(filepath)
    return {
        'xyz': data['xyz'],
        'rgb': data['rgb'],
        'frame_ids': data['frame_ids'],
        'scales': data['scales'],
        'opacities': data['opacities'],
        'num_frames': int(data['num_frames']),
        'frame_indices': data['frame_indices'],
    }


def load_frame_point_cloud(filepath):
    """Load a single frame point cloud from .npz file."""
    data = np.load(filepath)
    return {
        'xyz': data['xyz'],
        'rgb': data['rgb'],
        'scales': data['scales'],
        'opacities': data['opacities'],
        'frame_idx': int(data['frame_idx']),
    }


def create_point_cloud(xyz, colors):
    """Create an Open3D point cloud from numpy arrays."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def get_time_colors(frame_ids, num_frames):
    """Generate time-based colors (blue -> red gradient)."""
    # Normalize frame IDs to [0, 1]
    if num_frames > 1:
        t = frame_ids / (num_frames - 1)
    else:
        t = np.zeros_like(frame_ids, dtype=np.float32)
    
    # Blue (early) to Red (late) gradient
    colors = np.zeros((len(frame_ids), 3), dtype=np.float32)
    colors[:, 0] = t  # Red increases with time
    colors[:, 2] = 1 - t  # Blue decreases with time
    colors[:, 1] = 0.3  # Small green component
    
    return colors


class TemporalPointCloudVisualizer:
    """Interactive visualizer for temporal point clouds."""
    
    def __init__(self, data, opacity_threshold=0.0, playback_fps=10.0):
        self.data = data
        self.opacity_threshold = opacity_threshold
        self.current_frame_idx = 0
        self.playing = False
        self.use_time_colors = False
        self.use_opacity_filter = opacity_threshold > 0
        self.play_speed = playback_fps  # frames per second
        self.last_frame_time = 0
        
        # Get unique frame indices
        self.frame_indices = sorted(np.unique(data['frame_ids']))
        self.num_frames = len(self.frame_indices)
        
        print(f"Loaded {len(data['xyz'])} points across {self.num_frames} frames")
        print(f"Frame indices: {self.frame_indices}")
        
    def get_frame_mask(self, frame_id):
        """Get mask for points in a specific frame."""
        return self.data['frame_ids'] == frame_id
    
    def filter_by_opacity(self, mask):
        """Further filter mask by opacity threshold."""
        if self.use_opacity_filter:
            opacity_mask = self.data['opacities'].flatten() >= self.opacity_threshold
            return mask & opacity_mask
        return mask
    
    def get_colors(self, mask):
        """Get colors for points (RGB or time-based)."""
        if self.use_time_colors:
            return get_time_colors(self.data['frame_ids'][mask], self.num_frames)
        else:
            return self.data['rgb'][mask]
    
    def update_point_cloud(self, pcd, show_all=False):
        """Update point cloud for current frame or all frames."""
        if show_all:
            mask = np.ones(len(self.data['xyz']), dtype=bool)
        else:
            frame_id = self.frame_indices[self.current_frame_idx]
            mask = self.get_frame_mask(frame_id)
        
        mask = self.filter_by_opacity(mask)
        
        xyz = self.data['xyz'][mask]
        colors = self.get_colors(mask)
        
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return len(xyz)
    
    def run(self, show_all_frames=False):
        """Run the interactive visualizer."""
        # Create initial point cloud
        pcd = create_point_cloud(
            self.data['xyz'][:1],  # Start with minimal points
            self.data['rgb'][:1]
        )
        
        # Create visualizer
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window("Temporal Point Cloud Viewer", width=1280, height=720)
        vis.add_geometry(pcd)
        
        # Set render options
        render_option = vis.get_render_option()
        render_option.point_size = 3.0
        render_option.background_color = np.array([0.1, 0.1, 0.1])
        
        # Key callbacks
        def toggle_play(vis):
            self.playing = not self.playing
            print(f"{'Playing' if self.playing else 'Paused'}")
            return False
        
        def next_frame(vis):
            self.current_frame_idx = (self.current_frame_idx + 1) % self.num_frames
            n_points = self.update_point_cloud(pcd, show_all_frames)
            vis.update_geometry(pcd)
            print(f"Frame {self.frame_indices[self.current_frame_idx]} ({n_points} points)")
            return False
        
        def prev_frame(vis):
            self.current_frame_idx = (self.current_frame_idx - 1) % self.num_frames
            n_points = self.update_point_cloud(pcd, show_all_frames)
            vis.update_geometry(pcd)
            print(f"Frame {self.frame_indices[self.current_frame_idx]} ({n_points} points)")
            return False
        
        def toggle_time_colors(vis):
            self.use_time_colors = not self.use_time_colors
            n_points = self.update_point_cloud(pcd, show_all_frames)
            vis.update_geometry(pcd)
            print(f"Color mode: {'Time-based' if self.use_time_colors else 'RGB'}")
            return False
        
        def toggle_opacity_filter(vis):
            self.use_opacity_filter = not self.use_opacity_filter
            n_points = self.update_point_cloud(pcd, show_all_frames)
            vis.update_geometry(pcd)
            print(f"Opacity filter: {'ON' if self.use_opacity_filter else 'OFF'} (threshold={self.opacity_threshold})")
            return False
        
        def increase_point_size(vis):
            render_option.point_size = min(render_option.point_size + 1, 20)
            print(f"Point size: {render_option.point_size}")
            return False
        
        def decrease_point_size(vis):
            render_option.point_size = max(render_option.point_size - 1, 1)
            print(f"Point size: {render_option.point_size}")
            return False
        
        # Register key callbacks
        vis.register_key_callback(ord(' '), toggle_play)  # Space
        vis.register_key_callback(262, next_frame)  # Right arrow
        vis.register_key_callback(263, prev_frame)  # Left arrow
        vis.register_key_callback(ord('C'), toggle_time_colors)
        vis.register_key_callback(ord('O'), toggle_opacity_filter)
        vis.register_key_callback(ord('='), increase_point_size)  # +
        vis.register_key_callback(ord('-'), decrease_point_size)  # -
        
        # Initial update
        if show_all_frames:
            self.use_time_colors = True  # Default to time colors when showing all
        n_points = self.update_point_cloud(pcd, show_all_frames)
        vis.update_geometry(pcd)
        
        print("\nControls:")
        print("  Space       - Play/Pause animation")
        print("  Left/Right  - Previous/Next frame")
        print("  C           - Toggle color mode (RGB / time-based)")
        print("  O           - Toggle opacity filtering")
        print("  +/-         - Increase/Decrease point size")
        print("  Q/Escape    - Quit")
        print(f"\nShowing frame {self.frame_indices[self.current_frame_idx]} ({n_points} points)")
        
        # Main loop
        while True:
            if not vis.poll_events():
                break
            vis.update_renderer()
            
            # Handle animation playback
            if self.playing and not show_all_frames:
                current_time = time.time()
                if current_time - self.last_frame_time >= 1.0 / self.play_speed:
                    self.current_frame_idx = (self.current_frame_idx + 1) % self.num_frames
                    n_points = self.update_point_cloud(pcd, show_all_frames)
                    vis.update_geometry(pcd)
                    self.last_frame_time = current_time
            
            time.sleep(0.01)  # Small delay to prevent CPU spinning
        
        vis.destroy_window()


def main(config: VisualizationConfig):
    """Main entry point for visualization."""
    
    # Load data
    if os.path.isfile(config.input):
        print(f"Loading temporal point cloud from {config.input}")
        data = load_temporal_point_cloud(config.input)
    elif os.path.isdir(config.input):
        # Load all frame files in directory
        import glob
        frame_files = sorted(glob.glob(os.path.join(config.input, "*_frame_*.npz")))
        if not frame_files:
            print(f"No frame files found in {config.input}")
            return
        
        print(f"Loading {len(frame_files)} frame files from {config.input}")
        
        all_xyz = []
        all_rgb = []
        all_frame_ids = []
        all_scales = []
        all_opacities = []
        
        for filepath in frame_files:
            frame_data = load_frame_point_cloud(filepath)
            n_points = len(frame_data['xyz'])
            all_xyz.append(frame_data['xyz'])
            all_rgb.append(frame_data['rgb'])
            all_frame_ids.append(np.full(n_points, frame_data['frame_idx'], dtype=np.int32))
            all_scales.append(frame_data['scales'])
            all_opacities.append(frame_data['opacities'])
        
        data = {
            'xyz': np.concatenate(all_xyz, axis=0),
            'rgb': np.concatenate(all_rgb, axis=0),
            'frame_ids': np.concatenate(all_frame_ids, axis=0),
            'scales': np.concatenate(all_scales, axis=0),
            'opacities': np.concatenate(all_opacities, axis=0),
            'num_frames': len(frame_files),
            'frame_indices': np.array(sorted([load_frame_point_cloud(f)['frame_idx'] for f in frame_files])),
        }
    else:
        print(f"Input not found: {config.input}")
        return
    
    # Filter to specific frame if requested
    if config.frame is not None:
        mask = data['frame_ids'] == config.frame
        if not np.any(mask):
            print(f"Frame {config.frame} not found in data")
            return
        data = {
            'xyz': data['xyz'][mask],
            'rgb': data['rgb'][mask],
            'frame_ids': data['frame_ids'][mask],
            'scales': data['scales'][mask],
            'opacities': data['opacities'][mask],
            'num_frames': 1,
            'frame_indices': np.array([config.frame]),
        }
    
    # Create and run visualizer
    visualizer = TemporalPointCloudVisualizer(
        data, 
        opacity_threshold=config.opacity_threshold,
        playback_fps=config.playback_fps
    )
    visualizer.run(show_all_frames=config.show_all)


if __name__ == "__main__":
    tyro.cli(main)

"""
Visualization utilities for the SAM3D-Objects pipeline.

This module provides functions for plotting refinement histories
and other diagnostic visualizations.
"""

from __future__ import annotations

from typing import Any, Dict, List

import matplotlib.pyplot as plt


def plot_refinement_history(
    refinement_data: Dict[str, Any],
    output_path: str,
) -> None:
    """
    Plot refinement loss history for all frames and objects in a grid.

    Creates a figure where:
    - Rows: frames (sorted by frame index)
    - Columns: loss types (total, rgb, ssim, silhouette, regularization)

    Parameters
    ----------
    refinement_data : dict
        Refinement history data with structure:
        {
            'objects': {
                '0': {
                    '0': {'loss_history': [...], 'best_iteration': int},
                    '1': {...},
                },
                '1': {...},
            }
        }
    output_path : str
        Path to save the output figure.

    Examples
    --------
    >>> plot_refinement_history(data, "refinement_losses.png")
    Saved refinement loss plot to refinement_losses.png
    """
    # Collect all frames across all objects
    all_frames: List[Dict[str, Any]] = []
    for obj_idx, obj_data in refinement_data["objects"].items():
        for frame_idx, frame_data in obj_data.items():
            all_frames.append(
                {
                    "obj_idx": int(obj_idx),
                    "frame_idx": int(frame_idx),
                    "loss_history": frame_data["loss_history"],
                    "best_iteration": frame_data["best_iteration"],
                }
            )

    if not all_frames:
        print("No refinement data to plot")
        return

    # Sort by object index, then frame index
    all_frames.sort(key=lambda x: (x["obj_idx"], x["frame_idx"]))

    # Loss types to plot (ssim may not exist in older data)
    loss_types = ["total", "rgb", "ssim", "silhouette", "regularization"]
    loss_titles = ["Total Loss", "RGB Loss", "SSIM Loss", "Silhouette Loss", "Regularization"]

    n_frames = len(all_frames)
    n_cols = len(loss_types)

    # Create figure with subplots
    fig_height = max(3, 1.5 * n_frames)
    fig, axes = plt.subplots(n_frames, n_cols, figsize=(3.5 * n_cols, fig_height), squeeze=False)

    for row_idx, frame_info in enumerate(all_frames):
        loss_history = frame_info["loss_history"]
        best_iter = frame_info["best_iteration"]
        iterations = list(range(len(loss_history)))

        for col_idx, (loss_type, title) in enumerate(zip(loss_types, loss_titles)):
            ax = axes[row_idx, col_idx]

            # Extract loss values for this type (handle missing keys for backward compat)
            if loss_type in loss_history[0]:
                values = [h[loss_type] for h in loss_history]
            else:
                # Loss type not present (e.g., ssim in older data)
                values = [0.0] * len(loss_history)
                ax.text(
                    0.5, 0.5, "N/A", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="gray"
                )

            # Plot the loss curve
            ax.plot(iterations, values, "b-", linewidth=1)

            # Mark best iteration
            if best_iter < len(values):
                ax.axvline(x=best_iter, color="r", linestyle="--", alpha=0.7, linewidth=0.8)
                ax.scatter([best_iter], [values[best_iter]], color="r", s=20, zorder=5)

            # Labels
            if row_idx == 0:
                ax.set_title(title, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"Obj {frame_info['obj_idx']}, F{frame_info['frame_idx']}", fontsize=8)
            if row_idx == n_frames - 1:
                ax.set_xlabel("Iteration", fontsize=8)

            # Formatting
            ax.tick_params(axis="both", labelsize=7)
            ax.grid(True, alpha=0.3)

            # Scientific notation for small values (regularization)
            if loss_type == "regularization":
                ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved refinement loss plot to {output_path}")


__all__ = [
    "plot_refinement_history",
]

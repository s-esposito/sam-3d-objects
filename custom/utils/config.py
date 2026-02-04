"""
Configuration classes for the SAM3D-Objects pipeline.

This module contains dataclasses and configuration objects used across
the pipeline for pose refinement and other operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import yaml


@dataclass
class RefinementConfig:
    """
    Configuration for pose refinement optimization.

    This dataclass contains all hyperparameters for the differentiable
    rendering-based pose refinement process, including learning rates,
    loss weights, and optimization settings.

    Attributes
    ----------
    num_iterations : int
        Number of optimization iterations. Default: 100.
    lr_rotation : float
        Learning rate for rotation parameters. Default: 0.01.
    lr_translation : float
        Learning rate for translation parameters. Default: 0.001.
    lr_scale : float
        Learning rate for scale parameters. Default: 0.001.
    refine_scale : Literal["none", "perframe", "global"]
        Scale refinement mode:
        - "none": Do not refine scale, use initial predicted scale
        - "perframe": Refine scale independently for each frame
        - "global": Optimize a single scale for all frames across the sequence
    batch_size : int
        Batch size for global scale refinement. 0 or negative = all frames.
    rgb_loss_type : Literal["l1", "l2"]
        Type of RGB loss function. "l1" is more robust to outliers than "l2" (MSE).
        Default: "l1".
    rgb_multiscale : bool
        Whether to compute RGB loss at multiple scales. Helps escape local minima
        by capturing both coarse alignment and fine details. Default: True.
    rgb_multiscale_scales : tuple of float
        Scale factors for multi-scale loss. Default: (1.0, 0.5, 0.25).
        1.0 = full resolution, 0.5 = half resolution, etc.
    rgb_multiscale_weights : tuple of float
        Weights for each scale level. Should sum to 1.0 for consistent loss magnitude.
        Default: (0.5, 0.3, 0.2) - emphasizes full resolution while using coarse
        scales to guide global alignment.
    rgb_ssim_weight : float
        Weight for SSIM (structural similarity) loss. SSIM captures perceptual
        similarity and is more robust to small misalignments than pixel-wise losses.
        Set > 0 to enable. Default: 0.0.
    silhouette_weight : float
        Master weight for all silhouette losses. Set > 0 to enable. Default: 0.0.
    silhouette_com_weight : float
        Weight for center-of-mass loss within silhouette loss. This loss directly
        guides translation by matching the centroid of rendered alpha with GT mask.
        Effective even with zero overlap. Default: 1.0.
    silhouette_sdt_weight : float
        Weight for signed distance transform loss. Uses precomputed distance field
        to guide rendered pixels toward the mask boundary. Helps escape local
        minima when initially misaligned. Default: 0.1.
    silhouette_iou_weight : float
        Weight for soft IoU (intersection over union) loss. Provides fine-grained
        shape matching once roughly aligned. Default: 1.0.
    use_regularization : bool
        Whether to use regularization loss (penalizes deviation from initial pose).
    regularization_weight : float
        Weight for regularization loss. Default: 0.001.
    use_flow : bool
        Whether to use optical flow loss in refinement. Default: False.
    flow_weight : float
        Weight for optical flow correspondence loss. Default: 0.1.
    flow_model : Optional[Any]
        Cached flow model for correspondence loss (lazy loaded).
    verbose : bool
        Whether to print detailed logs during optimization.
    log_interval : int
        How often to log progress (every N iterations).

    Examples
    --------
    >>> config = RefinementConfig(
    ...     num_iterations=50,
    ...     refine_scale="perframe",
    ...     use_regularization=True
    ... )
    >>> config.lr_rotation
    0.01
    """

    # Optimization iterations
    num_iterations: int = 100

    # Learning rates
    lr_rotation: float = 0.01
    lr_translation: float = 0.001
    lr_scale: float = 0.001

    # Scale refinement mode
    refine_scale: Literal["none", "perframe", "global"] = "none"

    # Batch size for global scale refinement (0 or negative = all frames)
    batch_size: int = 0

    # RGB loss configuration
    # Note: GT background is always masked to black for full-image comparison
    rgb_loss_type: Literal["l1", "l2"] = "l1"
    rgb_multiscale: bool = True
    rgb_multiscale_scales: Tuple[float, ...] = (1.0, 0.5, 0.25)
    rgb_multiscale_weights: Tuple[float, ...] = (0.5, 0.3, 0.2)
    rgb_ssim_weight: float = 0.0  # SSIM loss weight (0 = disabled)

    # Silhouette loss weights
    # The silhouette loss combines three components for robust pose optimization:
    # 1. Center-of-mass loss: directly guides translation
    # 2. Signed distance transform loss: guides all params, escapes local minima
    # 3. Soft IoU loss: fine-grained shape matching once roughly aligned
    silhouette_weight: float = 0.0  # Master weight for all silhouette losses
    silhouette_com_weight: float = 1.0  # Center-of-mass loss weight
    silhouette_sdt_weight: float = 0.1  # Signed distance transform loss weight
    silhouette_iou_weight: float = 1.0  # Soft IoU loss weight

    # Regularization
    use_regularization: bool = False
    regularization_weight: float = 0.001

    # Optical flow correspondence loss
    use_flow: bool = False
    flow_weight: float = 0.1

    # Flow model for correspondence loss (lazy loaded)
    flow_model: Optional[Any] = field(default=None, repr=False)

    # Logging
    verbose: bool = True
    log_interval: int = 20

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        if self.refine_scale not in ("none", "perframe", "global"):
            raise ValueError(
                f"refine_scale must be 'none', 'perframe', or 'global', "
                f"got '{self.refine_scale}'"
            )
        if self.num_iterations <= 0:
            raise ValueError(f"num_iterations must be positive, got {self.num_iterations}")
        if self.rgb_loss_type not in ("l1", "l2"):
            raise ValueError(
                f"rgb_loss_type must be 'l1' or 'l2', got '{self.rgb_loss_type}'"
            )
        if self.rgb_multiscale:
            if len(self.rgb_multiscale_scales) != len(self.rgb_multiscale_weights):
                raise ValueError(
                    f"rgb_multiscale_scales and rgb_multiscale_weights must have same length, "
                    f"got {len(self.rgb_multiscale_scales)} and {len(self.rgb_multiscale_weights)}"
                )

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "RefinementConfig":
        """
        Load configuration from a YAML file.

        Parameters
        ----------
        yaml_path : str or Path
            Path to the YAML configuration file.

        Returns
        -------
        RefinementConfig
            Configuration object loaded from the YAML file.

        Examples
        --------
        >>> config = RefinementConfig.from_yaml("configs/refinement.yaml")
        >>> config.num_iterations
        100
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        # Convert lists to tuples for tuple fields
        if "rgb_multiscale_scales" in data and isinstance(data["rgb_multiscale_scales"], list):
            data["rgb_multiscale_scales"] = tuple(data["rgb_multiscale_scales"])
        if "rgb_multiscale_weights" in data and isinstance(data["rgb_multiscale_weights"], list):
            data["rgb_multiscale_weights"] = tuple(data["rgb_multiscale_weights"])

        # Remove any keys not in the dataclass (e.g., comments or extra fields)
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        return cls(**filtered_data)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary (for JSON serialization).

        Returns
        -------
        dict
            Dictionary representation of the configuration.
        """
        return {
            "num_iterations": self.num_iterations,
            "lr_rotation": self.lr_rotation,
            "lr_translation": self.lr_translation,
            "lr_scale": self.lr_scale,
            "refine_scale": self.refine_scale,
            "batch_size": self.batch_size,
            "rgb_loss_type": self.rgb_loss_type,
            "rgb_multiscale": self.rgb_multiscale,
            "rgb_multiscale_scales": list(self.rgb_multiscale_scales),
            "rgb_multiscale_weights": list(self.rgb_multiscale_weights),
            "rgb_ssim_weight": self.rgb_ssim_weight,
            "silhouette_weight": self.silhouette_weight,
            "silhouette_com_weight": self.silhouette_com_weight,
            "silhouette_sdt_weight": self.silhouette_sdt_weight,
            "silhouette_iou_weight": self.silhouette_iou_weight,
            "use_regularization": self.use_regularization,
            "regularization_weight": self.regularization_weight,
            "use_flow": self.use_flow,
            "flow_weight": self.flow_weight,
            "verbose": self.verbose,
            "log_interval": self.log_interval,
        }


def load_refinement_config(config_path: Optional[Union[str, Path]] = None) -> RefinementConfig:
    """
    Load refinement configuration from a YAML file or return defaults.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to the YAML configuration file. If None, returns default config.

    Returns
    -------
    RefinementConfig
        Configuration object.
    """
    if config_path is None:
        return RefinementConfig()
    return RefinementConfig.from_yaml(config_path)


__all__ = ["RefinementConfig", "load_refinement_config"]

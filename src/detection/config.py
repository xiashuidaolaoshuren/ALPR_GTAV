"""
Detection Module Configuration

Handles loading and validation of detection-specific configuration parameters
from the pipeline configuration file.
"""

import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class DetectionConfig:
    """Configuration class for the detection module."""

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize detection configuration from dictionary.

        Args:
            config_dict: Dictionary containing detection configuration parameters.
                        Expected to have 'detection' key with nested parameters.

        Raises:
            KeyError: If required configuration keys are missing.
            ValueError: If configuration values are invalid.
        """
        if "detection" not in config_dict:
            raise KeyError("Missing 'detection' key in configuration")

        detection_config = config_dict["detection"]

        # Load configuration parameters
        self.model_path = detection_config.get("model_path", "models/detection/yolov8n.pt")
        self.confidence_threshold = detection_config.get("confidence_threshold", 0.25)
        self.iou_threshold = detection_config.get("iou_threshold", 0.45)
        self.img_size = detection_config.get("img_size", 640)
        self.device = detection_config.get("device", "cuda")
        self.max_det = detection_config.get("max_det", 100)

        logger.info(
            f"Detection config loaded: model_path={self.model_path}, "
            f"conf={self.confidence_threshold}, iou={self.iou_threshold}"
        )

    def validate(self) -> bool:
        """
        Validate configuration parameters.

        Returns:
            True if configuration is valid.

        Raises:
            FileNotFoundError: If model_path does not exist.
            ValueError: If threshold values are out of valid range.
        """
        # Validate model path existence
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found: {self.model_path}")
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}. "
                "Please download the model or update the path in configs/pipeline_config.yaml"
            )

        # Validate confidence threshold
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                "Confidence threshold must be between 0.0 and 1.0, "
                f"got {self.confidence_threshold}"
            )

        # Validate IOU threshold
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError(
                "IOU threshold must be between 0.0 and 1.0, " f"got {self.iou_threshold}"
            )

        # Validate image size
        if self.img_size <= 0:
            raise ValueError(f"Image size must be positive, got {self.img_size}")

        # Validate device
        if self.device not in ["cuda", "cpu", "auto"]:
            logger.warning(
                f"Device '{self.device}' not in ['cuda', 'cpu', 'auto']. "
                "Will attempt to use it anyway."
            )

        logger.info("Detection configuration validated successfully")
        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary containing all configuration parameters.
        """
        return {
            "model_path": self.model_path,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "img_size": self.img_size,
            "device": self.device,
            "max_det": self.max_det,
        }

    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"DetectionConfig(model_path='{self.model_path}', "
            f"confidence_threshold={self.confidence_threshold}, "
            f"iou_threshold={self.iou_threshold}, "
            f"img_size={self.img_size}, "
            f"device='{self.device}', "
            f"max_det={self.max_det})"
        )

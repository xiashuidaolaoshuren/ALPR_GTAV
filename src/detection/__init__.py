"""
License Plate Detection Module

Uses YOLOv8 for detecting license plates in images/video frames.
Provides model loading, inference, and visualization capabilities.
"""

import logging

# Configure module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Import main functions for convenient access
from .model import load_detection_model, detect_plates, batch_detect_plates, validate_model
from .utils import (
    draw_bounding_boxes,
    compute_iou,
    filter_detections_by_size,
    crop_detections
)
from .config import DetectionConfig

__all__ = [
    # Model functions
    'load_detection_model',
    'detect_plates',
    'batch_detect_plates',
    'validate_model',
    # Utility functions
    'draw_bounding_boxes',
    'compute_iou',
    'filter_detections_by_size',
    'crop_detections',
    # Configuration
    'DetectionConfig',
]

__version__ = '0.1.0'

logger.info(f"Detection module initialized (version {__version__})")

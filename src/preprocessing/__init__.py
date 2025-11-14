"""
Image Preprocessing Module

Provides utilities for license plate image enhancement to improve OCR accuracy.
Includes grayscale conversion, resizing, CLAHE enhancement, and validation.

Main Functions:
    - preprocess_plate: Main preprocessing pipeline
    - resize_maintaining_aspect: Aspect ratio-preserving resize
    - apply_clahe: CLAHE contrast enhancement
    - validate_image: Image validation
    - batch_preprocess_plates: Batch processing
    - save_preprocessed_image: Save to disk
    - calculate_image_stats: Statistical analysis

Note:
    Cropping functionality is in src.detection.utils.crop_detections() and should be reused.

"""

import logging

# Import main preprocessing functions
from .image_enhancement import preprocess_plate, resize_maintaining_aspect, apply_clahe

# Import utility functions
from .utils import (
    validate_image,
    batch_preprocess_plates,
    save_preprocessed_image,
    calculate_image_stats,
)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Package metadata
__version__ = "0.1.0"
__all__ = [
    "preprocess_plate",
    "resize_maintaining_aspect",
    "apply_clahe",
    "validate_image",
    "batch_preprocess_plates",
    "save_preprocessed_image",
    "calculate_image_stats",
]

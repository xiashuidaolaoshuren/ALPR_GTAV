"""
License Plate Recognition Module

Uses PaddleOCR for recognizing text from cropped license plate images.
Provides OCR model loading, text inference, and post-processing capabilities.
"""

import logging

# Configure module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Import main functions for convenient access
from .model import load_ocr_model, recognize_text
from .utils import filter_by_regex, score_candidate, select_best_candidate
from .config import RecognitionConfig

__all__ = [
    # Model functions
    "load_ocr_model",
    "recognize_text",
    # Utility functions
    "filter_by_regex",
    "score_candidate",
    "select_best_candidate",
    # Configuration
    "RecognitionConfig",
]

__version__ = "0.1.0"

logger.info(f"Recognition module initialized (version {__version__})")

"""
Recognition Model Module

Core functionality for license plate recognition using PaddleOCR.
Provides OCR model loading and text inference capabilities.
"""

import logging
from typing import Tuple, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


def load_ocr_model(config: dict) -> Any:
    """
    Load PaddleOCR model for license plate text recognition.
    
    Args:
        config: Configuration dictionary containing OCR parameters. Expected keys:
               - use_gpu (bool): Enable GPU acceleration (default: True)
               - use_angle_cls (bool): Enable text angle classification (default: True)
               - lang (str): Language model to use (default: 'en' for English)
               - show_log (bool): Display PaddleOCR logs (default: False)
               - use_rec (bool): Enable text recognition (default: True)
    
    Returns:
        PaddleOCR: Loaded PaddleOCR model instance ready for inference.
    
    Raises:
        ImportError: If PaddleOCR is not installed.
        RuntimeError: If model initialization fails or GPU requested but unavailable.
        ValueError: If config contains invalid parameters.
    
    Example:
        >>> config = {'use_gpu': True, 'use_angle_cls': True, 'lang': 'en'}
        >>> ocr_model = load_ocr_model(config)
        >>> print(f"OCR model loaded successfully")
    
    Note:
        - First-time usage may download model files (~200MB for English)
        - GPU acceleration provides 10x speedup over CPU
        - Angle classification essential for rotated license plates
        - Falls back to CPU gracefully if GPU unavailable
    """
    from paddleocr import PaddleOCR
    
    # Extract configuration parameters with defaults
    use_gpu = config.get('use_gpu', True)
    use_angle_cls = config.get('use_angle_cls', True)
    lang = config.get('lang', 'en')
    show_log = config.get('show_log', False)
    use_rec = config.get('use_rec', True)
    
    logger.info(f"Loading PaddleOCR model with config: use_gpu={use_gpu}, "
                f"use_angle_cls={use_angle_cls}, lang={lang}, use_rec={use_rec}")
    
    # Determine device based on GPU availability
    device = 'gpu' if use_gpu else 'cpu'
    
    # Check GPU availability if requested
    if use_gpu:
        try:
            import paddle
            if not paddle.device.is_compiled_with_cuda():
                logger.warning("CUDA not available, falling back to CPU")
                device = 'cpu'
                use_gpu = False
            elif paddle.device.cuda.device_count() == 0:
                logger.warning("No GPU devices found, falling back to CPU")
                device = 'cpu'
                use_gpu = False
            else:
                logger.info(f"GPU available, using device: {device}")
        except ImportError:
            logger.warning("PaddlePaddle not found, cannot check GPU availability. Using CPU.")
            device = 'cpu'
            use_gpu = False
        except Exception as e:
            logger.warning(f"Error checking GPU availability: {e}. Falling back to CPU.")
            device = 'cpu'
            use_gpu = False
    
    try:
        # Initialize PaddleOCR
        # Note: PaddleOCR 3.x uses 'device' parameter (not 'use_gpu')
        # Note: use_angle_cls enables text angle classification for rotated text
        # Note: show_log parameter removed in PaddleOCR 3.x (controlled via logging)
        ocr_model = PaddleOCR(
            device=device,  # 'gpu' or 'cpu'
            use_angle_cls=use_angle_cls,
            lang=lang
        )
        
        logger.info(f"PaddleOCR model loaded successfully on {device}")
        
        # First-time usage note
        if not show_log:
            logger.info("Note: First-time usage may download model files (~200MB for English)")
        
        return ocr_model
        
    except Exception as e:
        logger.error(f"Failed to initialize PaddleOCR: {e}")
        raise RuntimeError(f"Failed to load OCR model: {str(e)}") from e


def recognize_text(
    preprocessed_image: np.ndarray,
    ocr_model: Any,
    config: dict
) -> Tuple[Optional[str], float]:
    """
    Recognize text from preprocessed license plate image.
    
    Performs OCR inference, applies post-processing filters (regex, scoring),
    and returns the best text candidate with confidence score.
    
    Args:
        preprocessed_image: Preprocessed plate image as numpy array (BGR or grayscale).
                          Should be output from preprocessing pipeline.
        ocr_model: Loaded PaddleOCR model instance from load_ocr_model().
        config: Configuration dictionary containing recognition parameters. Expected keys:
               - regex (str): Regex pattern for plate validation (default: '^[A-Z0-9]{6,8}$')
               - min_conf (float): Minimum confidence threshold (default: 0.3)
               - prefer_largest_box (bool): Prefer text from largest bbox (default: True)
    
    Returns:
        Tuple containing:
        - recognized_text (str or None): Recognized plate text in uppercase,
                                        or None if no valid text found
        - confidence (float): OCR confidence score [0.0-1.0],
                             or 0.0 if recognition failed
    
    Raises:
        ValueError: If preprocessed_image is invalid (wrong shape/type).
        RuntimeError: If OCR inference fails.
    
    Example:
        >>> import cv2
        >>> image = cv2.imread('cropped_plate.jpg')
        >>> text, conf = recognize_text(image, ocr_model, config)
        >>> if text:
        ...     print(f"Plate: {text} (confidence: {conf:.2f})")
        ... else:
        ...     print("No valid plate text detected")
    
    Note:
        - OCR may return multiple text lines; post-processing selects best candidate
        - Filtering uses regex pattern to validate plate format
        - Scoring formula: p * h * min(L/8, 1) where p=confidence, h=normalized height, L=length
        - Returns None if no candidates pass regex filter or meet confidence threshold
    """
    pass

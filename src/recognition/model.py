"""
Recognition Model Module

Core functionality for license plate recognition using PaddleOCR.
Provides OCR model loading and text inference capabilities.
"""

import logging
from typing import Tuple, Optional, Any
import numpy as np
import cv2

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
    # Pre-import torch to avoid PaddleOCR DLL load issues on Windows environments.
    import torch  # type: ignore
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
               - regex (str): Regex pattern for plate validation (default: r'^\\d{2}[A-Z]{3}\\d{3}$')
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
        - Filtering uses regex pattern to validate plate format (GTA V: r'^\\d{2}[A-Z]{3}\\d{3}$')
        - Scoring formula: p * h * min(L/8, 1) where p=confidence, h=normalized height, L=length
        - Returns None if no candidates pass regex filter or meet confidence threshold
    """
    from src.recognition.utils import filter_by_regex, score_candidate, select_best_candidate
    
    # Validate input image
    if preprocessed_image is None or not isinstance(preprocessed_image, np.ndarray):
        raise ValueError("preprocessed_image must be a valid numpy array")
    if preprocessed_image.size == 0:
        raise ValueError("preprocessed_image is empty")
    if len(preprocessed_image.shape) not in [2, 3]:
        raise ValueError(f"preprocessed_image must be 2D or 3D array, got shape {preprocessed_image.shape}")
    
    # Extract configuration parameters with defaults
    regex_pattern = config.get('regex', r'^\d{2}[A-Z]{3}\d{3}$')  # GTA V format
    min_conf = config.get('min_conf', 0.3)
    
    logger.info(f"Running OCR inference with regex pattern: {regex_pattern}, min_conf: {min_conf}")
    
    # Step 0: Convert grayscale to BGR if needed (PaddleOCR 3.x requires 3-channel images)
    if len(preprocessed_image.shape) == 2:
        logger.debug("Converting grayscale to BGR for PaddleOCR compatibility")
        preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2BGR)
    
    # Step 1: Run OCR inference
    try:
        # Use predict() method (PaddleOCR 3.x)
        result = ocr_model.predict(preprocessed_image)
        logger.debug(f"OCR inference completed, result type: {type(result)}")
    except IndexError as e:
        # PaddleOCR sometimes throws IndexError (e.g., "tuple index out of range") 
        # for certain image characteristics - handle gracefully
        logger.warning(f"OCR inference failed with IndexError (likely PaddleOCR internal issue): {e}")
        return None, 0.0
    except Exception as e:
        logger.error(f"OCR inference failed: {e}")
        raise RuntimeError(f"OCR inference failed: {str(e)}") from e
    
    # Step 2: Handle empty results
    if not result:
        logger.debug("OCR returned empty result")
        return None, 0.0
    
    # PaddleOCR predict() returns a list of dict results
    # Each dict contains: 'rec_texts', 'rec_scores', 'rec_polys', etc.
    # For single image: result = [{'rec_texts': [...], 'rec_scores': [...], 'rec_polys': [...]}]
    
    if not isinstance(result, list) or len(result) == 0:
        logger.debug("OCR returned empty or invalid result")
        return None, 0.0
    
    # Get first image's results
    ocr_result = result[0]
    
    # Check if result is a dict with expected keys
    if not isinstance(ocr_result, dict):
        logger.error(f"Unexpected OCR result format: {type(ocr_result)}")
        return None, 0.0
    
    # Extract recognized texts, scores, and polygons
    rec_texts = ocr_result.get('rec_texts', [])
    rec_scores = ocr_result.get('rec_scores', [])
    rec_polys = ocr_result.get('rec_polys', [])
    
    if not rec_texts or not rec_scores or not rec_polys:
        logger.debug("No text detected by OCR")
        return None, 0.0
    
    # Step 3: Extract candidates from all detected text lines
    candidates = []
    image_height = preprocessed_image.shape[0]
    
    logger.debug(f"Processing {len(rec_texts)} text lines from OCR result")
    
    # Iterate over all detected text lines
    for i, (text, confidence, bbox_points) in enumerate(zip(rec_texts, rec_scores, rec_polys)):
        try:
            # Normalize text to uppercase
            text = str(text).upper().strip()
            confidence = float(confidence)
            
            # Skip empty text or low confidence
            if not text:
                logger.debug(f"Skipping empty text in line {i}")
                continue
            if confidence < min_conf:
                logger.debug(f"Skipping low confidence text '{text}' ({confidence:.3f} < {min_conf})")
                continue
            
            # Calculate bounding box height from polygon points
            bbox_array = np.array(bbox_points)
            if bbox_array.shape[0] < 2:
                logger.warning(f"Skipping line {i}: insufficient bbox points")
                continue
            
            # Height = max_y - min_y
            bbox_height = float(np.max(bbox_array[:, 1]) - np.min(bbox_array[:, 1]))
            
            # Store candidate
            candidates.append({
                'text': text,
                'confidence': confidence,
                'bbox_height': bbox_height,
                'bbox': bbox_points
            })
            
            logger.debug(f"Extracted candidate {i}: text='{text}', conf={confidence:.3f}, "
                        f"bbox_height={bbox_height:.1f}")
        
        except Exception as e:
            logger.warning(f"Error parsing line {i}: {e}")
            continue
    
    if not candidates:
        logger.debug("No valid candidates extracted from OCR results")
        return None, 0.0
    
    logger.info(f"Extracted {len(candidates)} candidates before filtering")
    
    # Step 4: Filter candidates by regex pattern
    valid_candidates = []
    for candidate in candidates:
        if filter_by_regex(candidate['text'], regex_pattern):
            valid_candidates.append(candidate)
            logger.debug(f"Candidate '{candidate['text']}' passed regex filter")
        else:
            logger.debug(f"Candidate '{candidate['text']}' failed regex filter")
    
    if not valid_candidates:
        logger.info(f"No candidates match GTA V plate format: {regex_pattern}")
        logger.debug(f"Rejected candidates: {[c['text'] for c in candidates]}")
        return None, 0.0
    
    logger.info(f"{len(valid_candidates)} candidates passed regex filter")
    
    # Step 5: Score each valid candidate
    for candidate in valid_candidates:
        try:
            candidate['score'] = score_candidate(
                text=candidate['text'],
                confidence=candidate['confidence'],
                bbox_height=candidate['bbox_height'],
                image_height=image_height
            )
        except Exception as e:
            logger.warning(f"Error scoring candidate '{candidate['text']}': {e}")
            candidate['score'] = 0.0
    
    # Step 6: Select best candidate
    best_text, best_confidence = select_best_candidate(valid_candidates)
    
    if best_text:
        logger.info(f"Final recognition result: '{best_text}' (confidence: {best_confidence:.3f})")
    else:
        logger.info("No valid plate text recognized after filtering and scoring")
    
    return best_text, best_confidence

"""
Detection Model Module

Core functionality for license plate detection using YOLOv8.
Provides model loading and inference capabilities.
"""

import os
import logging
from typing import List, Tuple
import numpy as np
from ultralytics import YOLO
import torch

logger = logging.getLogger(__name__)


def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: First box as (x1, y1, x2, y2)
        box2: Second box as (x1, y1, x2, y2)

    Returns:
        IoU score between 0.0 and 1.0
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection coordinates
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    # Calculate intersection area
    if x2_i < x1_i or y2_i < y1_i:
        intersection = 0.0
    else:
        intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    # Avoid division by zero
    if union == 0:
        return 0.0

    return intersection / union


def apply_nms(
    detections: List[Tuple[int, int, int, int, float]], iou_threshold: float = 0.3
) -> List[Tuple[int, int, int, int, float]]:
    """
    Apply Non-Maximum Suppression to remove duplicate/overlapping detections.

    This is an additional NMS pass after YOLOv8's internal NMS to ensure
    we only keep one detection per unique plate (removes near-duplicates).

    Args:
        detections: List of (x1, y1, x2, y2, confidence) tuples
        iou_threshold: IoU threshold for considering boxes as duplicates.
                      Lower values are more aggressive (0.3 recommended).

    Returns:
        Filtered list of detections with duplicates removed
    """
    if not detections:
        return []

    # Sort by confidence (descending)
    sorted_dets = sorted(detections, key=lambda x: x[4], reverse=True)

    keep = []

    for current in sorted_dets:
        # Check if current box overlaps significantly with any kept box
        should_keep = True

        for kept in keep:
            iou = calculate_iou(current[:4], kept[:4])

            if iou > iou_threshold:
                # Overlaps too much with an already kept detection
                should_keep = False
                logger.debug(f"Filtered duplicate detection (IoU={iou:.3f} > {iou_threshold})")
                break

        if should_keep:
            keep.append(current)

    if len(keep) < len(sorted_dets):
        logger.info(
            f"NMS filtered {len(sorted_dets) - len(keep)} duplicate detections "
            f"({len(sorted_dets)} → {len(keep)})"
        )

    return keep


def load_detection_model(model_path: str, device: str = "auto"):
    """
    Load pre-trained YOLOv8 model for license plate detection.

    Args:
        model_path: Path to YOLOv8 .pt weights file. Should be a valid file path
                   pointing to a trained YOLOv8 model in PyTorch format.
        device: Device to load model on. Options:
               - 'cuda': Use GPU acceleration (requires CUDA)
               - 'cpu': Use CPU only
               - 'auto': Automatically select GPU if available, otherwise CPU

    Returns:
        Loaded YOLO model instance ready for inference.

    Raises:
        FileNotFoundError: If model_path does not exist.
        RuntimeError: If model fails to load due to compatibility issues
                     or corrupted weights.
        ImportError: If required dependencies (ultralytics) are not installed.

    Example:
        >>> model = load_detection_model('models/detection/yolov8n.pt')
        >>> print(f"Model loaded on device: {model.device}")

    Note:
        - The model file can be large (10-100 MB) and should not be committed to Git
        - First-time usage may download additional dependencies
        - GPU acceleration significantly improves inference speed
    """
    # Validate model path
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Please download the model using: python models/detection/download_model.py"
        )

    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Auto-selected device: {device}")

    logger.info(f"Loading YOLOv8 model from: {model_path}")
    logger.info(f"Target device: {device}")

    try:
        # Load YOLO model
        model = YOLO(model_path)

        # YOLOv8 handles device placement internally, but we can specify it
        if hasattr(model, "to"):
            model.to(device)

        # Log model information
        logger.info("✓ Model loaded successfully")
        logger.info(f"  Model type: {type(model).__name__}")

        # Get model details if available
        if hasattr(model, "names"):
            logger.info(f"  Classes: {len(model.names) if model.names else 'N/A'}")

        return model

    except Exception as e:
        logger.error(f"Failed to load model: {type(e).__name__}: {str(e)}")
        raise RuntimeError(
            f"Model loading failed: {str(e)}\n"
            "The model file may be corrupted or incompatible. "
            "Try re-downloading it using: python models/detection/download_model.py"
        )


def detect_plates(
    frame: np.ndarray,
    model,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    min_area: int = 1000,
) -> List[Tuple[int, int, int, int, float]]:
    """
    Detect license plates in a single frame.

    Args:
        frame: Input image in BGR format (OpenCV convention). Should be a
              numpy array with shape (height, width, 3).
        model: Loaded YOLO model instance from load_detection_model().
        conf_threshold: Confidence threshold for detections (0.0 to 1.0).
                       Detections with confidence below this value are filtered out.
                       Lower values detect more plates but increase false positives.
        iou_threshold: IOU (Intersection over Union) threshold for Non-Maximum
                      Suppression (0.0 to 1.0). Controls how much overlap is allowed
                      between detections. Higher values allow more overlapping boxes.
        min_area: Minimum bounding box area (pixels) to filter tiny false positives.
                 Default 1000px filters boxes smaller than ~32×32. Set to 0 to disable.

    Returns:
        List of detections, where each detection is a tuple:
        (x1, y1, x2, y2, confidence)
        - x1, y1: Top-left corner coordinates (integers)
        - x2, y2: Bottom-right corner coordinates (integers)
        - confidence: Detection confidence score (float, 0.0 to 1.0)

        Returns empty list if no plates are detected.

    Raises:
        ValueError: If frame is not a valid numpy array or has incorrect shape.
        RuntimeError: If inference fails due to model or input issues.

    Example:
        >>> import cv2
        >>> frame = cv2.imread('test_image.jpg')
        >>> detections = detect_plates(frame, model, conf_threshold=0.5)
        >>> print(f"Detected {len(detections)} license plates")
        >>> for x1, y1, x2, y2, conf in detections:
        ...     print(f"Plate at ({x1},{y1})-({x2},{y2}) with confidence {conf:.2f}")

    Note:
        - Input frame should not be preprocessed (model handles resizing internally)
        - Coordinates are in the original image space (not normalized)
        - For batch processing, call this function for each frame
    """
    # Validate input frame
    if not isinstance(frame, np.ndarray):
        raise ValueError(f"Frame must be a numpy array, got {type(frame)}")

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Frame must have shape (height, width, 3), got {frame.shape}")

    logger.debug(f"Running detection on frame of shape {frame.shape}")

    try:
        # Run YOLOv8 inference
        results = model.predict(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)

        detections = []

        # Parse results
        for result in results:
            # Check if boxes exist
            if not hasattr(result, "boxes") or result.boxes is None:
                logger.debug("No boxes found in results")
                continue

            boxes = result.boxes

            # Extract each detection
            for box in boxes:
                # Get coordinates in xyxy format (x1, y1, x2, y2)
                if hasattr(box, "xyxy") and len(box.xyxy) > 0:
                    coords = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = coords

                    # Get confidence score
                    confidence = float(box.conf[0].cpu().numpy())

                    # Convert to integers and append
                    detections.append((int(x1), int(y1), int(x2), int(y2), confidence))

        logger.info(f"Detected {len(detections)} plates with conf >= {conf_threshold:.2f}")

        # Filter by minimum area to remove tiny false positives
        if min_area > 0:
            original_count = len(detections)
            detections = [
                det for det in detections if (det[2] - det[0]) * (det[3] - det[1]) >= min_area
            ]
            if len(detections) < original_count:
                logger.info(
                    f"Filtered {original_count - len(detections)} small detections "
                    f"(area < {min_area}px)"
                )

        # Apply additional NMS to remove near-duplicate detections
        # (more aggressive than YOLOv8's internal NMS)
        if len(detections) > 1:
            detections = apply_nms(detections, iou_threshold=0.3)

        # Log detection details in debug mode
        if detections:
            logger.debug("Detection details:")
            for i, (x1, y1, x2, y2, conf) in enumerate(detections, 1):
                w, h = x2 - x1, y2 - y1
                logger.debug(f"  {i}. ({x1},{y1})-({x2},{y2}) size={w}x{h} conf={conf:.3f}")

        return detections

    except Exception as e:
        logger.error(f"Detection failed: {type(e).__name__}: {str(e)}")
        raise RuntimeError(
            f"Detection inference failed: {str(e)}\n"
            "Please check that the model and input are compatible."
        )


def batch_detect_plates(
    frames: List[np.ndarray], model, conf_threshold: float = 0.25, iou_threshold: float = 0.45
) -> List[List[Tuple[int, int, int, int, float]]]:
    """
    Detect license plates in multiple frames (batch processing).

    Args:
        frames: List of input images in BGR format.
        model: Loaded YOLO model instance.
        conf_threshold: Confidence threshold for detections.
        iou_threshold: IOU threshold for NMS.

    Returns:
        List of detection lists, one for each input frame.
        Each detection list has the same format as detect_plates().

    Raises:
        ValueError: If frames list is empty or contains invalid arrays.
        RuntimeError: If batch inference fails.

    Note:
        - Batch processing is more efficient than processing frames individually
        - All frames should have similar dimensions for optimal performance
        - May require significant GPU memory for large batches
    """
    # Implementation for future optimization
    logger.warning("batch_detect_plates() not yet implemented - future task")


def validate_model(model) -> bool:
    """
    Validate that the loaded model can perform inference.

    Args:
        model: Loaded YOLO model instance to validate.

    Returns:
        True if model validation succeeds, False otherwise.

    Raises:
        RuntimeError: If validation fails critically.

    Note:
        - Creates a dummy input (e.g., 640x640 random image) for testing
        - Useful for debugging model loading issues
        - Should be called after load_detection_model() in production
    """
    logger.info("Validating model with dummy input...")

    try:
        # Create dummy input image (640x640 RGB)
        dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # Run inference
        results = model.predict(dummy_input, verbose=False)

        # Check if results are valid
        if results is None:
            logger.error("Model returned None results")
            return False

        logger.info("✓ Model validation successful")
        logger.info(f"  Inference test passed with dummy {dummy_input.shape} input")

        return True

    except Exception as e:
        logger.error(f"Model validation failed: {type(e).__name__}: {str(e)}")
        raise RuntimeError(
            f"Model validation failed: {str(e)}\n"
            "The model may not be compatible with the current environment."
        )

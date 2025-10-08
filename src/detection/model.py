"""
Detection Model Module

Core functionality for license plate detection using YOLOv8.
Provides model loading and inference capabilities.
"""

import os
import logging
from typing import List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


def load_detection_model(model_path: str, device: str = 'auto'):
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
    try:
        from ultralytics import YOLO
        import torch
    except ImportError as e:
        logger.error(f"Required dependencies not found: {e}")
        raise ImportError(
            "Ultralytics YOLO not installed. Please install with: pip install ultralytics torch"
        )
    
    # Validate model path
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Please download the model using: python models/detection/download_model.py"
        )
    
    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Auto-selected device: {device}")
    
    logger.info(f"Loading YOLOv8 model from: {model_path}")
    logger.info(f"Target device: {device}")
    
    try:
        # Load YOLO model
        model = YOLO(model_path)
        
        # YOLOv8 handles device placement internally, but we can specify it
        if hasattr(model, 'to'):
            model.to(device)
        
        # Log model information
        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Model type: {type(model).__name__}")
        
        # Get model details if available
        if hasattr(model, 'names'):
            logger.info(f"  Classes: {len(model.names) if model.names else 'N/A'}")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {type(e).__name__}: {str(e)}")
        raise RuntimeError(
            f"Model loading failed: {str(e)}\n"
            f"The model file may be corrupted or incompatible. "
            f"Try re-downloading it using: python models/detection/download_model.py"
        )


def detect_plates(
    frame: np.ndarray,
    model,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
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
        raise ValueError(
            f"Frame must have shape (height, width, 3), got {frame.shape}"
        )
    
    logger.debug(f"Running detection on frame of shape {frame.shape}")
    
    try:
        # Run YOLOv8 inference
        results = model.predict(
            frame,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        detections = []
        
        # Parse results
        for result in results:
            # Check if boxes exist
            if not hasattr(result, 'boxes') or result.boxes is None:
                logger.debug("No boxes found in results")
                continue
            
            boxes = result.boxes
            
            # Extract each detection
            for box in boxes:
                # Get coordinates in xyxy format (x1, y1, x2, y2)
                if hasattr(box, 'xyxy') and len(box.xyxy) > 0:
                    coords = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = coords
                    
                    # Get confidence score
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # Convert to integers and append
                    detections.append((
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                        confidence
                    ))
        
        logger.info(f"Detected {len(detections)} plates with conf >= {conf_threshold:.2f}")
        
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
            f"Please check that the model and input are compatible."
        )


def batch_detect_plates(
    frames: List[np.ndarray],
    model,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
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
    pass


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
            f"The model may not be compatible with the current environment."
        )

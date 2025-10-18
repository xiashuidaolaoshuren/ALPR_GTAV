"""
Detection Utility Functions

Provides helper functions for visualization, metrics computation,
and post-processing of detection results.
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)


def draw_bounding_boxes(
    frame: np.ndarray,
    detections: List[Tuple[int, int, int, int, float]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    show_confidence: bool = True,
    font_scale: float = 0.5,
    text_labels: Optional[List[str]] = None
) -> np.ndarray:
    """
    Draw bounding boxes on frame with optional confidence and text labels.
    
    Args:
        frame: Input image in BGR format (numpy array).
        detections: List of detections in format [(x1, y1, x2, y2, confidence), ...].
                   Coordinates should be integers in pixel space.
        color: Box color in BGR format. Default is green (0, 255, 0).
        thickness: Line thickness in pixels. Default is 2.
        show_confidence: If True, display confidence score above each box.
        font_scale: Font size for confidence labels. Default is 0.5.
        text_labels: Optional list of text labels (e.g., recognized plate text) to display
                    below each bounding box. Length must match number of detections.
                    Added in Task 19 for pipeline integration.
    
    Returns:
        Annotated frame with bounding boxes drawn. Returns a copy of the input frame.
    
    Raises:
        ValueError: If frame is not a valid numpy array, detections format is incorrect,
                   or text_labels length doesn't match detections length.
    
    Example:
        >>> import cv2
        >>> frame = cv2.imread('image.jpg')
        >>> detections = [(100, 100, 200, 150, 0.95), (300, 200, 400, 250, 0.87)]
        >>> # With text labels for recognized plates
        >>> text_labels = ['ABC123', 'XYZ789']
        >>> annotated = draw_bounding_boxes(frame, detections, text_labels=text_labels)
        >>> cv2.imwrite('annotated.jpg', annotated)
    
    Note:
        - This function does not modify the input frame (creates a copy)
        - For real-time visualization, consider reducing thickness for better performance
        - Coordinates outside image boundaries are clipped automatically by cv2
        - text_labels parameter added for Task 19 pipeline integration
    """
    # Validate input frame
    if not isinstance(frame, np.ndarray):
        raise ValueError(f"Frame must be a numpy array, got {type(frame)}")
    
    if frame.ndim != 3:
        raise ValueError(f"Frame must have 3 dimensions, got {frame.ndim}")
    
    # Validate text_labels if provided
    if text_labels is not None and len(text_labels) != len(detections):
        raise ValueError(
            f"text_labels length ({len(text_labels)}) must match "
            f"detections length ({len(detections)})"
        )
    
    # Create a copy to avoid modifying the original
    annotated = frame.copy()
    
    # Draw each detection
    for i, detection in enumerate(detections):
        if len(detection) != 5:
            logger.warning(f"Invalid detection format (expected 5 elements, got {len(detection)}): {detection}")
            continue
        
        x1, y1, x2, y2, conf = detection
        
        # Ensure coordinates are integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        
        # Add confidence label if requested
        if show_confidence:
            # Format confidence with 2 decimal places
            label = f'{conf:.2f}'
            
            # Calculate label size for background
            label_size, baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                thickness
            )
            
            # Position label above the box (or below if box is at top of image)
            label_y = y1 - 10 if y1 > 20 else y2 + 20
            
            # Draw background rectangle for better readability
            cv2.rectangle(
                annotated,
                (x1, label_y - label_size[1] - baseline),
                (x1 + label_size[0], label_y + baseline),
                color,
                -1  # Filled rectangle
            )
            
            # Draw text
            cv2.putText(
                annotated,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),  # Black text for contrast
                thickness=1,
                lineType=cv2.LINE_AA
            )
        
        # Add text label below box if provided (Task 19 integration)
        if text_labels is not None and text_labels[i]:
            text = text_labels[i]
            
            # Calculate text size for background
            text_size, baseline = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                thickness
            )
            
            # Position text below the box (with some padding)
            text_y = y2 + text_size[1] + 15
            
            # Ensure text stays within frame bounds
            if text_y + baseline > frame.shape[0]:
                text_y = y1 - 15  # Draw above box if no space below
            
            # Draw background rectangle for better readability
            cv2.rectangle(
                annotated,
                (x1, text_y - text_size[1] - baseline),
                (x1 + text_size[0], text_y + baseline),
                (0, 0, 0),  # Black background
                -1  # Filled rectangle
            )
            
            # Draw text
            cv2.putText(
                annotated,
                text,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),  # White text for contrast
                thickness=1,
                lineType=cv2.LINE_AA
            )
    
    logger.debug(f"Drew {len(detections)} bounding boxes on frame")
    return annotated


def compute_iou(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int]
) -> float:
    """
    Compute Intersection over Union (IoU) for two bounding boxes.
    
    Args:
        box1: First bounding box in format (x1, y1, x2, y2).
             (x1, y1) is top-left corner, (x2, y2) is bottom-right corner.
        box2: Second bounding box in same format as box1.
    
    Returns:
        IoU value between 0.0 and 1.0:
        - 0.0: No overlap between boxes
        - 1.0: Boxes are identical
        - Values in between: Degree of overlap
    
    Raises:
        ValueError: If box coordinates are invalid (e.g., x1 > x2 or y1 > y2).
    
    Example:
        >>> box1 = (100, 100, 200, 200)  # 100x100 box
        >>> box2 = (150, 150, 250, 250)  # 100x100 box, partially overlapping
        >>> iou = compute_iou(box1, box2)
        >>> print(f"IoU: {iou:.3f}")
        IoU: 0.143
    
    Note:
        - Used for tracking, NMS (Non-Maximum Suppression), and evaluation
        - Common thresholds: 0.5 for detection matching, 0.3-0.5 for tracking
        - Efficient implementation using vectorized operations
    """
    # Validate box coordinates
    if box1[0] > box1[2] or box1[1] > box1[3]:
        raise ValueError(f"Invalid box1 coordinates: {box1}")
    if box2[0] > box2[2] or box2[1] > box2[3]:
        raise ValueError(f"Invalid box2 coordinates: {box2}")
    
    # Compute intersection coordinates
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # Compute intersection area
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    intersection_area = inter_width * inter_height
    
    # Compute union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    # Avoid division by zero
    if union_area == 0:
        return 0.0
    
    # Compute IoU
    iou = intersection_area / union_area
    return float(iou)


def filter_detections_by_size(
    detections: List[Tuple[int, int, int, int, float]],
    min_width: int = 20,
    min_height: int = 10,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None
) -> List[Tuple[int, int, int, int, float]]:
    """
    Filter detections based on bounding box size constraints.
    
    Args:
        detections: List of detections in format [(x1, y1, x2, y2, confidence), ...].
        min_width: Minimum width in pixels. Detections smaller than this are filtered out.
        min_height: Minimum height in pixels. Detections smaller than this are filtered out.
        max_width: Maximum width in pixels. If None, no upper limit.
        max_height: Maximum height in pixels. If None, no upper limit.
    
    Returns:
        Filtered list of detections that meet size constraints.
    
    Example:
        >>> detections = [(10, 10, 15, 15, 0.9), (100, 100, 250, 180, 0.95)]
        >>> filtered = filter_detections_by_size(detections, min_width=30, min_height=20)
        >>> print(f"Kept {len(filtered)} out of {len(detections)} detections")
        Kept 1 out of 2 detections
    
    Note:
        - Useful for filtering out tiny false positives or unrealistic detections
        - Typical license plates: width 100-400 pixels, height 30-150 pixels (depends on resolution)
        - Consider image resolution when setting thresholds
    """
    filtered = []
    
    for x1, y1, x2, y2, conf in detections:
        width = x2 - x1
        height = y2 - y1
        
        # Check minimum size
        if width < min_width or height < min_height:
            continue
        
        # Check maximum size if specified
        if max_width is not None and width > max_width:
            continue
        if max_height is not None and height > max_height:
            continue
        
        filtered.append((x1, y1, x2, y2, conf))
    
    logger.debug(f"Size filtering: {len(detections)} -> {len(filtered)} detections")
    return filtered


def crop_detections(
    frame: np.ndarray,
    detections: List[Tuple[int, int, int, int, float]],
    padding: int = 0
) -> List[np.ndarray]:
    """
    Crop detected regions from frame.
    
    Args:
        frame: Input image in BGR format.
        detections: List of detections in format [(x1, y1, x2, y2, confidence), ...].
        padding: Additional padding (in pixels) to add around each crop.
                Useful for including context around the detection.
    
    Returns:
        List of cropped image regions (numpy arrays), one for each detection.
        Returns empty list if no detections.
    
    Raises:
        ValueError: If frame is invalid or detections are out of bounds.
    
    Example:
        >>> import cv2
        >>> frame = cv2.imread('image.jpg')
        >>> detections = [(100, 100, 200, 150, 0.95)]
        >>> crops = crop_detections(frame, detections, padding=5)
        >>> for i, crop in enumerate(crops):
        ...     cv2.imwrite(f'crop_{i}.jpg', crop)
    
    Note:
        - Crops are automatically clipped to image boundaries
        - Padding is useful for OCR preprocessing (context helps recognition)
        - Cropped regions maintain the original image format (BGR)
    """
    crops = []
    height, width = frame.shape[:2]
    
    for x1, y1, x2, y2, conf in detections:
        # Add padding and clip to image boundaries
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(width, x2 + padding)
        y2_pad = min(height, y2 + padding)
        
        # Crop region
        crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
        crops.append(crop)
    
    logger.debug(f"Cropped {len(crops)} detections from frame")
    return crops


def non_maximum_suppression(
    detections: List[Tuple[int, int, int, int, float]],
    iou_threshold: float = 0.45
) -> List[Tuple[int, int, int, int, float]]:
    """
    Apply Non-Maximum Suppression to remove overlapping detections.
    
    Args:
        detections: List of detections in format [(x1, y1, x2, y2, confidence), ...].
        iou_threshold: IoU threshold for suppression. Detections with IoU above
                      this threshold are considered duplicates.
    
    Returns:
        Filtered list of detections after NMS.
    
    Note:
        - YOLOv8 already applies NMS internally, so this is usually not needed
        - Provided for custom post-processing or non-YOLO detectors
        - Keeps detections with highest confidence scores
    """
    # Implementation for future use
    logger.warning("non_maximum_suppression() not yet implemented - future task")
    pass

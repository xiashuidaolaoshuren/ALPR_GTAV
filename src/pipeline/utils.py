"""
Pipeline Utility Functions

This module provides helper functions for pipeline operations including:
- Visualization (drawing results on frames)
- Result serialization (converting tracks to JSON/dict)
- Logging and statistics
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
import json
from datetime import datetime

from src.tracking.tracker import PlateTrack

logger = logging.getLogger(__name__)


def draw_tracks_on_frame(
    frame: np.ndarray,
    tracks: Dict[int, PlateTrack],
    show_text: bool = True,
    show_track_id: bool = True,
    show_confidence: bool = True,
    box_color: Tuple[int, int, int] = (0, 255, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    box_thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """
    Draw tracked plates with annotations on frame.

    Visualizes all active tracks by drawing bounding boxes and optional
    text labels (plate text, track ID, confidence) on a copy of the frame.

    Args:
        frame: Input frame in BGR format (will be copied)
        tracks: Dictionary of active PlateTrack objects
        show_text: Display recognized plate text
        show_track_id: Display track ID number
        show_confidence: Display OCR confidence score
        box_color: BGR color for bounding boxes (default: green)
        text_color: BGR color for text labels (default: white)
        box_thickness: Line thickness for boxes
        font_scale: Scale factor for text size

    Returns:
        np.ndarray: Annotated frame copy with visualizations

    Example:
        >>> annotated = draw_tracks_on_frame(frame, tracks, show_text=True)
        >>> cv2.imshow('ALPR Results', annotated)
        >>> cv2.waitKey(1)

    Note:
        - Only draws active tracks (is_active=True)
        - Text is drawn below the bounding box
        - Colors use BGR format (OpenCV convention)
    """
    # Create copy to avoid modifying original
    annotated = frame.copy()

    # Draw each active track
    for track_id, track in tracks.items():
        if not track.is_active:
            continue

        # Extract bbox coordinates
        x1, y1, x2, y2 = track.bbox

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, box_thickness)

        # Build text label
        label_parts = []

        if show_track_id:
            label_parts.append(f"ID:{track_id}")

        if show_text and track.text:
            label_parts.append(track.text)

        if show_confidence and track.text:
            label_parts.append(f"({track.ocr_confidence:.2f})")

        # Draw text label if there's anything to show
        if label_parts:
            label = " ".join(label_parts)

            # Calculate text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, 1)

            # Draw text background (black rectangle)
            text_x = x1
            text_y = y2 + text_height + 10

            # Ensure text stays within frame
            if text_y + baseline > frame.shape[0]:
                text_y = y1 - 10  # Draw above box if no space below

            cv2.rectangle(
                annotated,
                (text_x, text_y - text_height - baseline),
                (text_x + text_width, text_y + baseline),
                (0, 0, 0),  # Black background
                cv2.FILLED,
            )

            # Draw text
            cv2.putText(
                annotated, label, (text_x, text_y), font, font_scale, text_color, 1, cv2.LINE_AA
            )

    return annotated


def serialize_tracks(tracks: Dict[int, PlateTrack], include_inactive: bool = False) -> List[dict]:
    """
    Convert tracks to serializable dictionary format.

    Useful for saving results to JSON or logging.

    Args:
        tracks: Dictionary of PlateTrack objects
        include_inactive: Include lost/inactive tracks in output

    Returns:
        List of dictionaries, each containing:
        - track_id: Unique track identifier
        - bbox: Bounding box [x1, y1, x2, y2]
        - text: Recognized plate text (or null)
        - ocr_confidence: OCR confidence score
        - detection_confidence: Detection confidence
        - age: Track age in frames
        - is_active: Active status

    Example:
        >>> tracks_json = serialize_tracks(tracks)
        >>> with open('results.json', 'w') as f:
        ...     json.dump(tracks_json, f, indent=2)
    """
    serialized = []

    for track_id, track in tracks.items():
        # Skip inactive if not requested
        if not include_inactive and not track.is_active:
            continue

        serialized.append(
            {
                "track_id": track.id,
                "bbox": list(track.bbox),
                "text": track.text,
                "ocr_confidence": round(track.ocr_confidence, 3),
                "detection_confidence": round(track.detection_confidence, 3),
                "age": track.age,
                "frames_since_last_ocr": track.frames_since_last_ocr,
                "is_active": track.is_active,
            }
        )

    return serialized


def save_results_json(
    tracks: Dict[int, PlateTrack],
    output_path: str,
    frame_number: Optional[int] = None,
    timestamp: Optional[str] = None,
) -> None:
    """
    Save pipeline results to JSON file.

    Args:
        tracks: Dictionary of PlateTrack objects
        output_path: Path to output JSON file
        frame_number: Optional frame number for context
        timestamp: Optional timestamp string

    Example:
        >>> save_results_json(tracks, 'outputs/results_frame_100.json', frame_number=100)
    """
    # Build output dictionary
    output = {
        "timestamp": timestamp or datetime.now().isoformat(),
        "frame_number": frame_number,
        "track_count": len(tracks),
        "tracks": serialize_tracks(tracks, include_inactive=False),
    }

    # Write to file
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise


def format_track_summary(tracks: Dict[int, PlateTrack], verbose: bool = False) -> str:
    """
    Format track information as human-readable string.

    Args:
        tracks: Dictionary of PlateTrack objects
        verbose: Include detailed per-track information

    Returns:
        str: Formatted summary string

    Example:
        >>> summary = format_track_summary(tracks, verbose=True)
        >>> print(summary)
        Active Tracks: 3
        ================
        Track 1: "ABC123" (conf=0.95, age=45)
        Track 2: "XYZ789" (conf=0.87, age=12)
        Track 3: No text (age=2)
    """
    active_tracks = {tid: t for tid, t in tracks.items() if t.is_active}

    lines = []
    lines.append(f"Active Tracks: {len(active_tracks)}")

    if verbose and active_tracks:
        lines.append("=" * 40)
        for track_id, track in sorted(active_tracks.items()):
            if track.text:
                lines.append(
                    f'Track {track_id}: "{track.text}" '
                    f"(conf={track.ocr_confidence:.2f}, age={track.age})"
                )
            else:
                lines.append(f"Track {track_id}: No text (age={track.age})")

    return "\n".join(lines)


def create_side_by_side_comparison(
    original_frame: np.ndarray, annotated_frame: np.ndarray
) -> np.ndarray:
    """
    Create side-by-side comparison of original and annotated frames.

    Useful for visualization and debugging.

    Args:
        original_frame: Original input frame
        annotated_frame: Frame with annotations

    Returns:
        np.ndarray: Horizontally concatenated frames with labels

    Example:
        >>> comparison = create_side_by_side_comparison(frame, annotated)
        >>> cv2.imwrite('comparison.jpg', comparison)
    """
    # Ensure both frames have same height
    h1, h2 = original_frame.shape[0], annotated_frame.shape[0]

    if h1 != h2:
        # Resize to match height
        if h1 > h2:
            annotated_frame = cv2.resize(
                annotated_frame, (int(annotated_frame.shape[1] * h1 / h2), h1)
            )
        else:
            original_frame = cv2.resize(
                original_frame, (int(original_frame.shape[1] * h2 / h1), h2)
            )

    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(original_frame, "Original", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(annotated_frame, "ALPR Results", (10, 30), font, 1, (255, 255, 255), 2)

    # Concatenate horizontally
    comparison = np.hstack([original_frame, annotated_frame])

    return comparison


def log_pipeline_performance(
    frame_count: int, processing_time: float, track_count: int, ocr_count: int
) -> None:
    """
    Log pipeline performance metrics.

    Args:
        frame_count: Number of frames processed
        processing_time: Total processing time in seconds
        track_count: Number of active tracks
        ocr_count: Number of OCR operations performed

    Example:
        >>> import time
        >>> start = time.time()
        >>> # ... process frames ...
        >>> elapsed = time.time() - start
        >>> log_pipeline_performance(100, elapsed, 5, 23)
    """
    fps = frame_count / processing_time if processing_time > 0 else 0
    avg_time = processing_time / frame_count if frame_count > 0 else 0

    logger.info("=" * 60)
    logger.info("Pipeline Performance Summary")
    logger.info("=" * 60)
    logger.info(f"Total frames processed: {frame_count}")
    logger.info(f"Total processing time: {processing_time:.2f}s")
    logger.info(f"Average FPS: {fps:.2f}")
    logger.info(f"Average time per frame: {avg_time * 1000:.2f}ms")
    logger.info(f"Active tracks: {track_count}")
    logger.info(f"OCR operations: {ocr_count}")
    logger.info("=" * 60)

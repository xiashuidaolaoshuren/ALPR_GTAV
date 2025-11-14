"""
Tracking module for license plate tracking with OCR trigger logic.

This module implements the PlateTrack class which manages the state of tracked
license plates and determines when OCR should be performed based on configurable
criteria defined in shrimp-rules.md.
"""

from typing import Optional, Tuple


class PlateTrack:
    """
    Manages the state and lifecycle of a tracked license plate.

    This class maintains all information about a tracked plate including its
    bounding box, detection confidence, OCR results, and timing information
    for intelligent OCR triggering.

    Attributes:
        id (int): Unique track identifier assigned by ByteTrack
        bbox (Tuple[int, int, int, int]): Bounding box coordinates (x1, y1, x2, y2)
        detection_confidence (float): Latest detection confidence score
        text (Optional[str]): Recognized plate text (None if not yet recognized)
        ocr_confidence (float): Confidence score from OCR recognition
        age (int): Number of frames since track was first detected
        frames_since_last_ocr (int): Frames elapsed since last OCR run
        is_active (bool): Whether track is currently active (not lost)
    """

    def __init__(self, track_id: int, bbox: Tuple[int, int, int, int], confidence: float):
        """
        Initialize a new plate track.

        Args:
            track_id: Unique identifier for this track
            bbox: Initial bounding box (x1, y1, x2, y2)
            confidence: Initial detection confidence score
        """
        self.id = track_id
        self.bbox = bbox
        self.detection_confidence = confidence
        self.text: Optional[str] = None
        self.ocr_confidence: float = 0.0
        self.age: int = 0  # Frames since first detection
        self.frames_since_last_ocr: int = 0
        self.is_active: bool = True

    def update(self, bbox: Tuple[int, int, int, int], confidence: float) -> None:
        """
        Update track with new detection information.

        This method is called every frame when the track is successfully detected
        by the tracker. It updates the bounding box, confidence, and increments
        frame counters.

        Args:
            bbox: Updated bounding box coordinates (x1, y1, x2, y2)
            confidence: Updated detection confidence score
        """
        self.bbox = bbox
        self.detection_confidence = confidence
        self.age += 1
        self.frames_since_last_ocr += 1

    def update_text(self, text: Optional[str], confidence: float) -> None:
        """
        Update track with OCR recognition results.

        This method is called after OCR is performed on the tracked plate.
        It stores the recognized text and confidence, and resets the OCR timer.

        Args:
            text: Recognized plate text (None if recognition failed)
            confidence: OCR confidence score (0.0 if recognition failed)
        """
        self.text = text
        self.ocr_confidence = confidence
        self.frames_since_last_ocr = 0

    def should_run_ocr(self, config: dict) -> bool:
        """
        Determine whether OCR should be run for this track based on decision rules.

        This method implements the OCR trigger logic defined in shrimp-rules.md:

        1. New track detected (age == 0) → Always run OCR
        2. Stale recognition (frames_since_last_ocr >= ocr_interval) → Run OCR to refresh
        3. Low confidence retry (ocr_confidence < threshold) → Run OCR to improve
        4. High detection confidence + no text → Run OCR for strong detections
        5. Track lost (is_active == False) → Never run OCR (handled by caller)

        Args:
            config: Tracking configuration dictionary containing:
                - ocr_interval: Maximum frames between OCR runs (default: 30)
                - ocr_confidence_threshold: Minimum acceptable confidence (default: 0.7)

        Returns:
            bool: True if OCR should be run, False otherwise
        """
        # Condition 1: New track (age 0) - always recognize immediately
        if self.age == 0:
            return True

        # Condition 2: Refresh stale recognition
        ocr_interval = config.get("ocr_interval", 30)
        if self.frames_since_last_ocr >= ocr_interval:
            return True

        # Condition 3: Low confidence, retry to improve recognition
        threshold = config.get("ocr_confidence_threshold", 0.7)
        if self.text is not None and self.ocr_confidence < threshold:
            return True

        # Condition 4: High detection confidence but no text yet
        # This catches cases where we have a strong detection but haven't recognized yet
        if self.detection_confidence > 0.9 and self.text is None:
            return True

        return False

    def mark_lost(self) -> None:
        """
        Mark this track as lost (no longer actively tracked).

        This is called when ByteTrack determines the track has been lost.
        Once lost, OCR should not be run on this track (enforced by caller).
        """
        self.is_active = False

    def __repr__(self) -> str:
        """String representation for debugging."""
        status = "active" if self.is_active else "lost"
        text_display = f"'{self.text}'" if self.text else "None"
        return (
            f"PlateTrack(id={self.id}, age={self.age}, "
            f"text={text_display}, conf={self.ocr_confidence:.2f}, "
            f"status={status})"
        )

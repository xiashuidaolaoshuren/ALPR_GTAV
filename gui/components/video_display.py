"""
Video Display Component

Handles the display of processed video frames with bounding boxes
and recognized license plate text overlaid in real-time.

Author: Felix (xiashuidaolaoshuren)
Date: 2025-11-12
"""

import streamlit as st
import cv2
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class VideoDisplay:
    """
    Component for displaying video frames with ALPR overlays.
    
    This class handles:
    - Real-time frame updates using st.empty()
    - Drawing bounding boxes around detected plates
    - Overlaying recognized text with confidence scores
    - Converting frames from BGR to RGB for Streamlit
    """
    
    # Color constants (BGR format for OpenCV)
    COLOR_BOX = (0, 255, 0)        # Green for bounding boxes
    COLOR_TEXT_BG = (0, 255, 0)    # Green background for text
    COLOR_TEXT = (0, 0, 0)         # Black text
    COLOR_NO_DETECTION = (0, 165, 255)  # Orange for boxes without text
    
    # Font settings
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2
    TEXT_PADDING = 5
    
    def __init__(self, placeholder):
        """
        Initialize the video display component.
        
        Args:
            placeholder: Streamlit st.empty() placeholder for frame updates
        """
        self.placeholder = placeholder
        self._frame_count = 0
    
    def draw_detection(
        self, 
        frame: np.ndarray, 
        bbox: Tuple[int, int, int, int], 
        plate_text: Optional[str] = None, 
        confidence: Optional[float] = None,
        track_id: Optional[int] = None
    ) -> np.ndarray:
        """
        Draw detection bounding box and text overlay on frame.
        
        Args:
            frame: Video frame (BGR format)
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            plate_text: Recognized plate text (optional)
            confidence: Recognition confidence score (optional)
            track_id: Tracking ID (optional)
        
        Returns:
            Modified frame with overlays
        """
        try:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Choose color based on whether we have plate text
            box_color = self.COLOR_BOX if plate_text else self.COLOR_NO_DETECTION
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Draw track ID if available (top-left corner)
            if track_id is not None:
                track_label = f"ID:{track_id}"
                (tw, th), _ = cv2.getTextSize(
                    track_label, self.FONT, self.FONT_SCALE * 0.7, self.FONT_THICKNESS - 1
                )
                # Draw background rectangle for track ID
                cv2.rectangle(
                    frame, 
                    (x1, y1 - th - self.TEXT_PADDING * 2), 
                    (x1 + tw + self.TEXT_PADDING * 2, y1), 
                    (255, 255, 0),  # Yellow background
                    -1
                )
                # Draw track ID text
                cv2.putText(
                    frame, 
                    track_label, 
                    (x1 + self.TEXT_PADDING, y1 - self.TEXT_PADDING), 
                    self.FONT, 
                    self.FONT_SCALE * 0.7, 
                    (0, 0, 0),  # Black text
                    self.FONT_THICKNESS - 1
                )
            
            # Draw plate text and confidence if available
            if plate_text:
                # Build label string
                label = f"{plate_text}"
                if confidence is not None:
                    label += f" ({confidence:.1%})"
                
                # Calculate text size
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, self.FONT, self.FONT_SCALE, self.FONT_THICKNESS
                )
                
                # Calculate text position (above bounding box)
                text_x = x1
                text_y = y1 - text_height - self.TEXT_PADDING * 2
                
                # Adjust if text goes above frame
                if text_y < 0:
                    text_y = y2 + text_height + self.TEXT_PADDING * 2
                
                # Draw background rectangle for text
                cv2.rectangle(
                    frame,
                    (text_x, text_y - self.TEXT_PADDING),
                    (text_x + text_width + self.TEXT_PADDING * 2, text_y + text_height + baseline),
                    self.COLOR_TEXT_BG,
                    -1
                )
                
                # Draw text
                cv2.putText(
                    frame,
                    label,
                    (text_x + self.TEXT_PADDING, text_y + text_height),
                    self.FONT,
                    self.FONT_SCALE,
                    self.COLOR_TEXT,
                    self.FONT_THICKNESS
                )
            
            return frame
            
        except Exception as e:
            logger.error(f"Error drawing detection: {e}")
            return frame
    
    def draw_multiple_detections(
        self, 
        frame: np.ndarray, 
        detections: list
    ) -> np.ndarray:
        """
        Draw multiple detections on a single frame.
        
        Args:
            frame: Video frame (BGR format)
            detections: List of detection dictionaries, each containing:
                - bbox: (x1, y1, x2, y2)
                - plate_text: Optional[str]
                - confidence: Optional[float]
                - track_id: Optional[int]
        
        Returns:
            Modified frame with all overlays
        """
        for det in detections:
            frame = self.draw_detection(
                frame,
                bbox=det['bbox'],
                plate_text=det.get('plate_text'),
                confidence=det.get('confidence'),
                track_id=det.get('track_id')
            )
        return frame
    
    def update_frame(self, frame: np.ndarray, use_container_width: bool = True):
        """
        Update the displayed frame in Streamlit.
        
        Args:
            frame: Video frame to display (BGR format)
            use_container_width: Whether to use full container width
        """
        try:
            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Update placeholder with new frame
            self.placeholder.image(
                frame_rgb, 
                channels="RGB", 
                use_container_width=use_container_width
            )
            
            self._frame_count += 1
            
        except Exception as e:
            logger.error(f"Error updating frame: {e}")
    
    def display_message(self, message: str, message_type: str = "info"):
        """
        Display a message in the video placeholder.
        
        Args:
            message: Message to display
            message_type: Type of message ("info", "warning", "error", "success")
        """
        if message_type == "info":
            self.placeholder.info(message)
        elif message_type == "warning":
            self.placeholder.warning(message)
        elif message_type == "error":
            self.placeholder.error(message)
        elif message_type == "success":
            self.placeholder.success(message)
        else:
            self.placeholder.write(message)
    
    def clear(self):
        """Clear the video display."""
        self.placeholder.empty()
    
    def get_frame_count(self) -> int:
        """
        Get the number of frames displayed.
        
        Returns:
            Number of frames displayed since initialization
        """
        return self._frame_count

"""
Video Handler Utility

Handles video file upload, temporary storage, and metadata extraction
for the GTA V ALPR Streamlit application.

Author: Felix (xiashuidaolaoshuren)
Date: 2025-11-12
"""

import tempfile
import cv2
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class VideoHandler:
    """
    Handles video file operations for the Streamlit GUI.
    
    This class manages:
    - Saving uploaded files to temporary storage
    - Extracting video metadata (fps, resolution, duration)
    - Providing video capture interface
    - Cleanup of temporary files
    """
    
    def __init__(self, uploaded_file):
        """
        Initialize VideoHandler with an uploaded file.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
        """
        self.uploaded_file = uploaded_file
        self.temp_path: Optional[str] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self._video_info: Optional[Dict] = None
    
    def save_temp_file(self) -> str:
        """
        Save uploaded file to a temporary location.
        
        Returns:
            str: Path to the temporary file
            
        Raises:
            Exception: If file saving fails
        """
        try:
            # Get file extension from uploaded file
            suffix = Path(self.uploaded_file.name).suffix
            
            # Create temporary file with appropriate extension
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                # Write uploaded file content
                tmp.write(self.uploaded_file.read())
                self.temp_path = tmp.name
            
            logger.info(f"Saved uploaded video to: {self.temp_path}")
            return self.temp_path
            
        except Exception as e:
            logger.error(f"Failed to save temporary file: {e}")
            raise
    
    def get_video_info(self) -> Dict:
        """
        Extract video metadata using OpenCV.
        
        Returns:
            Dict containing:
                - fps: Frames per second
                - frame_count: Total number of frames
                - width: Video width in pixels
                - height: Video height in pixels
                - duration: Video duration in seconds
                - filename: Original filename
                
        Raises:
            Exception: If video cannot be opened or metadata extraction fails
        """
        if self._video_info is not None:
            return self._video_info
        
        if self.temp_path is None:
            raise ValueError("Must call save_temp_file() before get_video_info()")
        
        try:
            # Open video file
            self.cap = cv2.VideoCapture(self.temp_path)
            
            if not self.cap.isOpened():
                raise ValueError(f"Failed to open video file: {self.temp_path}")
            
            # Extract metadata
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate duration
            duration = frame_count / fps if fps > 0 else 0
            
            self._video_info = {
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration': duration,
                'filename': self.uploaded_file.name
            }
            
            logger.info(f"Video info extracted: {self._video_info}")
            return self._video_info
            
        except Exception as e:
            logger.error(f"Failed to extract video metadata: {e}")
            if self.cap:
                self.cap.release()
            raise
    
    def get_capture(self) -> cv2.VideoCapture:
        """
        Get OpenCV VideoCapture object for frame reading.
        
        Returns:
            cv2.VideoCapture: Video capture object
            
        Raises:
            ValueError: If video hasn't been initialized
        """
        if self.cap is None or not self.cap.isOpened():
            if self.temp_path:
                self.cap = cv2.VideoCapture(self.temp_path)
            else:
                raise ValueError("Video not initialized. Call save_temp_file() first.")
        
        return self.cap
    
    def cleanup(self):
        """
        Clean up resources: release video capture and delete temporary file.
        """
        try:
            # Release video capture
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            
            # Delete temporary file
            if self.temp_path:
                temp_file = Path(self.temp_path)
                if temp_file.exists():
                    temp_file.unlink()
                    logger.info(f"Deleted temporary file: {self.temp_path}")
                self.temp_path = None
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor: ensure cleanup when object is deleted."""
        self.cleanup()

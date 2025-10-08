"""
Video I/O utilities for reading and writing video files.

This module provides VideoReader and VideoWriter classes for handling
video processing operations with proper resource management.
"""

import cv2
import logging
from typing import Generator, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class VideoReader:
    """
    Video reader class for frame-by-frame video processing.
    
    Attributes:
        video_path (str): Path to the input video file
        cap (cv2.VideoCapture): OpenCV video capture object
        fps (float): Frames per second of the video
        total_frames (int): Total number of frames in the video
        width (int): Frame width in pixels
        height (int): Frame height in pixels
    """
    
    def __init__(self, video_path: str):
        """
        Initialize the video reader.
        
        Args:
            video_path: Path to the input video file
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video cannot be opened
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.video_path = str(video_path)
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video: {self.video_path}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Opened video: {self.video_path}")
        logger.info(f"Properties: {self.width}x{self.height} @ {self.fps:.2f} fps, {self.total_frames} frames")
    
    def read_frames(self, sample_rate: int = 1) -> Generator[Tuple[int, any], None, None]:
        """
        Generate frames from the video.
        
        Args:
            sample_rate: Process every Nth frame (1 = all frames, 2 = every other frame, etc.)
            
        Yields:
            Tuple of (frame_index, frame) where frame is a numpy array (BGR format)
        """
        frame_idx = 0
        frames_processed = 0
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            # Sample frames based on sample_rate
            if frame_idx % sample_rate == 0:
                yield frame_idx, frame
                frames_processed += 1
            
            frame_idx += 1
        
        logger.info(f"Read {frames_processed} frames (sampled from {frame_idx} total frames)")
    
    def get_frame_at(self, frame_number: int) -> Optional[any]:
        """
        Get a specific frame by frame number.
        
        Args:
            frame_number: The frame number to retrieve (0-indexed)
            
        Returns:
            Frame as numpy array, or None if frame cannot be read
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def release(self):
        """Release the video capture resources."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            logger.info(f"Released video: {self.video_path}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
    
    def __del__(self):
        """Destructor to ensure resources are released."""
        if hasattr(self, 'cap'):
            self.release()


class VideoWriter:
    """
    Video writer class for saving processed video frames.
    
    Attributes:
        output_path (str): Path to the output video file
        fps (float): Frames per second for the output video
        width (int): Frame width in pixels
        height (int): Frame height in pixels
        writer (cv2.VideoWriter): OpenCV video writer object
    """
    
    def __init__(self, output_path: str, fps: float, width: int, height: int, 
                 codec: str = 'mp4v'):
        """
        Initialize the video writer.
        
        Args:
            output_path: Path to the output video file
            fps: Frames per second
            width: Frame width in pixels
            height: Frame height in pixels
            codec: FourCC codec code (default: 'mp4v')
            
        Raises:
            ValueError: If video writer cannot be initialized
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.width = width
        self.height = height
        
        # Create output directory if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            str(self.output_path), 
            fourcc, 
            fps, 
            (width, height)
        )
        
        if not self.writer.isOpened():
            raise ValueError(f"Failed to initialize video writer: {self.output_path}")
        
        self.frames_written = 0
        logger.info(f"Initialized video writer: {self.output_path}")
        logger.info(f"Properties: {width}x{height} @ {fps:.2f} fps, codec: {codec}")
    
    def write_frame(self, frame: any):
        """
        Write a single frame to the output video.
        
        Args:
            frame: Frame as numpy array (BGR format)
        """
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            logger.warning(
                f"Frame size mismatch: expected {self.width}x{self.height}, "
                f"got {frame.shape[1]}x{frame.shape[0]}. Resizing frame."
            )
            frame = cv2.resize(frame, (self.width, self.height))
        
        self.writer.write(frame)
        self.frames_written += 1
    
    def release(self):
        """Release the video writer resources."""
        if self.writer is not None and self.writer.isOpened():
            self.writer.release()
            logger.info(f"Released video writer: {self.output_path} ({self.frames_written} frames written)")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
    
    def __del__(self):
        """Destructor to ensure resources are released."""
        if hasattr(self, 'writer'):
            self.release()

"""
Unit tests for video I/O utilities.

Tests VideoReader and VideoWriter classes for proper video processing.
"""

import unittest
import tempfile
import shutil
from pathlib import Path

import cv2
import numpy as np

from src.utils.video_io import VideoReader, VideoWriter


class TestVideoIO(unittest.TestCase):
    """Test cases for video I/O utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_video_path = self.test_dir / "test_video.mp4"
        
        # Create a simple test video
        self._create_test_video(self.test_video_path, num_frames=30, fps=10.0)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def _create_test_video(self, output_path: Path, num_frames: int = 30, 
                          fps: float = 10.0, width: int = 640, height: int = 480):
        """
        Create a simple test video with colored frames.
        
        Args:
            output_path: Path to save the video
            num_frames: Number of frames to create
            fps: Frames per second
            width: Frame width
            height: Frame height
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        for i in range(num_frames):
            # Create a frame with gradient colors
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = (i * 255 // num_frames)  # Blue channel
            frame[:, :, 1] = 128  # Green channel
            frame[:, :, 2] = (255 - i * 255 // num_frames)  # Red channel
            writer.write(frame)
        
        writer.release()
    
    def test_video_reader_initialization(self):
        """Test VideoReader initialization."""
        reader = VideoReader(str(self.test_video_path))
        
        self.assertEqual(reader.width, 640)
        self.assertEqual(reader.height, 480)
        self.assertEqual(reader.fps, 10.0)
        self.assertEqual(reader.total_frames, 30)
        
        reader.release()
    
    def test_video_reader_nonexistent_file(self):
        """Test VideoReader with nonexistent file."""
        with self.assertRaises(FileNotFoundError):
            VideoReader(str(self.test_dir / "nonexistent.mp4"))
    
    def test_video_reader_context_manager(self):
        """Test VideoReader as context manager."""
        with VideoReader(str(self.test_video_path)) as reader:
            self.assertTrue(reader.cap.isOpened())
        
        # After context exit, video should be released
        self.assertFalse(reader.cap.isOpened())
    
    def test_video_reader_read_frames(self):
        """Test reading frames from video."""
        reader = VideoReader(str(self.test_video_path))
        
        frames = list(reader.read_frames())
        self.assertEqual(len(frames), 30)
        
        # Check frame indices and shapes
        for frame_idx, (idx, frame) in enumerate(frames):
            self.assertEqual(idx, frame_idx)
            self.assertEqual(frame.shape, (480, 640, 3))
        
        reader.release()
    
    def test_video_reader_sample_rate(self):
        """Test frame sampling."""
        reader = VideoReader(str(self.test_video_path))
        
        # Read every 2nd frame
        frames = list(reader.read_frames(sample_rate=2))
        self.assertEqual(len(frames), 15)  # 30 / 2 = 15
        
        # Check frame indices (should be 0, 2, 4, 6, ...)
        for i, (frame_idx, frame) in enumerate(frames):
            self.assertEqual(frame_idx, i * 2)
        
        reader.release()
    
    def test_video_reader_get_frame_at(self):
        """Test getting specific frame."""
        reader = VideoReader(str(self.test_video_path))
        
        # Get frame at index 10
        frame = reader.get_frame_at(10)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (480, 640, 3))
        
        # Get frame beyond video length
        frame = reader.get_frame_at(100)
        self.assertIsNone(frame)
        
        reader.release()
    
    def test_video_writer_initialization(self):
        """Test VideoWriter initialization."""
        output_path = self.test_dir / "output.mp4"
        writer = VideoWriter(str(output_path), fps=10.0, width=640, height=480)
        
        self.assertEqual(writer.fps, 10.0)
        self.assertEqual(writer.width, 640)
        self.assertEqual(writer.height, 480)
        self.assertEqual(writer.frames_written, 0)
        
        writer.release()
        self.assertTrue(output_path.exists())
    
    def test_video_writer_context_manager(self):
        """Test VideoWriter as context manager."""
        output_path = self.test_dir / "output.mp4"
        
        with VideoWriter(str(output_path), fps=10.0, width=640, height=480) as writer:
            self.assertTrue(writer.writer.isOpened())
        
        # After context exit, video should be released
        self.assertFalse(writer.writer.isOpened())
        self.assertTrue(output_path.exists())
    
    def test_video_writer_write_frames(self):
        """Test writing frames to video."""
        output_path = self.test_dir / "output.mp4"
        writer = VideoWriter(str(output_path), fps=10.0, width=640, height=480)
        
        # Write 10 frames
        for i in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:, :, :] = i * 25  # Gradual brightness increase
            writer.write_frame(frame)
        
        self.assertEqual(writer.frames_written, 10)
        writer.release()
        
        # Verify the written video
        reader = VideoReader(str(output_path))
        self.assertEqual(reader.total_frames, 10)
        reader.release()
    
    def test_video_writer_frame_resize(self):
        """Test automatic frame resizing."""
        output_path = self.test_dir / "output.mp4"
        writer = VideoWriter(str(output_path), fps=10.0, width=640, height=480)
        
        # Write frame with different size (should be resized)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        writer.write_frame(frame)
        
        self.assertEqual(writer.frames_written, 1)
        writer.release()
    
    def test_video_round_trip(self):
        """Test reading and writing a complete video."""
        output_path = self.test_dir / "output.mp4"
        
        # Read original video
        reader = VideoReader(str(self.test_video_path))
        
        # Write to new video
        writer = VideoWriter(
            str(output_path),
            fps=reader.fps,
            width=reader.width,
            height=reader.height
        )
        
        for frame_idx, frame in reader.read_frames():
            writer.write_frame(frame)
        
        reader.release()
        writer.release()
        
        # Verify output video properties
        output_reader = VideoReader(str(output_path))
        self.assertEqual(output_reader.width, 640)
        self.assertEqual(output_reader.height, 480)
        self.assertEqual(output_reader.total_frames, 30)
        output_reader.release()


if __name__ == '__main__':
    unittest.main()

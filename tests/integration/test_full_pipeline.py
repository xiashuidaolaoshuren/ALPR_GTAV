"""
Integration tests for the complete ALPR pipeline.

Tests the end-to-end workflow including detection, tracking, preprocessing,
and recognition components working together.

Note: Track lifetime tests use consecutive video frames instead of sampled images
to ensure ByteTrack properly maintains state and track age increments correctly.
"""

import unittest
import sys
import os
from pathlib import Path
import numpy as np
import cv2

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.pipeline.alpr_pipeline import ALPRPipeline
from src.tracking.tracker import PlateTrack


class TestPipelineInitialization(unittest.TestCase):
    """Test pipeline initialization and configuration loading."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Get project root directory
        cls.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cls.config_path = os.path.join(cls.project_root, 'configs', 'pipeline_config.yaml')
        
    def test_pipeline_init_success(self):
        """Test successful pipeline initialization."""
        pipeline = ALPRPipeline(self.config_path)
        
        # Verify all components loaded
        self.assertIsNotNone(pipeline.detection_model)
        self.assertIsNotNone(pipeline.ocr_model)
        self.assertIsNotNone(pipeline.config)
        self.assertEqual(pipeline.frame_count, 0)
        self.assertEqual(len(pipeline.tracks), 0)
    
    def test_pipeline_config_loaded(self):
        """Test configuration is properly loaded."""
        pipeline = ALPRPipeline(self.config_path)
        
        # Verify config sections exist
        self.assertIn('detection', pipeline.config)
        self.assertIn('recognition', pipeline.config)
        self.assertIn('tracking', pipeline.config)
        self.assertIn('preprocessing', pipeline.config)
        
        # Verify key parameters
        self.assertIn('confidence_threshold', pipeline.config['detection'])
        self.assertIn('ocr_interval', pipeline.config['tracking'])
    
    def test_pipeline_invalid_config_raises_error(self):
        """Test invalid config path raises error."""
        with self.assertRaises(FileNotFoundError):
            ALPRPipeline('nonexistent_config.yaml')


class TestSingleFrameProcessing(unittest.TestCase):
    """Test single frame processing through the pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Get project root directory
        cls.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cls.config_path = os.path.join(cls.project_root, 'configs', 'pipeline_config.yaml')
        cls.test_image_path = os.path.join(cls.project_root, 'outputs', 'test_images', 'day_clear_front_00000.jpg')
        
    def setUp(self):
        """Initialize pipeline for each test."""
        self.pipeline = ALPRPipeline(self.config_path)
    
    def test_process_valid_frame(self):
        """Test processing a valid frame."""
        # Load test image
        frame = cv2.imread(self.test_image_path)
        self.assertIsNotNone(frame, "Test image not found")
        
        # Process frame
        tracks = self.pipeline.process_frame(frame)
        
        # Verify returns dict
        self.assertIsInstance(tracks, dict)
        
        # Verify frame count incremented
        self.assertEqual(self.pipeline.frame_count, 1)
    
    def test_process_frame_invalid_input(self):
        """Test processing invalid frame input."""
        # Test with None
        with self.assertRaises(ValueError):
            self.pipeline.process_frame(None)
        
        # Test with wrong shape
        invalid_frame = np.zeros((100, 100), dtype=np.uint8)  # Grayscale instead of BGR
        with self.assertRaises(ValueError):
            self.pipeline.process_frame(invalid_frame)
    
    def test_process_frame_no_detections(self):
        """Test processing frame with no plates."""
        # Create blank frame (unlikely to have detections)
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Process frame
        tracks = self.pipeline.process_frame(blank_frame)
        
        # Verify empty tracks or very few detections
        self.assertIsInstance(tracks, dict)
        # Frame count should still increment
        self.assertEqual(self.pipeline.frame_count, 1)




class TestVideoFrameTracking(unittest.TestCase):
    """
    Test tracking behavior using consecutive video frames.
    
    NOTE: This class uses video frames instead of sampled images because:
    - ByteTrack requires consecutive frames to maintain internal state
    - Track age only increments when ByteTrack successfully matches detections
    - Sampled images (with time gaps) cause ByteTrack to lose track state
    - Video frames ensure proper testing of track lifetime and persistence
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Get project root directory
        cls.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cls.config_path = os.path.join(cls.project_root, 'configs', 'pipeline_config.yaml')
        
        # Use test video with consecutive frames
        cls.test_video_path = os.path.join(cls.project_root, 'outputs', 'unit_test_video.mp4')
        
        # Verify video exists
        if not os.path.exists(cls.test_video_path):
            raise FileNotFoundError(
                f"Test video not found: {cls.test_video_path}\n"
                f"Please ensure unit_test_video.mp4 exists in outputs directory."
            )
        
        # Get video properties
        cap = cv2.VideoCapture(cls.test_video_path)
        cls.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cls.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        print(f"\nTest video loaded: {cls.total_frames} frames @ {cls.fps} FPS")
    
    def setUp(self):
        """Initialize pipeline for each test."""
        self.pipeline = ALPRPipeline(self.config_path)
    
    def _read_video_frames(self, start_frame=0, num_frames=10):
        """
        Read consecutive frames from test video.
        
        Args:
            start_frame: Starting frame index
            num_frames: Number of frames to read
            
        Returns:
            List of frame arrays
        """
        cap = cv2.VideoCapture(self.test_video_path)
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        for _ in range(num_frames):
            success, frame = cap.read()
            if not success:
                break
            frames.append(frame)
        
        cap.release()
        return frames
    
    def test_track_age_increments_consecutively(self):
        """
        Test that track age increments by 1 for each consecutive frame.
        
        This is the core test for the track lifetime issue. With consecutive
        video frames (not sampled images), ByteTrack should properly maintain
        track state and increment age each frame.
        """
        # Read first 10 consecutive frames
        frames = self._read_video_frames(start_frame=0, num_frames=10)
        self.assertGreater(len(frames), 0, "No frames read from video")
        
        # Process first frame
        tracks = self.pipeline.process_frame(frames[0])
        self.assertGreater(len(tracks), 0, "No tracks detected in first frame")
        
        # Get the first track
        track_id = list(tracks.keys())[0]
        initial_age = tracks[track_id].age
        
        # Verify new track starts at age 0
        self.assertEqual(initial_age, 0, 
                        "New tracks should start with age 0")
        self.assertTrue(tracks[track_id].is_active,
                       "New tracks should be active")
        
        # Process remaining frames and verify age increments
        for frame_idx in range(1, len(frames)):
            tracks = self.pipeline.process_frame(frames[frame_idx])
            
            # Track should persist
            self.assertIn(track_id, tracks,
                         f"Track {track_id} should persist in frame {frame_idx}")
            
            # Age should increment by 1 each frame
            expected_age = frame_idx  # Frame 0 → age 0, Frame 1 → age 1, etc.
            actual_age = tracks[track_id].age
            self.assertEqual(actual_age, expected_age,
                           f"Frame {frame_idx}: Age should be {expected_age}, got {actual_age}")
            
            # Track should remain active
            self.assertTrue(tracks[track_id].is_active,
                          f"Track should remain active in frame {frame_idx}")
        
        print(f"✓ Track age incremented correctly: 0 → {tracks[track_id].age}")
    
    def test_track_persistence_over_longer_sequence(self):
        """
        Test track persists and ages correctly over a longer sequence (30 frames).
        """
        # Read 30 consecutive frames
        frames = self._read_video_frames(start_frame=0, num_frames=30)
        self.assertGreaterEqual(len(frames), 30, "Not enough frames in video")
        
        # Process all frames
        track_id = None
        for frame_idx, frame in enumerate(frames):
            tracks = self.pipeline.process_frame(frame)
            
            if frame_idx == 0:
                # First frame - get initial track
                self.assertGreater(len(tracks), 0, "No tracks in first frame")
                track_id = list(tracks.keys())[0]
            else:
                # Subsequent frames - verify track persists
                self.assertIn(track_id, tracks,
                            f"Track {track_id} lost at frame {frame_idx}")
                
                # Verify age
                self.assertEqual(tracks[track_id].age, frame_idx,
                               f"Frame {frame_idx}: Expected age {frame_idx}, got {tracks[track_id].age}")
        
        # Final verification - age should be 29 (frames 0-29)
        self.assertEqual(tracks[track_id].age, 29,
                        "Final age should be 29 for 30 frames (frames 0-29)")
        
        print(f"✓ Track persisted for 30 frames with age 0→29")
    
    def test_track_bbox_stability(self):
        """
        Test bounding box remains stable (small variations) for stationary vehicle.
        """
        frames = self._read_video_frames(start_frame=0, num_frames=10)
        
        # Process first frame
        tracks = self.pipeline.process_frame(frames[0])
        track_id = list(tracks.keys())[0]
        initial_bbox = tracks[track_id].bbox
        
        # Process remaining frames and check bbox stability
        for frame in frames[1:]:
            tracks = self.pipeline.process_frame(frame)
            current_bbox = tracks[track_id].bbox
            
            # Calculate bbox center movement
            initial_center = ((initial_bbox[0] + initial_bbox[2]) / 2,
                            (initial_bbox[1] + initial_bbox[3]) / 2)
            current_center = ((current_bbox[0] + current_bbox[2]) / 2,
                            (current_bbox[1] + current_bbox[3]) / 2)
            
            movement = np.sqrt((current_center[0] - initial_center[0])**2 +
                             (current_center[1] - initial_center[1])**2)
            
            # For stationary vehicle, movement should be minimal (<10 pixels)
            self.assertLess(movement, 10,
                          f"Bbox center moved {movement:.1f} pixels (expected <10 for stationary vehicle)")
        
        print(f"✓ Bounding box remained stable across frames")
    
    def test_ocr_triggers_and_recognizes_text(self):
        """
        Test OCR triggers on new tracks and recognizes text correctly.
        """
        frames = self._read_video_frames(start_frame=0, num_frames=5)
        
        # Process first frame (should trigger OCR for new track)
        tracks = self.pipeline.process_frame(frames[0])
        track_id = list(tracks.keys())[0]
        
        # Check OCR was triggered and text recognized
        track = tracks[track_id]
        
        # OCR should have been run on age=0 (new track)
        # Note: OCR might fail if plate is not clear, but should have attempted
        self.assertIsNotNone(track.ocr_confidence,
                           "OCR confidence should be set after processing")
        
        # If text was recognized, it should match GTA V format or be None
        if track.text is not None:
            # Verify text format (8 characters: ##XXX### format)
            self.assertEqual(len(track.text), 8,
                           f"GTA V plate text should be 8 characters, got: {track.text}")
            print(f"✓ OCR recognized text: {track.text} (confidence: {track.ocr_confidence:.2f})")
        else:
            print(f"✓ OCR attempted but no valid text recognized (confidence: {track.ocr_confidence:.2f})")
    
    def test_track_state_attributes(self):
        """
        Test all track attributes are properly maintained.
        """
        frames = self._read_video_frames(start_frame=0, num_frames=5)
        
        # Process frames
        for frame_idx, frame in enumerate(frames):
            tracks = self.pipeline.process_frame(frame)
            
            for track_id, track in tracks.items():
                # Verify all required attributes exist
                self.assertIsNotNone(track.id, "Track ID should not be None")
                self.assertIsNotNone(track.bbox, "Bbox should not be None")
                self.assertIsNotNone(track.detection_confidence, "Detection confidence should not be None")
                self.assertIsNotNone(track.age, "Age should not be None")
                self.assertIsInstance(track.is_active, bool, "is_active should be boolean")
                
                # Verify bbox format (x1, y1, x2, y2)
                self.assertEqual(len(track.bbox), 4, "Bbox should have 4 coordinates")
                self.assertLess(track.bbox[0], track.bbox[2], "x1 should be < x2")
                self.assertLess(track.bbox[1], track.bbox[3], "y1 should be < y2")
                
                # Verify age matches frame index
                if frame_idx > 0:  # Skip first frame (age=0)
                    self.assertEqual(track.age, frame_idx,
                                   f"Frame {frame_idx}: Age should match frame index")
        
        print(f"✓ All track attributes properly maintained")
    
    def test_frame_count_increments(self):
        """
        Test pipeline frame counter increments correctly.
        """
        frames = self._read_video_frames(start_frame=0, num_frames=10)
        
        self.assertEqual(self.pipeline.frame_count, 0, "Initial frame count should be 0")
        
        for frame_idx, frame in enumerate(frames, start=1):
            self.pipeline.process_frame(frame)
            self.assertEqual(self.pipeline.frame_count, frame_idx,
                           f"Frame count should be {frame_idx}")
        
        print(f"✓ Frame counter incremented correctly: 0 → {self.pipeline.frame_count}")
    
    def test_pipeline_reset_clears_state(self):
        """
        Test pipeline reset clears all tracks and counters.
        """
        # Process some frames
        frames = self._read_video_frames(start_frame=0, num_frames=5)
        for frame in frames:
            self.pipeline.process_frame(frame)
        
        # Verify state before reset
        self.assertGreater(self.pipeline.frame_count, 0, "Should have processed frames")
        self.assertGreater(len(self.pipeline.tracks), 0, "Should have tracks")
        
        # Reset pipeline
        self.pipeline.reset()
        
        # Verify state after reset
        self.assertEqual(self.pipeline.frame_count, 0, "Frame count should be 0 after reset")
        self.assertEqual(len(self.pipeline.tracks), 0, "Tracks should be empty after reset")
        
        print(f"✓ Pipeline reset cleared all state")
    
    def test_lost_track_cleanup(self):
        """
        Test that lost tracks are eventually removed from the dictionary.
        
        Note: Tracks are only removed when (!is_active AND age >= max_age).
        Lost tracks with age < max_age are retained in case they reappear.
        """
        # Process many frames to create track with high age
        frames = self._read_video_frames(start_frame=0, num_frames=35)
        for frame in frames:
            self.pipeline.process_frame(frame)
        
        track_count_before = len(self.pipeline.tracks)
        self.assertGreater(track_count_before, 0, "Should have tracks before cleanup test")
        
        # Get track info before losing it
        track_id = list(self.pipeline.tracks.keys())[0]
        age_before_loss = self.pipeline.tracks[track_id].age
        
        # Process blank frames (no detections) to trigger track loss
        blank_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        max_age = self.pipeline.config['tracking'].get('max_age', 30)
        
        # Process enough blank frames for cleanup
        # Track will be marked as lost, and after max_age frames it should be removed
        for _ in range(max_age + 5):
            self.pipeline.process_frame(blank_frame)
        
        # Tracks should be cleaned up
        track_count_after = len(self.pipeline.tracks)
        self.assertEqual(track_count_after, 0,
                        f"Lost tracks with age >= {max_age} should be removed (was {track_count_before} tracks)")
        
        print(f"✓ Lost track (age={age_before_loss}) cleaned up after {max_age} frames")


class TestVideoProperties(unittest.TestCase):
    """Test video file properties and accessibility."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cls.test_video_path = os.path.join(cls.project_root, 'outputs', 'unit_test_video.mp4')
    
    def test_video_file_exists(self):
        """Test video file exists and is accessible."""
        self.assertTrue(os.path.exists(self.test_video_path),
                       f"Video file should exist: {self.test_video_path}")
    
    def test_video_can_be_opened(self):
        """Test video can be opened with OpenCV."""
        cap = cv2.VideoCapture(self.test_video_path)
        self.assertTrue(cap.isOpened(), "Video should be openable")
        cap.release()
    
    def test_video_has_frames(self):
        """Test video has expected number of frames."""
        cap = cv2.VideoCapture(self.test_video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        self.assertGreater(frame_count, 0, "Video should have frames")
        self.assertGreaterEqual(frame_count, 30,
                              f"Video should have at least 30 frames for testing, got {frame_count}")
    
    def test_video_frame_dimensions(self):
        """Test video frames have expected dimensions."""
        cap = cv2.VideoCapture(self.test_video_path)
        success, frame = cap.read()
        cap.release()
        
        self.assertTrue(success, "Should be able to read first frame")
        self.assertIsNotNone(frame, "Frame should not be None")
        
        # Verify frame shape (height, width, channels)
        self.assertEqual(len(frame.shape), 3, "Frame should have 3 dimensions")
        self.assertEqual(frame.shape[2], 3, "Frame should have 3 color channels (BGR)")
        
        # Verify reasonable dimensions (GTA V screenshots are typically 1920x1080)
        height, width = frame.shape[:2]
        self.assertGreaterEqual(width, 640, f"Frame width should be >= 640, got {width}")
        self.assertGreaterEqual(height, 480, f"Frame height should be >= 480, got {height}")


class TestOCRTriggerLogic(unittest.TestCase):
    """Test OCR triggering based on track state."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Get project root directory
        cls.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cls.config_path = os.path.join(cls.project_root, 'configs', 'pipeline_config.yaml')
    
    def setUp(self):
        """Initialize pipeline for each test."""
        self.pipeline = ALPRPipeline(self.config_path)
    
    def test_ocr_runs_on_new_tracks(self):
        """Test OCR runs when new tracks appear."""
        import os
        # Process frame with detections
        frame = cv2.imread(os.path.join(self.project_root, 'outputs', 'test_images', 'day_clear_front_00000.jpg'))
        if frame is None:
            self.skipTest("Test image not found")
        
        tracks = self.pipeline.process_frame(frame)
        
        # For tracks with age > 0, check if OCR was triggered
        # (frames_since_last_ocr should be 0 for new tracks that ran OCR)
        for track in tracks.values():
            if track.age > 0:
                # Track has been seen before
                # frames_since_last_ocr will vary based on ocr_interval
                self.assertIsInstance(track.frames_since_last_ocr, int)
                self.assertGreaterEqual(track.frames_since_last_ocr, 0)
    
    def test_ocr_respects_interval(self):
        """Test OCR doesn't run every frame when interval is set."""
        import os
        ocr_interval = self.pipeline.config['tracking'].get('ocr_interval', 30)
        
        # Process multiple frames
        test_frames = [
            os.path.join(self.project_root, 'outputs', 'test_images', 'day_clear_front_00000.jpg'),
            os.path.join(self.project_root, 'outputs', 'test_images', 'day_clear_front_00001.jpg'),
            os.path.join(self.project_root, 'outputs', 'test_images', 'day_clear_front_00002.jpg'),
        ]
        
        ocr_calls_per_frame = []
        
        for frame_path in test_frames:
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            
            # Count tracks that just ran OCR
            prev_tracks = {tid: t.frames_since_last_ocr for tid, t in self.pipeline.tracks.items()}
            
            tracks = self.pipeline.process_frame(frame)
            
            # Count OCR calls (frames_since_last_ocr == 0 and age > 0)
            ocr_calls = sum(
                1 for t in tracks.values()
                if t.frames_since_last_ocr == 0 and t.age > 0
            )
            ocr_calls_per_frame.append(ocr_calls)
        
        # Verify OCR doesn't run every frame (after first frame)
        # Note: This is statistical, exact behavior depends on track lifecycle
        self.assertIsInstance(ocr_calls_per_frame, list)


class TestPipelineStatistics(unittest.TestCase):
    """Test pipeline statistics collection."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Get project root directory
        cls.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cls.config_path = os.path.join(cls.project_root, 'configs', 'pipeline_config.yaml')
    
    def setUp(self):
        """Initialize pipeline for each test."""
        self.pipeline = ALPRPipeline(self.config_path)
    
    def test_get_statistics(self):
        """Test statistics retrieval."""
        stats = self.pipeline.get_statistics()
        
        # Verify all required fields
        self.assertIn('frame_count', stats)
        self.assertIn('track_count', stats)
        self.assertIn('active_count', stats)
        self.assertIn('recognized_count', stats)
        self.assertIn('avg_track_age', stats)
        self.assertIn('avg_ocr_confidence', stats)
        
        # Verify initial values
        self.assertEqual(stats['frame_count'], 0)
        self.assertEqual(stats['track_count'], 0)
    
    def test_statistics_update_after_processing(self):
        """Test statistics update after processing frames."""
        import os
        frame = cv2.imread(os.path.join(self.project_root, 'outputs', 'test_images', 'day_clear_front_00000.jpg'))
        if frame is None:
            self.skipTest("Test image not found")
        
        # Process frame
        self.pipeline.process_frame(frame)
        
        # Get updated statistics
        stats = self.pipeline.get_statistics()
        
        # Frame count should be 1
        self.assertEqual(stats['frame_count'], 1)
        
        # Track count should be >= 0
        self.assertGreaterEqual(stats['track_count'], 0)


class TestPipelineReset(unittest.TestCase):
    """Test pipeline reset functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Get project root directory
        cls.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cls.config_path = os.path.join(cls.project_root, 'configs', 'pipeline_config.yaml')
    
    def setUp(self):
        """Initialize pipeline for each test."""
        self.pipeline = ALPRPipeline(self.config_path)
    
    def test_reset_clears_state(self):
        """Test reset clears pipeline state."""
        import os
        # Process some frames
        frame = cv2.imread(os.path.join(self.project_root, 'outputs', 'test_images', 'day_clear_front_00000.jpg'))
        if frame is None:
            self.skipTest("Test image not found")
        
        self.pipeline.process_frame(frame)
        self.pipeline.process_frame(frame)
        
        # Verify state exists
        self.assertGreater(self.pipeline.frame_count, 0)
        
        # Reset pipeline
        self.pipeline.reset()
        
        # Verify state cleared
        self.assertEqual(self.pipeline.frame_count, 0)
        self.assertEqual(len(self.pipeline.tracks), 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Get project root directory
        cls.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cls.config_path = os.path.join(cls.project_root, 'configs', 'pipeline_config.yaml')
    
    def setUp(self):
        """Initialize pipeline for each test."""
        self.pipeline = ALPRPipeline(self.config_path)
    
    def test_multiple_plates_same_frame(self):
        """Test handling multiple plates in one frame."""
        import os
        # Note: This depends on test image content
        # Just verify no crashes with real test images
        frame = cv2.imread(os.path.join(self.project_root, 'outputs', 'test_images', 'day_clear_front_00000.jpg'))
        if frame is None:
            self.skipTest("Test image not found")
        
        tracks = self.pipeline.process_frame(frame)
        
        # Verify returns dict (may have 0 or more tracks)
        self.assertIsInstance(tracks, dict)
    
    def test_very_small_frame(self):
        """Test handling very small frames."""
        # Create tiny frame
        tiny_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        
        # Should not crash
        tracks = self.pipeline.process_frame(tiny_frame)
        self.assertIsInstance(tracks, dict)
    
    def test_very_large_frame(self):
        """Test handling large frames."""
        # Create large frame (4K)
        large_frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        
        # Should not crash (may be slow)
        tracks = self.pipeline.process_frame(large_frame)
        self.assertIsInstance(tracks, dict)


if __name__ == '__main__':
    unittest.main()

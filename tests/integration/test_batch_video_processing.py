"""
Unit tests for batch video processing script (process_video.py).

Tests all core functionality including frame sampling, video I/O,
JSON/CSV export, progress tracking, and error handling.
"""

import unittest
import sys
import os
import json
import csv
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the functions we need to test
from scripts.process_video import export_json, export_csv, parse_args


class TestVideoProcessingExports(unittest.TestCase):
    """Test JSON and CSV export functionality."""
    
    def setUp(self):
        """Create temporary directory for test outputs."""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)
    
    def test_export_json_basic(self):
        """Test basic JSON export with minimal data."""
        results = [
            {
                'frame': 0,
                'timestamp': 0.0,
                'tracks': [
                    {
                        'id': 1,
                        'text': 'ABC123',
                        'ocr_confidence': 0.95,
                        'detection_confidence': 0.85,
                        'bbox': [100, 100, 200, 150],
                        'age': 0
                    }
                ]
            }
        ]
        
        stats = {
            'frames_processed': 1,
            'total_frames': 100,
            'sample_rate': 10,
            'elapsed_time': 5.0,
            'fps': 0.2,
            'plates_detected': 1,
            'plates_recognized': 1,
            'unique_plates': 1
        }
        
        output_path = os.path.join(self.test_dir, 'test.json')
        export_json(results, output_path, stats)
        
        # Verify file exists and is valid JSON
        self.assertTrue(os.path.exists(output_path))
        
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(data['statistics']['frames_processed'], 1)
        self.assertEqual(data['statistics']['plates_detected'], 1)
        self.assertEqual(len(data['frames']), 1)
        self.assertEqual(data['frames'][0]['tracks'][0]['text'], 'ABC123')
    
    def test_export_json_empty_results(self):
        """Test JSON export with no detections."""
        results = []
        stats = {
            'frames_processed': 0,
            'total_frames': 0,
            'sample_rate': 1,
            'elapsed_time': 0.0,
            'fps': 0.0,
            'plates_detected': 0,
            'plates_recognized': 0,
            'unique_plates': 0
        }
        
        output_path = os.path.join(self.test_dir, 'empty.json')
        export_json(results, output_path, stats)
        
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(len(data['frames']), 0)
        self.assertEqual(data['statistics']['plates_detected'], 0)
    
    def test_export_csv_basic(self):
        """Test basic CSV export with track data."""
        results = [
            {
                'frame': 0,
                'timestamp': 0.0,
                'tracks': [
                    {
                        'id': 1,
                        'text': 'ABC123',
                        'ocr_confidence': 0.95,
                        'detection_confidence': 0.85,
                        'bbox': [100, 100, 200, 150],
                        'age': 0
                    }
                ]
            },
            {
                'frame': 30,
                'timestamp': 1.0,
                'tracks': [
                    {
                        'id': 2,
                        'text': 'XYZ789',
                        'ocr_confidence': 0.90,
                        'detection_confidence': 0.80,
                        'bbox': [150, 120, 250, 170],
                        'age': 1
                    }
                ]
            }
        ]
        
        output_path = os.path.join(self.test_dir, 'test.csv')
        export_csv(results, output_path)
        
        # Verify file exists and has correct data
        self.assertTrue(os.path.exists(output_path))
        
        with open(output_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['track_id'], '1')
        self.assertEqual(rows[0]['text'], 'ABC123')
        self.assertEqual(rows[1]['track_id'], '2')
        self.assertEqual(rows[1]['text'], 'XYZ789')
    
    def test_export_csv_empty_text(self):
        """Test CSV export with tracks that have no recognized text."""
        results = [
            {
                'frame': 0,
                'timestamp': 0.0,
                'tracks': [
                    {
                        'id': 1,
                        'text': None,
                        'ocr_confidence': 0.0,
                        'detection_confidence': 0.75,
                        'bbox': [100, 100, 200, 150],
                        'age': 0
                    }
                ]
            }
        ]
        
        output_path = os.path.join(self.test_dir, 'empty_text.csv')
        export_csv(results, output_path)
        
        with open(output_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['text'], '')  # None becomes empty string
        self.assertEqual(rows[0]['ocr_confidence'], '0.0')


class TestArgumentParsing(unittest.TestCase):
    """Test command-line argument parsing."""
    
    def test_parse_args_required_only(self):
        """Test parsing with only required arguments."""
        test_args = ['--input', 'video.mp4', '--output', 'output.mp4']
        
        with patch('sys.argv', ['process_video.py'] + test_args):
            args = parse_args()
        
        self.assertEqual(args.input, 'video.mp4')
        self.assertEqual(args.output, 'output.mp4')
        self.assertEqual(args.sample_rate, 1)  # Default
        self.assertFalse(args.no_video)  # Default
    
    def test_parse_args_all_options(self):
        """Test parsing with all available options."""
        test_args = [
            '--input', 'video.mp4',
            '--output', 'output.mp4',
            '--config', 'custom_config.yaml',
            '--sample-rate', '5',
            '--no-video',
            '--export-json', 'results.json',
            '--export-csv', 'data.csv',
            '--show-track-id',
            '--show-confidence'
        ]
        
        with patch('sys.argv', ['process_video.py'] + test_args):
            args = parse_args()
        
        self.assertEqual(args.input, 'video.mp4')
        self.assertEqual(args.output, 'output.mp4')
        self.assertEqual(args.config, 'custom_config.yaml')
        self.assertEqual(args.sample_rate, 5)
        self.assertTrue(args.no_video)
        self.assertEqual(args.export_json, 'results.json')
        self.assertEqual(args.export_csv, 'data.csv')
        self.assertTrue(args.show_track_id)
        self.assertTrue(args.show_confidence)


class TestVideoProcessingIntegration(unittest.TestCase):
    """Integration tests for complete video processing pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    @patch('scripts.process_video.ALPRPipeline')
    @patch('scripts.process_video.VideoReader')
    @patch('scripts.process_video.VideoWriter')
    def test_fps_calculation_with_sampling(self, mock_writer, mock_reader, mock_pipeline):
        """Test that output FPS is correctly adjusted for frame sampling."""
        # Mock VideoReader
        mock_reader_instance = MagicMock()
        mock_reader_instance.fps = 30.0
        mock_reader_instance.width = 1920
        mock_reader_instance.height = 1080
        mock_reader_instance.total_frames = 300  # Add total_frames
        mock_reader_instance.read_frames.return_value = []
        mock_reader.return_value = mock_reader_instance
        
        # Mock VideoWriter to capture FPS
        mock_writer_instance = MagicMock()
        mock_writer.return_value = mock_writer_instance
        
        # Mock pipeline
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Import and run process_video function
        from scripts.process_video import process_video
        
        # Create mock args
        class MockArgs:
            input = 'test.mp4'
            output = 'output.mp4'
            config = 'configs/pipeline_config.yaml'
            sample_rate = 10
            no_video = False
            export_json = None
            export_csv = None
            show_track_id = False
            show_confidence = False
        
        # Process video
        with patch('builtins.print'):  # Suppress print statements
            process_video(MockArgs())
        
        # Verify VideoWriter was called with adjusted FPS
        mock_writer.assert_called_once()
        call_args = mock_writer.call_args[0]
        output_fps = call_args[1]
        
        # FPS should be original_fps / sample_rate = 30 / 10 = 3.0
        self.assertEqual(output_fps, 3.0)


class TestErrorHandling(unittest.TestCase):
    """Test error handling in video processing."""
    
    @patch('scripts.process_video.VideoReader')
    def test_invalid_video_path(self, mock_reader):
        """Test handling of invalid video file path."""
        mock_reader.side_effect = FileNotFoundError("Video file not found")
        
        from scripts.process_video import process_video
        
        class MockArgs:
            input = 'nonexistent.mp4'
            output = 'output.mp4'
            config = 'configs/pipeline_config.yaml'
            sample_rate = 1
            no_video = False
            export_json = None
            export_csv = None
            show_track_id = False
            show_confidence = False
        
        # Should raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            process_video(MockArgs())
    
    @patch('scripts.process_video.ALPRPipeline')
    def test_invalid_config_path(self, mock_pipeline):
        """Test handling of invalid configuration file."""
        mock_pipeline.side_effect = FileNotFoundError("Config not found")
        
        from scripts.process_video import process_video
        
        class MockArgs:
            input = 'test.mp4'
            output = 'output.mp4'
            config = 'nonexistent.yaml'
            sample_rate = 1
            no_video = False
            export_json = None
            export_csv = None
            show_track_id = False
            show_confidence = False
        
        # Should raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            process_video(MockArgs())


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestVideoProcessingExports))
    suite.addTests(loader.loadTestsFromTestCase(TestArgumentParsing))
    suite.addTests(loader.loadTestsFromTestCase(TestVideoProcessingIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_tests()
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)

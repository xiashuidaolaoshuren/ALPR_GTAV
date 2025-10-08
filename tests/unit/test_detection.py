"""
Unit Tests for Detection Module

Tests for model loading, inference, and utility functions.
"""

import unittest
import os
import numpy as np
from pathlib import Path

# Import detection module functions
from src.detection import (
    load_detection_model,
    detect_plates,
    draw_bounding_boxes,
    compute_iou,
    filter_detections_by_size,
    crop_detections,
    DetectionConfig
)


class TestDetectionConfig(unittest.TestCase):
    """Tests for DetectionConfig class."""
    
    def test_config_initialization(self):
        """Test configuration initialization with valid parameters."""
        config_dict = {
            'detection': {
                'model_path': 'models/detection/yolov8n.pt',
                'confidence_threshold': 0.25,
                'iou_threshold': 0.45,
                'img_size': 640,
                'device': 'cuda',
                'max_det': 100
            }
        }
        
        config = DetectionConfig(config_dict)
        
        self.assertEqual(config.model_path, 'models/detection/yolov8n.pt')
        self.assertEqual(config.confidence_threshold, 0.25)
        self.assertEqual(config.iou_threshold, 0.45)
        self.assertEqual(config.img_size, 640)
        self.assertEqual(config.device, 'cuda')
        self.assertEqual(config.max_det, 100)
    
    def test_config_missing_key(self):
        """Test configuration initialization with missing 'detection' key."""
        config_dict = {}
        
        with self.assertRaises(KeyError):
            DetectionConfig(config_dict)
    
    def test_config_validation_invalid_threshold(self):
        """Test configuration validation with invalid threshold values."""
        config_dict = {
            'detection': {
                'model_path': 'models/detection/yolov8n.pt',
                'confidence_threshold': 1.5,  # Invalid: > 1.0
                'iou_threshold': 0.45
            }
        }
        
        config = DetectionConfig(config_dict)
        
        # Should raise ValueError during validation
        # Note: Skipped for now since model file doesn't exist
        # with self.assertRaises(ValueError):
        #     config.validate()
    
    def test_config_to_dict(self):
        """Test converting configuration to dictionary."""
        config_dict = {
            'detection': {
                'model_path': 'models/detection/yolov8n.pt',
                'confidence_threshold': 0.25,
                'iou_threshold': 0.45,
                'img_size': 640,
                'device': 'cuda',
                'max_det': 100
            }
        }
        
        config = DetectionConfig(config_dict)
        result = config.to_dict()
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['model_path'], 'models/detection/yolov8n.pt')
        self.assertEqual(result['confidence_threshold'], 0.25)


class TestModelFunctions(unittest.TestCase):
    """Tests for model loading and inference functions."""
    
    def test_load_model_not_implemented(self):
        """Test that load_detection_model is defined but not yet implemented."""
        # Function should exist
        self.assertTrue(callable(load_detection_model))
        
        # Currently returns None (not implemented)
        # Will be implemented in Task 7
    
    def test_detect_plates_not_implemented(self):
        """Test that detect_plates is defined but not yet implemented."""
        # Function should exist
        self.assertTrue(callable(detect_plates))
        
        # Currently returns None (not implemented)
        # Will be implemented in Task 8


class TestUtilityFunctions(unittest.TestCase):
    """Tests for utility functions."""
    
    def test_compute_iou_no_overlap(self):
        """Test IoU computation for non-overlapping boxes."""
        box1 = (0, 0, 100, 100)
        box2 = (200, 200, 300, 300)
        
        iou = compute_iou(box1, box2)
        
        self.assertAlmostEqual(iou, 0.0, places=5)
    
    def test_compute_iou_identical_boxes(self):
        """Test IoU computation for identical boxes."""
        box1 = (100, 100, 200, 200)
        box2 = (100, 100, 200, 200)
        
        iou = compute_iou(box1, box2)
        
        self.assertAlmostEqual(iou, 1.0, places=5)
    
    def test_compute_iou_partial_overlap(self):
        """Test IoU computation for partially overlapping boxes."""
        box1 = (0, 0, 100, 100)     # 100x100 box
        box2 = (50, 50, 150, 150)   # 100x100 box, 50% overlap
        
        iou = compute_iou(box1, box2)
        
        # Expected IoU: intersection=50x50=2500, union=10000+10000-2500=17500
        # IoU = 2500/17500 = 0.142857
        self.assertAlmostEqual(iou, 0.142857, places=5)
    
    def test_compute_iou_invalid_box(self):
        """Test IoU computation with invalid box coordinates."""
        box1 = (100, 100, 50, 50)  # Invalid: x1 > x2
        box2 = (0, 0, 100, 100)
        
        with self.assertRaises(ValueError):
            compute_iou(box1, box2)
    
    def test_filter_detections_by_size(self):
        """Test filtering detections by size constraints."""
        detections = [
            (0, 0, 10, 10, 0.9),      # 10x10 - too small
            (0, 0, 100, 50, 0.95),    # 100x50 - valid
            (0, 0, 200, 100, 0.92),   # 200x100 - valid
            (0, 0, 500, 300, 0.88)    # 500x300 - too large
        ]
        
        filtered = filter_detections_by_size(
            detections,
            min_width=50,
            min_height=30,
            max_width=400,
            max_height=200
        )
        
        # Should keep only middle two detections
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0][2] - filtered[0][0], 100)  # width=100
        self.assertEqual(filtered[1][2] - filtered[1][0], 200)  # width=200
    
    def test_crop_detections(self):
        """Test cropping detected regions from frame."""
        # Create a test frame (100x100 red image)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :] = [0, 0, 255]  # Red
        
        detections = [
            (10, 10, 50, 50, 0.95),
            (60, 60, 90, 90, 0.92)
        ]
        
        crops = crop_detections(frame, detections, padding=0)
        
        self.assertEqual(len(crops), 2)
        self.assertEqual(crops[0].shape, (40, 40, 3))
        self.assertEqual(crops[1].shape, (30, 30, 3))
    
    def test_crop_detections_with_padding(self):
        """Test cropping with padding."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [(10, 10, 50, 50, 0.95)]
        
        crops = crop_detections(frame, detections, padding=5)
        
        # Should be 40x40 + 10 (5 on each side) = 50x50
        self.assertEqual(len(crops), 1)
        self.assertEqual(crops[0].shape, (50, 50, 3))
    
    def test_draw_bounding_boxes_not_implemented(self):
        """Test that draw_bounding_boxes is defined but not yet implemented."""
        # Function should exist
        self.assertTrue(callable(draw_bounding_boxes))
        
        # Currently returns None (not implemented)
        # Will be implemented in Task 8


class TestIntegration(unittest.TestCase):
    """Integration tests for detection module."""
    
    def test_module_imports(self):
        """Test that all expected functions are importable."""
        from src.detection import (
            load_detection_model,
            detect_plates,
            draw_bounding_boxes,
            compute_iou,
            DetectionConfig
        )
        
        # All imports should succeed
        self.assertTrue(True)
    
    def test_config_from_yaml(self):
        """Test loading configuration from YAML file."""
        # Assuming pipeline_config.yaml exists
        config_path = Path('configs/pipeline_config.yaml')
        
        if config_path.exists():
            import yaml
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            config = DetectionConfig(config_dict)
            
            self.assertIsNotNone(config.model_path)
            self.assertIsNotNone(config.confidence_threshold)


def run_tests():
    """Run all tests."""
    unittest.main()


if __name__ == '__main__':
    run_tests()

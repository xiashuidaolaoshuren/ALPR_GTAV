"""
Unit tests for OCR inference and post-processing (Task 16).

Tests recognize_text() function and utility functions for filtering,
scoring, and selecting best text candidates from OCR results.
"""

import unittest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.recognition.model import recognize_text
from src.recognition.utils import filter_by_regex, score_candidate, select_best_candidate


class TestFilterByRegex(unittest.TestCase):
    """Test regex filtering for GTA V license plate format."""
    
    def test_valid_gta_v_plate(self):
        """Test that valid GTA V plates pass the filter."""
        pattern = r'^\d{2}[A-Z]{3}\d{3}$'
        
        valid_plates = [
            '12ABC345',
            '99ZZZ999',
            '00AAA000',
            '12XYZ789'
        ]
        
        for plate in valid_plates:
            with self.subTest(plate=plate):
                self.assertTrue(filter_by_regex(plate, pattern),
                              f"Valid plate '{plate}' should pass filter")
        
        print(f"✓ All {len(valid_plates)} valid GTA V plates passed filter")
    
    def test_invalid_gta_v_plates(self):
        """Test that invalid plates are rejected."""
        pattern = r'^\d{2}[A-Z]{3}\d{3}$'
        
        invalid_plates = [
            'ABC123',      # Wrong format
            '12ABC34',     # Too short
            '12ABC3456',   # Too long
            '12abc345',    # Lowercase
            '1A2BC345',    # Wrong digit position
            '12AB3456',    # Wrong letter count
            'HEADER',      # Not a plate
            '',            # Empty
            '12-ABC-345',  # Has separators
        ]
        
        for plate in invalid_plates:
            with self.subTest(plate=plate):
                self.assertFalse(filter_by_regex(plate, pattern),
                               f"Invalid plate '{plate}' should be rejected")
        
        print(f"✓ All {len(invalid_plates)} invalid plates rejected")
    
    def test_empty_inputs(self):
        """Test handling of empty text or pattern."""
        pattern = r'^\d{2}[A-Z]{3}\d{3}$'
        
        self.assertFalse(filter_by_regex('', pattern))
        self.assertFalse(filter_by_regex('12ABC345', ''))
        self.assertFalse(filter_by_regex('', ''))
        
        print("✓ Empty input handling successful")
    
    def test_invalid_regex_pattern(self):
        """Test error handling for invalid regex."""
        with self.assertRaises(Exception):
            filter_by_regex('12ABC345', '[invalid(regex')
        
        print("✓ Invalid regex error handling successful")


class TestScoreCandidate(unittest.TestCase):
    """Test OCR candidate scoring formula."""
    
    def test_perfect_score(self):
        """Test scoring for ideal candidate (8 chars, high conf, large bbox)."""
        # Perfect GTA V plate: 8 chars, 100% confidence, half image height
        score = score_candidate(
            text='12ABC345',
            confidence=1.0,
            bbox_height=50.0,
            image_height=100.0
        )
        
        # Expected: 1.0 * 0.5 * 1.0 = 0.5
        self.assertAlmostEqual(score, 0.5, places=3)
        
        print(f"✓ Perfect score test passed: score={score:.3f}")
    
    def test_score_components(self):
        """Test that score formula balances confidence, height, and length."""
        image_height = 100.0
        
        # Test confidence impact
        score_high_conf = score_candidate('12ABC345', 0.9, 50.0, image_height)
        score_low_conf = score_candidate('12ABC345', 0.5, 50.0, image_height)
        self.assertGreater(score_high_conf, score_low_conf)
        
        # Test height impact
        score_tall = score_candidate('12ABC345', 0.9, 60.0, image_height)
        score_short = score_candidate('12ABC345', 0.9, 30.0, image_height)
        self.assertGreater(score_tall, score_short)
        
        # Test length impact (capped at 8)
        score_8char = score_candidate('12ABC345', 0.9, 50.0, image_height)
        score_4char = score_candidate('1234', 0.9, 50.0, image_height)
        self.assertGreater(score_8char, score_4char)
        
        print("✓ Score component tests passed")
    
    def test_length_capping(self):
        """Test that length is capped at 8 characters."""
        image_height = 100.0
        
        # 8 characters and 12 characters should have same length contribution
        score_8 = score_candidate('12ABC345', 1.0, 50.0, image_height)
        score_12 = score_candidate('12ABC345WXYZ', 1.0, 50.0, image_height)
        
        self.assertAlmostEqual(score_8, score_12, places=3)
        
        print("✓ Length capping test passed")
    
    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        # Confidence out of range
        with self.assertRaises(ValueError):
            score_candidate('12ABC345', 1.5, 50.0, 100.0)
        
        with self.assertRaises(ValueError):
            score_candidate('12ABC345', -0.1, 50.0, 100.0)
        
        # Non-positive heights
        with self.assertRaises(ValueError):
            score_candidate('12ABC345', 0.9, 0.0, 100.0)
        
        with self.assertRaises(ValueError):
            score_candidate('12ABC345', 0.9, 50.0, -100.0)
        
        print("✓ Invalid input error handling successful")
    
    def test_empty_text(self):
        """Test that empty text returns zero score."""
        score = score_candidate('', 0.9, 50.0, 100.0)
        self.assertEqual(score, 0.0)
        
        print("✓ Empty text handling successful")


class TestSelectBestCandidate(unittest.TestCase):
    """Test best candidate selection."""
    
    def test_select_highest_score(self):
        """Test that candidate with highest score is selected."""
        candidates = [
            {'text': '12ABC345', 'confidence': 0.95, 'score': 0.475, 'bbox': []},
            {'text': '12XYZ789', 'confidence': 0.80, 'score': 0.320, 'bbox': []},
            {'text': '99ZZZ999', 'confidence': 0.90, 'score': 0.400, 'bbox': []}
        ]
        
        text, conf = select_best_candidate(candidates)
        
        self.assertEqual(text, '12ABC345')
        self.assertEqual(conf, 0.95)
        
        print("✓ Highest score selection successful")
    
    def test_empty_candidates(self):
        """Test handling of empty candidate list."""
        text, conf = select_best_candidate([])
        
        self.assertIsNone(text)
        self.assertEqual(conf, 0.0)
        
        print("✓ Empty candidates handling successful")
    
    def test_single_candidate(self):
        """Test with single candidate."""
        candidates = [
            {'text': '12ABC345', 'confidence': 0.95, 'score': 0.475, 'bbox': []}
        ]
        
        text, conf = select_best_candidate(candidates)
        
        self.assertEqual(text, '12ABC345')
        self.assertEqual(conf, 0.95)
        
        print("✓ Single candidate selection successful")
    
    def test_missing_keys(self):
        """Test error handling for malformed candidates."""
        candidates = [
            {'text': '12ABC345', 'confidence': 0.95}  # Missing 'score'
        ]
        
        with self.assertRaises(KeyError):
            select_best_candidate(candidates)
        
        print("✓ Missing keys error handling successful")


class TestRecognizeText(unittest.TestCase):
    """Test main OCR inference and post-processing function."""
    
    def test_valid_plate_recognition(self):
        """Test recognition of valid GTA V plate."""
        # Create mock OCR model
        mock_model = Mock()
        
        # Mock OCR result with one valid plate
        # Format: [{'rec_texts': [...], 'rec_scores': [...], 'rec_polys': [...]}]
        mock_model.predict.return_value = [{
            'rec_texts': ['12ABC345'],
            'rec_scores': [0.95],
            'rec_polys': [[[10, 20], [110, 20], [110, 50], [10, 50]]]
        }]
        
        # Create test image
        test_image = np.zeros((100, 200, 3), dtype=np.uint8)
        
        # Config
        config = {
            'regex': r'^\d{2}[A-Z]{3}\d{3}$',
            'min_conf': 0.3
        }
        
        # Run recognition
        text, conf = recognize_text(test_image, mock_model, config)
        
        self.assertEqual(text, '12ABC345')
        self.assertGreater(conf, 0.9)
        
        print(f"✓ Valid plate recognition successful: {text} ({conf:.3f})")
    
    def test_multiple_text_lines_filtering(self):
        """Test that filtering selects correct plate from multiple text lines."""
        mock_model = Mock()
        
        # Mock OCR with header + plate + footer
        mock_model.predict.return_value = [{
            'rec_texts': ['HEADER', '12ABC345', 'FOOTER'],
            'rec_scores': [0.85, 0.95, 0.80],
            'rec_polys': [
                [[10, 5], [110, 5], [110, 15], [10, 15]],    # Small header
                [[10, 20], [110, 20], [110, 55], [10, 55]],  # Large plate
                [[10, 60], [110, 60], [110, 70], [10, 70]]   # Small footer
            ]
        }]
        
        test_image = np.zeros((100, 200, 3), dtype=np.uint8)
        config = {'regex': r'^\d{2}[A-Z]{3}\d{3}$', 'min_conf': 0.3}
        
        text, conf = recognize_text(test_image, mock_model, config)
        
        self.assertEqual(text, '12ABC345')
        
        print("✓ Multiple text lines filtering successful")
    
    def test_no_valid_candidates(self):
        """Test when no text matches GTA V format."""
        mock_model = Mock()
        
        # Mock OCR with only invalid text
        mock_model.predict.return_value = [{
            'rec_texts': ['INVALID'],
            'rec_scores': [0.95],
            'rec_polys': [[[10, 20], [110, 20], [110, 50], [10, 50]]]
        }]
        
        test_image = np.zeros((100, 200, 3), dtype=np.uint8)
        config = {'regex': r'^\d{2}[A-Z]{3}\d{3}$', 'min_conf': 0.3}
        
        text, conf = recognize_text(test_image, mock_model, config)
        
        self.assertIsNone(text)
        self.assertEqual(conf, 0.0)
        
        print("✓ No valid candidates handling successful")
    
    def test_empty_ocr_result(self):
        """Test handling of empty OCR results."""
        mock_model = Mock()
        mock_model.predict.return_value = []
        
        test_image = np.zeros((100, 200, 3), dtype=np.uint8)
        config = {'regex': r'^\d{2}[A-Z]{3}\d{3}$', 'min_conf': 0.3}
        
        text, conf = recognize_text(test_image, mock_model, config)
        
        self.assertIsNone(text)
        self.assertEqual(conf, 0.0)
        
        print("✓ Empty OCR result handling successful")
    
    def test_low_confidence_filtering(self):
        """Test that low confidence results are filtered out."""
        mock_model = Mock()
        
        # Mock OCR with low confidence
        mock_model.predict.return_value = [{
            'rec_texts': ['12ABC345'],
            'rec_scores': [0.2],  # Below min_conf threshold
            'rec_polys': [[[10, 20], [110, 20], [110, 50], [10, 50]]]
        }]
        
        test_image = np.zeros((100, 200, 3), dtype=np.uint8)
        config = {'regex': r'^\d{2}[A-Z]{3}\d{3}$', 'min_conf': 0.3}
        
        text, conf = recognize_text(test_image, mock_model, config)
        
        self.assertIsNone(text)
        
        print("✓ Low confidence filtering successful")
    
    def test_invalid_image(self):
        """Test error handling for invalid images."""
        mock_model = Mock()
        config = {'regex': r'^\d{2}[A-Z]{3}\d{3}$', 'min_conf': 0.3}
        
        # None image
        with self.assertRaises(ValueError):
            recognize_text(None, mock_model, config)
        
        # Empty image
        with self.assertRaises(ValueError):
            recognize_text(np.array([]), mock_model, config)
        
        # Wrong dimensions
        with self.assertRaises(ValueError):
            recognize_text(np.zeros((10,)), mock_model, config)
        
        print("✓ Invalid image error handling successful")
    
    def test_case_normalization(self):
        """Test that lowercase text is normalized to uppercase."""
        mock_model = Mock()
        
        # Mock OCR with lowercase text
        mock_model.predict.return_value = [{
            'rec_texts': ['12abc345'],  # Lowercase
            'rec_scores': [0.95],
            'rec_polys': [[[10, 20], [110, 20], [110, 50], [10, 50]]]
        }]
        
        test_image = np.zeros((100, 200, 3), dtype=np.uint8)
        config = {'regex': r'^\d{2}[A-Z]{3}\d{3}$', 'min_conf': 0.3}
        
        text, conf = recognize_text(test_image, mock_model, config)
        
        self.assertEqual(text, '12ABC345')  # Should be uppercase
        
        print("✓ Case normalization successful")
    
    def test_scoring_prefers_larger_text(self):
        """Test that scoring prefers text from larger bounding boxes."""
        mock_model = Mock()
        
        # Two valid plates: one small, one large
        mock_model.predict.return_value = [{
            'rec_texts': ['99ZZZ999', '12ABC345'],
            'rec_scores': [0.90, 0.90],
            'rec_polys': [
                [[10, 40], [110, 40], [110, 60], [10, 60]],  # Small (h=20)
                [[10, 20], [110, 20], [110, 70], [10, 70]]   # Large (h=50)
            ]
        }]
        
        test_image = np.zeros((100, 200, 3), dtype=np.uint8)
        config = {'regex': r'^\d{2}[A-Z]{3}\d{3}$', 'min_conf': 0.3}
        
        text, conf = recognize_text(test_image, mock_model, config)
        
        # Should select larger text
        self.assertEqual(text, '12ABC345')
        
        print("✓ Scoring prefers larger text successful")


if __name__ == '__main__':
    print("="*70)
    print("Running OCR Inference and Post-Processing Tests (Task 16)")
    print("="*70)
    print()
    
    # Run tests
    unittest.main(verbosity=2, exit=False)
    
    print()
    print("="*70)
    print("✅ ALL TASK 16 TESTS COMPLETED")
    print("="*70)

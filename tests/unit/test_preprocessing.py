"""
Unit Tests for Preprocessing Module

Tests for license plate image preprocessing functions including:
- Image enhancement (grayscale, resize, CLAHE)
- Image validation
- Batch processing
- Statistical analysis

"""

import unittest
import cv2
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing import (
    preprocess_plate,
    resize_maintaining_aspect,
    apply_clahe,
    validate_image,
    batch_preprocess_plates,
    save_preprocessed_image,
    calculate_image_stats
)


class TestPreprocessPlate(unittest.TestCase):
    """Test cases for preprocess_plate function."""
    
    def setUp(self):
        """Create test fixtures."""
        # Create a small color plate image
        self.small_color_plate = np.random.randint(
            0, 255, (50, 150, 3), dtype=np.uint8
        )
        
        # Create a small grayscale plate image
        self.small_gray_plate = np.random.randint(
            0, 255, (50, 150), dtype=np.uint8
        )
        
        # Default config
        self.config_default = {
            'min_width': 200,
            'use_clahe': False
        }
        
        # Config with CLAHE enabled
        self.config_clahe = {
            'min_width': 200,
            'use_clahe': True,
            'clahe_clip_limit': 2.0,
            'clahe_tile_grid_size': [8, 8]
        }
        
        # Config with all features
        self.config_full = {
            'min_width': 200,
            'use_clahe': True,
            'clahe_clip_limit': 3.0,
            'clahe_tile_grid_size': [8, 8],
            'use_gaussian_blur': True,
            'gaussian_kernel_size': [3, 3],
            'use_sharpening': True,
            'sharpen_strength': 1.0
        }
    
    def test_preprocess_basic(self):
        """Test basic preprocessing without CLAHE."""
        result = preprocess_plate(self.small_color_plate, self.config_default)
        
        # Should be grayscale
        self.assertEqual(len(result.shape), 2, "Output should be grayscale")
        
        # Width should be at least min_width
        self.assertGreaterEqual(
            result.shape[1], self.config_default['min_width'],
            "Width should be at least min_width"
        )
        
        # Aspect ratio should be preserved
        original_ratio = self.small_color_plate.shape[0] / self.small_color_plate.shape[1]
        result_ratio = result.shape[0] / result.shape[1]
        self.assertAlmostEqual(
            original_ratio, result_ratio, places=2,
            msg="Aspect ratio should be preserved"
        )
    
    def test_preprocess_with_clahe(self):
        """Test preprocessing with CLAHE enhancement."""
        result = preprocess_plate(self.small_color_plate, self.config_clahe)
        
        # Should be grayscale
        self.assertEqual(len(result.shape), 2, "Output should be grayscale")
        
        # Should be uint8
        self.assertEqual(result.dtype, np.uint8, "Output should be uint8")
        
        # Width should meet minimum
        self.assertGreaterEqual(
            result.shape[1], self.config_clahe['min_width'],
            "Width should be at least min_width"
        )
    
    def test_preprocess_full_features(self):
        """Test preprocessing with all features enabled."""
        result = preprocess_plate(self.small_color_plate, self.config_full)
        
        # Should be grayscale
        self.assertEqual(len(result.shape), 2, "Output should be grayscale")
        
        # Should be uint8
        self.assertEqual(result.dtype, np.uint8, "Output should be uint8")
        
        # Width should meet minimum
        self.assertGreaterEqual(
            result.shape[1], self.config_full['min_width'],
            "Width should be at least min_width"
        )
    
    def test_preprocess_already_grayscale(self):
        """Test preprocessing when input is already grayscale."""
        result = preprocess_plate(self.small_gray_plate, self.config_default)
        
        # Should remain grayscale
        self.assertEqual(len(result.shape), 2, "Output should be grayscale")
        
        # Width should meet minimum
        self.assertGreaterEqual(
            result.shape[1], self.config_default['min_width'],
            "Width should be at least min_width"
        )
    
    def test_preprocess_no_resize_needed(self):
        """Test when image already meets min_width."""
        large_plate = np.random.randint(0, 255, (100, 400, 3), dtype=np.uint8)
        result = preprocess_plate(large_plate, self.config_default)
        
        # Width should not be significantly changed (only grayscale conversion)
        # Allow small difference due to CLAHE processing
        self.assertAlmostEqual(
            result.shape[1], large_plate.shape[1], delta=10,
            msg="Large image should not be resized significantly"
        )
    
    def test_preprocess_invalid_input(self):
        """Test error handling for invalid inputs."""
        # Test with non-numpy array
        with self.assertRaises(TypeError):
            preprocess_plate([1, 2, 3], self.config_default)
        
        # Test with empty array
        empty_img = np.array([], dtype=np.uint8)
        with self.assertRaises(ValueError):
            preprocess_plate(empty_img, self.config_default)
    
    def test_preprocess_does_not_modify_original(self):
        """Test that preprocessing doesn't modify the original image."""
        original_copy = self.small_color_plate.copy()
        _ = preprocess_plate(self.small_color_plate, self.config_default)
        
        np.testing.assert_array_equal(
            self.small_color_plate, original_copy,
            err_msg="Original image should not be modified"
        )


class TestResizeMaintainingAspect(unittest.TestCase):
    """Test cases for resize_maintaining_aspect function."""
    
    def test_resize_upscale(self):
        """Test upscaling a small image."""
        img = np.random.randint(0, 255, (100, 300), dtype=np.uint8)
        result = resize_maintaining_aspect(img, 600)
        
        # Width should match target
        self.assertEqual(result.shape[1], 600, "Width should match target")
        
        # Height should scale proportionally
        expected_height = int(100 * (600 / 300))
        self.assertEqual(result.shape[0], expected_height, "Height should scale")
        
        # Aspect ratio preserved
        original_ratio = img.shape[0] / img.shape[1]
        result_ratio = result.shape[0] / result.shape[1]
        self.assertAlmostEqual(original_ratio, result_ratio, places=5)
    
    def test_resize_downscale(self):
        """Test downscaling a large image."""
        img = np.random.randint(0, 255, (200, 800), dtype=np.uint8)
        result = resize_maintaining_aspect(img, 400)
        
        # Width should match target
        self.assertEqual(result.shape[1], 400, "Width should match target")
        
        # Height should scale proportionally
        expected_height = int(200 * (400 / 800))
        self.assertEqual(result.shape[0], expected_height, "Height should scale")
    
    def test_resize_color_image(self):
        """Test resizing a color image."""
        img = np.random.randint(0, 255, (100, 300, 3), dtype=np.uint8)
        result = resize_maintaining_aspect(img, 600)
        
        # Should remain color
        self.assertEqual(len(result.shape), 3, "Should remain color")
        self.assertEqual(result.shape[2], 3, "Should have 3 channels")
        
        # Width should match
        self.assertEqual(result.shape[1], 600, "Width should match target")
    
    def test_resize_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        img = np.random.randint(0, 255, (100, 300), dtype=np.uint8)
        
        # Test with invalid target width
        with self.assertRaises(ValueError):
            resize_maintaining_aspect(img, 0)
        
        with self.assertRaises(ValueError):
            resize_maintaining_aspect(img, -100)
        
        # Test with non-numpy input
        with self.assertRaises(TypeError):
            resize_maintaining_aspect([1, 2, 3], 200)


class TestApplyCLAHE(unittest.TestCase):
    """Test cases for apply_clahe function."""
    
    def test_clahe_basic(self):
        """Test basic CLAHE application."""
        # Create low-contrast grayscale image
        img = np.random.randint(100, 150, (200, 400), dtype=np.uint8)
        result = apply_clahe(img, clip_limit=2.0, grid_size=(8, 8))
        
        # Should be same shape
        self.assertEqual(result.shape, img.shape, "Shape should be preserved")
        
        # Should be uint8
        self.assertEqual(result.dtype, np.uint8, "Should be uint8")
        
        # Contrast should improve (std should increase)
        self.assertGreater(
            np.std(result), np.std(img),
            "Contrast (std) should improve"
        )
    
    def test_clahe_different_parameters(self):
        """Test CLAHE with different parameters."""
        img = np.random.randint(50, 200, (200, 400), dtype=np.uint8)
        
        # Different clip limits
        result_low = apply_clahe(img, clip_limit=1.0, grid_size=(8, 8))
        result_high = apply_clahe(img, clip_limit=4.0, grid_size=(8, 8))
        
        # Both should be valid
        self.assertEqual(result_low.dtype, np.uint8)
        self.assertEqual(result_high.dtype, np.uint8)
        
        # Different grid sizes
        result_small = apply_clahe(img, clip_limit=2.0, grid_size=(4, 4))
        result_large = apply_clahe(img, clip_limit=2.0, grid_size=(16, 16))
        
        # Both should be valid
        self.assertEqual(result_small.shape, img.shape)
        self.assertEqual(result_large.shape, img.shape)
    
    def test_clahe_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        valid_img = np.random.randint(0, 255, (200, 400), dtype=np.uint8)
        
        # Test with color image (3 channels)
        color_img = np.random.randint(0, 255, (200, 400, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            apply_clahe(color_img)
        
        # Test with wrong dtype
        float_img = valid_img.astype(np.float32)
        with self.assertRaises(ValueError):
            apply_clahe(float_img)
        
        # Test with non-numpy input
        with self.assertRaises(TypeError):
            apply_clahe([1, 2, 3])


class TestValidateImage(unittest.TestCase):
    """Test cases for validate_image function."""
    
    def test_validate_valid_images(self):
        """Test validation of valid images."""
        # Valid grayscale
        gray = np.random.randint(0, 255, (100, 200), dtype=np.uint8)
        is_valid, msg = validate_image(gray)
        self.assertTrue(is_valid, "Valid grayscale should pass")
        self.assertEqual(msg, "", "Valid image should have empty error message")
        
        # Valid color
        color = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        is_valid, msg = validate_image(color)
        self.assertTrue(is_valid, "Valid color should pass")
        self.assertEqual(msg, "")
    
    def test_validate_size_constraints(self):
        """Test size constraint validation."""
        # Too small width
        small = np.random.randint(0, 255, (50, 40), dtype=np.uint8)
        is_valid, msg = validate_image(small, min_width=50, min_height=20)
        self.assertFalse(is_valid, "Image below min_width should fail")
        self.assertIn("width", msg.lower())
        
        # Too small height
        short = np.random.randint(0, 255, (10, 100), dtype=np.uint8)
        is_valid, msg = validate_image(short, min_width=50, min_height=20)
        self.assertFalse(is_valid, "Image below min_height should fail")
        self.assertIn("height", msg.lower())
        
        # Too large
        huge = np.random.randint(0, 255, (100, 3000), dtype=np.uint8)
        is_valid, msg = validate_image(huge, max_width=2000)
        self.assertFalse(is_valid, "Image exceeding max_width should fail")
        self.assertIn("width", msg.lower())
    
    def test_validate_type_errors(self):
        """Test validation of type errors."""
        # Non-numpy array
        is_valid, msg = validate_image([1, 2, 3])
        self.assertFalse(is_valid)
        self.assertIn("numpy", msg.lower())
        
        # Empty array
        empty = np.array([], dtype=np.uint8)
        is_valid, msg = validate_image(empty)
        self.assertFalse(is_valid)
        self.assertIn("empty", msg.lower())
        
        # Wrong dtype
        float_img = np.random.rand(100, 200).astype(np.float32)
        is_valid, msg = validate_image(float_img)
        self.assertFalse(is_valid)
        self.assertIn("uint8", msg.lower())


class TestBatchPreprocessPlates(unittest.TestCase):
    """Test cases for batch_preprocess_plates function."""
    
    def test_batch_all_valid(self):
        """Test batch processing with all valid images."""
        plates = [
            np.random.randint(0, 255, (50, 150, 3), dtype=np.uint8),
            np.random.randint(0, 255, (60, 180, 3), dtype=np.uint8),
            np.random.randint(0, 255, (55, 160, 3), dtype=np.uint8)
        ]
        config = {'min_width': 200, 'use_clahe': False}
        
        results = batch_preprocess_plates(plates, config)
        
        # Should have same length
        self.assertEqual(len(results), len(plates), "Output length should match")
        
        # All should be successful
        self.assertTrue(
            all(r is not None for r in results),
            "All valid images should be processed"
        )
        
        # All should be grayscale and meet min_width
        for result in results:
            self.assertEqual(len(result.shape), 2, "Should be grayscale")
            self.assertGreaterEqual(result.shape[1], 200, "Should meet min_width")
    
    def test_batch_mixed_validity(self):
        """Test batch processing with mixed valid/invalid images."""
        plates = [
            np.random.randint(0, 255, (50, 150, 3), dtype=np.uint8),  # Valid
            np.array([]),  # Invalid - empty
            np.random.randint(0, 255, (60, 180, 3), dtype=np.uint8),  # Valid
        ]
        config = {'min_width': 200, 'use_clahe': False}
        
        results = batch_preprocess_plates(plates, config, validate=True)
        
        # Should have same length
        self.assertEqual(len(results), len(plates))
        
        # First and third should succeed, second should fail
        self.assertIsNotNone(results[0], "First image should succeed")
        self.assertIsNone(results[1], "Second image should fail")
        self.assertIsNotNone(results[2], "Third image should succeed")
    
    def test_batch_no_validation(self):
        """Test batch processing without validation."""
        plates = [
            np.random.randint(0, 255, (50, 150, 3), dtype=np.uint8),
            np.random.randint(0, 255, (60, 180, 3), dtype=np.uint8)
        ]
        config = {'min_width': 200, 'use_clahe': False}
        
        results = batch_preprocess_plates(plates, config, validate=False)
        
        # All should be processed (no validation checks)
        self.assertEqual(len(results), len(plates))
        self.assertTrue(all(r is not None for r in results))


class TestSavePreprocessedImage(unittest.TestCase):
    """Test cases for save_preprocessed_image function."""
    
    def test_save_image(self):
        """Test saving an image to disk."""
        img = np.random.randint(0, 255, (100, 200), dtype=np.uint8)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_plate.jpg')
            success = save_preprocessed_image(img, output_path)
            
            self.assertTrue(success, "Save should succeed")
            self.assertTrue(os.path.exists(output_path), "File should exist")
            
            # Verify can read back
            loaded = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
            self.assertIsNotNone(loaded, "Should be able to read back")
            self.assertEqual(loaded.shape, img.shape, "Shape should match")
    
    def test_save_create_directories(self):
        """Test that directories are created automatically."""
        img = np.random.randint(0, 255, (100, 200), dtype=np.uint8)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Path with nested directories
            output_path = os.path.join(tmpdir, 'subdir1', 'subdir2', 'test.jpg')
            success = save_preprocessed_image(img, output_path, create_dirs=True)
            
            self.assertTrue(success, "Save should succeed")
            self.assertTrue(os.path.exists(output_path), "File should exist")


class TestCalculateImageStats(unittest.TestCase):
    """Test cases for calculate_image_stats function."""
    
    def test_stats_grayscale(self):
        """Test statistics calculation for grayscale image."""
        img = np.random.randint(50, 200, (100, 200), dtype=np.uint8)
        stats = calculate_image_stats(img)
        
        # Check all keys present
        required_keys = ['mean', 'std', 'min', 'max', 'shape', 'dtype']
        for key in required_keys:
            self.assertIn(key, stats, f"Stats should contain '{key}'")
        
        # Check value ranges
        self.assertGreaterEqual(stats['mean'], 0)
        self.assertLessEqual(stats['mean'], 255)
        self.assertGreaterEqual(stats['std'], 0)
        self.assertGreaterEqual(stats['min'], 0)
        self.assertLessEqual(stats['max'], 255)
        self.assertEqual(stats['shape'], img.shape)
    
    def test_stats_color(self):
        """Test statistics calculation for color image."""
        img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        stats = calculate_image_stats(img)
        
        # Should work with color images too
        self.assertIn('mean', stats)
        self.assertIn('shape', stats)
        self.assertEqual(stats['shape'], img.shape)
    
    def test_stats_uniform_image(self):
        """Test statistics for uniform image (all same value)."""
        img = np.full((100, 200), 128, dtype=np.uint8)
        stats = calculate_image_stats(img)
        
        # Mean should be 128, std should be 0
        self.assertAlmostEqual(stats['mean'], 128.0, places=1)
        self.assertAlmostEqual(stats['std'], 0.0, places=5)
        self.assertEqual(stats['min'], 128)
        self.assertEqual(stats['max'], 128)


class TestIntegrationWithRealPlates(unittest.TestCase):
    """Integration tests with real GTA V plate crops."""
    
    def setUp(self):
        """Set up paths to test images."""
        self.test_images_dir = Path(project_root) / 'outputs' / 'test_images'
        self.config = {
            'min_width': 200,
            'use_clahe': True,
            'clahe_clip_limit': 2.0,
            'clahe_tile_grid_size': [8, 8]
        }
    
    def test_preprocess_real_gta_plate(self):
        """Test preprocessing with real GTA V test images if available."""
        if not self.test_images_dir.exists():
            self.skipTest("Test images directory not found")
        
        # Find first .jpg file
        image_files = list(self.test_images_dir.glob('*.jpg'))
        if not image_files:
            self.skipTest("No test images found")
        
        # Load first image
        img_path = image_files[0]
        img = cv2.imread(str(img_path))
        
        if img is None:
            self.skipTest(f"Could not load image: {img_path}")
        
        # This is a full frame, not a crop, so we'll just test that preprocessing works
        result = preprocess_plate(img, self.config)
        
        # Should produce valid output
        self.assertIsNotNone(result, "Preprocessing should succeed")
        self.assertEqual(len(result.shape), 2, "Output should be grayscale")
        self.assertEqual(result.dtype, np.uint8, "Output should be uint8")
        
        # Width should meet minimum
        self.assertGreaterEqual(
            result.shape[1], self.config['min_width'],
            "Width should meet minimum"
        )


def run_tests():
    """Run all unit tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPreprocessPlate))
    suite.addTests(loader.loadTestsFromTestCase(TestResizeMaintainingAspect))
    suite.addTests(loader.loadTestsFromTestCase(TestApplyCLAHE))
    suite.addTests(loader.loadTestsFromTestCase(TestValidateImage))
    suite.addTests(loader.loadTestsFromTestCase(TestBatchPreprocessPlates))
    suite.addTests(loader.loadTestsFromTestCase(TestSavePreprocessedImage))
    suite.addTests(loader.loadTestsFromTestCase(TestCalculateImageStats))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationWithRealPlates))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)

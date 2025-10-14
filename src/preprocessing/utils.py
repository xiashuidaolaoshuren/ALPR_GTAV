"""
Preprocessing Utility Functions

This module provides helper functions for batch preprocessing, image validation,
and utility operations for license plate image enhancement.

"""

import logging
from typing import List, Tuple, Optional
import cv2
import numpy as np
from pathlib import Path

from .image_enhancement import preprocess_plate

logger = logging.getLogger(__name__)


def validate_image(
    image: np.ndarray,
    min_width: int = 50,
    min_height: int = 20,
    max_width: int = 2000,
    max_height: int = 1000
) -> Tuple[bool, str]:
    """
    Validate image properties for preprocessing.
    
    Checks if an image meets basic requirements for license plate preprocessing:
    - Is a valid numpy array
    - Has correct dimensions (2D or 3D)
    - Has reasonable size constraints
    - Has valid data type (uint8)
    
    Args:
        image (np.ndarray): Input image to validate.
        min_width (int, optional): Minimum acceptable width. Default: 50.
        min_height (int, optional): Minimum acceptable height. Default: 20.
        max_width (int, optional): Maximum acceptable width. Default: 2000.
        max_height (int, optional): Maximum acceptable height. Default: 1000.
    
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
            - is_valid: True if image passes all checks, False otherwise
            - error_message: Empty string if valid, error description if invalid
    
    Example:
        >>> import numpy as np
        >>> valid_img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        >>> is_valid, msg = validate_image(valid_img)
        >>> is_valid
        True
        >>> msg
        ''
        >>> 
        >>> invalid_img = np.random.randint(0, 255, (10, 20), dtype=np.uint8)
        >>> is_valid, msg = validate_image(invalid_img)
        >>> is_valid
        False
        >>> 'height' in msg.lower()
        True
    
    Note:
        - GTA V plates are typically 200-400px wide when cropped
        - Validation prevents processing of invalid or corrupted crops
    """
    # Type check
    if not isinstance(image, np.ndarray):
        return False, f"Image must be numpy array, got {type(image)}"
    
    # Empty check
    if image.size == 0:
        return False, "Image is empty"
    
    # Dimension check
    if len(image.shape) not in [2, 3]:
        return False, f"Image must be 2D or 3D array, got shape {image.shape}"
    
    # Color channel check
    if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
        return False, f"Invalid number of channels: {image.shape[2]}"
    
    # Data type check
    if image.dtype != np.uint8:
        return False, f"Image must be uint8, got {image.dtype}"
    
    # Size constraints
    height, width = image.shape[:2]
    
    if width < min_width:
        return False, f"Width {width} below minimum {min_width}"
    
    if height < min_height:
        return False, f"Height {height} below minimum {min_height}"
    
    if width > max_width:
        return False, f"Width {width} exceeds maximum {max_width}"
    
    if height > max_height:
        return False, f"Height {height} exceeds maximum {max_height}"
    
    return True, ""


def batch_preprocess_plates(
    cropped_plates: List[np.ndarray],
    config: dict,
    validate: bool = True
) -> List[Optional[np.ndarray]]:
    """
    Preprocess multiple cropped license plate images in batch.
    
    Applies the preprocessing pipeline to a list of cropped plate images.
    Invalid images are skipped with None returned in their position.
    
    Args:
        cropped_plates (List[np.ndarray]): List of cropped plate images.
        config (dict): Preprocessing configuration (same as preprocess_plate).
        validate (bool, optional): Whether to validate images before processing.
            If True, invalid images return None. Default: True.
    
    Returns:
        List[Optional[np.ndarray]]: List of preprocessed images.
            None values indicate images that failed validation or processing.
    
    Example:
        >>> import numpy as np
        >>> plates = [
        ...     np.random.randint(0, 255, (50, 150, 3), dtype=np.uint8),
        ...     np.random.randint(0, 255, (60, 180, 3), dtype=np.uint8)
        ... ]
        >>> config = {'min_width': 200, 'use_clahe': False}
        >>> results = batch_preprocess_plates(plates, config)
        >>> len(results) == len(plates)
        True
        >>> all(r is not None for r in results)  # All valid
        True
    
    Note:
        - Processes images sequentially (not parallelized)
        - Logs warnings for failed images
        - Returns list with same length as input (preserves ordering)
    """
    results = []
    
    for i, plate in enumerate(cropped_plates):
        try:
            # Validate if requested
            if validate:
                is_valid, error_msg = validate_image(plate)
                if not is_valid:
                    logger.warning(f"Plate {i} failed validation: {error_msg}")
                    results.append(None)
                    continue
            
            # Preprocess
            preprocessed = preprocess_plate(plate, config)
            results.append(preprocessed)
            
        except Exception as e:
            logger.error(f"Error preprocessing plate {i}: {e}")
            results.append(None)
    
    successful = sum(1 for r in results if r is not None)
    logger.info(f"Batch preprocessing: {successful}/{len(cropped_plates)} successful")
    
    return results


def save_preprocessed_image(
    image: np.ndarray,
    output_path: str,
    create_dirs: bool = True
) -> bool:
    """
    Save preprocessed image to disk.
    
    Args:
        image (np.ndarray): Preprocessed image to save.
        output_path (str): Output file path (e.g., 'outputs/preprocessed/plate_001.jpg').
        create_dirs (bool, optional): Whether to create parent directories if they
            don't exist. Default: True.
    
    Returns:
        bool: True if save was successful, False otherwise.
    
    Example:
        >>> import numpy as np
        >>> import tempfile
        >>> import os
        >>> img = np.random.randint(0, 255, (100, 200), dtype=np.uint8)
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = os.path.join(tmpdir, 'test.jpg')
        ...     success = save_preprocessed_image(img, path)
        ...     success and os.path.exists(path)
        True
    
    Note:
        - Supports common formats: .jpg, .png, .bmp
        - JPEG quality is set to 95 for minimal compression loss
    """
    try:
        # Create directories if needed
        if create_dirs:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save image
        success = cv2.imwrite(output_path, image)
        
        if success:
            logger.debug(f"Saved preprocessed image to {output_path}")
        else:
            logger.error(f"Failed to save image to {output_path}")
        
        return success
        
    except Exception as e:
        logger.error(f"Error saving image to {output_path}: {e}")
        return False


def calculate_image_stats(image: np.ndarray) -> dict:
    """
    Calculate statistical properties of an image.
    
    Computes useful statistics for analyzing preprocessing effects:
    - Mean intensity
    - Standard deviation (contrast indicator)
    - Min/max values
    - Histogram information
    
    Args:
        image (np.ndarray): Input image (grayscale or color).
    
    Returns:
        dict: Dictionary with keys:
            - 'mean': Mean pixel intensity
            - 'std': Standard deviation (contrast)
            - 'min': Minimum pixel value
            - 'max': Maximum pixel value
            - 'shape': Image dimensions
            - 'dtype': Data type
    
    Example:
        >>> import numpy as np
        >>> img = np.random.randint(0, 255, (100, 200), dtype=np.uint8)
        >>> stats = calculate_image_stats(img)
        >>> 'mean' in stats and 'std' in stats
        True
        >>> 0 <= stats['mean'] <= 255
        True
    
    Note:
        - Useful for comparing before/after preprocessing
        - Higher std typically indicates better contrast
    """
    stats = {
        'mean': float(np.mean(image)),
        'std': float(np.std(image)),
        'min': int(np.min(image)),
        'max': int(np.max(image)),
        'shape': image.shape,
        'dtype': str(image.dtype)
    }
    
    return stats

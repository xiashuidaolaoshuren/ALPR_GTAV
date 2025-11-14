"""
Image Enhancement Module for License Plate Preprocessing

This module provides preprocessing functions for cropped license plate images
to improve OCR accuracy. Includes grayscale conversion, resizing, and CLAHE
(Contrast Limited Adaptive Histogram Equalization).

Note:
    Cropping functionality is provided by src.detection.utils.crop_detections()
    and should be reused. This module focuses only on enhancement operations.

"""

import logging
from typing import Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def preprocess_plate(cropped_image: np.ndarray, config: dict) -> np.ndarray:
    """
    Apply preprocessing pipeline to a cropped license plate image.

    This function applies a series of optional preprocessing steps to enhance
    license plate images for improved OCR accuracy. Steps include:
    1. Grayscale conversion (if image is color)
    2. Resizing to minimum width while maintaining aspect ratio
    3. CLAHE enhancement for improved contrast

    All steps are optional and controlled by configuration flags.

    Args:
        cropped_image (np.ndarray): Input image (cropped plate) in BGR or grayscale.
            Shape: (H, W, 3) for color or (H, W) for grayscale.
        config (dict): Preprocessing configuration with keys:
            - 'use_clahe' (bool): Enable CLAHE enhancement
            - 'clahe_clip_limit' (float): CLAHE clip limit (default: 2.0)
            - 'clahe_tile_grid_size' (list): CLAHE grid size (default: [8, 8])
            - 'min_width' (int): Minimum width in pixels (default: 200)
            - 'use_gaussian_blur' (bool): Enable Gaussian blur (optional)
            - 'gaussian_kernel_size' (list): Blur kernel size (optional)
            - 'use_sharpening' (bool): Enable sharpening (optional)
            - 'sharpen_strength' (float): Sharpening strength (optional)

    Returns:
        np.ndarray: Preprocessed image, typically grayscale.
            Shape: (H', W') where W' >= min_width and aspect ratio is preserved.

    Raises:
        ValueError: If input image is empty or invalid.
        TypeError: If input is not a numpy array.

    Example:
        >>> import cv2
        >>> import numpy as np
        >>> # Simulate a small cropped plate
        >>> plate = np.random.randint(0, 255, (50, 150, 3), dtype=np.uint8)
        >>> config = {
        ...     'min_width': 200,
        ...     'use_clahe': True,
        ...     'clahe_clip_limit': 2.0,
        ...     'clahe_tile_grid_size': [8, 8]
        ... }
        >>> enhanced = preprocess_plate(plate, config)
        >>> enhanced.shape[1] >= 200  # Width should be at least 200
        True
        >>> len(enhanced.shape) == 2  # Should be grayscale
        True

    Note:
        - The function creates a copy of the input to avoid modifying the original
        - Grayscale conversion is applied automatically if CLAHE is enabled
        - Optional features (blur, sharpening) are applied if enabled in config
    """
    if not isinstance(cropped_image, np.ndarray):
        raise TypeError(f"Input must be numpy array, got {type(cropped_image)}")

    if cropped_image.size == 0:
        raise ValueError("Input image is empty")

    # Create a copy to avoid modifying original
    image = cropped_image.copy()

    # Step 1: Convert to grayscale if needed (for CLAHE or OCR)
    # Most OCR engines work better with grayscale
    if len(image.shape) == 3:
        logger.debug("Converting BGR to grayscale")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Resize if below minimum width (maintain aspect ratio)
    height, width = image.shape[:2]
    min_width = config.get("min_width", 200)

    if width < min_width:
        logger.debug(f"Resizing from width {width} to {min_width} (aspect ratio maintained)")
        image = resize_maintaining_aspect(image, min_width)

    # Step 3: Optional Gaussian blur for noise reduction
    if config.get("use_gaussian_blur", False):
        kernel_size = tuple(config.get("gaussian_kernel_size", [3, 3]))
        logger.debug(f"Applying Gaussian blur with kernel {kernel_size}")
        image = cv2.GaussianBlur(image, kernel_size, 0)

    # Step 4: CLAHE enhancement (optional)
    if config.get("use_clahe", False):
        logger.debug("Applying CLAHE enhancement")
        clip_limit = config.get("clahe_clip_limit", 2.0)
        grid_size = tuple(config.get("clahe_tile_grid_size", [8, 8]))
        image = apply_clahe(image, clip_limit, grid_size)

    # Step 5: Optional sharpening
    if config.get("use_sharpening", False):
        strength = config.get("sharpen_strength", 1.0)
        logger.debug(f"Applying sharpening with strength {strength}")
        # Create sharpening kernel
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
        kernel = kernel * strength
        kernel[1, 1] = 9 - (strength - 1) * 8  # Adjust center
        image = cv2.filter2D(image, -1, kernel)

    return image


def resize_maintaining_aspect(
    image: np.ndarray, target_width: int, interpolation: int = cv2.INTER_CUBIC
) -> np.ndarray:
    """
    Resize image to target width while maintaining aspect ratio.

    This function scales an image to a specified width while preserving
    the original aspect ratio. Uses high-quality interpolation by default.

    Args:
        image (np.ndarray): Input image (grayscale or color).
            Shape: (H, W) or (H, W, C).
        target_width (int): Desired width in pixels. Must be positive.
        interpolation (int, optional): OpenCV interpolation method.
            Default: cv2.INTER_CUBIC (high quality).
            Options: INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_LANCZOS4.

    Returns:
        np.ndarray: Resized image with target width and scaled height.
            Shape: (H', target_width) or (H', target_width, C).

    Raises:
        ValueError: If target_width <= 0 or image is empty.
        TypeError: If input is not a numpy array.

    Example:
        >>> import numpy as np
        >>> # Create a 100x300 image
        >>> img = np.random.randint(0, 255, (100, 300), dtype=np.uint8)
        >>> resized = resize_maintaining_aspect(img, 600)
        >>> resized.shape
        (200, 600)
        >>> # Aspect ratio preserved: 100/300 == 200/600

    Note:
        - INTER_CUBIC is recommended for upscaling (smoother results)
        - INTER_AREA is recommended for downscaling (better quality)
        - For small images being upscaled significantly, INTER_LANCZOS4 may be better
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Input must be numpy array, got {type(image)}")

    if image.size == 0:
        raise ValueError("Input image is empty")

    if target_width <= 0:
        raise ValueError(f"Target width must be positive, got {target_width}")

    height, width = image.shape[:2]

    # Calculate scaling factor
    scale = target_width / width
    new_height = int(height * scale)
    new_size = (target_width, new_height)

    logger.debug(f"Resizing from ({width}, {height}) to {new_size}, scale={scale:.2f}")

    # Perform resize
    resized = cv2.resize(image, new_size, interpolation=interpolation)

    return resized


def apply_clahe(
    gray_image: np.ndarray, clip_limit: float = 2.0, grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to grayscale image.

    CLAHE enhances local contrast by dividing the image into tiles and applying
    histogram equalization to each tile with a contrast limiting threshold.
    This improves visibility of license plate text in poor lighting conditions.

    Args:
        gray_image (np.ndarray): Input grayscale image.
            Shape: (H, W). Must be single-channel uint8.
        clip_limit (float, optional): Contrast limiting threshold.
            Higher values increase contrast. Range: [1.0, 40.0]. Default: 2.0.
        grid_size (Tuple[int, int], optional): Size of grid for histogram equalization.
            Smaller tiles = more local adaptation, larger = more global.
            Default: (8, 8).

    Returns:
        np.ndarray: Enhanced grayscale image with improved local contrast.
            Shape: (H, W), dtype: uint8.

    Raises:
        ValueError: If image is not grayscale (single-channel) or not uint8.
        TypeError: If input is not a numpy array.

    Example:
        >>> import cv2
        >>> import numpy as np
        >>> # Create a low-contrast grayscale image
        >>> img = np.random.randint(100, 150, (200, 400), dtype=np.uint8)
        >>> enhanced = apply_clahe(img, clip_limit=3.0, grid_size=(8, 8))
        >>> enhanced.dtype
        dtype('uint8')
        >>> # Contrast should be improved
        >>> np.std(enhanced) > np.std(img)
        True

    Note:
        - Input must be grayscale (convert color images first)
        - clip_limit=1.0 is equivalent to standard histogram equalization
        - Typical range: 2.0-4.0 for most images
        - Smaller grid_size (e.g., 4x4) for more aggressive local adaptation
        - Larger grid_size (e.g., 16x16) for smoother global adaptation
    """
    if not isinstance(gray_image, np.ndarray):
        raise TypeError(f"Input must be numpy array, got {type(gray_image)}")

    if len(gray_image.shape) != 2:
        raise ValueError(f"Input must be grayscale (2D array), got shape {gray_image.shape}")

    if gray_image.dtype != np.uint8:
        raise ValueError(f"Input must be uint8, got {gray_image.dtype}")

    if gray_image.size == 0:
        raise ValueError("Input image is empty")

    logger.debug(f"Applying CLAHE with clip_limit={clip_limit}, grid_size={grid_size}")

    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

    # Apply to image
    enhanced = clahe.apply(gray_image)

    return enhanced

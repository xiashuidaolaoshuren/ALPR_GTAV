# Utilities Module

Shared utilities and helper functions used across all scripts and the main pipeline.

## Overview

The `utils/` module provides:
- Configuration loading and validation
- Video I/O operations (reading/writing frames)
- Visualization utilities (drawing bounding boxes, annotations)
- Logging configuration and utilities
- Data processing helpers
- Path and file management utilities

## Key Modules

### config_utils.py
**Configuration loading, validation, and management**

**Functions:**
- `load_config(path)` - Load YAML configuration
- `validate_config(config)` - Validate config structure and values
- `merge_configs(base, override)` - Merge configuration dictionaries
- `get_default_config()` - Return default configuration template
- `save_config(config, path)` - Save configuration to YAML

**Usage:**
```python
from src.utils.config_utils import load_config, validate_config

config = load_config('configs/pipeline_config.yaml')
validate_config(config)
```

---

### video_io.py
**Video reading, writing, and frame processing utilities**

**Classes:**
- `VideoReader` - Read frames from video file
- `VideoWriter` - Write frames to video file
- `FrameBuffer` - Efficient frame buffering

**Functions:**
- `get_video_info(video_path)` - Get video metadata (FPS, resolution, duration)
- `extract_frame_range(video_path, start, end)` - Extract specific frame range
- `resize_frame(frame, size)` - Resize with aspect ratio preservation
- `pad_frame(frame, size)` - Pad frame to exact size

**Usage:**
```python
from src.utils.video_io import VideoReader, get_video_info

# Get video information
info = get_video_info('video.mp4')
print(f"FPS: {info['fps']}, Size: {info['width']}x{info['height']}")

# Read frames
with VideoReader('video.mp4') as reader:
    for frame in reader:
        # Process frame
        pass
```

---

### visualization.py
**Drawing utilities for annotations and visualizations**

**Functions:**
- `draw_bounding_box(frame, box, color, thickness)` - Draw detection box
- `draw_text(frame, text, position, font_size, color)` - Draw text with background
- `draw_plate_annotation(frame, plate, text, confidence)` - Draw plate with text
- `draw_grid(frame, grid_size)` - Draw reference grid
- `create_comparison_grid(images)` - Create side-by-side comparison
- `draw_trajectory(frame, trajectory, color)` - Draw tracking trajectory

**Usage:**
```python
from src.utils.visualization import draw_bounding_box, draw_text

frame = cv2.imread('image.jpg')
# Draw detection
frame = draw_bounding_box(frame, [100, 100, 200, 150], (0, 255, 0), 2)
# Draw text
frame = draw_text(frame, 'License Plate', (100, 90), font_size=1, color=(0, 255, 0))
```

---

### logging_config.py
**Logging setup and utilities**

**Functions:**
- `setup_logger(name, level, file_path)` - Configure logger
- `get_logger(name)` - Get configured logger instance
- `log_to_file(message, file_path)` - Log to file
- `setup_tensorboard(log_dir)` - Setup TensorBoard logging

**Usage:**
```python
from src.utils.logging_config import setup_logger, get_logger

# Setup once
setup_logger('alpr', level='INFO', file_path='logs/alpr.log')

# Use anywhere
logger = get_logger('alpr')
logger.info('Processing started')
```

---

### data_utils.py
**Data loading, preprocessing, and transformation utilities**

**Functions:**
- `load_image(path, resize)` - Load and optionally resize image
- `normalize_image(image, mean, std)` - Normalize pixel values
- `convert_bbox_format(bbox, from_fmt, to_fmt)` - Convert bbox formats
- `split_dataset(data_dir, ratios)` - Split dataset into train/val/test
- `create_dataloader(dataset, batch_size, shuffle)` - Create DataLoader

**Usage:**
```python
from src.utils.data_utils import load_image, normalize_image

image = load_image('image.jpg', resize=(640, 640))
image = normalize_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

---

### path_utils.py
**Path and file management utilities**

**Functions:**
- `ensure_dir(path)` - Create directory if it doesn't exist
- `get_files(directory, pattern)` - Get files matching pattern
- `safe_rename(old_path, new_path)` - Rename with overwrite prevention
- `get_file_size(path)` - Get file size in MB/GB
- `get_absolute_path(relative_path)` - Convert to absolute path

**Usage:**
```python
from src.utils.path_utils import ensure_dir, get_files

# Create output directory
ensure_dir('outputs/results')

# Get all images
images = get_files('datasets/images', '*.jpg')
```

---

### metrics.py
**Calculation of metrics and evaluation utilities**

**Functions:**
- `calculate_iou(box1, box2)` - Intersection over union
- `calculate_precision_recall(predictions, ground_truth)` - Precision/recall
- `calculate_map(predictions, ground_truth)` - Mean average precision
- `calculate_ocr_accuracy(predictions, ground_truth)` - OCR accuracy
- `confuse_matrix(predictions, ground_truth, num_classes)` - Confusion matrix

**Usage:**
```python
from src.utils.metrics import calculate_iou, calculate_ocr_accuracy

# Calculate detection IOU
iou = calculate_iou([100, 100, 200, 150], [110, 110, 210, 160])

# Calculate OCR accuracy
accuracy = calculate_ocr_accuracy(predicted_texts, ground_truth_texts)
```

---

## Usage Examples

### Example 1: Load Config and Create Logger

```python
from src.utils.config_utils import load_config
from src.utils.logging_config import setup_logger, get_logger

# Setup
config = load_config('configs/pipeline_config.yaml')
setup_logger('app', level='INFO', file_path='logs/app.log')
logger = get_logger('app')

logger.info(f"Loaded config: {config}")
```

### Example 2: Read Video and Process Frames

```python
from src.utils.video_io import VideoReader, get_video_info
from src.utils.visualization import draw_bounding_box
import cv2

# Get info
info = get_video_info('video.mp4')
print(f"Video: {info['width']}x{info['height']} @ {info['fps']} FPS")

# Process frames
with VideoReader('video.mp4') as reader:
    for frame_idx, frame in enumerate(reader):
        # Your detection code here
        detections = model.detect(frame)
        
        # Draw and save
        for det in detections:
            frame = draw_bounding_box(frame, det['box'], (0, 255, 0), 2)
        
        if frame_idx % 10 == 0:
            cv2.imwrite(f'outputs/frame_{frame_idx}.jpg', frame)
```

### Example 3: Data Processing Pipeline

```python
from src.utils.data_utils import load_image, normalize_image, create_dataloader
from src.utils.path_utils import get_files, ensure_dir

# Prepare directories
ensure_dir('outputs/processed')

# Load and process images
image_paths = get_files('datasets/images', '*.jpg')
images = []

for path in image_paths:
    image = load_image(path, resize=(640, 640))
    image = normalize_image(image)
    images.append(image)

# Create dataloader
dataloader = create_dataloader(images, batch_size=32, shuffle=True)
```

## Dependencies

The utils module requires:
- `opencv-python` - Image/video processing
- `numpy` - Numerical operations
- `pyyaml` - Configuration files
- `torch` - Deep learning framework (for some utilities)

Install with:
```powershell
pip install -r requirements.txt
```

## Adding New Utilities

When adding new utilities:

1. **Create a new module file** (e.g., `new_utils.py`)
2. **Add comprehensive docstrings** with parameter descriptions
3. **Include type hints** for function signatures
4. **Add usage examples** in docstrings
5. **Test independently** before integration
6. **Update this README** with new module documentation

Example template:
```python
"""
Module description

Functions:
    function_name: Brief description
"""

from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)

def function_name(arg1: str, arg2: Optional[int] = None) -> str:
    """
    Brief description of function.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2 (default: None)
    
    Returns:
        Description of return value
    
    Example:
        >>> result = function_name('value', 10)
        >>> print(result)
    """
    # Implementation
    pass
```

## Testing Utilities

Each utility module includes doctests and unit tests:

```powershell
# Run doctests
python -m doctest src/utils/config_utils.py -v

# Run unit tests
python -m pytest tests/unit/test_config_utils.py -v
```

## Performance Considerations

- **Video I/O:** Uses efficient frame buffering to minimize memory
- **Image Processing:** Leverages OpenCV's optimized implementations
- **Data Loading:** Supports multiprocessing for faster batch creation
- **Visualization:** Renders efficiently for real-time preview

## References

- **[Main Pipeline](../../src/pipeline/alpr_pipeline.py)**
- **[Configuration](../../configs/pipeline_config.yaml)**
- **[Project Structure](../../docs/project_structure.md)**

---

*Last updated: November 15, 2025*

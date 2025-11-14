# API Reference

Complete API documentation for the GTA V ALPR system. This reference covers all public modules, classes, and functions.

## Table of Contents

- [Overview](#overview)
- [Pipeline Module](#pipeline-module)
- [Detection Module](#detection-module)
- [Recognition Module](#recognition-module)
- [Tracking Module](#tracking-module)
- [Preprocessing Module](#preprocessing-module)
- [Utils Module](#utils-module)
- [Common Patterns](#common-patterns)

---

## Overview

The GTA V ALPR system is organized into six main modules:

- **pipeline**: High-level orchestration of the complete ALPR workflow
- **detection**: YOLOv8-based license plate detection
- **recognition**: PaddleOCR-based text recognition
- **tracking**: ByteTrack-based plate tracking and OCR optimization
- **preprocessing**: Image enhancement for improved OCR accuracy
- **utils**: Configuration management and video I/O utilities

### Installation

```python
# Import the main pipeline
from src.pipeline.alpr_pipeline import ALPRPipeline

# Import specific components as needed
from src.detection.model import load_detection_model
from src.recognition.model import load_ocr_model
from src.tracking.tracker import PlateTrack
```

---

## Pipeline Module

### `src.pipeline.alpr_pipeline`

Main orchestration module for the complete ALPR workflow.

#### Class: `ALPRPipeline`

End-to-end Automatic License Plate Recognition pipeline that coordinates detection, tracking, preprocessing, and recognition.

**Attributes:**
- `config` (dict): Full pipeline configuration loaded from YAML
- `detection_model`: Loaded YOLOv8 model for plate detection
- `ocr_model`: Loaded PaddleOCR model for text recognition
- `tracks` (Dict[int, PlateTrack]): Active tracks indexed by track ID
- `frame_count` (int): Number of frames processed

**Methods:**

##### `__init__(config_path: str)`

Initialize ALPR pipeline with configuration.

**Parameters:**
- `config_path` (str): Path to pipeline configuration YAML file

**Raises:**
- `FileNotFoundError`: If config file doesn't exist
- `ValueError`: If configuration validation fails
- `RuntimeError`: If model initialization fails

**Example:**
```python
from src.pipeline.alpr_pipeline import ALPRPipeline

# Initialize with default config
pipeline = ALPRPipeline('configs/pipeline_config.yaml')

# Or use a custom config
pipeline = ALPRPipeline('configs/examples/high_accuracy.yaml')
```

##### `process_frame(frame: np.ndarray) -> Dict[int, PlateTrack]`

Process a single video frame through the complete ALPR pipeline.

**Parameters:**
- `frame` (np.ndarray): Input frame in BGR format (OpenCV convention), shape (H, W, 3)

**Returns:**
- Dict[int, PlateTrack]: Dictionary of active tracks indexed by track ID

**Raises:**
- `ValueError`: If frame is invalid
- `RuntimeError`: If processing fails

**Example:**
```python
import cv2

# Read video frame
frame = cv2.imread('test_frame.jpg')

# Process frame
tracks = pipeline.process_frame(frame)

# Access results
for track_id, track in tracks.items():
    if track.text:
        print(f"Track {track_id}: {track.text}")
        print(f"  Confidence: {track.ocr_confidence:.3f}")
        print(f"  BBox: {track.bbox}")
```

##### `reset() -> None`

Reset pipeline state (clear all tracks and counters).

**Example:**
```python
# Process first video
for frame in video1:
    tracks = pipeline.process_frame(frame)

# Reset before processing second video
pipeline.reset()

# Process second video
for frame in video2:
    tracks = pipeline.process_frame(frame)
```

##### `get_statistics() -> dict`

Get current pipeline statistics.

**Returns:**
- dict: Statistics including:
  - `frame_count`: Total frames processed
  - `track_count`: Current active track count
  - `active_count`: Number of active tracks
  - `recognized_count`: Tracks with recognized text
  - `avg_track_age`: Average track age
  - `avg_ocr_confidence`: Average OCR confidence

**Example:**
```python
# Get statistics
stats = pipeline.get_statistics()

print(f"Processed {stats['frame_count']} frames")
print(f"Active tracks: {stats['active_count']}")
print(f"Recognized: {stats['recognized_count']}")
print(f"Avg confidence: {stats['avg_ocr_confidence']:.3f}")
```

---

## Detection Module

### `src.detection.model`

YOLOv8-based license plate detection.

#### Function: `load_detection_model(model_path: str, device: str = 'cuda') -> YOLO`

Load YOLOv8 detection model from weights file.

**Parameters:**
- `model_path` (str): Path to model weights file (.pt)
- `device` (str): Computation device ('cuda' or 'cpu')

**Returns:**
- YOLO: Loaded YOLOv8 model ready for inference

**Raises:**
- `FileNotFoundError`: If model file doesn't exist
- `RuntimeError`: If model loading fails

**Example:**
```python
from src.detection.model import load_detection_model

# Load fine-tuned model
model = load_detection_model(
    model_path='models/detection/yolov8_finetuned_v2_best.pt',
    device='cuda'
)

# Or load baseline model
baseline = load_detection_model(
    model_path='models/detection/yolov8n.pt',
    device='cpu'
)
```

#### Function: `detect_plates(model: YOLO, image: np.ndarray, conf: float = 0.25, iou: float = 0.45) -> List[Tuple]`

Detect license plates in a single image.

**Parameters:**
- `model` (YOLO): Loaded YOLOv8 model
- `image` (np.ndarray): Input image in BGR format
- `conf` (float): Confidence threshold (0.0-1.0)
- `iou` (float): IOU threshold for NMS (0.0-1.0)

**Returns:**
- List[Tuple]: List of detections, each (bbox, confidence, class_id)

**Example:**
```python
import cv2
from src.detection.model import load_detection_model, detect_plates

# Load model and image
model = load_detection_model('models/detection/yolov8n.pt')
image = cv2.imread('test_image.jpg')

# Detect plates
detections = detect_plates(model, image, conf=0.30, iou=0.45)

# Process results
for bbox, confidence, class_id in detections:
    x1, y1, x2, y2 = bbox
    print(f"Plate detected: confidence={confidence:.3f}")
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
```

#### Function: `batch_detect_plates(model: YOLO, images: List[np.ndarray], conf: float = 0.25, iou: float = 0.45) -> List[List[Tuple]]`

Batch detection for multiple images (GPU-optimized).

**Parameters:**
- `model` (YOLO): Loaded YOLOv8 model
- `images` (List[np.ndarray]): List of images in BGR format
- `conf` (float): Confidence threshold
- `iou` (float): IOU threshold for NMS

**Returns:**
- List[List[Tuple]]: Detections for each image

**Example:**
```python
# Load batch of images
images = [cv2.imread(f'frame_{i:04d}.jpg') for i in range(32)]

# Batch detect (faster than sequential)
all_detections = batch_detect_plates(model, images, conf=0.25)

# Process each frame's detections
for frame_idx, detections in enumerate(all_detections):
    print(f"Frame {frame_idx}: {len(detections)} plates")
```

---

## Recognition Module

### `src.recognition.model`

PaddleOCR-based text recognition with post-processing.

#### Function: `load_ocr_model(config: dict) -> PaddleOCR`

Load PaddleOCR model with specified configuration.

**Parameters:**
- `config` (dict): Recognition configuration dict containing:
  - `use_gpu` (bool): Enable GPU acceleration
  - `lang` (str): Language model ('en', 'ch', etc.)
  - `use_textline_orientation` (bool): Enable angle classification
  - `show_log` (bool): Show PaddleOCR logs

**Returns:**
- PaddleOCR: Loaded OCR model

**Example:**
```python
from src.recognition.model import load_ocr_model

# Load with config
config = {
    'use_gpu': True,
    'lang': 'en',
    'use_textline_orientation': True,
    'show_log': False
}
ocr_model = load_ocr_model(config)
```

#### Function: `recognize_text(preprocessed_image: np.ndarray, ocr_model: PaddleOCR, config: dict, enable_adaptive_preprocessing: bool = False, preprocessing_config: dict = None) -> Tuple[Optional[str], float]`

Recognize text from a preprocessed plate crop.

**Parameters:**
- `preprocessed_image` (np.ndarray): Cropped plate image in BGR format
- `ocr_model` (PaddleOCR): Loaded OCR model
- `config` (dict): Recognition configuration with:
  - `rec_threshold` (float): Recognition confidence threshold
  - `regex` (str): Validation regex pattern (e.g., '^\d{2}[A-Z]{3}\d{3}$')
  - `prefer_largest_box` (bool): Use text from largest bounding box
  - `mask_top_ratio` (float): Mask top portion of image (0.0-1.0)
  - `min_conf` (float): Minimum confidence after filtering
- `enable_adaptive_preprocessing` (bool): Apply image enhancement
- `preprocessing_config` (dict): Preprocessing parameters

**Returns:**
- Tuple[Optional[str], float]: (recognized_text, confidence) or (None, 0.0) if no valid text

**Example:**
```python
import cv2
from src.recognition.model import load_ocr_model, recognize_text

# Load OCR model
config = {'use_gpu': True, 'lang': 'en', 'rec_threshold': 0.5}
ocr_model = load_ocr_model(config)

# Crop plate region
plate_crop = frame[y1:y2, x1:x2]

# Recognize text
text, confidence = recognize_text(
    preprocessed_image=plate_crop,
    ocr_model=ocr_model,
    config={
        'rec_threshold': 0.5,
        'regex': r'^\d{2}[A-Z]{3}\d{3}$',
        'prefer_largest_box': True,
        'mask_top_ratio': 0.0,
        'min_conf': 0.3
    },
    enable_adaptive_preprocessing=True
)

if text:
    print(f"Recognized: {text} (conf={confidence:.3f})")
else:
    print("No valid text recognized")
```

---

## Tracking Module

### `src.tracking.tracker`

License plate tracking with intelligent OCR triggering.

#### Class: `PlateTrack`

Manages the state and lifecycle of a tracked license plate.

**Attributes:**
- `id` (int): Unique track identifier assigned by ByteTrack
- `bbox` (Tuple[int, int, int, int]): Bounding box coordinates (x1, y1, x2, y2)
- `detection_confidence` (float): Latest detection confidence score
- `text` (Optional[str]): Recognized plate text (None if not yet recognized)
- `ocr_confidence` (float): Confidence score from OCR recognition
- `age` (int): Number of frames since track was first detected
- `frames_since_last_ocr` (int): Frames elapsed since last OCR run
- `is_active` (bool): Whether track is currently active (not lost)

**Methods:**

##### `__init__(track_id: int, bbox: Tuple[int, int, int, int], confidence: float)`

Initialize a new plate track.

**Example:**
```python
from src.tracking.tracker import PlateTrack

# Create new track
track = PlateTrack(
    track_id=1,
    bbox=(100, 200, 250, 300),
    confidence=0.85
)
```

##### `update(bbox: Tuple[int, int, int, int], confidence: float) -> None`

Update track with new detection information.

**Parameters:**
- `bbox` (Tuple[int, int, int, int]): New bounding box
- `confidence` (float): New detection confidence

**Example:**
```python
# Update track with new detection
track.update(
    bbox=(105, 205, 255, 305),
    confidence=0.90
)
```

##### `update_text(text: str, confidence: float) -> None`

Update track with OCR recognition results.

**Parameters:**
- `text` (str): Recognized text
- `confidence` (float): OCR confidence score

**Example:**
```python
# Update track with OCR result
track.update_text(text="12ABC345", confidence=0.87)
```

##### `should_run_ocr(config: dict) -> bool`

Determine if OCR should be run for this track.

**Parameters:**
- `config` (dict): Tracking configuration with:
  - `ocr_interval` (int): Minimum frames between OCR runs
  - `ocr_confidence_threshold` (float): Minimum confidence to skip OCR

**Returns:**
- bool: True if OCR should be run

**Logic:**
1. Always run OCR on first detection
2. Run if no text recognized yet
3. Run if `ocr_interval` frames have passed since last OCR
4. Skip if recent high-confidence text exists

**Example:**
```python
# Check if OCR should run
config = {'ocr_interval': 30, 'ocr_confidence_threshold': 0.7}

if track.should_run_ocr(config):
    # Run OCR
    text, conf = recognize_text(plate_crop, ocr_model, config)
    track.update_text(text, conf)
```

##### `mark_lost() -> None`

Mark track as lost (no longer active).

**Example:**
```python
# Track disappeared from frame
track.mark_lost()
print(f"Track {track.id} lost, age={track.age}")
```

### `src.tracking.utils`

#### Function: `get_track_summary(tracks: Dict[int, PlateTrack]) -> dict`

Get summary statistics for a collection of tracks.

**Parameters:**
- `tracks` (Dict[int, PlateTrack]): Dictionary of tracks

**Returns:**
- dict: Summary with keys:
  - `total`: Total number of tracks
  - `active`: Number of active tracks
  - `recognized`: Number of tracks with recognized text
  - `avg_age`: Average track age
  - `avg_ocr_confidence`: Average OCR confidence

**Example:**
```python
from src.tracking.utils import get_track_summary

# Get summary
summary = get_track_summary(pipeline.tracks)

print(f"Total tracks: {summary['total']}")
print(f"Active: {summary['active']}")
print(f"Recognized: {summary['recognized']}")
print(f"Avg OCR confidence: {summary['avg_ocr_confidence']:.3f}")
```

---

## Preprocessing Module

### `src.preprocessing.image_enhancement`

Image enhancement functions for improved OCR accuracy.

#### Function: `apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray`

Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

**Parameters:**
- `image` (np.ndarray): Input image (grayscale or BGR)
- `clip_limit` (float): Contrast enhancement limit (1.0-4.0)
- `tile_grid_size` (Tuple[int, int]): Grid size for localized enhancement

**Returns:**
- np.ndarray: Enhanced image

**Example:**
```python
from src.preprocessing.image_enhancement import apply_clahe

# Enhance contrast for better OCR
enhanced = apply_clahe(
    plate_crop,
    clip_limit=2.0,
    tile_grid_size=(8, 8)
)
```

#### Function: `apply_sharpening(image: np.ndarray, strength: float = 1.0) -> np.ndarray`

Apply sharpening filter to enhance text edges.

**Parameters:**
- `image` (np.ndarray): Input image
- `strength` (float): Sharpening intensity (0.5-2.0)

**Returns:**
- np.ndarray: Sharpened image

**Example:**
```python
from src.preprocessing.image_enhancement import apply_sharpening

# Sharpen text edges
sharpened = apply_sharpening(plate_crop, strength=1.0)
```

#### Function: `preprocess_plate(image: np.ndarray, config: dict) -> np.ndarray`

Apply full preprocessing pipeline to a plate crop.

**Parameters:**
- `image` (np.ndarray): Input plate crop
- `config` (dict): Preprocessing configuration with:
  - `enable_enhancement` (bool): Enable enhancement
  - `min_width` (int): Minimum width for upscaling
  - `use_clahe` (bool): Enable CLAHE
  - `clahe_clip_limit` (float): CLAHE clip limit
  - `clahe_tile_grid_size` (Tuple[int, int]): CLAHE grid size
  - `use_gaussian_blur` (bool): Enable Gaussian blur
  - `gaussian_kernel_size` (Tuple[int, int]): Blur kernel size
  - `use_sharpening` (bool): Enable sharpening
  - `sharpen_strength` (float): Sharpening strength

**Returns:**
- np.ndarray: Preprocessed image

**Example:**
```python
from src.preprocessing.image_enhancement import preprocess_plate

# Full preprocessing pipeline
config = {
    'enable_enhancement': True,
    'min_width': 200,
    'use_clahe': True,
    'clahe_clip_limit': 2.0,
    'clahe_tile_grid_size': (8, 8),
    'use_gaussian_blur': False,
    'use_sharpening': True,
    'sharpen_strength': 1.0
}

preprocessed = preprocess_plate(plate_crop, config)
```

---

## Utils Module

### `src.utils.config`

Configuration loading and validation.

#### Function: `load_config(config_path: str) -> dict`

Load and validate pipeline configuration from YAML.

**Parameters:**
- `config_path` (str): Path to configuration YAML file

**Returns:**
- dict: Validated configuration dictionary

**Raises:**
- `FileNotFoundError`: If config file doesn't exist
- `ValueError`: If configuration is invalid

**Example:**
```python
from src.utils.config import load_config

# Load and validate config
config = load_config('configs/pipeline_config.yaml')

# Access configuration
print(f"Detection confidence: {config['detection']['confidence_threshold']}")
print(f"OCR interval: {config['tracking']['ocr_interval']}")
```

### `src.utils.video_io`

Video reading and writing utilities.

#### Function: `read_video_frames(video_path: str) -> Generator[np.ndarray, None, None]`

Read video frames one at a time (memory-efficient).

**Parameters:**
- `video_path` (str): Path to video file

**Yields:**
- np.ndarray: Video frames in BGR format

**Example:**
```python
from src.utils.video_io import read_video_frames
from src.pipeline.alpr_pipeline import ALPRPipeline

# Initialize pipeline
pipeline = ALPRPipeline('configs/pipeline_config.yaml')

# Process video
for frame in read_video_frames('input_video.mp4'):
    tracks = pipeline.process_frame(frame)
    
    # Process results
    for track_id, track in tracks.items():
        if track.text:
            print(f"Frame {pipeline.frame_count}: {track.text}")
```

#### Function: `write_annotated_video(input_path: str, output_path: str, tracks_by_frame: List[Dict[int, PlateTrack]], fps: int = 30) -> None`

Write video with annotated bounding boxes and text.

**Parameters:**
- `input_path` (str): Path to input video
- `output_path` (str): Path to output video
- `tracks_by_frame` (List[Dict[int, PlateTrack]]): Tracks for each frame
- `fps` (int): Output video frame rate

**Example:**
```python
from src.utils.video_io import write_annotated_video

# Collect tracks for each frame
all_tracks = []
for frame in read_video_frames('input.mp4'):
    tracks = pipeline.process_frame(frame)
    all_tracks.append(tracks)

# Write annotated video
write_annotated_video(
    input_path='input.mp4',
    output_path='output_annotated.mp4',
    tracks_by_frame=all_tracks,
    fps=30
)
```

---

## Common Patterns

### Pattern 1: Complete Video Processing

```python
from src.pipeline.alpr_pipeline import ALPRPipeline
from src.utils.video_io import read_video_frames
import cv2

# Initialize pipeline
pipeline = ALPRPipeline('configs/pipeline_config.yaml')

# Process video
for frame in read_video_frames('gameplay.mp4'):
    # Run pipeline
    tracks = pipeline.process_frame(frame)
    
    # Annotate frame
    for track_id, track in tracks.items():
        if track.text:
            x1, y1, x2, y2 = track.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, track.text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display (optional)
    cv2.imshow('ALPR', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Print statistics
stats = pipeline.get_statistics()
print(f"Processed {stats['frame_count']} frames")
print(f"Recognized {stats['recognized_count']} plates")
```

### Pattern 2: Custom Detection Only

```python
from src.detection.model import load_detection_model, detect_plates
import cv2

# Load model
model = load_detection_model('models/detection/yolov8_finetuned_v2_best.pt')

# Read image
image = cv2.imread('test_image.jpg')

# Detect plates
detections = detect_plates(model, image, conf=0.30, iou=0.45)

# Process detections
for bbox, confidence, class_id in detections:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, f'{confidence:.2f}', (x1, y1 - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow('Detections', image)
cv2.waitKey(0)
```

### Pattern 3: Custom OCR with Preprocessing

```python
from src.recognition.model import load_ocr_model, recognize_text
from src.preprocessing.image_enhancement import preprocess_plate
import cv2

# Load OCR model
config = {
    'use_gpu': True,
    'lang': 'en',
    'use_textline_orientation': True,
    'show_log': False
}
ocr_model = load_ocr_model(config)

# Read plate crop
plate_crop = cv2.imread('plate_crop.jpg')

# Preprocess
preprocessing_config = {
    'enable_enhancement': True,
    'min_width': 200,
    'use_clahe': True,
    'clahe_clip_limit': 2.0,
    'clahe_tile_grid_size': (8, 8),
    'use_sharpening': True,
    'sharpen_strength': 1.0
}
preprocessed = preprocess_plate(plate_crop, preprocessing_config)

# Recognize text
recognition_config = {
    'rec_threshold': 0.5,
    'regex': r'^\d{2}[A-Z]{3}\d{3}$',
    'prefer_largest_box': True,
    'mask_top_ratio': 0.0,
    'min_conf': 0.3
}
text, confidence = recognize_text(preprocessed, ocr_model, recognition_config)

if text:
    print(f"Recognized: {text} (confidence={confidence:.3f})")
else:
    print("No valid text recognized")
```

### Pattern 4: Batch Processing

```python
from src.detection.model import load_detection_model, batch_detect_plates
import cv2
import glob

# Load model
model = load_detection_model('models/detection/yolov8_finetuned_v2_best.pt')

# Load batch of images
image_paths = glob.glob('test_images/*.jpg')
images = [cv2.imread(path) for path in image_paths]

# Batch detect (GPU-optimized)
all_detections = batch_detect_plates(model, images, conf=0.25, iou=0.45)

# Process results
for path, detections in zip(image_paths, all_detections):
    print(f"{path}: {len(detections)} plates detected")
    for bbox, confidence, class_id in detections:
        print(f"  BBox: {bbox}, Conf: {confidence:.3f}")
```

### Pattern 5: Custom Configuration

```python
from src.pipeline.alpr_pipeline import ALPRPipeline
import yaml

# Load base config
with open('configs/pipeline_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Customize
config['detection']['confidence_threshold'] = 0.30
config['tracking']['ocr_interval'] = 20
config['preprocessing']['use_clahe'] = True

# Save custom config
with open('configs/custom_config.yaml', 'w') as f:
    yaml.dump(config, f)

# Use custom config
pipeline = ALPRPipeline('configs/custom_config.yaml')
```

---

## Additional Resources

- **Configuration Guide**: [docs/configuration_guide.md](configuration_guide.md)
- **User Guide**: [docs/user_guide.md](user_guide.md)
- **Developer Guide**: [docs/developer_guide.md](developer_guide.md)
- **Troubleshooting**: [docs/troubleshooting.md](troubleshooting.md)
- **Generated HTML Docs**: [docs/api/src/index.html](api/src/index.html)

---

*Last updated: November 14, 2025*

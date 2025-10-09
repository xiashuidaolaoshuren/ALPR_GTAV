# Project Structure Documentation

This document provides detailed explanations of the GTA V ALPR project directory structure, module responsibilities, and file placement guidelines.

## Directory Overview

```
ALPR_GTA5/
├── src/                    # Source code modules
├── datasets/               # Training and validation data
├── models/                 # Model weights storage
├── configs/                # Configuration files
├── scripts/                # Componentized CLI tools (no wrappers)
├── tests/                  # Test suite
├── docs/                   # Project documentation
├── outputs/                # Processing results and logs
├── .venv/                  # Python virtual environment
└── shrimp_data/            # Task management data
```

---

## Source Code (`src/`)

The main source code organized into functional modules following a modular architecture pattern.

### `src/detection/`
**Purpose**: License plate detection using YOLOv8

**Key Responsibilities:**
- Load pre-trained YOLOv8 detection model
- Perform inference on images/video frames
- Parse detection results and extract bounding boxes
- Apply confidence filtering and Non-Maximum Suppression (NMS)
- Visualization of detection results

**Expected Files:**
- `model.py` - YOLOv8 model loading and inference
- `utils.py` - Detection utilities (NMS, coordinate conversion, visualization)
- `config.py` - Detection-specific configuration handling
- `__init__.py` - Module initialization

**Example Usage:**
```python
from src.detection import LicensePlateDetector

detector = LicensePlateDetector(config)
detections = detector.detect(image)
```

### `src/recognition/`
**Purpose**: Text recognition from detected license plate regions using PaddleOCR

**Key Responsibilities:**
- Initialize PaddleOCR engine
- Preprocess cropped plate images for OCR
- Extract text from plate images
- Post-process OCR results (filtering, formatting)
- Confidence scoring

**Expected Files:**
- `model.py` - PaddleOCR initialization and inference
- `preprocessor.py` - Image preprocessing for OCR
- `postprocessor.py` - Text cleaning and validation
- `utils.py` - Recognition utilities
- `__init__.py` - Module initialization

**Example Usage:**
```python
from src.recognition import PlateRecognizer

recognizer = PlateRecognizer(config)
text, confidence = recognizer.recognize(plate_crop)
```

### `src/tracking/`
**Purpose**: Multi-object tracking to maintain plate identity across frames

**Key Responsibilities:**
- Implement ByteTrack or IOU-based tracking
- Associate detections across frames
- Manage track lifecycle (creation, update, deletion)
- Optimize OCR calls using tracking information
- Track state management

**Expected Files:**
- `tracker.py` - Main tracking implementation
- `track.py` - Track object definition
- `association.py` - Detection-to-track matching algorithms
- `utils.py` - Tracking utilities
- `__init__.py` - Module initialization

**Example Usage:**
```python
from src.tracking import PlateTracker

tracker = PlateTracker(config)
tracked_plates = tracker.update(detections, frame_id)
```

### `src/preprocessing/`
**Purpose**: Image enhancement and preprocessing utilities

**Key Responsibilities:**
- Image quality enhancement (CLAHE, sharpening)
- Noise reduction (Gaussian blur, denoising)
- Image normalization and resizing
- Color space conversions
- Data augmentation (during training)

**Expected Files:**
- `enhancer.py` - Image enhancement functions
- `augmentation.py` - Data augmentation pipeline
- `transforms.py` - Image transformation utilities
- `utils.py` - Preprocessing utilities
- `__init__.py` - Module initialization

**Example Usage:**
```python
from src.preprocessing import ImageEnhancer

enhancer = ImageEnhancer(config)
enhanced_image = enhancer.enhance(image)
```

### `src/pipeline/`
**Purpose**: Orchestrate the complete ALPR workflow

**Key Responsibilities:**
- Integrate detection, recognition, and tracking modules
- Manage end-to-end processing pipeline
- Handle video/batch processing
- Coordinate module interactions
- Result aggregation and output

**Expected Files:**
- `alpr_pipeline.py` - Main pipeline class
- `video_processor.py` - Video processing utilities
- `batch_processor.py` - Batch image processing
- `result_handler.py` - Result formatting and export
- `__init__.py` - Module initialization

**Example Usage:**
```python
from src.pipeline import ALPRPipeline

pipeline = ALPRPipeline(config)
results = pipeline.process_video(video_path)
```

### `src/utils/`
**Purpose**: Common utilities and helper functions

**Key Responsibilities:**
- Configuration file loading and validation (ConfigLoader)
- Logging setup and management
- File I/O operations
- Visualization utilities
- Common helper functions

**Expected Files:**
- `config.py` - ConfigLoader class (✓ implemented)
- `logger.py` - Logging configuration
- `visualization.py` - Drawing and annotation utilities
- `io_utils.py` - File reading/writing helpers
- `metrics.py` - Evaluation metrics
- `__init__.py` - Module initialization

---

## Datasets (`datasets/`)

Training, validation, and test data for the ALPR system.

### `datasets/lpr/`
**Purpose**: YOLO format dataset for license plate detection

**Structure:**
```
lpr/
├── data.yaml              # Dataset configuration
├── train/
│   ├── images/           # Training images (.jpg, .png)
│   └── labels/           # YOLO annotations (.txt)
├── valid/
│   ├── images/           # Validation images
│   └── labels/           # YOLO annotations
└── test/
    ├── images/           # Test images
    └── labels/           # YOLO annotations (optional)
```

**YOLO Annotation Format:**
Each `.txt` file contains one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```
- `class_id`: Always 0 for license plates
- Coordinates: Normalized to [0.0, 1.0] relative to image dimensions
- Example: `0 0.512 0.384 0.156 0.089`

**data.yaml Format:**
```yaml
path: datasets/lpr
train: train/images
val: valid/images
test: test/images

nc: 1  # Number of classes
names: ['license_plate']  # Class names
```

### `datasets/ocr/`
**Purpose**: OCR training data (optional, for fine-tuning)

**Structure:**
```
ocr/
└── images/               # Cropped license plate images
    ├── plate_001.jpg
    ├── plate_002.jpg
    └── ...
```

---

## Models (`models/`)

Storage for trained model weights and checkpoints.

### `models/detection/`
**Purpose**: YOLOv8 detection model weights

**Expected Files:**
- `yolov8n.pt` - Nano model (fastest, less accurate)
- `yolov8s.pt` - Small model (balanced)
- `yolov8m.pt` - Medium model (more accurate)
- `best.pt` - Best checkpoint from training
- `last.pt` - Latest checkpoint

**Note:** Models are not tracked in Git (excluded by `.gitignore`).

### `models/recognition/`
**Purpose**: PaddleOCR model weights

**Expected Files:**
- PaddleOCR models are automatically downloaded to user cache
- Custom fine-tuned models can be placed here
- Model configuration files

---

## Configuration (`configs/`)

YAML configuration files for all modules.

### `pipeline_config.yaml`
**Purpose**: Main configuration file for the ALPR pipeline

**Sections:**
- `detection`: YOLOv8 model path, thresholds, device settings
- `recognition`: PaddleOCR language, GPU settings, thresholds
- `tracking`: Tracker type, matching thresholds, OCR optimization
- `preprocessing`: Enhancement settings, augmentation parameters
- `pipeline`: Output settings, logging configuration

**Usage:**
```python
from src.utils.config import ConfigLoader

config = ConfigLoader.load_yaml('configs/pipeline_config.yaml')
detection_config = config['detection']
```

---

## Scripts (`scripts/`)

All runnable CLI entrypoints live directly inside their functional subfolders.
There are no root-level Python shims—call the scripts from their component
directory paths.

```
scripts/
├── data_ingestion/      # Frame extraction, metadata generation, cleaning
├── annotation/          # Label Studio helpers and converters
├── inference/           # Single-image and video inference utilities
├── evaluation/          # Evaluation, reporting, visualisation scripts
└── diagnostics/         # GPU checks, dataset quality analysers
```

Each subdirectory ships with its own README describing responsibilities,
required arguments, and output conventions. Add new tooling inside the
appropriate folder so the invocation path mirrors the workflow grouping.

---

## Tests (`tests/`)

Test suite following pytest conventions.

### `tests/unit/`
**Purpose**: Unit tests for individual components

**Expected Test Files:**
- `test_detection.py` - Detection module tests
- `test_recognition.py` - Recognition module tests
- `test_tracking.py` - Tracking module tests
- `test_preprocessing.py` - Preprocessing tests
- `test_config.py` - ConfigLoader tests

**Example Test:**
```python
def test_config_loader():
    config = ConfigLoader.load_yaml('configs/pipeline_config.yaml')
    assert 'detection' in config
    assert config['detection']['confidence_threshold'] == 0.25
```

### `tests/integration/`
**Purpose**: Integration tests for module interactions

**Expected Test Files:**
- `test_pipeline.py` - End-to-end pipeline tests
- `test_detection_recognition.py` - Detection + OCR workflow
- `test_video_processing.py` - Video processing tests

### `tests/data/`
**Purpose**: Test fixtures and sample data

**Contents:**
- Sample images for testing
- Expected output files
- Mock configuration files

---

## Documentation (`docs/`)

Project documentation and guides.

### Expected Documents:

**`project_structure.md`** (✓ this file)
- Directory structure explanation
- Module responsibilities
- File placement guidelines

**`data_collection_strategy.md`** (Week 1, Task 4)
- GTA V data collection methods
- ScriptHookV integration guide
- Capture scenarios and best practices

**`api_reference.md`** (Future)
- API documentation for all modules
- Function signatures and examples

**`training_guide.md`** (Week 3+)
- How to fine-tune YOLOv8
- PaddleOCR training instructions
- Best practices and tips

---

## Outputs (`outputs/`)

Processing results, logs, and intermediate files.

### Expected Contents:

**Logs:**
- `alpr_pipeline.log` - Main pipeline logs
- `environment_setup.log` - GPU verification logs

**Results:**
- `annotated_images/` - Images with bounding boxes and text
- `annotated_videos/` - Processed video files
- `crops/` - Extracted license plate crops
- `results.csv` - Detection/recognition results table

**Checkpoints:**
- `training_checkpoints/` - Model training checkpoints (Week 3+)

**Visualizations:**
- `detection_plots/` - Performance plots
- `confusion_matrices/` - Evaluation visualizations

---

## File Naming Conventions

### Images:
```
<scene>_<condition>_<frame_number>.jpg
Example: highway_day_00123.jpg, parking_night_00456.jpg
```

### Labels (YOLO):
```
<image_name>.txt (matching image filename)
Example: highway_day_00123.txt
```

### Scripts:
```
<action>_<target>.py
Example: detect_image.py, train_detection_model.py
```

### Modules:
```
lowercase_with_underscores.py
Example: license_plate_detector.py, ocr_recognizer.py
```

### Test Files:
```
test_<module_name>.py
Example: test_detection.py, test_pipeline.py
```

---

## Module Import Patterns

### Recommended Import Style:

```python
# Absolute imports from project root
from src.detection import LicensePlateDetector
from src.recognition import PlateRecognizer
from src.tracking import PlateTracker
from src.utils.config import ConfigLoader
from src.utils.logger import setup_logger

# Use module-level imports to expose key classes
# in each module's __init__.py
```

### Module __init__.py Pattern:

```python
# src/detection/__init__.py
from .model import LicensePlateDetector
from .utils import visualize_detections, apply_nms

__all__ = ['LicensePlateDetector', 'visualize_detections', 'apply_nms']
```

---

## Best Practices

### Code Organization:

1. **One class per file** for main components
2. **Utils files** for helper functions
3. **Config files** separate from logic
4. **Type hints** for all function signatures
5. **Docstrings** (Google style) for all public APIs

### File Placement:

- **Models**: Save to `models/<module>/`
- **Datasets**: Organize in `datasets/<dataset_name>/`
- **Outputs**: Write to `outputs/<category>/`
- **Logs**: Write to `outputs/<module>_<date>.log`
- **Tests**: Mirror src structure in tests/unit/

### Configuration:

- **No hardcoded paths** - use config files
- **Environment-specific settings** in config
- **Validation** for all loaded configs
- **Defaults** provided for optional parameters

---

## Development Workflow

### Adding a New Module:

1. Create directory in `src/<module_name>/`
2. Add `__init__.py` with module docstring
3. Implement main functionality in `<feature>.py`
4. Add utilities in `utils.py`
5. Create unit tests in `tests/unit/test_<module_name>.py`
6. Update `configs/pipeline_config.yaml` if needed
7. Document in this file

### Adding a New Script:

1. Create script in the appropriate component folder
    (e.g. `scripts/data_ingestion/<script_name>.py`)
2. Add argparse for CLI parameters
3. Import from src modules (not copy code)
4. Add logging and error handling
5. Document usage in README.md
6. Create integration test if applicable

### Adding New Data:

1. Organize in appropriate datasets/ subdirectory
2. Follow naming conventions
3. Update data.yaml if YOLO format
4. Add to .gitignore if large files
5. Document in README.md

---

## Version Control Strategy

### Tracked Files:
- Source code (`src/`)
- Configuration templates (`configs/`)
- Scripts (`scripts/` component subpackages + wrappers)
- Tests (`tests/`)
- Documentation (`docs/`)
- Requirements (`requirements.txt`)
- Project files (`.gitignore`, `README.md`, etc.)

### Ignored Files:
- Model weights (`models/`)
- Datasets (`datasets/`)
- Outputs (`outputs/`)
- Virtual environments (`.venv/`)
- Cache files (`__pycache__/`, `*.pyc`)
- IDE settings (`.vscode/`, `.idea/`)

---

## References

- **Coding Standards**: Follow `shrimp-rules.md`
- **YOLO Format**: [Ultralytics Documentation](https://docs.ultralytics.com/)
- **PaddleOCR**: [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- **ByteTrack**: [ByteTrack Paper](https://arxiv.org/abs/2110.06864)

---

**Document Version**: 1.2  
**Last Updated**: 2025-10-10 (Wrapper removal + docs refresh)  
**Status**: Up-to-date with wrapper-free scripts layout

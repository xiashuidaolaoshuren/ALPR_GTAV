# GTA V ALPR Project Development Standards

> **For AI Agent Use Only** - This document provides project-specific rules and decision-making guidelines for automated coding agents.

## Project Overview

- **Technology Stack:** Python 3.9+, YOLOv8 (Ultralytics), PaddleOCR, OpenCV, Albumentations
- **Architecture:** Two-stage pipeline (Detection → Recognition) with tracking
- **Target:** License plate detection and recognition from GTA V gameplay footage

---

## Directory Structure Standards

### MUST Follow This Structure

```
ALPR_GTA5/
├── src/
│   ├── detection/          # YOLOv8 license plate detection module
│   ├── recognition/        # PaddleOCR text recognition module
│   ├── tracking/           # ByteTrack or IOU-based tracking
│   ├── preprocessing/      # Image preprocessing utilities
│   ├── pipeline/           # Main integration pipeline
│   └── utils/              # Common utilities (video I/O, visualization)
├── datasets/
│   ├── lpr/
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   ├── valid/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── test/
│   │       ├── images/
│   │       └── labels/
│   └── ocr/
│       ├── images/         # Cropped license plate images
│       └── labels.txt      # Filename-to-text mapping
├── models/
│   ├── detection/          # YOLOv8 weights
│   └── recognition/        # PaddleOCR configs
├── configs/                # Configuration files
├── scripts/                # Utility scripts (data collection, annotation)
├── tests/                  # Unit and integration tests
└── outputs/                # Results, logs, visualizations
```

### Rules

- **MUST** place detection logic in `src/detection/`
- **MUST** place OCR logic in `src/recognition/`
- **MUST** place tracking logic in `src/tracking/`
- **MUST** place the integrated pipeline in `src/pipeline/`
- **DO NOT** mix detection and recognition logic in the same module
- **DO NOT** place model weights in the Git repository (use `.gitignore`)

---

## Pipeline Architecture Rules

### Two-Stage Pipeline Flow

**MUST follow this exact sequence:**

1. **Detection Stage:**
   - Input: Video frame (BGR format from OpenCV)
   - Process: YOLOv8 inference → Bounding box extraction
   - Output: List of bounding boxes `[(x1, y1, x2, y2, confidence), ...]`

2. **Recognition Stage:**
   - Input: Cropped license plate image from bounding box
   - Process: Preprocessing → PaddleOCR inference
   - Output: Recognized text string

3. **Tracking Stage:**
   - Input: Current frame detections + historical tracks
   - Process: Associate detections with existing tracks (ByteTrack/IOU)
   - Output: Track ID assignment, OCR trigger decision

### Integration Rules

- **MUST** crop the license plate region using detection bounding boxes before passing to OCR
- **MUST** use tracking to avoid redundant OCR on the same plate across frames
- **MUST** maintain separate functions for detection, cropping, preprocessing, and recognition
- **DO NOT** run OCR on every frame; use tracking to determine when OCR is needed
- **DO NOT** pass full frames directly to PaddleOCR

### Example (Correct Flow)

```python
# Correct: Modular pipeline
frame = read_frame(video)
detections = detect_plates(frame)  # YOLOv8
tracks = update_tracks(detections)  # Tracking

for track in tracks:
    if should_run_ocr(track):  # Decision logic
        cropped = crop_plate(frame, track.bbox)
        preprocessed = preprocess_plate(cropped)
        text = recognize_text(preprocessed)  # PaddleOCR
        track.text = text
```

### Example (Incorrect - Prohibited)

```python
# WRONG: Running OCR on every detection without tracking
for bbox in detections:
    cropped = crop_plate(frame, bbox)
    text = recognize_text(cropped)  # Redundant OCR
```

---

## Dataset Organization Rules

### Detection Dataset (YOLOv8 Format)

**MUST adhere to this structure:**

- **Images:** `datasets/lpr/{train|valid|test}/images/*.jpg`
- **Labels:** `datasets/lpr/{train|valid|test}/labels/*.txt`
- **Label Format:** One file per image, each line: `<class_id> <x_center> <y_center> <width> <height>`
  - Coordinates normalized to [0, 1]
  - `class_id` for license plate is `0`

**MUST create `datasets/lpr/data.yaml`:**

```yaml
train: ../datasets/lpr/train/images
val: ../datasets/lpr/valid/images

nc: 1

names: ['license_plate']
```

**Rules:**

- **ALWAYS** normalize bounding box coordinates to [0, 1] range
- **DO NOT** use absolute pixel coordinates in label files
- **MUST** ensure one-to-one correspondence between image and label files
- **MUST** update `data.yaml` when adding new classes (though only `license_plate` is expected)

### OCR Dataset (PaddleOCR Format)

**MUST adhere to this structure:**

- **Images:** `datasets/ocr/images/*.png` (cropped plates)
- **Labels:** `datasets/ocr/labels.txt`

**Label Format (labels.txt):**

```
image_001.png	SA8821A
image_002.png	46EEK827
image_003.png	LS4289V
```

- **MUST** use tab (`\t`) as delimiter between filename and text
- **DO NOT** include file paths, only filenames
- **DO NOT** add headers or extra formatting

---

## Model Integration Standards

### YOLOv8 Detection Module

**Pre-trained Model Source:**

- **MUST** use `yasirfaizahmed/license-plate-object-detection` from Hugging Face as the base model
- **MUST** load model using: `from ultralytics import YOLO; model = YOLO('path/to/weights.pt')`

**Inference Configuration:**

- **MUST** set appropriate confidence threshold (start with `conf=0.25`)
- **MUST** set IOU threshold for NMS (start with `iou=0.45`)
- **MUST** use `model.track()` for tracking integration (ByteTrack is built-in)

**Rules:**

- **DO NOT** modify YOLOv8 architecture without documented justification
- **DO NOT** train from scratch; always start with pre-trained weights
- **MUST** use `model.predict()` for single-frame inference or `model.track()` for video

### PaddleOCR Recognition Module

**Initialization:**

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_angle_cls=True,  # Enable text angle detection
    lang='en',           # English text
    use_gpu=True         # Use GPU if available
)
```

**Inference:**

- **MUST** pass preprocessed cropped images to `ocr.ocr(image, cls=True)`
- **MUST** extract text from nested result structure: `result[0][0][1][0]`

**Rules:**

- **ALWAYS** enable angle classification (`use_angle_cls=True`) for rotated plates
- **DO NOT** pass full-resolution frames to PaddleOCR
- **MUST** handle empty results (no text detected) gracefully

---

## Tracking Implementation Rules

### When to Trigger OCR

**MUST implement logic based on these conditions:**

| Condition | Action | Rationale |
|-----------|--------|-----------|
| New track detected | Run OCR | First occurrence of plate |
| Track age > N frames (e.g., 30) | Run OCR | Refresh stale recognition |
| Detection confidence > threshold | Run OCR | High-quality detection |
| Previous OCR confidence < threshold | Run OCR | Low-confidence result, retry |
| Track lost for M frames | Do NOT run OCR | Plate no longer visible |

**Example Decision Function:**

```python
def should_run_ocr(track, frame_count):
    if track.is_new:
        return True
    if track.frames_since_last_ocr > 30:
        return True
    if track.last_ocr_confidence < 0.7:
        return True
    return False
```

### Tracking Algorithm Choice

- **PREFER** ByteTrack (integrated with YOLOv8: `model.track(source, tracker='bytetrack.yaml')`)
- **ALTERNATIVE** Simple IOU-based tracker if ByteTrack fails
- **DO NOT** implement tracking from scratch unless necessary

---

## Image Preprocessing Standards

### For OCR Module

**MUST apply these preprocessing steps IN ORDER:**

1. **Grayscale Conversion** (if color information not needed):
   ```python
   gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
   ```

2. **Resizing** (maintain aspect ratio):
   ```python
   height, width = gray.shape
   if width < 200:  # Minimum width for OCR
       scale = 200 / width
       gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
   ```

3. **Contrast Enhancement** (optional, test performance):
   ```python
   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
   enhanced = clahe.apply(gray)
   ```

### When to Apply Preprocessing

- **ALWAYS** resize if width < 200px
- **TEST** grayscale vs. color (PaddleOCR supports both)
- **TEST** CLAHE enhancement on low-light frames
- **DO NOT** apply aggressive augmentation during inference

---

## Code Organization Standards

### Module Structure

**Each module MUST have:**

- `__init__.py` for package initialization
- `model.py` for core model logic (loading, inference)
- `utils.py` for helper functions
- `config.py` for module-specific configuration (if needed)

**Example (`src/detection/`):**

```
detection/
├── __init__.py
├── model.py          # YOLOv8 loading, inference
├── utils.py          # Bounding box utilities, visualization
└── config.py         # Detection thresholds, model paths
```

### Function Naming Conventions

- **Detection functions:** `detect_plates()`, `load_detection_model()`, `draw_bounding_boxes()`
- **Recognition functions:** `recognize_text()`, `preprocess_plate()`, `load_ocr_model()`
- **Tracking functions:** `update_tracks()`, `associate_detections()`, `should_run_ocr()`
- **Pipeline functions:** `process_frame()`, `process_video()`, `initialize_pipeline()`

### Import Standards

- **MUST** use absolute imports: `from src.detection.model import detect_plates`
- **DO NOT** use relative imports across modules
- **MUST** organize imports: stdlib → third-party → local
- **MUST** let critical imports fail fast—avoid wrapping dependency imports in try/except; allow `ImportError` to propagate so setup issues surface immediately.

---

## Configuration Management

### Configuration File Structure

**MUST create `configs/pipeline_config.yaml`:**

```yaml
detection:
  model_path: models/detection/yolov8n.pt
  confidence_threshold: 0.25
  iou_threshold: 0.45
  
recognition:
  use_gpu: true
  use_angle_cls: true
  lang: en
  
tracking:
  tracker_type: bytetrack
  max_age: 30
  min_hits: 3
  iou_threshold: 0.3
  ocr_interval: 30  # Frames between OCR runs
  
preprocessing:
  min_width: 200
  use_clahe: false
  clahe_clip_limit: 2.0
```

**Rules:**

- **MUST** load configuration at pipeline initialization
- **DO NOT** hardcode parameters in source files
- **MUST** validate configuration values on load

---

## Multi-File Coordination Rules

### When Adding New Features

| Action | Files to Update |
|--------|----------------|
| Add new detection model | `src/detection/model.py`, `configs/pipeline_config.yaml`, `models/detection/` (add weights), README.md (document model) |
| Add new preprocessing step | `src/preprocessing/utils.py`, `configs/pipeline_config.yaml`, tests for new function |
| Modify dataset structure | `datasets/lpr/data.yaml`, README.md (dataset section), annotation scripts |
| Change pipeline flow | `src/pipeline/`, `configs/pipeline_config.yaml`, integration tests, README.md (usage section) |
| Add dependencies | `requirements.txt`, README.md (installation section), `.github/copilot-instructions.md` (if affects development) |

### Documentation Synchronization

- **WHEN** modifying pipeline architecture → **MUST** update `GTA_V_ALPR_Project_Plan.md` Section 4 (Methodology)
- **WHEN** changing dataset format → **MUST** update `GTA_V_ALPR_Project_Plan.md` Section 8 (Datasets)
- **WHEN** adding new dependencies → **MUST** update `GTA_V_ALPR_Project_Plan.md` Section 3.2 (Core Libraries)

---

## Testing Standards

### Test Organization

- **Unit tests:** `tests/unit/test_{module}.py`
- **Integration tests:** `tests/integration/test_pipeline.py`
- **Test data:** `tests/data/` (small sample images/videos)

### Required Tests

- **MUST** test detection module with sample images (day/night)
- **MUST** test OCR module with pre-cropped plates
- **MUST** test tracking logic with sequential frames
- **MUST** test full pipeline end-to-end

### Test Conditions (GTA V Specific)

**MUST include test cases for:**

- Day lighting
- Night lighting (artificial street lamps)
- Rainy weather
- Different vehicle angles (front/rear, side)
- Partially occluded plates

---

## Prohibited Actions

### Architecture Violations

- ❌ **DO NOT** merge detection and recognition into a single module
- ❌ **DO NOT** pass full frames to OCR without detection-based cropping
- ❌ **DO NOT** skip tracking and run OCR on every frame
- ❌ **DO NOT** implement custom YOLO architecture (use Ultralytics)

### Dataset Violations

- ❌ **DO NOT** use absolute pixel coordinates in YOLO labels
- ❌ **DO NOT** mix detection and OCR datasets in the same directory
- ❌ **DO NOT** commit large dataset files to Git (use `.gitignore`)

### Code Quality Violations

- ❌ **DO NOT** hardcode file paths (use config files or arguments)
- ❌ **DO NOT** use global variables for model instances
- ❌ **DO NOT** ignore exceptions without logging

### Model Usage Violations

- ❌ **DO NOT** train YOLOv8 from scratch without transfer learning
- ❌ **DO NOT** modify PaddleOCR model architecture
- ❌ **DO NOT** use non-standard model file formats

---

## AI Decision-Making Guidelines

### When Uncertain About Implementation

**Decision Tree:**

1. **Does it affect the two-stage pipeline flow?**
   - YES → Review "Pipeline Architecture Rules" section
   - NO → Continue

2. **Does it involve dataset handling?**
   - YES → Review "Dataset Organization Rules" section
   - NO → Continue

3. **Does it involve model inference?**
   - YES → Review "Model Integration Standards" section
   - NO → Continue

4. **Does it require preprocessing?**
   - YES → Review "Image Preprocessing Standards" section
   - NO → Continue

5. **Still uncertain?**
   - Consult `GTA_V_ALPR_Project_Plan.md` for high-level guidance
   - Check existing code in `src/` for patterns
   - Use context7 MCP to verify current API usage

### Priority Guidelines

**When multiple tasks are pending:**

1. **Highest Priority:** Core pipeline functionality (detection → recognition → tracking)
2. **High Priority:** Dataset preparation and model integration
3. **Medium Priority:** Preprocessing optimization and performance tuning
4. **Low Priority:** Visualization and logging enhancements

### Conflict Resolution

**When project plan conflicts with this document:**

- **This document (shrimp-rules.md) takes precedence** for implementation details
- **Project plan takes precedence** for timeline and milestone definitions

---

## Environment Setup Rules

### Python Environment

- **MUST** use Python 3.9 or higher
- **PREFER** virtual environment (`venv` or `conda`)
- **MUST** install dependencies: `pip install -r requirements.txt`

### GPU Configuration

- **MUST** check GPU availability before initializing models
- **MUST** handle CPU fallback gracefully
- **DO NOT** assume GPU is always available

### Required Packages

```
ultralytics>=8.0.0
opencv-python>=4.8.0
paddlepaddle-gpu>=2.5.0  # or paddlepaddle for CPU
paddleocr>=2.7.0
albumentations>=1.3.0
pyyaml>=6.0
```

---

## Version Control Rules

### Git Workflow

- **MUST** commit functional changes incrementally
- **MUST** write descriptive commit messages
- **DO NOT** commit model weights (add to `.gitignore`)
- **DO NOT** commit dataset files (add to `.gitignore`)

### `.gitignore` Requirements

**MUST include:**

```
# Models
models/**/*.pt
models/**/*.pdparams

# Datasets
datasets/

# Environment
.venv/
venv/
__pycache__/

# Outputs
outputs/
*.log
```

---

## Performance Optimization Guidelines

### When to Optimize

- **DO NOT** optimize prematurely
- **MUST** profile code before optimization
- **MUST** maintain code readability during optimization

### Optimization Priorities

1. **Reduce redundant OCR calls** (use tracking effectively)
2. **Optimize preprocessing** (vectorize operations, use GPU)
3. **Batch inference** (process multiple detections together if possible)
4. **Model selection** (use smaller YOLO models like YOLOv8n for speed)

### Benchmarking

- **MUST** measure FPS (frames per second) for video processing
- **MUST** measure detection mAP on validation set
- **MUST** measure OCR character error rate (CER)

---

## Logging and Debugging Standards

### Logging Configuration

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/pipeline.log'),
        logging.StreamHandler()
    ]
)
```

### What to Log

- **INFO:** Pipeline initialization, model loading, video processing start/end
- **DEBUG:** Frame-by-frame processing details, tracking updates
- **WARNING:** Low-confidence detections, failed OCR attempts
- **ERROR:** Model loading failures, file I/O errors, exceptions

### What NOT to Log

- ❌ **DO NOT** log every frame's raw pixel data
- ❌ **DO NOT** log sensitive information (if any)
- ❌ **DO NOT** spam logs with redundant messages

---

## End of Document

**This document is a living standard. Update it when:**

- Adding new modules or significantly changing architecture
- Modifying dataset formats or structures
- Introducing new dependencies or tools
- Discovering new project-specific patterns or anti-patterns

**Last Updated:** 2025-10-08

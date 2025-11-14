# Configuration Guide

This guide explains all configuration parameters for the GTA V ALPR pipeline and provides recommendations for different use cases.

## Table of Contents
- [Overview](#overview)
- [Configuration Files](#configuration-files)
- [Detection Module](#detection-module)
- [Recognition Module](#recognition-module)
- [Tracking Module](#tracking-module)
- [Preprocessing Module](#preprocessing-module)
- [Pipeline Settings](#pipeline-settings)
- [Tuning for Different Scenarios](#tuning-for-different-scenarios)
- [Validation](#validation)

---

## Overview

The ALPR pipeline uses YAML configuration files to control all aspects of detection, recognition, tracking, and preprocessing. This modular approach allows you to easily tune the pipeline for different scenarios (speed vs. accuracy, different lighting conditions, etc.) without changing code.

### Configuration Flow
```
Video Input → Detection (YOLOv8) → Tracking → Preprocessing → Recognition (PaddleOCR) → Output
              ↑ detection config   ↑ tracking   ↑ preprocessing  ↑ recognition config
```

---

## Configuration Files

The project includes several configuration files:

- **Main config**: `configs/pipeline_config.yaml` - Default production configuration
- **Example configs**: `configs/examples/`
  - `high_performance.yaml` - Speed-optimized (60+ FPS)
  - `high_accuracy.yaml` - Quality-optimized (lower FPS, best accuracy)
  - `balanced.yaml` - Production-ready balance

### Loading a Configuration

**Python API:**
```python
from src.pipeline.alpr_pipeline import ALPRPipeline

# Use default config
pipeline = ALPRPipeline()

# Load custom config
pipeline = ALPRPipeline(config_path="configs/examples/high_accuracy.yaml")
```

**Command-line:**
```bash
# Use default config
python scripts/process_video.py --input video.mp4

# Load custom config
python scripts/process_video.py --input video.mp4 --config configs/examples/high_performance.yaml
```

---

## Detection Module

The detection module uses YOLOv8 to locate license plates in video frames.

### Parameters

#### `model_path`
- **Type**: String (file path)
- **Description**: Path to YOLOv8 weights file
- **Options**:
  - `models/detection/yolov8n.pt` - Pretrained baseline (fast, lower accuracy)
  - `models/detection/yolov8_finetuned_best.pt` - Fine-tuned v1
  - `models/detection/yolov8_finetuned_v2_best.pt` - Fine-tuned v2 (best accuracy)
- **Recommended**: `yolov8_finetuned_v2_best.pt`

#### `confidence_threshold`
- **Type**: Float (0.0 - 1.0)
- **Description**: Minimum confidence score for a detection to be considered valid
- **Trade-off**:
  - **Lower** (0.15-0.25): More detections, higher false positive rate
  - **Higher** (0.30-0.40): Fewer detections, lower false positive rate
- **Recommended**:
  - Speed: 0.20
  - Balanced: 0.25
  - Accuracy: 0.35

#### `iou_threshold`
- **Type**: Float (0.0 - 1.0)
- **Description**: IOU threshold for Non-Maximum Suppression (removes overlapping boxes)
- **Trade-off**:
  - **Lower** (0.35-0.40): Fewer overlapping detections, may merge close plates
  - **Higher** (0.45-0.55): More overlapping detections, may duplicate plates
- **Recommended**:
  - Speed: 0.50
  - Balanced: 0.45
  - Accuracy: 0.40

#### `img_size`
- **Type**: Integer (pixels)
- **Description**: Image size for inference (width/height)
- **Trade-off**:
  - **Smaller** (640): Faster inference, may miss small plates
  - **Larger** (1056, 1280): Better accuracy for small plates, slower inference
- **Recommended**:
  - Speed: 640
  - Balanced/Accuracy: 1056

#### `device`
- **Type**: String
- **Options**: `cuda` (GPU) or `cpu`
- **Recommended**: `cuda` (if available)

#### `max_det`
- **Type**: Integer
- **Description**: Maximum number of detections per frame
- **Recommended**: 100 (default)

### Example Configurations

**High Performance (Speed):**
```yaml
detection:
  model_path: models/detection/yolov8_finetuned_v2_best.pt
  confidence_threshold: 0.20
  iou_threshold: 0.50
  img_size: 640
  device: cuda
  max_det: 100
```

**High Accuracy:**
```yaml
detection:
  model_path: models/detection/yolov8_finetuned_v2_best.pt
  confidence_threshold: 0.35
  iou_threshold: 0.40
  img_size: 1056
  device: cuda
  max_det: 100
```

---

## Recognition Module

The recognition module uses PaddleOCR to read text from detected license plates.

### Parameters

#### `use_gpu`
- **Type**: Boolean
- **Description**: Enable GPU acceleration for OCR
- **Recommended**: `true` (if CUDA available)

#### `use_textline_orientation`
- **Type**: Boolean
- **Description**: Detect and correct text orientation
- **Trade-off**:
  - **true**: More robust to rotation, slightly slower
  - **false**: Faster, may fail on rotated text
- **Recommended**:
  - Speed: `false`
  - Balanced/Accuracy: `true`

#### `lang`
- **Type**: String
- **Options**: `en` (English), `ch` (Chinese), etc.
- **Recommended**: `en` (GTA V uses English alphabet)

#### `rec_threshold`
- **Type**: Float (0.0 - 1.0)
- **Description**: Minimum confidence for OCR result
- **Recommended**:
  - Speed: 0.4
  - Balanced: 0.5
  - Accuracy: 0.6

#### `show_log`
- **Type**: Boolean
- **Description**: Print PaddleOCR internal logs
- **Recommended**: `false` (reduces clutter)

#### `use_rec`
- **Type**: Boolean
- **Description**: Enable text recognition (set to `false` for detection-only mode)
- **Recommended**: `true`

### Post-processing Parameters

#### `regex`
- **Type**: String (regex pattern)
- **Description**: Validate recognized text against pattern
- **GTA V Format**: `^\d{2}[A-Z]{3}\d{3}$` (e.g., 12ABC345)
- **Tip**: Set to empty string `''` to disable filtering

#### `prefer_largest_box`
- **Type**: Boolean
- **Description**: When multiple text lines detected, use text from largest box
- **Recommended**: `true` (robust to headers/banners)

#### `mask_top_ratio`
- **Type**: Float (0.0 - 1.0)
- **Description**: Mask top portion of plate image (ignores headers)
- **Example**: 0.25 masks top 25% of image
- **Recommended**: 0.0 (GTA V plates have no headers)

#### `min_conf`
- **Type**: Float (0.0 - 1.0)
- **Description**: Minimum confidence after filtering
- **Recommended**: 0.3

### Example Configurations

**High Performance (Speed):**
```yaml
recognition:
  use_gpu: true
  use_textline_orientation: false
  lang: en
  rec_threshold: 0.4
  show_log: false
  use_rec: true
  regex: '^\d{2}[A-Z]{3}\d{3}$'
  prefer_largest_box: true
  mask_top_ratio: 0.0
  min_conf: 0.3
```

**High Accuracy:**
```yaml
recognition:
  use_gpu: true
  use_textline_orientation: true
  lang: en
  rec_threshold: 0.6
  show_log: false
  use_rec: true
  regex: '^\d{2}[A-Z]{3}\d{3}$'
  prefer_largest_box: true
  mask_top_ratio: 0.0
  min_conf: 0.4
```

---

## Tracking Module

The tracking module maintains identity of detected plates across frames and manages OCR calls.

### Parameters

#### `tracker_type`
- **Type**: String
- **Options**:
  - `bytetrack` - Fast and robust (recommended)
  - `botsort` - Advanced with ReID (slower)
  - `iou` - Simple IOU-based (fastest, less robust)
- **Recommended**: `bytetrack`

#### `max_age`
- **Type**: Integer (frames)
- **Description**: Maximum frames to keep lost tracks alive
- **Trade-off**:
  - **Lower** (20): Faster track cleanup, may lose occluded plates
  - **Higher** (40): More robust to occlusions, slower cleanup
- **Recommended**:
  - Speed: 20
  - Balanced: 30
  - Accuracy: 40

#### `min_hits`
- **Type**: Integer (frames)
- **Description**: Minimum detections before confirming track
- **Trade-off**:
  - **Lower** (2): More tracks, higher false positive rate
  - **Higher** (4): Fewer false tracks, may miss brief plates
- **Recommended**:
  - Speed: 2
  - Balanced: 3
  - Accuracy: 4

#### `iou_threshold`
- **Type**: Float (0.0 - 1.0)
- **Description**: Minimum IOU for matching detection to track
- **Trade-off**:
  - **Lower** (0.25): Looser matching, may merge different plates
  - **Higher** (0.35): Stricter matching, may create duplicates
- **Recommended**: 0.3 (balanced), 0.25 (accuracy)

#### `ocr_interval`
- **Type**: Integer (frames)
- **Description**: Frames between OCR calls for same plate
- **Trade-off**:
  - **Lower** (20): More frequent OCR, higher accuracy, slower
  - **Higher** (45): Less OCR, faster, may miss text changes
- **Recommended**:
  - Speed: 45
  - Balanced: 30
  - Accuracy: 20

#### `ocr_confidence_threshold`
- **Type**: Float (0.0 - 1.0)
- **Description**: Minimum confidence to save OCR result
- **Recommended**:
  - Speed: 0.6
  - Balanced: 0.7
  - Accuracy: 0.75

### Example Configurations

**High Performance (Speed):**
```yaml
tracking:
  tracker_type: bytetrack
  max_age: 20
  min_hits: 2
  iou_threshold: 0.3
  ocr_interval: 45
  ocr_confidence_threshold: 0.6
```

**High Accuracy:**
```yaml
tracking:
  tracker_type: bytetrack
  max_age: 40
  min_hits: 4
  iou_threshold: 0.25
  ocr_interval: 20
  ocr_confidence_threshold: 0.75
```

---

## Preprocessing Module

The preprocessing module applies image enhancement to plate crops before OCR.

### Parameters

#### `enable_enhancement`
- **Type**: Boolean
- **Description**: Enable full enhancement pipeline
- **Recommended**:
  - Speed: `false`
  - Balanced/Accuracy: `true`

#### `min_width`
- **Type**: Integer (pixels)
- **Description**: Minimum width for plate crops (upscales smaller crops)
- **Recommended**:
  - Speed: 150
  - Balanced: 200
  - Accuracy: 250

### CLAHE (Contrast Enhancement)

#### `use_clahe`
- **Type**: Boolean
- **Description**: Enable CLAHE for contrast enhancement
- **Recommended**: `true` (improves text visibility)

#### `clahe_clip_limit`
- **Type**: Float (1.0 - 4.0)
- **Description**: Maximum contrast enhancement
- **Trade-off**:
  - **Lower** (1.5): Conservative, preserves details
  - **Higher** (3.0): Aggressive, may amplify noise
- **Recommended**: 2.0

#### `clahe_tile_grid_size`
- **Type**: List [height, width]
- **Description**: Grid size for localized enhancement
- **Options**: `[4, 4]` (more localized) or `[8, 8]` (more uniform)
- **Recommended**: `[8, 8]`

### Gaussian Blur

#### `use_gaussian_blur`
- **Type**: Boolean
- **Description**: Enable Gaussian blur for noise reduction
- **Trade-off**: Reduces noise but may blur text
- **Recommended**: `false` (preserves text sharpness)

#### `gaussian_kernel_size`
- **Type**: List [height, width] (odd integers)
- **Options**: `[3, 3]` (light) or `[5, 5]` (moderate)
- **Recommended**: `[3, 3]`

### Sharpening

#### `use_sharpening`
- **Type**: Boolean
- **Description**: Enable sharpening filter for text edges
- **Recommended**: `true` (improves text clarity)

#### `sharpen_strength`
- **Type**: Float (0.5 - 2.0)
- **Description**: Sharpening intensity
- **Trade-off**:
  - **Lower** (0.7): Conservative, safer
  - **Higher** (1.5): Aggressive, may create artifacts
- **Recommended**: 1.0

### Example Configurations

**High Performance (Speed):**
```yaml
preprocessing:
  enable_enhancement: false
  min_width: 150
  use_clahe: false
  use_gaussian_blur: false
  use_sharpening: false
```

**High Accuracy:**
```yaml
preprocessing:
  enable_enhancement: true
  min_width: 250
  use_clahe: true
  clahe_clip_limit: 2.0
  clahe_tile_grid_size: [8, 8]
  use_gaussian_blur: false
  use_sharpening: true
  sharpen_strength: 1.0
```

---

## Pipeline Settings

Global pipeline configuration for video processing and output management.

### Parameters

#### `enable_full_pipeline`
- **Type**: Boolean
- **Description**: Run full ALPR pipeline (detection + recognition + tracking)
- **Tip**: Set to `false` for detection-only mode
- **Recommended**: `true`

#### `save_intermediate`
- **Type**: Boolean
- **Description**: Save cropped plates and annotated frames
- **Trade-off**:
  - **true**: Useful for debugging, uses more disk space
  - **false**: Faster, uses less disk space
- **Recommended**: `true` (development), `false` (production)

#### `output_dir`
- **Type**: String (directory path)
- **Description**: Output directory for all results
- **Recommended**: `outputs`

#### `batch_size`
- **Type**: Integer (frames)
- **Description**: Number of frames processed at once
- **Trade-off**:
  - **Lower** (16): Less memory, may reduce throughput
  - **Higher** (64): Better GPU utilization, more memory
- **Recommended**:
  - Limited memory: 16
  - Balanced: 32
  - High-end GPU: 64

### Logging

#### `enable_logging`
- **Type**: Boolean
- **Recommended**: `true`

#### `log_level`
- **Type**: String
- **Options**: `DEBUG`, `INFO`, `WARNING`, `ERROR`
- **Recommended**: `INFO` (balanced), `DEBUG` (troubleshooting)

#### `log_file`
- **Type**: String (file path)
- **Recommended**: `outputs/alpr_pipeline.log`

---

## Tuning for Different Scenarios

### Real-Time Performance (60+ FPS)

**Priority**: Speed  
**Use case**: Live video processing, real-time monitoring

```yaml
# Load: configs/examples/high_performance.yaml
detection:
  img_size: 640
  confidence_threshold: 0.20
  iou_threshold: 0.50

recognition:
  use_textline_orientation: false
  rec_threshold: 0.4

tracking:
  ocr_interval: 45
  ocr_confidence_threshold: 0.6

preprocessing:
  enable_enhancement: false

pipeline:
  batch_size: 64
```

**Expected**: 60+ FPS on RTX 3070 Ti

---

### High Accuracy (Best Quality)

**Priority**: Accuracy  
**Use case**: Forensic analysis, critical applications

```yaml
# Load: configs/examples/high_accuracy.yaml
detection:
  img_size: 1056
  confidence_threshold: 0.35
  iou_threshold: 0.40

recognition:
  use_textline_orientation: true
  rec_threshold: 0.6

tracking:
  max_age: 40
  min_hits: 4
  ocr_interval: 20
  ocr_confidence_threshold: 0.75

preprocessing:
  enable_enhancement: true
  use_clahe: true
  use_sharpening: true

pipeline:
  batch_size: 32
```

**Expected**: 15-25 FPS on RTX 3070 Ti, best accuracy

---

### Balanced (Production)

**Priority**: Balance  
**Use case**: General-purpose, production deployment

```yaml
# Load: configs/examples/balanced.yaml
detection:
  img_size: 1056
  confidence_threshold: 0.30
  iou_threshold: 0.45

recognition:
  use_textline_orientation: true
  rec_threshold: 0.5

tracking:
  max_age: 30
  min_hits: 3
  ocr_interval: 30
  ocr_confidence_threshold: 0.7

preprocessing:
  enable_enhancement: true
  use_clahe: true
  use_gaussian_blur: true
  use_sharpening: false

pipeline:
  batch_size: 48
```

**Expected**: 30-45 FPS on RTX 3070 Ti

---

### Night / Low Light

**Priority**: Visibility in dark conditions

```yaml
preprocessing:
  enable_enhancement: true
  use_clahe: true
  clahe_clip_limit: 3.0  # More aggressive contrast
  use_sharpening: true
  sharpen_strength: 1.5

recognition:
  rec_threshold: 0.4  # More lenient for noisy conditions
```

---

### Rain / Motion Blur

**Priority**: Robustness to blur

```yaml
preprocessing:
  enable_enhancement: true
  use_gaussian_blur: false  # Avoid additional blur
  use_sharpening: true
  sharpen_strength: 1.5

tracking:
  max_age: 40  # Longer tracking for occlusions
  ocr_interval: 20  # More frequent OCR
```

---

## Validation

Use the validation script to check configuration files for errors:

```bash
python scripts/utils/validate_config.py configs/pipeline_config.yaml
```

### Validation Checks

1. **Required Fields**: Ensures all required parameters are present
2. **Type Validation**: Checks parameter types (int, float, bool, string)
3. **Range Validation**: Verifies numeric values are within valid ranges
4. **Enum Validation**: Checks string parameters against allowed values
5. **File Paths**: Verifies model files exist
6. **Special Fields**: Validates CLAHE grid size and Gaussian kernel size

### Example Output

```
✓ Configuration is valid!

Summary:
- Detection: YOLOv8 fine-tuned v2, 1056px, conf=0.25
- Recognition: GPU enabled, English, conf=0.5
- Tracking: ByteTrack, ocr_interval=30
- Preprocessing: Enhancement enabled (CLAHE + sharpening)
```

### Integration into Pipeline

The validation is automatically run when loading a config:

```python
from src.pipeline.alpr_pipeline import ALPRPipeline

# Validation runs automatically, raises ValueError if invalid
pipeline = ALPRPipeline(config_path="configs/my_config.yaml")
```

---

## Troubleshooting

### Low FPS
- Reduce `img_size` to 640
- Increase `ocr_interval` to 45+
- Disable `enable_enhancement`
- Increase `batch_size` (if memory allows)

### Poor Detection Accuracy
- Increase `img_size` to 1056
- Lower `confidence_threshold` to 0.20-0.25
- Use fine-tuned model (`yolov8_finetuned_v2_best.pt`)

### Poor OCR Accuracy
- Enable `use_clahe` and `use_sharpening`
- Lower `rec_threshold` to 0.4
- Decrease `ocr_interval` to 20
- Enable `use_textline_orientation`

### Too Many False Positives
- Increase `confidence_threshold` to 0.30-0.35
- Increase `min_hits` to 4
- Enable `regex` filtering

### Duplicate Tracks
- Lower `iou_threshold` (detection) to 0.40
- Lower `iou_threshold` (tracking) to 0.25

---

## Additional Resources

- **Project Plan**: `GTA_V_ALPR_Project_Plan.md` - Technical architecture and methodology
- **API Reference**: `docs/api_reference.md` - Detailed API documentation
- **User Guide**: `docs/user_guide.md` - End-to-end usage examples
- **Developer Guide**: `docs/developer_guide.md` - Contributing and development workflow

---

*Last updated: 2025*

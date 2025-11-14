# GTA V ALPR (Automatic License Plate Recognition)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A sophisticated computer vision system for detecting and recognizing license plates in GTA V gameplay footage using state-of-the-art deep learning models.

## 📋 Project Overview

This project implements a complete two-stage ALPR pipeline specifically designed for processing GTA V game footage:

1. **Detection Stage**: YOLOv8 object detection model identifies license plate locations
2. **Recognition Stage**: PaddleOCR extracts text from detected plates
3. **Tracking Stage**: ByteTrack maintains plate identity across video frames

**Key Features:**
- 🎮 **Streamlit GUI** for interactive video processing
- 🚀 Real-time or batch processing of video/images
- ⚡ GPU-accelerated inference (CUDA 12.1+)
- 🎯 Multi-object tracking to avoid redundant OCR
- ⚙️ Configurable pipeline parameters via YAML or GUI
- 🏗️ Modular architecture for easy customization
- 📊 Performance profiling and optimization tools
- 📈 Comprehensive evaluation and reporting

## 🛠️ Technology Stack

- **Python**: 3.9+ (developed with 3.12.9)
- **Deep Learning Frameworks**:
  - PyTorch 2.5.1+ (CUDA 12.1)
  - PaddlePaddle 2.6.1 (GPU)
- **Detection**: Ultralytics YOLOv8 (fine-tuned on GTA V plates)
- **Recognition**: PaddleOCR 3.2.0
- **Tracking**: ByteTrack (integrated with YOLOv8)
- **GUI**: Streamlit 1.40.2
- **Computer Vision**: OpenCV 4.10.0
- **Data Augmentation**: Albumentations 2.0.8
- **Configuration**: PyYAML 6.0.2
- **Testing**: Pytest 8.4.2

---

## 🚀 Quick Start

### 1. Installation

**Prerequisites:**
- Python 3.9 or higher
- NVIDIA GPU with CUDA 12.1+ support (recommended)
- 8GB+ RAM, 4GB+ VRAM

**Step 1: Clone repository**
```powershell
git clone https://github.com/xiashuidaolaoshuren/ALPR_GTAV.git
cd ALPR_GTAV
```

**Step 2: Create virtual environment**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
```

**Step 3: Install dependencies**
```powershell
# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install all dependencies
pip install -r requirements.txt

# Or use UV for faster installation (recommended)
pip install uv
uv pip install -r requirements.txt
```

**Step 4: Verify installation**
```powershell
python scripts/diagnostics/verify_gpu.py
```

Expected output:
```
✓ PyTorch CUDA available (Device: NVIDIA GeForce RTX 3070 Ti)
✓ PaddlePaddle GPU available
✓ All required packages installed
```

### 2. Download Models

Models are downloaded automatically on first use. To manually download:

```powershell
# Detection model (YOLOv8 fine-tuned on GTA V)
python models/detection/download_model.py

# OCR models (PaddleOCR - auto-downloaded)
# No action needed, downloads on first OCR call
```

### 3. Run GUI Application

**Launch Streamlit interface:**
```powershell
streamlit run gui/app.py
```

This opens a web interface at `http://localhost:8501` with:
- Video file upload
- Real-time parameter adjustment
- Live processing visualization
- Results download

**📸 Screenshot:** [_Placeholder: GUI main interface showing video upload and control panel_]

### 4. Command-Line Usage

**Process single video:**
```powershell
python scripts/process_video.py `
  --input "outputs/raw_footage/day_clear/video1.mp4" `
  --output "outputs/processed_video.mp4" `
  --config "configs/pipeline_config.yaml"
```

**Batch processing:**
```powershell
python scripts/data_ingestion/batch_process_footage.py `
  --input-dir "outputs/raw_footage" `
  --output-dir "outputs/processed" `
  --fps 5
```

**Export results to JSON:**
```powershell
python scripts/process_video.py `
  --input "video.mp4" `
  --no-video `
  --export-json "results.json"
```

---

## 📁 Project Structure

```
ALPR_GTA5/
├── src/                    # Source code
│   ├── detection/          # YOLOv8 detection module
│   ├── recognition/        # PaddleOCR recognition module
│   ├── tracking/           # ByteTrack tracking module
│   ├── preprocessing/      # Image preprocessing utilities
│   ├── pipeline/           # Complete ALPR pipeline
│   └── utils/              # Configuration and helpers
├── datasets/               # Training/validation data
│   ├── lpr/               # YOLO format detection dataset
│   │   ├── train/         # Training images/labels
│   │   ├── valid/         # Validation images/labels
│   │   └── test/          # Test images/labels
│   └── ocr/               # OCR training data
├── models/                 # Model weights storage
│   ├── detection/         # YOLOv8 model files
│   └── recognition/       # PaddleOCR model files
├── configs/                # Configuration files
│   └── pipeline_config.yaml  # Main pipeline configuration
├── scripts/                # Componentized CLI tools (no root wrappers)
│   ├── data_ingestion/    # Frame extraction, metadata, cleaning
│   ├── annotation/        # Label Studio helpers
│   ├── inference/         # Image/video inference entrypoints
│   ├── evaluation/        # Evaluation + reporting utilities
│   ├── diagnostics/       # Environment + dataset checks
│   ├── profiling/         # Pipeline performance profiling and optimization
│   └── README.md          # Overview of script layout
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── data/              # Test data fixtures
├── docs/                   # Documentation
│   └── project_structure.md  # Detailed structure docs
└── outputs/                # Processing results
```

## 💻 Usage

### GUI Interface (Recommended for Beginners)

The Streamlit GUI provides an intuitive interface for video processing:

**Launch:**
```powershell
streamlit run gui/app.py
```

**Features:**
- **📁 Video Upload**: Drag & drop or browse for MP4/AVI/MOV files
- **⚙️ Parameter Control**: Real-time sliders for confidence, IOU, OCR interval
- **🎬 Live Preview**: See processing results as they happen
- **📊 Info Panel**: Track statistics, latest recognitions, live logs
- **💾 Export**: Download annotated video and results (JSON/CSV)

**📸 Screenshot:** [_Placeholder: GUI control panel with sliders and buttons_]

**Basic Workflow:**
1. Upload video file (e.g., `video.mp4`)
2. Adjust parameters (confidence: 0.25, IOU: 0.45, OCR interval: 30)
3. Click **Start Processing**
4. Monitor progress in Info Panel
5. Download results when complete

See [User Guide](docs/user_guide.md) for detailed GUI walkthrough.

---

### Command-Line Interface

For batch processing and automation:

#### Process Single Video

**Basic usage:**
```powershell
python scripts/process_video.py `
  --input "path/to/video.mp4" `
  --output "outputs/annotated_video.mp4"
```

**With custom config:**
```powershell
python scripts/process_video.py `
  --input "video.mp4" `
  --output "output.mp4" `
  --config "configs/my_custom_config.yaml"
```

**Skip video output (faster):**
```powershell
python scripts/process_video.py `
  --input "video.mp4" `
  --no-video `
  --export-json "results.json"
```

**Frame sampling:**
```powershell
# Process every 5th frame
python scripts/process_video.py `
  --input "video.mp4" `
  --output "output.mp4" `
  --sample-rate 5
```

#### Batch Processing

**Process multiple videos:**
```powershell
python scripts/data_ingestion/batch_process_footage.py `
  --input-dir "outputs/raw_footage" `
  --output-dir "outputs/processed" `
  --fps 5 `
  --quality 95
```

#### Single Image Detection

**Detect plates in image:**
```powershell
python scripts/inference/detect_image.py `
  --image "path/to/image.jpg" `
  --output "outputs/annotated.jpg" `
  --config "configs/pipeline_config.yaml"
```

---

### Configuration

The pipeline is highly configurable via YAML files:

**Main config:** `configs/pipeline_config.yaml`

```yaml
# Detection settings
detection:
  model_path: models/detection/yolov8_finetuned_v2_best.pt
  confidence_threshold: 0.25  # Lower = more detections
  iou_threshold: 0.45         # NMS threshold
  image_size: 640             # Input size (640/1056)

# Recognition settings
recognition:
  use_gpu: true
  lang: 'en'
  det_db_box_thresh: 0.5
  rec_batch_num: 6

# Tracking settings
tracking:
  max_age: 30                 # Frames to keep lost tracks
  min_hits: 3                 # Detections before confirmed
  ocr_interval: 30            # Frames between OCR runs
  ocr_confidence_threshold: 0.7

# Preprocessing
preprocessing:
  use_clahe: true
  use_sharpening: true
  use_denoising: false

# Device
device: cuda  # or 'cpu'
```

**See:** [Configuration Guide](docs/configuration_guide.md) for complete parameter reference.

---

### Output Formats

**JSON output:**
```json
{
  "video_info": {
    "path": "video.mp4",
    "fps": 30.0,
    "total_frames": 900
  },
  "recognitions": [
    {
      "track_id": 1,
      "text": "12ABC345",
      "confidence": 0.92,
      "first_frame": 10,
      "last_frame": 45,
      "bbox": [100, 200, 150, 250]
    }
  ]
}
```

**CSV output:**
```csv
track_id,text,confidence,first_frame,last_frame,bbox
1,12ABC345,0.92,10,45,"[100, 200, 150, 250]"
2,99XYZ123,0.88,60,95,"[300, 400, 350, 450]"
```

---

## 📊 Dataset Organization

### Detection Dataset (YOLO Format)

Place your annotated data in `datasets/lpr/` following this structure:

```
datasets/lpr/
├── data.yaml              # Dataset configuration
├── train/
│   ├── images/           # Training images (.jpg, .png)
│   └── labels/           # YOLO annotations (.txt)
├── valid/
│   ├── images/           # Validation images
│   └── labels/           # YOLO annotations
└── test/
    ├── images/           # Test images
    └── labels/           # YOLO annotations
```

**YOLO Annotation Format:**
```
<class_id> <x_center> <y_center> <width> <height>
```
All coordinates are normalized (0.0 - 1.0).

### OCR Dataset

Place cropped license plate images in `datasets/ocr/images/` for recognition model fine-tuning (optional).

## 🧪 Development

### Running Tests

```powershell
# Run all tests
pytest

# Run specific test suite
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only

# Run with coverage
pytest --cov=src --cov-report=html tests/

# Run specific test file
pytest tests/unit/test_detection.py -v

# Run with markers
pytest -m "not slow"  # Skip slow tests
```

### Code Standards

This project follows:
- **PEP 8** Python style guidelines (100-char line limit)
- **Type hints** for all function signatures
- **Google-style docstrings** for all public APIs
- **Modular design** with clear separation of concerns

**Example:**
```python
def recognize_text(image: np.ndarray, model: PaddleOCR, 
                   config: dict) -> Tuple[Optional[str], float]:
    """
    Recognize text from plate crop using PaddleOCR.
    
    Args:
        image: Cropped plate image in BGR format
        model: Loaded PaddleOCR model
        config: Recognition configuration dict
    
    Returns:
        Tuple[Optional[str], float]: (recognized_text, confidence) 
            or (None, 0.0) if no valid text
    
    Raises:
        ValueError: If image is invalid
    """
    ...
```

### Project Guidelines

See `.github/copilot-instructions.md` and `shrimp-rules.md` for:
- Coding standards and best practices
- Dependency management (no soft failures on critical imports)
- MCP server usage (context7, sequential-thinking, task-manager)
- Development workflow

### Adding New Features

**1. Create feature branch:**
```powershell
git checkout -b feature/your-feature-name
```

**2. Implement feature:**
- Follow module structure in `src/`
- Add config parameters to `configs/pipeline_config.yaml`
- Write unit tests in `tests/unit/`
- Update integration tests if needed

**3. Test locally:**
```powershell
pytest tests/
flake8 src/ gui/ scripts/
```

**4. Submit PR:**
- Use conventional commits: `feat:`, `fix:`, `docs:`, etc.
- Include description, test results, checklist
- Link related issues

See [Developer Guide](docs/developer_guide.md) for complete contribution guidelines.

---

## 📊 Evaluation & Performance

### Model Performance

**Detection (YOLOv8 Fine-tuned v2):**
- mAP@0.5: 0.95+
- Precision: 0.93
- Recall: 0.91
- Speed: 30-50 FPS (RTX 3070 Ti, 640px input)

**Recognition (PaddleOCR):**
- Character Accuracy: 85-90% (GTA V plates)
- Speed: 20-50ms per plate (GPU)

**Complete Pipeline:**
- Processing Speed: 30-40 FPS (1080p, 1-2 plates per frame)
- End-to-End Accuracy: ~80-85%

### Run Evaluation

**Evaluate detection model:**
```powershell
python scripts/evaluation/evaluate_detection.py `
  --model "models/detection/yolov8_finetuned_v2_best.pt" `
  --data "datasets/lpr/data.yaml" `
  --split "test" `
  --output "outputs/evaluation/detection_report.json"
```

**Evaluate OCR:**
```powershell
python scripts/evaluation/evaluate_ocr.py `
  --images "datasets/ocr/images" `
  --ground-truth "datasets/ocr/ground_truth.txt" `
  --output "outputs/evaluation/ocr_report.md"
```

**Generate reports:**
```powershell
python scripts/evaluation/generate_report.py `
  --detection-results "outputs/evaluation/detection_report.json" `
  --ocr-results "outputs/evaluation/ocr_results.json" `
  --output "outputs/evaluation/full_report.md"
```

### Performance Profiling

**Profile pipeline:**
```powershell
python scripts/profiling/profile_pipeline.py `
  --config "configs/pipeline_config.yaml" `
  --input "test_video.mp4" `
  --output "outputs/profiling/performance_report.md"
```

**Compare configurations:**
```powershell
python scripts/profiling/compare_configurations.py `
  --configs "configs/*.yaml" `
  --input "test_video.mp4" `
  --output "outputs/profiling/optimization_comparison.csv"
```

**Optimization summary:** See [outputs/profiling/OPTIMIZATION_SUMMARY.md](outputs/profiling/OPTIMIZATION_SUMMARY.md)

---

## 📈 Performance Expectations

### GPU Performance (NVIDIA RTX 3070 Ti, 8GB VRAM)

| Task | Resolution | FPS | Notes |
|------|-----------|-----|-------|
| Detection only | 640x640 | 60-120 | YOLOv8n baseline |
| Detection only | 1056x1056 | 30-50 | Fine-tuned model |
| OCR (per plate) | varies | 20-50ms | PaddleOCR GPU |
| Full Pipeline | 1080p | 30-40 | 1-2 plates/frame |
| Full Pipeline | 1080p | 15-25 | 3-5 plates/frame |

### CPU Performance (Intel i7-12700K)

| Task | Resolution | FPS | Notes |
|------|-----------|-----|-------|
| Detection only | 640x640 | 5-10 | YOLOv8n baseline |
| OCR (per plate) | varies | 50-150ms | PaddleOCR CPU |
| Full Pipeline | 1080p | 2-4 | 1-2 plates/frame |

### Memory Usage

- **GPU VRAM**: 2-4 GB (models + inference)
- **System RAM**: 1-2 GB (video frames + tracking)
- **Disk**: ~500MB (models + dependencies)

### Optimization Tips

1. **Lower image size** (640 vs 1056) for 2x speed boost
2. **Increase OCR interval** (60 vs 30 frames) for 40% speedup
3. **Use frame skipping** (process every 2nd frame) for real-time processing
4. **Disable preprocessing** (CLAHE, sharpening) if not needed
5. **Use CPU for small batches**, GPU for continuous processing

See [Configuration Guide](docs/configuration_guide.md) for tuning parameters.

---

## 🐛 Troubleshooting

### Common Issues

#### GPU not detected

**Problem:** `torch.cuda.is_available()` returns `False`

**Solutions:**
1. Check CUDA installation: `nvidia-smi`
2. Reinstall PyTorch with CUDA:
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
3. Verify in Python:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   print(torch.cuda.get_device_name(0))
   ```

#### Module not found errors

**Problem:** `ModuleNotFoundError: No module named 'src'`

**Solutions:**
1. Activate virtual environment: `.venv\Scripts\Activate.ps1`
2. Install in editable mode: `pip install -e .`
3. Set PYTHONPATH: `$env:PYTHONPATH = "$PWD"`

#### GUI won't start

**Problem:** Streamlit fails to launch or shows errors

**Solutions:**
1. Check Streamlit installation: `pip install --upgrade streamlit`
2. Try different port: `streamlit run gui/app.py --server.port 8502`
3. Clear cache: `streamlit cache clear`

#### No plates detected

**Problem:** Pipeline runs but no detections shown

**Solutions:**
1. Lower confidence threshold in config: `confidence_threshold: 0.15`
2. Verify model file exists: `Test-Path models/detection/yolov8_finetuned_v2_best.pt`
3. Test on known good image
4. Check video format/codec compatibility

#### OCR returns empty text

**Problem:** Plates detected but no text recognized

**Solutions:**
1. Enable preprocessing in config:
   ```yaml
   preprocessing:
     use_clahe: true
     use_sharpening: true
   ```
2. Check crop quality (save crops for inspection)
3. Adjust OCR parameters:
   ```yaml
   recognition:
     det_db_box_thresh: 0.3  # Lower threshold
     rec_batch_num: 6
   ```

#### Processing too slow

**Problem:** FPS < 10 on GPU

**Solutions:**
1. Reduce image size: `image_size: 640`
2. Increase OCR interval: `ocr_interval: 60`
3. Skip frames: `--sample-rate 2`
4. Disable preprocessing
5. Check GPU utilization: `nvidia-smi`

### Advanced Troubleshooting

See [Troubleshooting Guide](docs/troubleshooting.md) for:
- 24+ common issues with detailed solutions
- Installation problems (CUDA, PaddleOCR, version conflicts)
- Detection, recognition, and tracking issues
- GUI-specific problems (freezing, outdated results, too many reruns)
- Performance optimization tips
- Configuration errors
- Environment setup issues

---

## 📚 Documentation

### For Users
- **[User Guide](docs/user_guide.md)** - Complete GUI and CLI usage guide
- **[Configuration Guide](docs/configuration_guide.md)** - All config parameters explained
- **[Troubleshooting](docs/troubleshooting.md)** - Solutions to common problems

### For Developers
- **[Developer Guide](docs/developer_guide.md)** - Architecture, design decisions, contribution guidelines
- **[API Reference](docs/api_reference.md)** - Complete API documentation
- **[Project Structure](docs/project_structure.md)** - Codebase organization and module overview

### Additional Resources
- **[Annotation Guide](docs/annotation_guide.md)** - How to annotate GTA V plates
- **[Frame Sampling Guide](docs/frame_sampling_guide.md)** - Optimal frame extraction strategies
- **[Detection Comparison Report](docs/detection_comparison_report.md)** - Model performance analysis
- **[OCR Confusion Correction](docs/ocr_confusion_correction.md)** - Character recognition improvements

---

## 🙏 Acknowledgments

- **YOLOv8 Base Model**: [yasirfaizahmed/license-plate-object-detection](https://huggingface.co/yasirfaizahmed/license-plate-object-detection)
- **Ultralytics**: YOLOv8 implementation and training framework
- **PaddlePaddle**: OCR framework and pre-trained models
- **ByteTrack**: Multi-object tracking algorithm
- **Streamlit**: Interactive web application framework

## 📝 License

This project is for educational and research purposes. Please respect Rockstar Games' terms of service when using GTA V content.

# GTA V ALPR (Automatic License Plate Recognition)

A sophisticated computer vision system for detecting and recognizing license plates in GTA V gameplay footage using state-of-the-art deep learning models.

## 📋 Project Overview

This project implements a complete two-stage ALPR pipeline specifically designed for processing GTA V game footage:

1. **Detection Stage**: YOLOv8 object detection model identifies license plate locations
2. **Recognition Stage**: PaddleOCR extracts text from detected plates
3. **Tracking Stage**: ByteTrack maintains plate identity across video frames

**Key Features:**
- Real-time or batch processing of video/images
- GPU-accelerated inference for both detection and recognition
- Multi-object tracking to avoid redundant processing
- Configurable pipeline parameters via YAML
- Modular architecture for easy customization

## 🛠️ Technology Stack

- **Python**: 3.9+ (developed with 3.12.9)
- **Deep Learning Frameworks**:
  - PyTorch 2.5.1+ (CUDA 12.1)
  - PaddlePaddle 2.6.1 (GPU)
- **Detection**: Ultralytics YOLOv8 (yasirfaizahmed/license-plate-object-detection)
- **Recognition**: PaddleOCR 3.2.0
- **Computer Vision**: OpenCV 4.10.0
- **Data Augmentation**: Albumentations 2.0.8
- **Configuration**: PyYAML 6.0.2
- **Testing**: Pytest 8.4.2

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- NVIDIA GPU with CUDA 12.1+ support (recommended for performance)
- GTA V gameplay footage or screenshots

### Environment Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/ALPR_GTA5.git
cd ALPR_GTA5
```

2. **Create virtual environment:**
```bash
python -m venv .venv

# On Windows (PowerShell):
.\.venv\Scripts\activate

# On Linux/Mac:
source .venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Verify GPU installation:**
```bash
python scripts/diagnostics/verify_gpu.py
```

You should see:
- ✓ PyTorch CUDA available
- ✓ PaddlePaddle GPU available
- ✓ All packages installed

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

### Quick Start

**Single Image Detection:**
```powershell
python scripts/inference/detect_image.py --image path/to/image.jpg --output outputs/annotated.jpg
```

**Video Detection:**
```powershell
python scripts/inference/detect_video.py --video path/to/video.mp4 --output outputs/annotated.mp4
```

**Batch Frame Extraction:**
```powershell
python scripts/data_ingestion/batch_process_footage.py --fps 5 --quality 95
```

### Configuration

Edit `configs/pipeline_config.yaml` to customize:
- Detection confidence thresholds
- OCR language and parameters
- Tracking settings
- Preprocessing options
- Output formats

See [configuration documentation](docs/project_structure.md) for details.

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

```bash
# Run all tests
pytest

# Run specific test suite
pytest tests/unit/
pytest tests/integration/

# Run with coverage
pytest --cov=src tests/
```

### Code Standards

This project follows:
- **PEP 8** Python style guidelines
- **Type hints** for function signatures
- **Docstrings** (Google style) for all public APIs
- **Modular design** with clear separation of concerns

See `shrimp-rules.md` for complete development guidelines.

### Adding New Features

1. Follow the module structure in `src/`
2. Add configuration parameters to `configs/pipeline_config.yaml`
3. Write unit tests in `tests/unit/`
4. Update integration tests in `tests/integration/`
5. Document changes in `docs/`

## 📈 Performance Expectations

**GPU (NVIDIA RTX 3070 Ti):**
- Detection: ~60-120 FPS (640x640 input)
- Recognition: ~30-50 FPS (per plate crop)
- Full pipeline: ~30-40 FPS (1080p video with 1-2 plates)

**CPU (Modern multi-core):**
- Detection: ~5-10 FPS
- Recognition: ~3-5 FPS
- Full pipeline: ~2-4 FPS

*Performance varies based on image resolution, number of plates, and hardware.*

## 🙏 Acknowledgments

- **YOLOv8 Model**: [yasirfaizahmed/license-plate-object-detection](https://huggingface.co/yasirfaizahmed/license-plate-object-detection)
- **Ultralytics**: YOLOv8 implementation
- **PaddlePaddle**: OCR framework
- **ByteTrack**: Multi-object tracking algorithm

## 📝 License

This project is for educational and research purposes. Please respect Rockstar Games' terms of service when using GTA V content.

## 🐛 Troubleshooting

**GPU not detected:**
- Verify CUDA installation: `nvidia-smi`
- Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Check PaddlePaddle: `python -c "import paddle; print(paddle.device.is_compiled_with_cuda())"`

**Import errors:**
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

**Low accuracy:**
- Check detection confidence threshold (default: 0.25)
- Ensure good lighting and resolution in input footage
- Consider fine-tuning models on GTA V-specific data

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the development team.

---

**Current Status**: Week 1 Complete - Environment and structure ready for Week 2 implementation (detection module).

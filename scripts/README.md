# Scripts Directory - CLI Tools Reference

The `scripts/` directory contains modular CLI tools for the GTA V ALPR pipeline, organized by functional category. Each tool is self-contained with clear inputs/outputs and comprehensive documentation.

## Directory Structure

```
scripts/
├── annotation/          # Label Studio conversion and annotation helpers
├── data_ingestion/      # Frame extraction, video processing, dataset management
├── diagnostics/         # Environment verification, GPU checks, configuration validation
├── evaluation/          # Model evaluation, metrics calculation, report generation
├── inference/           # Image/video inference, detection, recognition
├── profiling/           # Performance profiling, optimization, benchmarking
├── training/            # Model training utilities (YOLO fine-tuning)
├── utils/               # Shared utilities and helper functions
│
├── process_video.py     # Main video processing script (legacy, see inference/)
├── README_process_video.md
├── demo_ocr_correction.py
├── test_ocr_correction.py
└── debug_multi_detection.py
```

## Quick Start

### 1. Setup
```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Verify installation
python scripts/diagnostics/verify_gpu.py
```

### 2. Common Workflows

**Process a single video:**
```powershell
python scripts/inference/process_video.py `
  --input "video.mp4" `
  --output "outputs/annotated.mp4"
```

**Detect plates in an image:**
```powershell
python scripts/inference/detect_image.py `
  --image "plate.jpg" `
  --output "outputs/annotated.jpg"
```

**Extract frames from video:**
```powershell
python scripts/data_ingestion/extract_frames.py `
  --input "video.mp4" `
  --output "outputs/frames" `
  --fps 5
```

**Evaluate detection model:**
```powershell
python scripts/evaluation/evaluate_detection.py `
  --model "models/detection/yolov8_finetuned_v2_best.pt" `
  --data "datasets/lpr/data.yaml"
```

**Profile pipeline performance:**
```powershell
python scripts/profiling/profile_pipeline.py `
  --config "configs/pipeline_config.yaml" `
  --input "test_video.mp4"
```

## Module Reference

### 📂 annotation/
**Label Studio conversion and annotation utilities**

Key scripts:
- `convert_labelstudio_to_yolo.py` - Convert Label Studio exports to YOLO format
- `validate_annotations.py` - Verify annotation quality
- `merge_datasets.py` - Combine multiple annotation exports

**Example:**
```powershell
python scripts/annotation/convert_labelstudio_to_yolo.py `
  --input "datasets/labelstudio_exports/project-1.json" `
  --output "datasets/lpr" `
  --split-ratio 0.8
```

---

### 📂 data_ingestion/
**Video processing, frame extraction, dataset management**

Key scripts:
- `extract_frames.py` - Extract frames from video at specified FPS
- `batch_process_footage.py` - Process multiple videos
- `clean_dataset.py` - Remove duplicates and invalid frames
- `generate_dataset_stats.py` - Compute dataset statistics

**Example:**
```powershell
# Extract frames at 5 FPS
python scripts/data_ingestion/extract_frames.py `
  --input "outputs/raw_footage/day_clear/video1.mp4" `
  --output "outputs/frames/day_clear" `
  --fps 5 `
  --quality 95

# Process entire directory
python scripts/data_ingestion/batch_process_footage.py `
  --input-dir "outputs/raw_footage" `
  --output-dir "outputs/processed" `
  --fps 5
```

---

### 🔧 diagnostics/
**Environment verification, dependency checks, configuration validation**

Key scripts:
- `verify_gpu.py` - Check CUDA/GPU availability
- `validate_config.py` - Validate configuration files
- `check_dependencies.py` - Verify all packages installed
- `system_info.py` - Print system information

**Example:**
```powershell
# Verify GPU
python scripts/diagnostics/verify_gpu.py

# Validate config
python scripts/diagnostics/validate_config.py configs/pipeline_config.yaml

# Check all dependencies
python scripts/diagnostics/check_dependencies.py
```

---

### 📊 evaluation/
**Model evaluation, metrics calculation, report generation**

Key scripts:
- `evaluate_detection.py` - Evaluate YOLOv8 model on test set
- `evaluate_ocr.py` - Evaluate PaddleOCR accuracy
- `evaluate_tracking.py` - Analyze tracking performance
- `generate_report.py` - Create comprehensive evaluation report

**Example:**
```powershell
# Evaluate detection
python scripts/evaluation/evaluate_detection.py `
  --model "models/detection/yolov8_finetuned_v2_best.pt" `
  --data "datasets/lpr/data.yaml" `
  --split "test"

# Evaluate OCR
python scripts/evaluation/evaluate_ocr.py `
  --images "datasets/ocr/images" `
  --ground-truth "datasets/ocr/ground_truth.txt"

# Generate full report
python scripts/evaluation/generate_report.py `
  --detection-results "outputs/evaluation/detection_results.json" `
  --ocr-results "outputs/evaluation/ocr_results.json"
```

---

### 🎯 inference/
**Image/video inference, plate detection, text recognition**

Key scripts:
- `detect_image.py` - Detect plates in single image
- `detect_video.py` - Process video with annotations
- `process_video.py` - Main video processing pipeline
- `recognize_plates.py` - OCR on detected plates

**Example:**
```powershell
# Detect image
python scripts/inference/detect_image.py `
  --image "photo.jpg" `
  --output "outputs/annotated.jpg" `
  --confidence 0.25

# Process video
python scripts/inference/process_video.py `
  --input "video.mp4" `
  --output "outputs/processed.mp4" `
  --config "configs/pipeline_config.yaml" `
  --export-json "results.json"

# Video with frame sampling
python scripts/inference/process_video.py `
  --input "video.mp4" `
  --output "output.mp4" `
  --sample-rate 2 `
  --no-video  # Skip video output for speed
```

---

### ⚡ profiling/
**Performance profiling, optimization analysis, benchmarking**

Key scripts:
- `profile_pipeline.py` - Profile pipeline execution time
- `compare_configurations.py` - Benchmark different configs
- `memory_profiler.py` - Analyze memory usage
- `optimization_report.py` - Generate optimization recommendations

**Example:**
```powershell
# Profile pipeline
python scripts/profiling/profile_pipeline.py `
  --config "configs/pipeline_config.yaml" `
  --input "test_video.mp4" `
  --output "outputs/profiling/report.md"

# Compare configurations
python scripts/profiling/compare_configurations.py `
  --configs "configs/test_*.yaml" `
  --input "test_video.mp4" `
  --output "optimization_results.csv"
```

---

### 🎓 training/
**Model training and fine-tuning utilities**

Key scripts:
- `train_detection.py` - Fine-tune YOLOv8 on custom dataset
- `train_ocr.py` - Fine-tune PaddleOCR (if applicable)
- `generate_training_config.py` - Create training configuration

**Example:**
```powershell
# Train detection model
python scripts/training/train_detection.py `
  --data "datasets/lpr/data.yaml" `
  --epochs 100 `
  --batch-size 16 `
  --device cuda
```

---

### 🔨 utils/
**Shared utilities and helper functions**

Contains:
- `config_utils.py` - Configuration loading/validation
- `video_io.py` - Video reading/writing
- `visualization.py` - Drawing utilities
- `logging_config.py` - Logging setup

---

## Command Patterns

### Get Help
```powershell
python scripts/<category>/<script>.py --help
```

### Common Flags
```powershell
--input, -i          Input file or directory path
--output, -o         Output file or directory path
--config, -c         Configuration YAML file
--device             cuda or cpu (default: cuda)
--verbose, -v        Enable verbose logging
--dry-run            Show what would be done (no execution)
```

## Best Practices

1. **Always activate virtual environment first**
   ```powershell
   .venv\Scripts\Activate.ps1
   ```

2. **Check script help before running**
   ```powershell
   python scripts/<category>/<script>.py --help
   ```

3. **Validate configuration**
   ```powershell
   python scripts/diagnostics/validate_config.py your_config.yaml
   ```

4. **Use absolute paths for clarity**
   ```powershell
   python scripts/inference/detect_image.py --image "D:\data\image.jpg"
   ```

5. **Save outputs to `outputs/` directory**
   ```powershell
   --output "outputs/<category>/<result_name>"
   ```

## Troubleshooting

**Script not found:**
- Ensure you're in the project root directory
- Check spelling of script name
- Verify the module category is correct

**Import errors:**
- Verify virtual environment is activated
- Run: `pip install -r requirements.txt`
- Check: `python scripts/diagnostics/check_dependencies.py`

**GPU not detected:**
- Run: `python scripts/diagnostics/verify_gpu.py`
- Check CUDA installation
- Set `--device cpu` as fallback

**Script hangs or crashes:**
- Add `--verbose` flag for debugging
- Check logs in `outputs/logs/`
- Try with smaller input file first
- Create issue on GitHub with error message

## Adding New Scripts

When adding a new script:

1. **Place in appropriate category folder** (or create new category)
2. **Use argparse for CLI arguments**
3. **Add comprehensive docstring with usage examples**
4. **Include error handling and logging**
5. **Update the category README.md**
6. **Test with `--help` flag**
7. **Add entry to this main README**

Example template:
```python
"""
Script Name - Brief description

Usage:
    python scripts/category/script_name.py --input file.txt --output output.txt

Arguments:
    --input, -i: Input file path
    --output, -o: Output file path
"""

import argparse
import logging

def main(args):
    # Implementation
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", required=True, help="Input file")
    parser.add_argument("--output", "-o", required=True, help="Output file")
    args = parser.parse_args()
    main(args)
```

---

## Additional Resources

- **[User Guide](../docs/user_guide.md)** - How to use scripts from user perspective
- **[Developer Guide](../docs/developer_guide.md)** - Development guidelines
- **[Configuration Guide](../docs/configuration_guide.md)** - Config parameter reference
- **[Troubleshooting](../docs/troubleshooting.md)** - Solutions to common issues

---

*Last updated: November 15, 2025*

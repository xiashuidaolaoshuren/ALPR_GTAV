# Scripts Directory Overview

This directory contains helper scripts for data collection and processing.

## Core Scripts

### 1. `extract_frames.py` ⭐ Core Tool
**Purpose:** Extract frames from video files for dataset creation.

**Key Features:**
- `FrameExtractor` class with configurable FPS and quality
- Can process single video or batch process directories
- CLI interface for direct usage

**Usage:**
```powershell
# Single video
python scripts/extract_frames.py --input video.mp4 --output images/ --fps 5

# Batch mode
python scripts/extract_frames.py --batch --input_dir videos/ --output_dir images/
```

**Used by:** `batch_process_footage.py` imports this

---

### 2. `batch_process_footage.py` 🎬 Project-Specific Wrapper
**Purpose:** Automate frame extraction for GTA V ALPR project structure.

**Key Features:**
- Automatically processes `outputs/raw_footage/` subdirectories
- Outputs to `outputs/test_images/`
- Project-aware folder scanning
- Summary reporting

**Usage:**
```powershell
python scripts/batch_process_footage.py --fps 5 --quality 95
```

**Relationship:** Imports `FrameExtractor` from `extract_frames.py`

---

### 3. `generate_metadata.py` 📝 Metadata Generator
**Purpose:** Auto-generate metadata templates from extracted images.

**Key Features:**
- Scans `outputs/test_images/` for images
- Infers conditions from filenames
- Creates/appends to `metadata.txt`
- Skips existing entries

**Usage:**
```powershell
python scripts/generate_metadata.py [--overwrite]
```

---

### 4. `check_dataset_quality.py` ✅ Quality Validator
**Purpose:** Validate dataset completeness and quality.

**Key Features:**
- Checks image count (50-100 target)
- Validates condition diversity
- Checks angle diversity
- Validates metadata completeness
- Basic image quality checks

**Usage:**
```powershell
python scripts/check_dataset_quality.py
```

---

### 5. `verify_gpu.py` 🖥️ Environment Check
**Purpose:** Verify GPU availability for deep learning.

**Usage:**
```powershell
python scripts/verify_gpu.py
```

---

## Documentation

### `data_collection_guide.md` 📚 User Guide
Comprehensive step-by-step guide for collecting GTA V gameplay footage and extracting frames.

**Contents:**
- OBS Studio setup
- GTA V configuration
- Recording workflow
- Frame extraction
- Quality checks
- Quick reference card

---

## Typical Workflow

```
1. Record Gameplay
   ↓
   Save videos to outputs/raw_footage/[condition]/
   
2. Extract Frames
   ↓
   python scripts/batch_process_footage.py
   
3. Generate Metadata
   ↓
   python scripts/generate_metadata.py
   
4. Review & Edit
   ↓
   Edit outputs/test_images/metadata.txt manually
   
5. Quality Check
   ↓
   python scripts/check_dataset_quality.py
   
6. Dataset Ready! ✅
```

---

## Script Dependencies

```
extract_frames.py (standalone)
    ↑
    │ imported by
    │
batch_process_footage.py (uses extract_frames)

generate_metadata.py (standalone)

check_dataset_quality.py (standalone)

verify_gpu.py (standalone)
```

---

## Notes

- All scripts use Python 3.9+ virtual environment (`.venv`)
- Scripts follow PEP 8 coding standards
- Comprehensive logging for debugging
- Error handling for common issues

For detailed usage, run any script with `--help` flag.

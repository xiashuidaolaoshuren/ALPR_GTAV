# User Guide

Complete guide for using the GTA V ALPR system. Covers both GUI and command-line interfaces with step-by-step tutorials.

## Table of Contents

- [Getting Started](#getting-started)
- [GUI Walkthrough](#gui-walkthrough)
- [Command-Line Usage](#command-line-usage)
- [Configuration](#configuration)
- [Best Practices](#best-practices)
- [Workflows](#workflows)
- [FAQ](#faq)

---

## Getting Started

### System Requirements

**Minimum:**
- CPU: Intel i5 or equivalent (4+ cores)
- RAM: 8GB
- GPU: NVIDIA GTX 1050 Ti (2GB VRAM) or CPU-only
- Storage: 2GB free space
- OS: Windows 10/11, Linux, macOS

**Recommended:**
- CPU: Intel i7 or AMD Ryzen 7 (8+ cores)
- RAM: 16GB
- GPU: NVIDIA RTX 3060+ (6GB+ VRAM)
- Storage: 10GB free space (for datasets)
- OS: Windows 11 with CUDA 12.1

### First-Time Setup

**1. Install Python 3.9+**

Download from [python.org](https://www.python.org/downloads/)

Verify installation:
```powershell
python --version  # Should show 3.9 or higher
```

**2. Clone Repository**
```powershell
git clone https://github.com/xiashuidaolaoshuren/ALPR_GTAV.git
cd ALPR_GTAV
```

**3. Create Virtual Environment**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
```

You should see `(.venv)` in your terminal prompt.

**4. Install Dependencies**
```powershell
# For GPU (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# For CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**5. Verify Installation**
```powershell
python scripts/diagnostics/verify_gpu.py
```

Expected output:
```
========================================
System Information:
========================================
Python Version: 3.12.9
Operating System: Windows-10-10.0.22631-SP0

========================================
GPU Information:
========================================
✓ PyTorch CUDA available
  Device: NVIDIA GeForce RTX 3070 Ti
  CUDA Version: 12.1
✓ PaddlePaddle GPU available

========================================
Package Versions:
========================================
✓ ultralytics: 8.3.45
✓ paddleocr: 2.10.0
✓ opencv-python: 4.10.0.84
✓ streamlit: 1.40.2

All checks passed! ✓
```

**6. Test GUI Launch**
```powershell
python -m streamlit run gui/app.py
```

Opens browser at `http://localhost:8501`

---

## GUI Walkthrough

The Streamlit GUI provides an intuitive interface for video processing with real-time visualization.

### Launching the GUI

**Method 1: PowerShell**
```powershell
cd ALPR_GTAV
.venv\Scripts\Activate.ps1
python -m streamlit run gui/app.py
```

**Method 2: Command Prompt**
```cmd
cd ALPR_GTAV
.venv\Scripts\activate.bat
python -m streamlit run gui/app.py
```

**Method 3: VS Code**
- Open project in VS Code
- Open `gui/app.py`
- Right-click → Run Python File in Terminal
- Or use integrated terminal with commands above

### GUI Components

**📸 Screenshot:** [_Placeholder: Full GUI interface showing all components labeled_]

The interface consists of three main areas:

1. **Left Sidebar (Control Panel)** - Configuration and controls
2. **Main Area (Video Display)** - Processing visualization
3. **Bottom Area (Info Panel)** - Status and results

### Control Panel (Sidebar)

**📸 Screenshot:** [_Placeholder: Control panel showing file uploader and sliders_]

#### 1. Video Upload

**Upload Button:**
- Click "Browse files" or drag & drop
- Supported formats: MP4, AVI, MOV, MKV
- Max size: 200MB (configurable)

**Video Information Display:**
```
📹 Video: example_video.mp4
⏱️ Duration: 00:02:30
📐 Resolution: 1920x1080
🎬 FPS: 30.0
📊 Total Frames: 4500
```

#### 2. Pipeline Parameters

**Confidence Threshold** (0.0 - 1.0)
- **Default**: 0.25
- **Description**: Minimum confidence for plate detection
- **Lower** = More detections (including false positives)
- **Higher** = Fewer detections (only high-confidence plates)
- **Recommended**: 0.20-0.30 for GTA V

**IOU Threshold** (0.0 - 1.0)
- **Default**: 0.45
- **Description**: Overlap threshold for Non-Maximum Suppression
- **Lower** = More aggressive suppression (fewer overlapping boxes)
- **Higher** = Keep more overlapping detections
- **Recommended**: 0.40-0.50

**OCR Interval** (1-120 frames)
- **Default**: 30
- **Description**: Frames to wait between OCR runs per track
- **Lower** = More OCR calls (slower, more accurate)
- **Higher** = Fewer OCR calls (faster, may miss text changes)
- **Recommended**: 30-60 for balance

**📸 Screenshot:** [_Placeholder: Slider controls showing default values_]

#### 3. Device Selection

**GPU (CUDA)** - Recommended
- Uses NVIDIA GPU for acceleration
- 10-30x faster than CPU
- Requires CUDA-compatible GPU

**CPU** - Fallback option
- Works on any system
- Slower processing (2-5 FPS)
- No GPU required

#### 4. Processing Controls

**Start Processing** Button
- Begins video processing
- Disables controls during processing
- Shows progress bar

**Stop Processing** Button
- Gracefully stops processing
- Saves progress up to current frame
- Re-enables controls

**📸 Screenshot:** [_Placeholder: Start/Stop buttons in different states_]

### Video Display (Main Area)

**📸 Screenshot:** [_Placeholder: Video display showing annotated frame with bounding boxes_]

**Features:**
- **Real-time Preview**: Shows annotated frames as they're processed
- **Bounding Boxes**: Green boxes around detected plates
- **Track IDs**: Numbers identifying each unique plate
- **Recognized Text**: Displayed above bounding boxes
- **Confidence Scores**: Shown next to text
- **FPS Counter**: Processing speed indicator

**Example Frame:**
```
┌───────────────────────────┐
│                           │
│    [Track 1: 12ABC345]   │  ← Recognized text
│    [Conf: 0.92]          │  ← Confidence
│    ┌─────────┐           │
│    │ Plate   │           │  ← Bounding box
│    └─────────┘           │
│                           │
│  FPS: 35.2               │  ← Processing speed
└───────────────────────────┘
```

### Info Panel (Tabs)

**📸 Screenshot:** [_Placeholder: Info panel showing both Status and Logs tabs_]

#### Status Tab

**Metrics:**
- **Active Tracks**: Number of plates currently being tracked
- **Recognized**: Total unique plates with text recognized
- **Processed Frames**: Current frame / Total frames
- **Processing FPS**: Real-time processing speed
- **Elapsed Time**: Time since processing started

**Latest Recognitions Table:**
| Track ID | Text | Confidence | First Seen | Last Seen |
|----------|------|------------|------------|-----------|
| 1 | 12ABC345 | 0.92 | Frame 10 | Frame 45 |
| 2 | 99XYZ123 | 0.88 | Frame 60 | Frame 95 |
| 3 | 77DEF456 | 0.85 | Frame 120 | Frame 155 |

#### Logs Tab

**Real-time log stream:**
```
[INFO] Pipeline initialized successfully
[INFO] Processing frame 100/4500 (2.22%)
[INFO] Track 1 OCR: 12ABC345 (conf=0.92)
[WARNING] Track 2 lost (age > max_age)
[INFO] Track 3 created
[INFO] Processing frame 200/4500 (4.44%)
```

**Log Levels:**
- **INFO**: Normal operation
- **WARNING**: Non-critical issues
- **ERROR**: Processing errors
- **DEBUG**: Detailed diagnostics (disabled by default)

### Processing Workflow

**Step-by-Step:**

1. **Upload Video**
   - Click "Browse files" in sidebar
   - Select video file (e.g., `gameplay.mp4`)
   - Wait for video info to load

2. **Adjust Parameters** (optional)
   - Move sliders to desired values
   - Default values work well for most cases

3. **Select Device**
   - Choose "GPU (CUDA)" if available
   - Use "CPU" as fallback

4. **Start Processing**
   - Click "Start Processing" button
   - Video display shows annotated frames
   - Progress bar appears below video
   - Info panel updates in real-time

5. **Monitor Progress**
   - Watch Status tab for metrics
   - Check Logs tab for issues
   - Observe FPS counter

6. **Stop or Wait for Completion**
   - Click "Stop Processing" to halt early
   - Or wait for automatic completion

7. **Download Results**
   - Click "Download Annotated Video" button
   - Click "Download Results (JSON)" for data
   - Files save to your Downloads folder

**📸 Screenshot:** [_Placeholder: Complete workflow showing all steps_]

---

## Command-Line Usage

For advanced users, automation, and batch processing.

### Basic Video Processing

**Process single video with defaults:**
```powershell
python scripts/process_video.py `
  --input "path/to/video.mp4" `
  --output "outputs/processed_video.mp4"
```

**Output:**
```
[INFO] Loading pipeline configuration...
[INFO] Initializing detection model...
[INFO] Initializing OCR model...
[INFO] Opening video: path/to/video.mp4
[INFO] Video info: 1920x1080 @ 30.0 FPS (4500 frames)
Processing: 100%|████████████████| 4500/4500 [02:15<00:00, 35.12 frames/s]
[INFO] Processing complete!
[INFO] Total tracks: 15
[INFO] Recognized plates: 12
[INFO] Output saved to: outputs/processed_video.mp4
```

### Command-Line Options

**All available flags:**
```powershell
python scripts/process_video.py --help
```

```
usage: process_video.py [-h] --input INPUT --output OUTPUT
                        [--config CONFIG] [--sample-rate SAMPLE_RATE]
                        [--no-video] [--export-json EXPORT_JSON]
                        [--export-csv EXPORT_CSV] [--show-preview]

optional arguments:
  --input INPUT         Input video path
  --output OUTPUT       Output video path
  --config CONFIG       Config file (default: configs/pipeline_config.yaml)
  --sample-rate N       Process every Nth frame (default: 1)
  --no-video            Skip video output (faster)
  --export-json FILE    Export results to JSON
  --export-csv FILE     Export results to CSV
  --show-preview        Show live preview window
```

### Advanced Examples

**1. Fast processing (skip video output):**
```powershell
python scripts/process_video.py `
  --input "video.mp4" `
  --output "dummy.mp4" `
  --no-video `
  --export-json "results.json"
```
- Skips video encoding (much faster)
- Only extracts data to JSON
- Use when you don't need annotated video

**2. Frame sampling for speed:**
```powershell
python scripts/process_video.py `
  --input "video.mp4" `
  --output "output.mp4" `
  --sample-rate 2
```
- Processes every 2nd frame
- 2x speedup
- May miss fast-moving plates

**3. Custom configuration:**
```powershell
python scripts/process_video.py `
  --input "video.mp4" `
  --output "output.mp4" `
  --config "configs/my_custom_config.yaml"
```
- Uses custom config file
- See [Configuration](#configuration) section

**4. Live preview (debugging):**
```powershell
python scripts/process_video.py `
  --input "video.mp4" `
  --output "output.mp4" `
  --show-preview
```
- Opens window showing processing
- Press 'q' to quit early
- Useful for debugging

**5. Multiple outputs:**
```powershell
python scripts/process_video.py `
  --input "video.mp4" `
  --output "output.mp4" `
  --export-json "results.json" `
  --export-csv "results.csv"
```
- Saves annotated video + JSON + CSV
- All formats contain same data

### Batch Processing

**Process all videos in a folder:**
```powershell
python scripts/data_ingestion/batch_process_footage.py `
  --input-dir "outputs/raw_footage" `
  --output-dir "outputs/processed" `
  --fps 5 `
  --quality 95
```

**What it does:**
- Finds all videos in `input-dir` (recursive)
- Extracts frames at 5 FPS
- Saves to `output-dir` with 95% JPEG quality
- Preserves folder structure

**Options:**
```
--fps N              Frames per second to extract (default: 5)
--quality Q          JPEG quality 1-100 (default: 95)
--extensions EXT     File extensions to process (default: .mp4,.avi,.mov)
--recursive          Process subfolders (default: True)
```

### Single Image Detection

**Detect plates in an image:**
```powershell
python scripts/inference/detect_image.py `
  --image "path/to/image.jpg" `
  --output "outputs/annotated.jpg" `
  --config "configs/pipeline_config.yaml"
```

**With custom parameters:**
```powershell
python scripts/inference/detect_image.py `
  --image "image.jpg" `
  --output "annotated.jpg" `
  --confidence 0.3 `
  --save-crops "outputs/crops"
```
- `--confidence`: Detection threshold
- `--save-crops`: Save cropped plate images

---

## Configuration

### Config File Structure

**Main config:** `configs/pipeline_config.yaml`

```yaml
# Device (cuda or cpu)
device: cuda

# Detection settings
detection:
  model_path: models/detection/yolov8_finetuned_v2_best.pt
  confidence_threshold: 0.25
  iou_threshold: 0.45
  image_size: 640  # 640 or 1056

# Recognition settings
recognition:
  use_gpu: true
  lang: 'en'
  det_db_box_thresh: 0.5
  rec_batch_num: 6
  use_angle_cls: false

# Tracking settings
tracking:
  tracker_type: bytetrack
  max_age: 30
  min_hits: 3
  iou_threshold: 0.3
  ocr_interval: 30
  ocr_confidence_threshold: 0.7

# Preprocessing
preprocessing:
  use_clahe: true
  clahe_clip_limit: 2.0
  clahe_tile_grid_size: [8, 8]
  use_sharpening: true
  sharpen_kernel_size: 3
  use_denoising: false

# Output settings
output:
  show_bbox: true
  show_track_id: true
  show_text: true
  show_confidence: true
  bbox_color: [0, 255, 0]  # Green
  text_color: [255, 255, 255]  # White
  bbox_thickness: 2
  text_font_scale: 0.9
```

### Key Parameters Explained

#### Detection Parameters

**`confidence_threshold`** (0.0 - 1.0)
- **Default**: 0.25
- **Description**: Minimum confidence score for plate detection
- **Impact**: 
  - Lower (0.15-0.20): More detections, more false positives
  - Default (0.25): Balanced
  - Higher (0.30-0.40): Fewer detections, high precision
- **When to adjust**:
  - Low light videos → Lower (0.20)
  - High quality videos → Higher (0.30)
  - Too many false positives → Increase
  - Missing plates → Decrease

**`iou_threshold`** (0.0 - 1.0)
- **Default**: 0.45
- **Description**: Intersection over Union threshold for NMS
- **Impact**:
  - Lower (0.30-0.40): More aggressive box suppression
  - Default (0.45): Balanced
  - Higher (0.50-0.60): Keep more overlapping boxes
- **When to adjust**:
  - Multiple boxes per plate → Decrease
  - Missing nearby plates → Increase

**`image_size`** (640 or 1056)
- **Default**: 640
- **Description**: Input size for detection model
- **Impact**:
  - 640: 2x faster, slightly lower accuracy
  - 1056: Slower, best accuracy for small plates
- **When to adjust**:
  - Small/distant plates → Use 1056
  - Real-time processing needed → Use 640

#### Recognition Parameters

**`use_gpu`** (true/false)
- **Default**: true
- **Description**: Use GPU for OCR
- **Impact**: 3-5x speedup with GPU

**`lang`** ('en', 'ch', 'french', etc.)
- **Default**: 'en'
- **Description**: OCR language
- **Impact**: Affects character recognition accuracy

**`det_db_box_thresh`** (0.0 - 1.0)
- **Default**: 0.5
- **Description**: Text detection threshold
- **Impact**:
  - Lower (0.3): Detect more text regions
  - Higher (0.7): Only high-confidence text
- **When to adjust**:
  - Missing text → Decrease to 0.3
  - Too many false detections → Increase to 0.6

**`rec_batch_num`** (1-12)
- **Default**: 6
- **Description**: Batch size for recognition
- **Impact**: Higher = faster on GPU, more memory

#### Tracking Parameters

**`max_age`** (1-100 frames)
- **Default**: 30
- **Description**: Frames to keep track alive without detection
- **Impact**:
  - Lower (15-20): Tracks lost quickly, more new IDs
  - Default (30): Balanced
  - Higher (50-60): Tracks persist longer
- **When to adjust**:
  - Tracks lost too quickly → Increase to 50
  - Too many stale tracks → Decrease to 20

**`min_hits`** (1-10)
- **Default**: 3
- **Description**: Detections needed before track confirmed
- **Impact**:
  - Lower (1-2): Tracks created faster, more false tracks
  - Default (3): Balanced
  - Higher (5-7): Only consistent detections tracked
- **When to adjust**:
  - Missing tracks → Decrease to 2
  - Too many false tracks → Increase to 5

**`ocr_interval`** (1-120 frames)
- **Default**: 30
- **Description**: Frames between OCR runs per track
- **Impact**:
  - Lower (10-20): More OCR, slower, more accurate
  - Default (30): Balanced
  - Higher (60-90): Less OCR, faster, may miss text
- **When to adjust**:
  - Missing text → Decrease to 15
  - Speed critical → Increase to 60

**`ocr_confidence_threshold`** (0.0 - 1.0)
- **Default**: 0.7
- **Description**: Skip OCR if confidence > threshold
- **Impact**:
  - Lower (0.5): Run OCR more often
  - Higher (0.9): Skip OCR if high confidence exists

#### Preprocessing Parameters

**`use_clahe`** (true/false)
- **Default**: true
- **Description**: Contrast Limited Adaptive Histogram Equalization
- **Impact**: Improves contrast in low-light images
- **When to use**: Night scenes, dark footage

**`use_sharpening`** (true/false)
- **Default**: true
- **Description**: Sharpening filter to enhance edges
- **Impact**: Makes text more readable
- **When to use**: Blurry footage, motion blur

**`use_denoising`** (true/false)
- **Default**: false
- **Description**: Non-local means denoising
- **Impact**: Reduces noise but slower
- **When to use**: Very noisy footage, compression artifacts

### Configuration Presets

**High Accuracy (Slow):**
```yaml
detection:
  confidence_threshold: 0.20
  image_size: 1056
tracking:
  ocr_interval: 15
preprocessing:
  use_clahe: true
  use_sharpening: true
  use_denoising: true
```

**Balanced (Default):**
```yaml
detection:
  confidence_threshold: 0.25
  image_size: 640
tracking:
  ocr_interval: 30
preprocessing:
  use_clahe: true
  use_sharpening: true
  use_denoising: false
```

**Fast Processing:**
```yaml
detection:
  confidence_threshold: 0.30
  image_size: 640
tracking:
  ocr_interval: 60
preprocessing:
  use_clahe: false
  use_sharpening: false
  use_denoising: false
```

**Night/Low Light:**
```yaml
detection:
  confidence_threshold: 0.20
  image_size: 1056
preprocessing:
  use_clahe: true
  clahe_clip_limit: 3.0
  use_sharpening: true
  use_denoising: true
```

---

## Best Practices

### Video Preparation

**Optimal Video Properties:**
- **Resolution**: 1080p (1920x1080) recommended
- **FPS**: 30 FPS ideal, 24-60 acceptable
- **Bitrate**: 5-10 Mbps for good quality
- **Codec**: H.264 (most compatible)
- **Format**: MP4 (best compatibility)

**If Source is Too Large:**
```powershell
# Use ffmpeg to compress
ffmpeg -i input.mp4 -vcodec libx264 -crf 23 -preset medium output.mp4
```
- `crf 23`: Good quality (lower = better, 18-28 range)
- `preset medium`: Balanced speed/size

### Performance Optimization

**1. Use Appropriate Image Size**
- **640px**: Fast processing, good for real-time
- **1056px**: Better accuracy, use for small/distant plates

**2. Adjust OCR Interval**
- **30 frames** (1 second @ 30fps): Default, good balance
- **60 frames**: 40% faster, acceptable for slow-moving plates
- **15 frames**: 25% slower, better for fast action

**3. Enable/Disable Preprocessing**
- **Enable** for: Low-light, blurry, or low-quality footage
- **Disable** for: High-quality, well-lit footage (faster)

**4. Frame Sampling**
- Process every 2nd or 3rd frame for 2-3x speedup
- Acceptable when plates are visible for many frames

**5. Skip Video Output**
- Use `--no-video` flag for data extraction only
- 2-3x faster (no encoding overhead)

### Accuracy Improvement

**1. Tune Confidence Threshold**
- Start at 0.25
- Lower to 0.20 if missing plates
- Raise to 0.30 if too many false positives

**2. Enable Preprocessing**
- Always use CLAHE for night scenes
- Use sharpening for blurry footage
- Use denoising sparingly (slow)

**3. Use High-Quality Source**
- Higher resolution = better accuracy
- Avoid heavily compressed footage
- Use original game recordings when possible

**4. Adjust Tracking Parameters**
- Increase `max_age` if tracks lost frequently
- Decrease `min_hits` if tracks not created
- Lower `ocr_interval` if text not recognized

---

## Workflows

### Workflow 1: Single Video Processing (GUI)

**Goal:** Process a single video with real-time visualization

**Steps:**
1. Launch GUI: `python -m streamlit run gui/app.py`
2. Upload video file
3. Adjust parameters if needed (usually defaults work)
4. Click "Start Processing"
5. Monitor progress in Info Panel
6. Download annotated video and results

**Use Case:** Quick analysis, demonstration, interactive adjustment

---

### Workflow 2: Batch Video Processing (CLI)

**Goal:** Process multiple videos automatically

**Steps:**
1. Organize videos in folders:
   ```
   raw_footage/
     day_clear/
       video1.mp4
       video2.mp4
     night_rain/
       video3.mp4
   ```

2. Run batch processor:
   ```powershell
   python scripts/data_ingestion/batch_process_footage.py `
     --input-dir "raw_footage" `
     --output-dir "processed" `
     --fps 5
   ```

3. Results saved to `processed/` with same structure

**Use Case:** Large datasets, automated pipelines

---

### Workflow 3: Fast Data Extraction (CLI)

**Goal:** Extract plate data without creating annotated video

**Steps:**
1. Process with `--no-video` flag:
   ```powershell
   python scripts/process_video.py `
     --input "video.mp4" `
     --output "dummy.mp4" `
     --no-video `
     --export-json "results.json"
   ```

2. Results in JSON:
   ```json
   {
     "recognitions": [
       {"track_id": 1, "text": "12ABC345", ...}
     ]
   }
   ```

**Use Case:** Data analysis, database import, research

---

### Workflow 4: Testing Different Configurations

**Goal:** Find optimal parameters for your footage

**Steps:**
1. Create test configs:
   ```
   configs/
     test_low_conf.yaml      # confidence: 0.20
     test_default.yaml       # confidence: 0.25
     test_high_conf.yaml     # confidence: 0.30
   ```

2. Run profiler:
   ```powershell
   python scripts/profiling/compare_configurations.py `
     --configs "configs/test_*.yaml" `
     --input "sample_video.mp4" `
     --output "optimization_results.csv"
   ```

3. Analyze results:
   ```csv
   config,fps,detections,recognitions,accuracy
   low_conf,28.5,45,38,0.84
   default,32.1,38,32,0.88
   high_conf,35.2,30,28,0.92
   ```

4. Choose best balance

**Use Case:** Optimization, benchmarking

---

### Workflow 5: Evaluation and Reporting

**Goal:** Measure model performance on test set

**Steps:**
1. Prepare test dataset:
   ```
   datasets/lpr/test/
     images/
       plate001.jpg
       plate002.jpg
     labels/
       plate001.txt
       plate002.txt
   ```

2. Run detection evaluation:
   ```powershell
   python scripts/evaluation/evaluate_detection.py `
     --model "models/detection/yolov8_finetuned_v2_best.pt" `
     --data "datasets/lpr/data.yaml" `
     --split "test"
   ```

3. Run OCR evaluation:
   ```powershell
   python scripts/evaluation/evaluate_ocr.py `
     --images "datasets/ocr/images" `
     --ground-truth "datasets/ocr/ground_truth.txt"
   ```

4. Generate report:
   ```powershell
   python scripts/evaluation/generate_report.py `
     --output "full_report.md"
   ```

**Use Case:** Model validation, performance tracking

---

## FAQ

### General Questions

**Q: What video formats are supported?**

A: MP4, AVI, MOV, MKV with H.264/H.265 codec. Use `ffmpeg` to convert if needed:
```powershell
ffmpeg -i input.avi -vcodec libx264 output.mp4
```

**Q: Can I use this without a GPU?**

A: Yes, but processing will be 10-30x slower. Set `device: cpu` in config or select CPU in GUI.

**Q: How much VRAM do I need?**

A: 2GB minimum (GTX 1050 Ti), 4GB+ recommended (RTX 3060+). Reduce `image_size` to 640 if running out of memory.

**Q: What's the difference between confidence and IOU thresholds?**

A: 
- **Confidence**: How certain the model is that a plate exists
- **IOU**: How much boxes can overlap before suppression (NMS)

### Performance Questions

**Q: Why is processing slow?**

A: Common causes:
1. Using CPU instead of GPU (check device setting)
2. Large image size (1056 vs 640)
3. Low OCR interval (too many OCR calls)
4. Preprocessing enabled unnecessarily
5. Writing video to slow storage (HDD vs SSD)

**Q: How can I speed up processing?**

A: Try these in order:
1. Use GPU if available
2. Reduce `image_size` to 640
3. Increase `ocr_interval` to 60
4. Use `--sample-rate 2` (process every 2nd frame)
5. Disable preprocessing (`use_clahe: false`, etc.)
6. Use `--no-video` if you only need data

**Q: What FPS should I expect?**

A:
- **GPU (RTX 3070 Ti)**: 30-40 FPS full pipeline
- **GPU (GTX 1660)**: 15-25 FPS full pipeline
- **CPU (i7)**: 2-4 FPS full pipeline

### Detection Questions

**Q: Why are no plates detected?**

A: Try:
1. Lower `confidence_threshold` to 0.20
2. Increase `image_size` to 1056
3. Check video quality (resolution, lighting)
4. Verify model file exists: `models/detection/yolov8_finetuned_v2_best.pt`

**Q: Why are there too many false positives?**

A: Try:
1. Raise `confidence_threshold` to 0.30-0.35
2. Lower `iou_threshold` to 0.35-0.40
3. Add post-processing filters (aspect ratio, area)

**Q: Plates are detected but immediately lost**

A: Tracking issue. Try:
1. Increase `max_age` to 50
2. Decrease `min_hits` to 2
3. Lower `iou_threshold` (tracking) to 0.25

### Recognition Questions

**Q: Why is OCR returning empty text?**

A: Try:
1. Enable preprocessing: `use_clahe: true`, `use_sharpening: true`
2. Lower `det_db_box_thresh` to 0.3
3. Save crops for inspection: Check if plates are readable
4. Verify PaddleOCR is installed correctly

**Q: Why is text incorrect?**

A: Common character confusions (0/O, 1/I, 8/B):
1. Enable post-processing correction (see `docs/ocr_confusion_correction.md`)
2. Use multiple OCR readings (lower `ocr_interval`)
3. Improve image quality (preprocessing, higher resolution)

**Q: How can I improve OCR accuracy?**

A: In order of impact:
1. Enable preprocessing (CLAHE + sharpening)
2. Use larger `image_size` (1056)
3. Lower `ocr_interval` (more readings)
4. Improve source video quality
5. Fine-tune OCR model on GTA V plates

### GUI Questions

**Q: GUI freezes during processing**

A: This is usually normal. Check:
1. Processing is running in background thread
2. Stop button should still be clickable
3. If completely frozen, restart Streamlit

**Q: Why don't I see updated results?**

A: Try:
1. Wait a few seconds (UI updates every 0.5s)
2. Check if processing is still running
3. Click "Stop" and restart if stuck

**Q: Can I adjust parameters during processing?**

A: No, parameters are locked during processing. Stop processing, adjust, then restart.

**Q: Where are results saved?**

A: Downloads folder in your browser's default location. Check browser settings if you can't find them.

### Configuration Questions

**Q: Which config file is used by GUI?**

A: `configs/pipeline_config.yaml` by default. GUI sliders override some parameters.

**Q: How do I create custom configs?**

A:
1. Copy `configs/pipeline_config.yaml`
2. Rename to `configs/my_config.yaml`
3. Edit parameters
4. Use with `--config` flag in CLI

**Q: What happens if I misconfigure?**

A: Pipeline will fail with error message. Use `validate_config.py` to check:
```powershell
python scripts/diagnostics/validate_config.py configs/my_config.yaml
```

### Error Messages

**Q: `ModuleNotFoundError: No module named 'ultralytics'`**

A: Virtual environment not activated or dependencies not installed:
```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Q: `RuntimeError: CUDA out of memory`**

A: GPU VRAM exhausted. Try:
1. Reduce `image_size` to 640
2. Close other GPU applications
3. Use CPU instead: `device: cpu`

**Q: `cv2.error: OpenCV(4.x) error`**

A: Video codec issue:
```powershell
# Convert video with ffmpeg
ffmpeg -i input.mp4 -vcodec libx264 output.mp4
```

**Q: `StreamlitAPIException: Too many reruns`**

A: GUI refresh loop issue. Restart Streamlit:
```powershell
# Press Ctrl+C in terminal
# Re-run: python -m streamlit run gui/app.py
```

---

## Additional Resources

- **[Configuration Guide](configuration_guide.md)** - Complete parameter reference
- **[Developer Guide](developer_guide.md)** - Architecture and contribution guidelines
- **[Troubleshooting Guide](troubleshooting.md)** - Detailed solutions for 24+ issues
- **[API Reference](api_reference.md)** - Python API documentation

---

*Last updated: November 14, 2025*

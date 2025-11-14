# Troubleshooting Guide

Comprehensive solutions to common issues when developing or using the GTA V ALPR system.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Detection Problems](#detection-problems)
- [Recognition Issues](#recognition-issues)
- [Tracking Problems](#tracking-problems)
- [GUI Issues](#gui-issues)
- [Performance Issues](#performance-issues)
- [Configuration Errors](#configuration-errors)
- [Model Loading Errors](#model-loading-errors)
- [Video Processing Issues](#video-processing-issues)
- [Environment Issues](#environment-issues)

---

## Installation Issues

### Issue 1: `ModuleNotFoundError: No module named 'ultralytics'`

**Symptoms:**
```python
ModuleNotFoundError: No module named 'ultralytics'
```

**Diagnosis:**
- Package not installed in current Python environment
- Using wrong Python interpreter

**Solution:**

1. **Verify Python environment:**
```powershell
# Check which Python is active
python --version
Get-Command python

# Check installed packages
pip list | Select-String ultralytics
```

2. **Activate virtual environment if using one:**
```powershell
# If using venv
.venv\Scripts\Activate.ps1

# If using conda
conda activate alpr
```

3. **Install missing package:**
```powershell
pip install ultralytics

# Or install all dependencies
pip install -r requirements.txt
```

**Prevention:**
- Always activate virtual environment before running scripts
- Add environment activation to launch scripts
- Use `python -m pip` to ensure correct environment

---

### Issue 2: CUDA Installation Errors

**Symptoms:**
```
AssertionError: Torch not compiled with CUDA enabled
RuntimeError: No CUDA GPUs are available
```

**Diagnosis:**
- PyTorch not installed with CUDA support
- CUDA drivers not installed
- CUDA version mismatch

**Solution:**

1. **Check CUDA availability:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

2. **Install PyTorch with CUDA (Windows):**
```powershell
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

3. **Verify CUDA drivers:**
```powershell
nvidia-smi
```

Should show GPU info and driver version.

4. **If no GPU available, switch to CPU:**
```yaml
# In configs/pipeline_config.yaml
device: cpu
```

**Prevention:**
- Check [PyTorch installation guide](https://pytorch.org/get-started/locally/)
- Match PyTorch CUDA version with installed NVIDIA drivers
- Test CUDA after installation with `torch.cuda.is_available()`

---

### Issue 3: PaddleOCR Installation Fails

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement paddlepaddle-gpu
```

**Diagnosis:**
- PaddlePaddle not available for current OS/Python version
- Need specific CUDA version

**Solution:**

1. **Check Python version (PaddlePaddle requires 3.7-3.9):**
```powershell
python --version
```

2. **Install PaddlePaddle (Windows, CUDA 11.2):**
```powershell
pip install paddlepaddle-gpu==2.5.1 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
```

3. **If GPU version fails, use CPU version:**
```powershell
pip install paddlepaddle
```

4. **Install PaddleOCR:**
```powershell
pip install paddleocr
```

5. **Update config to use CPU OCR:**
```yaml
# In configs/pipeline_config.yaml
recognition:
  use_gpu: false
```

**Prevention:**
- Check [PaddlePaddle compatibility](https://www.paddlepaddle.org.cn/en)
- Use Python 3.8 or 3.9 for best compatibility
- Test OCR loading after installation

---

## Detection Problems

### Issue 4: No Plates Detected in Video

**Symptoms:**
- Processing runs but no bounding boxes shown
- Empty tracks dictionary returned
- Logs show `Detections: 0`

**Diagnosis:**
- Confidence threshold too high
- Model not loaded correctly
- Input video corrupted or wrong format

**Solution:**

1. **Lower confidence threshold:**
```yaml
# In configs/pipeline_config.yaml
detection:
  confidence_threshold: 0.15  # Down from 0.25
```

2. **Test on known good image:**
```python
from src.detection.model import load_detection_model, detect_plates
import cv2

config = {'model_path': 'models/detection/yolov8_finetuned_v2_best.pt', 
          'device': 'cuda', 'confidence_threshold': 0.15}
model = load_detection_model(config)

# Test image
image = cv2.imread('tests/data/sample_images/test_plate.jpg')
detections = detect_plates(image, model, config)
print(f"Detections: {len(detections)}")
```

3. **Verify model file:**
```powershell
# Check model exists
Test-Path models/detection/yolov8_finetuned_v2_best.pt

# Check file size (should be >6MB)
(Get-Item models/detection/yolov8_finetuned_v2_best.pt).Length / 1MB
```

4. **Test with baseline model:**
```python
# Use pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # Download automatically
```

5. **Visualize detections:**
```python
# Save annotated image
results = model(image)
results[0].save('debug_detection.jpg')
```

**Prevention:**
- Always test detection on sample images before video processing
- Keep confidence threshold configurable
- Log detection counts for monitoring

---

### Issue 5: Too Many False Positives

**Symptoms:**
- Detecting non-plate objects (signs, windows, etc.)
- Many tracks with no recognized text
- Low precision, high recall

**Diagnosis:**
- Confidence threshold too low
- Model not fine-tuned for GTA V plates
- Need better NMS (Non-Maximum Suppression)

**Solution:**

1. **Increase confidence threshold:**
```yaml
detection:
  confidence_threshold: 0.35  # Up from 0.25
```

2. **Adjust NMS threshold:**
```yaml
detection:
  iou_threshold: 0.5  # Up from 0.45 (more aggressive NMS)
```

3. **Filter by detection size:**
```python
# In detect_plates()
MIN_PLATE_AREA = 500  # pixels
detections = [d for d in detections if d['area'] > MIN_PLATE_AREA]
```

4. **Use fine-tuned model:**
```yaml
detection:
  model_path: models/detection/yolov8_finetuned_v2_best.pt  # Not yolov8n.pt
```

5. **Filter by aspect ratio:**
```python
# Plates typically have width/height ratio between 2:1 and 5:1
MIN_ASPECT_RATIO = 1.5
MAX_ASPECT_RATIO = 6.0
detections = [d for d in detections 
              if MIN_ASPECT_RATIO < d['width']/d['height'] < MAX_ASPECT_RATIO]
```

**Prevention:**
- Fine-tune model on GTA V specific data
- Use validation metrics (precision/recall) to tune thresholds
- Implement post-processing filters

---

### Issue 6: Detection Too Slow

**Symptoms:**
- Processing speed < 5 FPS
- GPU utilization low
- CPU bottleneck

**Diagnosis:**
- Using CPU instead of GPU
- Image size too large
- Inefficient frame reading

**Solution:**

1. **Verify GPU is being used:**
```python
import torch
print(f"Using device: {model.device}")
```

2. **Reduce image size:**
```yaml
detection:
  image_size: 640  # Down from 1056
```

3. **Use batch processing:**
```python
# Process multiple frames at once
frames = [frame1, frame2, frame3]
results = model(frames, batch=4)
```

4. **Enable TensorRT optimization (if available):**
```python
model.export(format='engine')  # Export to TensorRT
model = YOLO('model.engine')    # Load optimized model
```

5. **Skip frames:**
```python
# Process every Nth frame
if frame_count % 2 == 0:
    detections = detect_plates(frame, model, config)
```

**Performance Benchmarks:**
- **Expected Speed**: 
  - GPU (RTX 3060): 30-50 FPS (640px), 15-25 FPS (1056px)
  - CPU (i7): 3-5 FPS (640px), 1-2 FPS (1056px)

**Prevention:**
- Always use GPU when available
- Profile code to identify bottlenecks
- Use appropriate image size for accuracy vs speed trade-off

---

## Recognition Issues

### Issue 7: OCR Returns Empty Text

**Symptoms:**
- Plates detected but `text = None` or `text = ""`
- OCR confidence very low
- Logs show `No valid text recognized`

**Diagnosis:**
- Plate crop too small or blurry
- PaddleOCR not loaded correctly
- Language setting incorrect

**Solution:**

1. **Inspect plate crops:**
```python
# Save crops for inspection
cv2.imwrite(f'debug/crop_{track_id}.jpg', crop)
```

2. **Enhance image before OCR:**
```yaml
preprocessing:
  use_clahe: true
  use_sharpening: true
  use_denoising: true
```

3. **Adjust OCR parameters:**
```yaml
recognition:
  lang: 'en'  # English only
  det_db_box_thresh: 0.3  # Lower detection threshold
  rec_batch_num: 6        # Process more characters at once
```

4. **Test OCR directly:**
```python
from src.recognition.model import load_ocr_model, recognize_text
import cv2

config = {'use_gpu': True, 'lang': 'en'}
ocr_model = load_ocr_model(config)

crop = cv2.imread('debug/crop_123.jpg')
text, conf = recognize_text(crop, ocr_model, config)
print(f"Text: {text}, Confidence: {conf}")
```

5. **Check minimum crop size:**
```python
# Ensure crop is at least 32x32 pixels
h, w = crop.shape[:2]
if h < 32 or w < 32:
    logger.warning(f"Crop too small: {w}x{h}")
```

**Prevention:**
- Use preprocessing to enhance plate images
- Log crop sizes and OCR results for debugging
- Validate OCR model loads correctly at startup

---

### Issue 8: Incorrect Character Recognition

**Symptoms:**
- OCR returns wrong characters (e.g., `0` instead of `O`)
- Common confusions: `8`/`B`, `5`/`S`, `1`/`I`
- Text partially correct

**Diagnosis:**
- OCR confusion between similar characters
- Need post-processing correction
- Insufficient training data for certain characters

**Solution:**

1. **Apply confusion correction:**
```python
# In src/recognition/ocr_correction.py
def correct_common_confusions(text):
    """Correct common OCR errors in license plates"""
    corrections = {
        '0': 'O',  # Zero to letter O
        'Q': 'O',  # Q to O
        '1': 'I',  # One to letter I
        '8': 'B',  # Eight to B (in certain positions)
        '5': 'S',  # Five to S
    }
    # Apply context-aware corrections
    return corrected_text
```

2. **Use GTA V plate format validation:**
```python
# GTA V plates typically: 12ABC345 format
import re

def validate_plate_format(text):
    """Check if text matches GTA V plate pattern"""
    pattern = r'^[0-9]{2}[A-Z]{3}[0-9]{3}$'  # Example pattern
    return re.match(pattern, text) is not None
```

3. **Implement voting mechanism:**
```python
# In PlateTrack class
def update_text(self, new_text, confidence):
    """Update text with voting (most common wins)"""
    self.text_history.append(new_text)
    
    if len(self.text_history) >= 3:
        # Use most common text
        from collections import Counter
        self.text = Counter(self.text_history).most_common(1)[0][0]
```

4. **Fine-tune OCR on GTA V plates:**
```bash
# Use PaddleOCR training tools
# See: https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/training_en.md
```

**Prevention:**
- Implement post-processing correction rules
- Collect ground truth data for OCR fine-tuning
- Use multiple OCR readings and voting

---

### Issue 9: OCR Too Slow

**Symptoms:**
- OCR takes >100ms per plate
- Overall FPS drops significantly
- GPU utilization spikes during OCR

**Diagnosis:**
- Running OCR on every frame
- Using GPU for small batches (inefficient)
- OCR model too large

**Solution:**

1. **Increase OCR interval:**
```yaml
tracking:
  ocr_interval: 60  # Run OCR every 60 frames (up from 30)
```

2. **Use tracking-based OCR triggering:**
```python
def should_run_ocr(self, config):
    # Only run OCR when necessary
    if self.ocr_confidence > 0.8:
        return False  # Skip if high confidence exists
    return True
```

3. **Optimize PaddleOCR settings:**
```yaml
recognition:
  use_gpu: true
  enable_mkldnn: true      # Enable Intel MKL-DNN
  cpu_threads: 4           # Parallel CPU threads
  use_tensorrt: true       # Use TensorRT (if available)
```

4. **Use lighter OCR model:**
```yaml
recognition:
  det_model: 'ch_PP-OCRv3_det'  # Lighter detection model
  rec_model: 'en_PP-OCRv3_rec'  # Lighter recognition model
```

5. **Profile OCR timing:**
```python
import time
start = time.time()
text, conf = recognize_text(crop, ocr_model, config)
elapsed = time.time() - start
logger.debug(f"OCR took {elapsed*1000:.1f}ms")
```

**Performance Benchmarks:**
- **Expected Speed**:
  - GPU: 20-50ms per plate
  - CPU: 50-150ms per plate

**Prevention:**
- Always use OCR interval > 1
- Implement smart OCR triggering based on tracking
- Cache OCR results per track

---

## Tracking Problems

### Issue 10: Tracks Lost Frequently

**Symptoms:**
- New track ID assigned every few frames for same plate
- Low `recognized_count` despite many detections
- Tracks disappear and reappear

**Diagnosis:**
- `max_age` too low (tracks expire too fast)
- `min_hits` too high (tracks require too many detections)
- IOU threshold too strict

**Solution:**

1. **Increase `max_age`:**
```yaml
tracking:
  max_age: 50  # Keep tracks alive for 50 frames without detection (up from 30)
```

2. **Decrease `min_hits`:**
```yaml
tracking:
  min_hits: 2  # Require only 2 detections to confirm track (down from 3)
```

3. **Relax IOU threshold:**
```yaml
tracking:
  iou_threshold: 0.3  # More lenient matching (down from 0.45)
```

4. **Debug track lifecycle:**
```python
for track_id, track in tracks.items():
    logger.debug(f"Track {track_id}: age={track.age}, "
                 f"hits={track.hits}, "
                 f"time_since_update={track.time_since_update}")
```

5. **Visualize track IDs:**
```python
# Draw track ID on frame
cv2.putText(frame, f"ID:{track_id}", (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
```

**Prevention:**
- Tune tracking parameters on validation videos
- Monitor track lifecycle metrics
- Use visualization to verify tracking continuity

---

### Issue 11: Multiple Tracks for Same Plate

**Symptoms:**
- Same plate assigned different track IDs
- Duplicate recognitions in results
- Inconsistent tracking behavior

**Diagnosis:**
- IOU threshold too high (not matching overlapping boxes)
- Bounding box jitter causing failed associations
- Tracker initialization issue

**Solution:**

1. **Lower IOU threshold:**
```yaml
tracking:
  iou_threshold: 0.2  # More aggressive association
```

2. **Enable bounding box smoothing:**
```python
def smooth_bbox(self, new_bbox, alpha=0.3):
    """Exponential moving average for bbox"""
    if self.bbox is None:
        self.bbox = new_bbox
    else:
        self.bbox = alpha * new_bbox + (1 - alpha) * self.bbox
```

3. **Use center distance metric:**
```python
def center_distance(bbox1, bbox2):
    """Distance between bbox centers"""
    cx1, cy1 = (bbox1[:2] + bbox1[2:]) / 2
    cx2, cy2 = (bbox2[:2] + bbox2[2:]) / 2
    return np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)

# Associate if center distance < threshold
if center_distance(det_bbox, track_bbox) < 50:
    associate(det, track)
```

4. **Implement duplicate detection:**
```python
def remove_duplicate_tracks(tracks):
    """Merge tracks with same text and close positions"""
    # Group tracks by text
    text_groups = defaultdict(list)
    for tid, track in tracks.items():
        if track.text:
            text_groups[track.text].append((tid, track))
    
    # Merge close tracks with same text
    for text, group in text_groups.items():
        if len(group) > 1:
            # Keep oldest track, merge others
            oldest = min(group, key=lambda x: x[1].first_seen)
            # ... merge logic ...
```

**Prevention:**
- Use robust distance metrics (IOU + center distance)
- Implement bbox smoothing
- Add duplicate track detection

---

## GUI Issues

### Issue 12: GUI Freezes During Processing

**Symptoms:**
- UI unresponsive after clicking "Start"
- Streamlit shows "Running..." indefinitely
- Can't click Stop button

**Diagnosis:**
- Processing not running in background thread
- Main thread blocked by long operation
- Missing `st.rerun()` calls

**Solution:**

1. **Ensure threaded processing:**
```python
# gui/utils/pipeline_wrapper.py
def start_processing(self, video_path, config):
    """Start processing in background thread"""
    thread = threading.Thread(
        target=self.process_video_threaded,
        args=(video_path, config),
        daemon=True  # Important: daemon thread
    )
    thread.start()
    return thread
```

2. **Add regular reruns:**
```python
# gui/app.py
while st.session_state.processing:
    # Update UI
    update_display()
    
    # Trigger rerun
    time.sleep(0.1)
    st.rerun()
```

3. **Check daemon flag:**
```python
# Thread must be daemon to not block app exit
thread = threading.Thread(..., daemon=True)
```

4. **Debug thread state:**
```python
import threading
logger.debug(f"Active threads: {threading.active_count()}")
for t in threading.enumerate():
    logger.debug(f"Thread: {t.name}, alive={t.is_alive()}")
```

**Prevention:**
- Always use background threads for long operations
- Call `st.rerun()` periodically
- Use daemon threads for automatic cleanup

---

### Issue 13: GUI Shows Outdated Results

**Symptoms:**
- Video finishes but UI shows old frame
- Metrics not updating
- Logs not appearing

**Diagnosis:**
- Session state not syncing with thread results
- Queue not being polled
- Missing state updates

**Solution:**

1. **Poll result queue regularly:**
```python
# In main loop
try:
    result = st.session_state.result_queue.get(timeout=0.1)
    # Update session state with result
    st.session_state.latest_frame = result['frame']
    st.session_state.tracks = result['tracks']
except queue.Empty:
    pass
```

2. **Force UI refresh:**
```python
if st.session_state.processing:
    st.rerun()
```

3. **Clear cache after processing:**
```python
# After processing completes
st.cache_data.clear()
st.cache_resource.clear()
```

4. **Debug queue state:**
```python
logger.debug(f"Queue size: {st.session_state.result_queue.qsize()}")
```

**Prevention:**
- Regularly poll queues for results
- Update session state immediately
- Implement heartbeat mechanism

---

### Issue 14: "Too Many Reruns" Error

**Symptoms:**
```
StreamlitAPIException: Too many reruns. Please contact support.
```

**Diagnosis:**
- Infinite rerun loop
- State change triggering reruns without exit condition
- Missing guards around `st.rerun()`

**Solution:**

1. **Add rerun guards:**
```python
# Track last rerun time
if 'last_rerun' not in st.session_state:
    st.session_state.last_rerun = time.time()

# Only rerun if enough time passed
if time.time() - st.session_state.last_rerun > 0.1:
    st.session_state.last_rerun = time.time()
    st.rerun()
```

2. **Use conditional reruns:**
```python
# Only rerun if processing
if st.session_state.get('processing', False):
    st.rerun()
```

3. **Limit rerun frequency:**
```python
# Max 10 reruns per second
time.sleep(0.1)
st.rerun()
```

4. **Remove unnecessary reruns:**
```python
# Don't rerun after every state update
# Only rerun when UI needs refreshing
```

**Prevention:**
- Always add time delays before reruns
- Use conditional rerun logic
- Avoid state changes that trigger automatic reruns

---

## Performance Issues

### Issue 15: High Memory Usage

**Symptoms:**
- Memory usage grows over time
- System becomes slow
- Out of memory errors

**Diagnosis:**
- Memory leaks in tracking
- Accumulated frames in queue
- Large cached models

**Solution:**

1. **Limit track history:**
```python
# In PlateTrack class
MAX_HISTORY_LENGTH = 100

def update(self, bbox, confidence):
    self.history.append(bbox)
    if len(self.history) > MAX_HISTORY_LENGTH:
        self.history.pop(0)  # Remove oldest
```

2. **Clear old tracks:**
```python
# Periodically remove inactive tracks
if frame_count % 100 == 0:
    tracks = {tid: t for tid, t in tracks.items() 
              if t.is_active or t.age < max_age}
```

3. **Limit queue size:**
```python
result_queue = queue.Queue(maxsize=10)  # Limit to 10 items
```

4. **Clear GPU cache:**
```python
import torch
if frame_count % 100 == 0:
    torch.cuda.empty_cache()
```

5. **Monitor memory:**
```python
import psutil
process = psutil.Process()
logger.info(f"Memory: {process.memory_info().rss / 1024**2:.1f} MB")
```

**Expected Memory Usage:**
- **GPU**: 2-4 GB (models + inference)
- **CPU**: 1-2 GB (video frames + tracking data)

**Prevention:**
- Implement maximum sizes for all collections
- Regularly clean up old data
- Profile memory usage

---

### Issue 16: Low FPS During Processing

**Symptoms:**
- Processing FPS < 10
- Video takes too long to process
- GPU utilization low

**Diagnosis:**
- I/O bottleneck (reading/writing video)
- Inefficient frame processing
- CPU-GPU transfer overhead

**Solution:**

1. **Profile bottlenecks:**
```python
import time

# Profile each stage
t1 = time.time()
frame = read_frame()
read_time = time.time() - t1

t2 = time.time()
detections = detect_plates(frame, model, config)
detect_time = time.time() - t2

# ... profile other stages ...
logger.info(f"Read: {read_time*1000:.1f}ms, "
            f"Detect: {detect_time*1000:.1f}ms")
```

2. **Optimize video reading:**
```python
# Use ffmpeg for faster decoding
import ffmpeg
stream = ffmpeg.input(video_path).output('pipe:', format='rawvideo', pix_fmt='bgr24')
```

3. **Batch processing:**
```python
# Accumulate frames and process in batches
frame_batch = []
if len(frame_batch) >= BATCH_SIZE:
    results = model(frame_batch)
```

4. **Reduce preprocessing:**
```yaml
preprocessing:
  use_clahe: false       # Disable expensive operations
  use_denoising: false
```

5. **Skip frames:**
```yaml
pipeline:
  frame_skip: 2  # Process every 2nd frame
```

**Performance Targets:**
- **Real-time (30 FPS)**: Possible on high-end GPU (RTX 3080+)
- **Fast processing (15-20 FPS)**: Mid-range GPU (RTX 3060)
- **Acceptable (10-15 FPS)**: Entry-level GPU (GTX 1660)

**Prevention:**
- Profile code to identify bottlenecks
- Use appropriate hardware for requirements
- Balance accuracy vs speed

---

## Configuration Errors

### Issue 17: `KeyError` When Loading Config

**Symptoms:**
```python
KeyError: 'detection'
```

**Diagnosis:**
- Configuration file incomplete
- Config schema changed
- Typo in config key

**Solution:**

1. **Validate config against schema:**
```python
python scripts/diagnostics/validate_config.py configs/pipeline_config.yaml
```

2. **Use default config as template:**
```powershell
cp configs/pipeline_config.yaml configs/my_config.yaml
# Edit my_config.yaml
```

3. **Add config validation:**
```python
def load_config(path):
    config = yaml.safe_load(open(path))
    
    # Validate required keys
    required_keys = ['detection', 'recognition', 'tracking']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key: {key}")
    
    return config
```

4. **Provide defaults:**
```python
def get_config_value(config, key, default):
    """Get config value with fallback"""
    return config.get(key, default)

# Usage
conf_thresh = get_config_value(config['detection'], 
                                'confidence_threshold', 0.25)
```

**Prevention:**
- Always validate config files before use
- Use schema validation (JSON Schema, Pydantic)
- Provide sensible defaults

---

### Issue 18: Invalid Configuration Values

**Symptoms:**
- Unexpected behavior
- Errors like "confidence must be 0-1"
- Model fails to load

**Diagnosis:**
- Configuration values out of valid range
- Wrong data types
- Path errors

**Solution:**

1. **Add value validation:**
```python
def validate_config(config):
    """Validate config values"""
    # Check confidence threshold
    conf = config['detection']['confidence_threshold']
    if not 0 <= conf <= 1:
        raise ValueError(f"Confidence must be 0-1, got {conf}")
    
    # Check model path
    model_path = config['detection']['model_path']
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Check device
    device = config['device']
    if device not in ['cuda', 'cpu']:
        raise ValueError(f"Invalid device: {device}")
```

2. **Use validation script:**
```powershell
python scripts/diagnostics/validate_config.py configs/pipeline_config.yaml
```

Output should show validation results and suggestions.

3. **Type checking:**
```python
from typing import Dict, Any

def load_config(path: str) -> Dict[str, Any]:
    """Load and validate configuration"""
    # Use Pydantic for type checking
    from pydantic import BaseModel, validator
    
    class Config(BaseModel):
        confidence_threshold: float
        
        @validator('confidence_threshold')
        def check_range(cls, v):
            if not 0 <= v <= 1:
                raise ValueError('Must be 0-1')
            return v
```

**Prevention:**
- Use schema validation
- Add type hints
- Provide example configs

---

## Model Loading Errors

### Issue 19: Model File Not Found

**Symptoms:**
```
FileNotFoundError: models/detection/yolov8_finetuned_v2_best.pt
```

**Diagnosis:**
- Model not downloaded
- Wrong file path
- File moved/deleted

**Solution:**

1. **Check model exists:**
```powershell
Test-Path models/detection/yolov8_finetuned_v2_best.pt
```

2. **Download model:**
```powershell
# If model is hosted online
Invoke-WebRequest -Uri "https://..." -OutFile models/detection/yolov8_finetuned_v2_best.pt

# Or use download script
python models/detection/download_model.py
```

3. **Use absolute path:**
```python
import os
model_path = os.path.abspath('models/detection/yolov8_finetuned_v2_best.pt')
```

4. **Fallback to default model:**
```python
def load_detection_model(config):
    model_path = config['detection']['model_path']
    
    if not os.path.exists(model_path):
        logger.warning(f"Model not found: {model_path}, using default")
        model_path = 'yolov8n.pt'  # Use pretrained model
    
    return YOLO(model_path)
```

**Prevention:**
- Include model download instructions in README
- Add model verification to setup script
- Use relative paths consistently

---

### Issue 20: Model Loading Fails

**Symptoms:**
```
RuntimeError: Error loading model
Segmentation fault (core dumped)
```

**Diagnosis:**
- Corrupted model file
- Version mismatch (ultralytics/YOLO)
- Insufficient memory

**Solution:**

1. **Verify model file integrity:**
```powershell
# Check file size
(Get-Item models/detection/yolov8_finetuned_v2_best.pt).Length / 1MB
# Should be >6 MB

# Try loading directly
python -c "from ultralytics import YOLO; YOLO('models/detection/yolov8_finetuned_v2_best.pt')"
```

2. **Update ultralytics:**
```powershell
pip install --upgrade ultralytics
```

3. **Re-download model:**
```powershell
# Backup old model
mv models/detection/yolov8_finetuned_v2_best.pt models/detection/backup.pt

# Download fresh copy
python models/detection/download_model.py
```

4. **Check memory:**
```python
import psutil
mem = psutil.virtual_memory()
logger.info(f"Available memory: {mem.available / 1024**3:.1f} GB")
if mem.available < 2e9:  # Less than 2GB
    logger.warning("Low memory, may fail to load model")
```

**Prevention:**
- Use checksum validation for downloads
- Keep ultralytics version pinned
- Test model loading in isolation

---

## Video Processing Issues

### Issue 21: Video Won't Open

**Symptoms:**
```
cv2.error: OpenCV(4.x) error
Video capture failed
```

**Diagnosis:**
- Unsupported video codec
- Corrupted video file
- Missing codec libraries

**Solution:**

1. **Check video file:**
```powershell
# Check file exists
Test-Path path/to/video.mp4

# Check file size
(Get-Item path/to/video.mp4).Length / 1MB
```

2. **Test with ffmpeg:**
```powershell
ffmpeg -i path/to/video.mp4
```

Shows codec info and any errors.

3. **Convert video:**
```powershell
# Convert to compatible format
ffmpeg -i input.mp4 -c:v libx264 -crf 23 -c:a aac output.mp4
```

4. **Install codec libraries:**
```powershell
# For Windows, install K-Lite Codec Pack
# Or use ffmpeg-python
pip install ffmpeg-python
```

5. **Use alternative reader:**
```python
# Use ffmpeg instead of OpenCV
import ffmpeg

probe = ffmpeg.probe(video_path)
video_stream = next((s for s in probe['streams'] 
                     if s['codec_type'] == 'video'), None)
```

**Prevention:**
- Use standard codecs (H.264, AAC)
- Validate videos before processing
- Provide codec requirements in docs

---

### Issue 22: Output Video Corrupted

**Symptoms:**
- Output video won't play
- Missing frames
- Wrong colors/resolution

**Diagnosis:**
- Incorrect codec settings
- FPS mismatch
- VideoWriter not properly released

**Solution:**

1. **Use correct codec:**
```python
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'x264'
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
```

2. **Match input video specs:**
```python
# Get input video properties
cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Use same for output
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
```

3. **Release writer properly:**
```python
try:
    # Write frames
    out.write(frame)
finally:
    out.release()  # Always release
```

4. **Use ffmpeg for encoding:**
```python
# Write raw frames to ffmpeg
import ffmpeg
process = (
    ffmpeg
    .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}')
    .output(output_path, vcodec='libx264', pix_fmt='yuv420p')
    .overwrite_output()
    .run_async(pipe_stdin=True)
)

# Write frames
process.stdin.write(frame.tobytes())
```

**Prevention:**
- Always match input/output specs
- Use reliable codecs (libx264)
- Properly release resources

---

## Environment Issues

### Issue 23: Import Errors in Different Environments

**Symptoms:**
```
ModuleNotFoundError: No module named 'src'
ImportError: attempted relative import beyond top-level package
```

**Diagnosis:**
- Package not installed in editable mode
- PYTHONPATH not set correctly
- Running from wrong directory

**Solution:**

1. **Install package in editable mode:**
```powershell
pip install -e .
```

This allows importing `src` from anywhere.

2. **Set PYTHONPATH:**
```powershell
$env:PYTHONPATH = "D:\Felix_stuff\ALPR_GTA5"
```

3. **Use absolute imports:**
```python
# Instead of:
from ..detection import load_model

# Use:
from src.detection.model import load_detection_model
```

4. **Run from project root:**
```powershell
cd D:\Felix_stuff\ALPR_GTA5
python scripts/process_video.py
```

**Prevention:**
- Always install package in editable mode
- Use absolute imports
- Add setup instructions to README

---

### Issue 24: Version Conflicts

**Symptoms:**
```
ERROR: Cannot install package due to conflicting dependencies
```

**Diagnosis:**
- Dependency version conflicts
- pip resolver issues
- Incompatible package versions

**Solution:**

1. **Use clean environment:**
```powershell
# Delete old environment
Remove-Item -Recurse -Force .venv

# Create new environment
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. **Install dependencies one by one:**
```powershell
# Install core dependencies first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
pip install paddlepaddle-gpu paddleocr
pip install streamlit opencv-python

# Then other dependencies
pip install -r requirements.txt
```

3. **Pin versions:**
```
# requirements.txt
torch==2.0.1
ultralytics==8.0.196
paddleocr==2.6.1
streamlit==1.28.0
```

4. **Check compatibility:**
```powershell
pip check
```

**Prevention:**
- Use pinned dependency versions
- Test on clean environments
- Document known version conflicts

---

## Additional Resources

- **Configuration Guide**: [docs/configuration_guide.md](configuration_guide.md)
- **Developer Guide**: [docs/developer_guide.md](developer_guide.md)
- **API Reference**: [docs/api_reference.md](api_reference.md)
- **Project Structure**: [docs/project_structure.md](project_structure.md)

---

**Still having issues?** 

1. Check logs in `outputs/logs/` for detailed error messages
2. Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`
3. Search existing issues: [GitHub Issues](https://github.com/xiashuidaolaoshuren/ALPR_GTAV/issues)
4. Create new issue with:
   - Error message
   - Configuration used
   - System specs (GPU, OS, Python version)
   - Steps to reproduce

---

*Last updated: November 14, 2025*

# Developer Guide

Complete guide for developers working on the GTA V ALPR system. Covers architecture, design decisions, development workflow, and contribution guidelines.

## Table of Contents

- [System Architecture](#system-architecture)
- [GUI Architecture](#gui-architecture)
- [Design Decisions](#design-decisions)
- [Contributing Guidelines](#contributing-guidelines)
- [Development Workflow](#development-workflow)
- [Extension Points](#extension-points)
- [Testing](#testing)
- [Debugging](#debugging)

---

## System Architecture

### High-Level Overview

The system follows a **modular pipeline architecture** with clean separation of concerns:

```
┌──────────────────────────────────────────────────────┐
│                   User Interfaces                    │
│  ┌─────────────────────┐  ┌────────────────────────┐ │
│  │   Streamlit GUI     │  │   Command-Line (CLI)   │ │
│  │  - Interactive UI   │  │  - Batch processing    │ │
│  │  - Real-time viz    │  │  - Script automation   │ │
│  └──────────┬──────────┘  └───────────┬────────────┘ │
└─────────────┼─────────────────────────┼──────────────┘
              │                         │
              └────────┬────────────────┘
                       ↓
         ┌──────────────────────────────┐
         │      ALPRPipeline (Core)     │
         │                              │
         │  ┌────────┐  ┌────────────┐  │
         │  │ Config │  │ Logging    │  │
         │  └────────┘  └────────────┘  │
         └───────────┬──────────────────┘
                     ↓
     ┌──────────────────────────────────┐
     │      Processing Modules          │
     │                                  │
     │  ┌──────────┐  ┌──────────────┐  │
     │  │ Detection│  │  Recognition │  │
     │  │ (YOLOv8) │  │ (PaddleOCR)  │  │
     │  └──────────┘  └──────────────┘  │
     │                                  │
     │  ┌───────────┐  ┌──────────────┐  │
     │  │ Tracking  │  │Preprocessing │  │
     │  │(ByteTrack)│  │ (Enhance)    │  │
     │  └───────────┘  └──────────────┘  │
     └──────────────────────────────────┘
                     ↓
              ┌──────────────┐
              │   Outputs    │
              │ - Logs       │
              │ - Results    │
              │ - Videos     │
              └──────────────┘
```

### Core Pipeline Flow

**Input**: Video frame (numpy array, BGR format)

**Stage 1: Detection + Tracking**
```python
results = model.track(frame, conf=0.25, iou=0.45, persist=True)
# Returns: Bounding boxes + Track IDs from ByteTrack
```

**Stage 2: Track Management**
```python
for detection in results:
    if track_id not in tracks:
        tracks[track_id] = PlateTrack(...)  # Create new track
    else:
        tracks[track_id].update(bbox, confidence)  # Update existing
```

**Stage 3: Conditional OCR**
```python
if track.should_run_ocr(config):
    crop = frame[y1:y2, x1:x2]
    text, conf = recognize_text(crop, ocr_model, config)
    track.update_text(text, conf)
```

**Stage 4: Track Cleanup**
```python
# Remove lost tracks older than max_age
tracks = {id: t for id, t in tracks.items() 
          if t.is_active or t.age < max_age}
```

**Output**: Dictionary of `{track_id: PlateTrack}`

### Module Responsibilities

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| **detection** | Plate location | `load_detection_model()`, `detect_plates()` |
| **recognition** | Text extraction | `load_ocr_model()`, `recognize_text()` |
| **tracking** | Identity management | `PlateTrack`, `should_run_ocr()` |
| **preprocessing** | Image enhancement | `apply_clahe()`, `apply_sharpening()` |
| **pipeline** | Orchestration | `ALPRPipeline`, `process_frame()` |
| **utils** | Common utilities | `load_config()`, `read_video_frames()` |

---

## GUI Architecture

### Component-Based Design

The GUI follows a **component-based architecture** with Streamlit:

```
app.py (Main Application)
  │
  ├── initialize_session_state()
  │   └── Creates all st.session_state variables
  │
  ├── Sidebar (ControlPanel)
  │   ├── FileUploader
  │   ├── ParameterSliders
  │   │   ├── confidence_threshold
  │   │   ├── iou_threshold
  │   │   └── ocr_interval
  │   ├── DeviceSelector (CUDA/CPU)
  │   └── ProcessingButtons (Start/Stop)
  │
  ├── Main Area
  │   ├── Header (Title + Instructions)
  │   ├── VideoDisplay
  │   │   └── st.empty() → Updated with annotated frames
  │   └── InfoPanel (Tabs)
  │       ├── Status Tab
  │       │   ├── Active Tracks Metric
  │       │   ├── Recognized Count Metric
  │       │   └── Latest Recognitions Table
  │       └── Logs Tab
  │           └── Real-time log stream
  │
  └── Background Thread
      └── GUIPipelineWrapper.process_video_threaded()
```

### Threading Model

**Why Threading?**
- Streamlit reruns entire script on each interaction
- Video processing is long-running (minutes)
- Need non-blocking UI for responsive experience

**Implementation:**
```python
# Main thread (Streamlit)
def start_processing():
    thread = threading.Thread(
        target=wrapper.process_video_threaded,
        args=(video_path, config, result_queue, stop_event),
        daemon=True
    )
    thread.start()
    st.session_state.processing = True

# Background thread
def process_video_threaded(video_path, config, queue, stop_event):
    pipeline = ALPRPipeline(config)
    for frame in read_frames(video_path):
        if stop_event.is_set():
            break
        tracks = pipeline.process_frame(frame)
        queue.put({'frame': frame, 'tracks': tracks})
```

**Communication:**
- **Queue**: `queue.Queue()` for thread-safe result passing
- **Stop Event**: `threading.Event()` for graceful shutdown
- **Session State**: `st.session_state.processing` flag

**Synchronization:**
```python
# Main thread polls queue
while st.session_state.processing:
    try:
        result = result_queue.get(timeout=0.1)
        # Update display with result
    except queue.Empty:
        time.sleep(0.01)
    st.rerun()  # Trigger UI update
```

### Session State Management

Streamlit's session state persists data across reruns:

**Critical Variables:**
```python
st.session_state.pipeline        # Cached ALPRPipeline (shared globally)
st.session_state.processing      # Boolean: Is processing active?
st.session_state.video_handler   # VideoHandler for file management
st.session_state.video_info      # Dict: {fps, frame_count, duration}
st.session_state.config          # Current pipeline configuration
st.session_state.results         # List of recent recognitions
st.session_state.log_handler     # StreamlitLogHandler instance
```

**Caching Strategy:**
```python
@st.cache_resource
def load_cached_pipeline(config_str, device):
    """Models loaded once, shared across all users"""
    return ALPRPipeline(config_path)
```

**Important**: `@st.cache_resource` creates a **global singleton** - all users share the same cached models. This is optimal for GPU memory usage.

### GUI Component Details

#### ControlPanel (`gui/components/control_panel.py`)

**Responsibilities:**
- Render sidebar controls
- Handle file uploads
- Manage configuration sliders
- Start/stop processing

**Key Methods:**
```python
def render():
    """Render entire control panel"""
    video_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
    
    # Configuration sliders
    conf = st.slider("Confidence", 0.0, 1.0, 0.25)
    iou = st.slider("IOU", 0.0, 1.0, 0.45)
    
    # Buttons
    if st.button("Start Processing"):
        start_processing_callback()
```

**State Management:**
- Updates `st.session_state.config` with slider values
- Validates uploaded files before processing
- Disables controls during processing

#### VideoDisplay (`gui/components/video_display.py`)

**Responsibilities:**
- Display annotated video frames
- Render bounding boxes and text overlays
- Show real-time FPS

**Key Methods:**
```python
def update_display(frame, tracks):
    """Update display with new frame and tracks"""
    annotated = draw_tracks_on_frame(frame, tracks)
    st.image(annotated, channels='BGR', use_column_width=True)
```

**Optimization:**
- Uses `st.empty()` placeholder for in-place updates (no flickering)
- Frame skipping: Only update every Nth frame (configurable)
- Resolution scaling: Downscale large frames for faster rendering

#### InfoPanel (`gui/components/info_panel.py`)

**Responsibilities:**
- Show processing statistics
- Display latest recognitions
- Stream real-time logs

**Tabs:**

**Status Tab:**
```python
# Metrics
st.metric("Active Tracks", len(active_tracks))
st.metric("Recognized", recognized_count)

# Latest recognitions table
st.dataframe(latest_recognitions)
```

**Logs Tab:**
```python
# Real-time log streaming
logs = st.session_state.log_handler.get_logs()
for log in logs[-50:]:  # Last 50 logs
    st.text(f"[{log.level}] {log.message}")
```

**Auto-scroll:**
- Uses JavaScript injection to scroll to bottom
- Updates every rerun automatically

---

## Design Decisions

### 1. Why Streamlit for GUI?

**Rationale:**
- **Rapid Development**: Build interactive UIs with pure Python (no HTML/CSS/JS)
- **Built-in Widgets**: File upload, sliders, buttons, tabs out-of-the-box
- **Real-time Updates**: Automatic reruns on state changes
- **Deployment**: Easy to deploy (streamlit share, Docker, Electron wrapper)

**Trade-offs:**
- ❌ Limited customization compared to React/Vue
- ❌ Full-page reruns can be inefficient (mitigated with caching)
- ✅ Perfect for data science demos and internal tools
- ✅ Python-only stack (no context switching)

**Alternatives Considered:**
- **Gradio**: Simpler but less flexible
- **Dash (Plotly)**: More complex, better for dashboards
- **Qt (PyQt5)**: Native GUI but steeper learning curve

### 2. Threading vs Multiprocessing

**Choice: Threading**

**Rationale:**
- **Shared Memory**: No need to serialize/deserialize frames
- **Simplicity**: Easier to debug than multiprocessing
- **GIL**: Not a bottleneck (video I/O and GPU ops release GIL)
- **Streamlit Compatibility**: Works well with session state

**When Multiprocessing Would Be Better:**
- CPU-intensive Python code (we use GPU for inference)
- Processing multiple videos in parallel
- Need true isolation between processes

### 3. Frame Skipping Strategies

**Challenge**: Processing every frame is slow, UI updates cause lag

**Solution: Adaptive Frame Skipping**

```python
# Only update UI every N frames
if frame_count % UPDATE_INTERVAL == 0:
    queue.put(result)

# Dynamically adjust based on FPS
if current_fps < target_fps:
    UPDATE_INTERVAL += 1
```

**Benefits:**
- ✅ Smoother UI experience
- ✅ Faster processing (less time spent on UI updates)
- ⚠️ Trade-off: May miss some plates in fast-moving scenes

**Configuration:**
- `gui/config.py`: `FRAME_SKIP_RATIO = 2` (process every 2nd frame)
- `gui/config.py`: `UI_UPDATE_INTERVAL = 5` (update UI every 5 frames)

### 4. Caching Strategy

**Problem**: Model loading is slow (5-10 seconds), users don't want to wait

**Solution: Multi-Level Caching**

**Level 1: Global Model Cache** (`@st.cache_resource`)
```python
@st.cache_resource
def load_cached_pipeline(config_str, device):
    return ALPRPipeline(config)
```
- Models loaded once per config/device combination
- Shared across all users
- Persists across sessions

**Level 2: Video Metadata Cache** (`@st.cache_data`)
```python
@st.cache_data
def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    return {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': cap.get(cv2.CAP_PROP_FRAME_COUNT)
    }
```
- Fast metadata extraction
- Per-file caching

**Level 3: Session State** (user-specific data)
```python
st.session_state.results  # Current session's results
st.session_state.config   # User's current config
```

**Cache Invalidation:**
- Automatic when config string changes (hash-based)
- Manual via "Clear Cache" button in sidebar

### 5. OCR Optimization via Tracking

**Problem**: Running OCR every frame is slow (10-20ms per plate)

**Solution: Intelligent OCR Triggering**

```python
def should_run_ocr(self, config):
    """Determine if OCR should run for this track"""
    # Always run on first detection
    if self.age == 0:
        return True
    
    # Run if no text recognized yet
    if not self.text:
        return True
    
    # Run if enough frames have passed
    if self.frames_since_last_ocr >= config['ocr_interval']:
        return True
    
    # Skip if recent high-confidence text exists
    if self.ocr_confidence > config['ocr_confidence_threshold']:
        return False
    
    return False
```

**Impact:**
- ✅ 70% reduction in OCR calls
- ✅ 3-5x faster processing
- ⚠️ May miss text changes (mitigated by re-running OCR periodically)

**Configuration:**
- `ocr_interval`: Frames between OCR runs (default: 30)
- `ocr_confidence_threshold`: Minimum confidence to skip OCR (default: 0.7)

### 6. Error Handling Strategy

**Philosophy**: Fail fast in development, graceful degradation in production

**Levels of Error Handling:**

**Level 1: Input Validation**
```python
def process_frame(self, frame):
    if not isinstance(frame, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(frame)}")
    if frame.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {frame.shape}")
```

**Level 2: Module-Level Try-Catch**
```python
try:
    results = model.track(frame, ...)
except Exception as e:
    logger.error(f"Detection failed: {e}")
    return []  # Return empty results, continue processing
```

**Level 3: Pipeline-Level Recovery**
```python
def process_frame(self, frame):
    try:
        # ... processing ...
    except Exception as e:
        logger.error(f"Frame processing failed: {e}")
        self.error_count += 1
        if self.error_count > MAX_ERRORS:
            raise  # Fail after too many errors
        return {}  # Return empty tracks
```

**GUI-Specific:**
```python
try:
    pipeline.process_frame(frame)
except Exception as e:
    st.error(f"Processing error: {e}")
    st.stop()  # Stop processing, show error to user
```

---

## Contributing Guidelines

### Code Style

**PEP 8 Compliance:**
- 4 spaces for indentation (no tabs)
- 100-character line limit
- 2 blank lines between top-level definitions
- Use `snake_case` for functions and variables
- Use `PascalCase` for classes

**Docstrings:**
- Google-style docstrings for all public functions/classes
- Include: description, parameters, returns, raises, examples

**Example:**
```python
def recognize_text(image: np.ndarray, model: PaddleOCR, 
                   config: dict) -> Tuple[Optional[str], float]:
    """
    Recognize text from a plate crop using PaddleOCR.
    
    Args:
        image: Cropped plate image in BGR format
        model: Loaded PaddleOCR model
        config: Recognition configuration dict
    
    Returns:
        Tuple[Optional[str], float]: (recognized_text, confidence) 
            or (None, 0.0) if no valid text
    
    Raises:
        ValueError: If image is invalid
    
    Example:
        >>> text, conf = recognize_text(crop, ocr_model, config)
        >>> if text:
        ...     print(f"Recognized: {text} (conf={conf:.3f})")
    """
    ...
```

**Type Hints:**
- Use type hints for all function signatures
- Import from `typing` for complex types
- Use `Optional[T]` for nullable values

### Adding New Features

#### Adding a New Preprocessing Method

1. **Create function in `src/preprocessing/image_enhancement.py`:**
```python
def apply_bilateral_filter(image: np.ndarray, d: int = 9, 
                           sigma_color: float = 75, 
                           sigma_space: float = 75) -> np.ndarray:
    """
    Apply bilateral filter for edge-preserving smoothing.
    
    Args:
        image: Input image
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space
    
    Returns:
        Filtered image
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
```

2. **Add to configuration schema (`configs/pipeline_config.yaml`):**
```yaml
preprocessing:
  use_bilateral_filter: true
  bilateral_d: 9
  bilateral_sigma_color: 75
  bilateral_sigma_space: 75
```

3. **Integrate into preprocessing pipeline:**
```python
def preprocess_plate(image, config):
    if config.get('use_bilateral_filter', False):
        image = apply_bilateral_filter(
            image,
            d=config['bilateral_d'],
            sigma_color=config['bilateral_sigma_color'],
            sigma_space=config['bilateral_sigma_space']
        )
    return image
```

4. **Add tests:**
```python
def test_bilateral_filter():
    image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    filtered = apply_bilateral_filter(image)
    assert filtered.shape == image.shape
    assert filtered.dtype == np.uint8
```

5. **Document in configuration guide:**
```markdown
#### `use_bilateral_filter`
- **Type**: Boolean
- **Description**: Apply edge-preserving smoothing
- **Recommended**: `true` for noisy images
```

#### Adding a New GUI Component

1. **Create component file (`gui/components/new_component.py`):**
```python
import streamlit as st

class NewComponent:
    """Description of component"""
    
    def __init__(self):
        pass
    
    def render(self):
        """Render component UI"""
        st.subheader("New Component")
        # Add widgets here
```

2. **Import in `gui/app.py`:**
```python
from gui.components.new_component import NewComponent
```

3. **Initialize in session state:**
```python
if 'new_component' not in st.session_state:
    st.session_state.new_component = NewComponent()
```

4. **Render in layout:**
```python
st.session_state.new_component.render()
```

### Pull Request Process

1. **Create Feature Branch:**
```bash
git checkout -b feature/your-feature-name
```

2. **Make Changes:**
- Follow code style guidelines
- Add/update tests
- Update documentation

3. **Test Locally:**
```bash
# Run tests
pytest tests/

# Check code style
flake8 src/ gui/ scripts/

# Run type checking (optional)
mypy src/
```

4. **Commit:**
```bash
git add .
git commit -m "feat: Add bilateral filter preprocessing"
```

**Commit Message Format:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Build/config changes

5. **Push and Create PR:**
```bash
git push origin feature/your-feature-name
```

6. **PR Description Template:**
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
```

---

## Development Workflow

### Setting Up Development Environment

1. **Clone Repository:**
```bash
git clone https://github.com/xiashuidaolaoshuren/ALPR_GTAV.git
cd ALPR_GTAV
```

2. **Create Virtual Environment:**
```bash
# Using venv
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate   # Linux/Mac

# Or using conda
conda create -n alpr python=3.9
conda activate alpr
```

3. **Install Dependencies:**
```bash
# Install all dependencies
pip install -r requirements.txt

# Or use UV for faster installation
uv pip install -r requirements.txt

# Install development tools
pip install pytest flake8 black mypy
```

4. **Download Models:**
```bash
# Detection model (if not already present)
python models/detection/download_model.py

# OCR models are downloaded automatically on first use
```

5. **Verify Installation:**
```bash
# Run tests
pytest tests/ -v

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Test GUI
python -m streamlit run gui/app.py
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/unit/test_detection.py

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run only fast tests (skip slow integration tests)
pytest tests/ -m "not slow"

# Verbose output
pytest tests/ -v -s
```

### Debugging Techniques

#### Debugging Pipeline

**1. Enable Debug Logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**2. Inspect Intermediate Results:**
```python
# In process_frame()
logger.debug(f"Detections: {len(detections)}")
logger.debug(f"Tracks: {list(tracks.keys())}")

# Save intermediate images
cv2.imwrite(f"debug/frame_{frame_count}.jpg", frame)
cv2.imwrite(f"debug/crop_{track_id}.jpg", crop)
```

**3. Use Breakpoints:**
```python
# In VS Code, set breakpoint and run debugger
# Or use pdb
import pdb; pdb.set_trace()
```

#### Debugging GUI

**1. Streamlit Debug Mode:**
```bash
python -m streamlit run gui/app.py --logger.level=debug
```

**2. Session State Inspection:**
```python
# In any Streamlit page
st.write("Session State:", st.session_state)
```

**3. Thread Debugging:**
```python
import threading
logger.debug(f"Active threads: {threading.enumerate()}")
logger.debug(f"Main thread: {threading.main_thread()}")
```

**4. Performance Profiling:**
```python
import time

start = time.time()
# ... code to profile ...
elapsed = time.time() - start
logger.info(f"Elapsed: {elapsed:.3f}s")
```

#### Common Debug Scenarios

**Issue: Model not loading**
```python
# Check paths
import os
print(os.path.exists('models/detection/yolov8_finetuned_v2_best.pt'))

# Check GPU
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

**Issue: OCR not working**
```python
# Test OCR directly
from src.recognition.model import load_ocr_model, recognize_text

config = {'use_gpu': True, 'lang': 'en', ...}
ocr_model = load_ocr_model(config)

# Test on sample image
import cv2
image = cv2.imread('test_plate.jpg')
text, conf = recognize_text(image, ocr_model, config)
print(f"Result: {text} (conf={conf})")
```

**Issue: Tracking not working**
```python
# Inspect track lifecycle
for track_id, track in tracks.items():
    print(f"Track {track_id}:")
    print(f"  Age: {track.age}")
    print(f"  Text: {track.text}")
    print(f"  Confidence: {track.ocr_confidence}")
    print(f"  Should run OCR: {track.should_run_ocr(config)}")
```

---

## Extension Points

### 1. Adding a New Detection Model

**Steps:**

1. **Create model loader:**
```python
# src/detection/model.py
def load_custom_model(model_path, device):
    """Load custom detection model"""
    model = CustomDetector(model_path)
    model.to(device)
    return model
```

2. **Add configuration option:**
```yaml
# configs/pipeline_config.yaml
detection:
  model_type: yolov8  # or 'custom'
  custom_model_path: models/detection/custom_model.pth
```

3. **Update pipeline initialization:**
```python
if config['detection']['model_type'] == 'yolov8':
    model = load_detection_model(...)
elif config['detection']['model_type'] == 'custom':
    model = load_custom_model(...)
```

### 2. Adding a New OCR Engine

**Steps:**

1. **Create OCR adapter:**
```python
# src/recognition/engines/tesseract_adapter.py
class TesseractAdapter:
    def __init__(self, config):
        self.config = config
        pytesseract.tesseract_cmd = config['tesseract_path']
    
    def recognize(self, image):
        text = pytesseract.image_to_string(image)
        return text, 1.0  # Tesseract doesn't return confidence
```

2. **Register adapter:**
```python
# src/recognition/model.py
OCR_ENGINES = {
    'paddleocr': PaddleOCRAdapter,
    'tesseract': TesseractAdapter,
    'easyocr': EasyOCRAdapter
}

def load_ocr_model(config):
    engine_type = config.get('engine', 'paddleocr')
    engine_class = OCR_ENGINES[engine_type]
    return engine_class(config)
```

### 3. Adding a New Tracker Algorithm

**Steps:**

1. **Implement tracker interface:**
```python
# src/tracking/algorithms/custom_tracker.py
class CustomTracker:
    def update(self, detections, frame_id):
        """Update tracks with new detections"""
        # Implement tracking logic
        return tracks
```

2. **Register tracker:**
```python
# src/tracking/tracker.py
TRACKERS = {
    'bytetrack': ByteTrack,
    'botsort': BotSort,
    'iou': IOUTracker,
    'custom': CustomTracker
}
```

3. **Update configuration:**
```yaml
tracking:
  tracker_type: custom
  custom_param1: value1
```

---

## Testing

### Test Organization

```
tests/
├── unit/                  # Fast, isolated tests
│   ├── test_detection.py
│   ├── test_recognition.py
│   ├── test_tracking.py
│   └── test_preprocessing.py
├── integration/           # Module interaction tests
│   ├── test_pipeline.py
│   └── test_gui.py
└── data/                  # Test fixtures
    ├── sample_images/
    └── sample_videos/
```

### Writing Tests

**Unit Test Example:**
```python
import pytest
import numpy as np
from src.preprocessing.image_enhancement import apply_clahe

def test_apply_clahe():
    """Test CLAHE enhancement"""
    # Arrange
    image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    
    # Act
    enhanced = apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8))
    
    # Assert
    assert enhanced.shape == image.shape
    assert enhanced.dtype == np.uint8
    assert not np.array_equal(enhanced, image)  # Image was modified

def test_apply_clahe_invalid_input():
    """Test CLAHE with invalid input"""
    with pytest.raises(ValueError):
        apply_clahe(None)
```

**Integration Test Example:**
```python
def test_complete_pipeline(sample_video_path):
    """Test end-to-end pipeline processing"""
    # Arrange
    pipeline = ALPRPipeline('configs/pipeline_config.yaml')
    
    # Act
    cap = cv2.VideoCapture(sample_video_path)
    ret, frame = cap.read()
    tracks = pipeline.process_frame(frame)
    
    # Assert
    assert isinstance(tracks, dict)
    assert all(isinstance(t, PlateTrack) for t in tracks.values())
```

### Test Fixtures

```python
@pytest.fixture
def sample_image():
    """Provide sample test image"""
    return cv2.imread('tests/data/sample_images/test_plate.jpg')

@pytest.fixture
def mock_pipeline(mocker):
    """Mock pipeline for testing"""
    mock = mocker.MagicMock()
    mock.process_frame.return_value = {}
    return mock
```

### Running Specific Tests

```bash
# Run tests matching pattern
pytest tests/ -k "test_detection"

# Run tests with markers
pytest tests/ -m "integration"

# Run failed tests only
pytest tests/ --lf

# Stop on first failure
pytest tests/ -x
```

---

## Debugging

### Common Issues

#### Issue 1: CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size in config
2. Use smaller image size (640 instead of 1056)
3. Close other GPU applications
4. Switch to CPU if necessary

```python
# Check GPU memory
import torch
print(torch.cuda.memory_allocated() / 1024**3, "GB")
print(torch.cuda.memory_reserved() / 1024**3, "GB")

# Clear cache
torch.cuda.empty_cache()
```

#### Issue 2: GUI Not Responsive

**Symptoms:**
- UI freezes during processing
- Buttons not working

**Solutions:**
1. Ensure processing runs in background thread
2. Check `st.rerun()` is called regularly
3. Reduce UI update frequency

```python
# Debug: Check if processing is stuck
if time.time() - last_update > 10:
    logger.warning("Processing may be stuck")
```

#### Issue 3: Tracks Lost Too Quickly

**Symptoms:**
- Plates detected but immediately lost
- Low recognized count

**Solutions:**
1. Increase `max_age` in tracking config
2. Lower `min_hits` requirement
3. Adjust `iou_threshold`

```python
# Debug: Monitor track lifecycle
logger.debug(f"Track {track_id}: age={track.age}, active={track.is_active}")
```

---

## Additional Resources

- **API Reference**: [docs/api_reference.md](api_reference.md)
- **Configuration Guide**: [docs/configuration_guide.md](configuration_guide.md)
- **User Guide**: [docs/user_guide.md](user_guide.md)
- **Troubleshooting**: [docs/troubleshooting.md](troubleshooting.md)
- **Project Structure**: [docs/project_structure.md](project_structure.md)

---

*Last updated: November 14, 2025*

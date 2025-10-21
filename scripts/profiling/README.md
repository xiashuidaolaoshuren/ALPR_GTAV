# ALPR Pipeline Profiling Tools

This directory contains tools for profiling and optimizing the ALPR pipeline performance.

## Overview

The profiling tools measure various performance metrics to identify bottlenecks and optimization opportunities:

- **FPS (Frames Per Second)**: Overall throughput
- **OCR Call Frequency**: Number of OCR operations per 100 frames
- **GPU Memory Usage**: GPU memory consumption
- **CPU Usage**: CPU utilization percentage
- **Frame Processing Time**: Time breakdown for each frame

## Quick Start

### Basic Profiling

Profile the pipeline with default configuration on 500 frames:

```bash
python scripts/profiling/profile_pipeline.py \
    --video outputs/raw_footage/day_clear/day_clear_airport_01.mp4 \
    --config configs/pipeline_config.yaml \
    --num-frames 500
```

### Test Multiple OCR Intervals

Compare performance with different `ocr_interval` settings:

```bash
python scripts/profiling/profile_pipeline.py \
    --video outputs/raw_footage/day_clear/day_clear_airport_01.mp4 \
    --config configs/pipeline_config.yaml \
    --num-frames 500 \
    --ocr-intervals 1,15,30,60
```

### Test Preprocessing Variations

Compare preprocessing configurations:

```bash
python scripts/profiling/profile_pipeline.py \
    --video outputs/raw_footage/day_clear/day_clear_airport_01.mp4 \
    --config configs/pipeline_config.yaml \
    --num-frames 500 \
    --test-preprocessing
```

## Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--video` | Path to input video file | Required |
| `--config` | Path to pipeline configuration YAML | Required |
| `--output-dir` | Output directory for profiling results | `outputs/profiling` |
| `--num-frames` | Number of frames to process | 500 |
| `--ocr-intervals` | Comma-separated list of ocr_interval values | None |
| `--test-preprocessing` | Test preprocessing variations | False |

## Output Files

Profiling generates the following outputs in `outputs/profiling/`:

### 1. Configuration Files

- `config_<test_name>.yaml`: Configuration used for each test

### 2. Comparison CSV

`optimization_comparison.csv`: Side-by-side comparison of all test results.

**Columns:**
- `test_name`: Test configuration name
- `fps`: Average frames per second
- `avg_frame_time_ms`: Average frame processing time (ms)
- `ocr_calls_per_100_frames`: OCR call frequency
- `total_ocr_calls`: Total OCR calls made
- `avg_gpu_memory_gb`: Average GPU memory (GB)
- `max_gpu_memory_gb`: Peak GPU memory (GB)
- `avg_cpu_percent`: Average CPU usage (%)
- `max_cpu_percent`: Peak CPU usage (%)
- `frames_processed`: Total frames processed

### 3. Performance Report

`performance_report.md`: Comprehensive Markdown report with:
- Executive summary
- Detailed metrics for each configuration
- Comparison table
- Optimization recommendations

## Understanding the Metrics

### FPS (Frames Per Second)

- **Target:** 15+ FPS for real-time processing
- **Higher is better**
- Calculated as: `1000 / avg_frame_time_ms`

### OCR Calls per 100 Frames

- **Target:** <10-15 calls per 100 frames
- **Lower is better** (reduced computational cost)
- Without tracking: ~100 calls per 100 frames (OCR every frame)
- With tracking: ~3-10 calls per 100 frames (80-90% reduction)

### GPU Memory Usage

- **Target:** <4GB for RTX 3070Ti
- **Lower is better**
- Typical range: 0.5-2GB for YOLOv8n + PaddleOCR

### CPU Usage

- **Target:** <80% average
- Monitor to ensure balanced GPU/CPU utilization
- High CPU usage may indicate preprocessing bottleneck

## Optimization Strategies

### 1. Reduce OCR Calls (Tracking Optimization)

**Configuration:** Increase `tracking.ocr_interval`

```yaml
tracking:
  ocr_interval: 30  # Run OCR every 30 frames instead of every frame
```

**Impact:**
- ✅ Reduces OCR calls by 80-90%
- ✅ Improves FPS by 30-50%
- ⚠️ May miss rapidly changing plates

**Recommended values:**
- **Conservative:** 15 frames (high accuracy, moderate performance)
- **Balanced:** 30 frames (good accuracy, good performance)
- **Aggressive:** 60 frames (lower accuracy, high performance)

### 2. Preprocessing Optimization

**Configuration:** Disable unnecessary preprocessing

```yaml
preprocessing:
  enable_enhancement: false  # Disable if image quality is good
  use_clahe: false           # CLAHE adds ~20ms per plate
```

**Impact:**
- ✅ Reduces preprocessing time by 50-70%
- ✅ Improves FPS by 10-20%
- ⚠️ May reduce OCR accuracy on low-quality images

### 3. Confidence Threshold Tuning

**Configuration:** Adjust OCR confidence threshold

```yaml
tracking:
  ocr_confidence_threshold: 0.8  # Higher = fewer re-runs
```

**Impact:**
- ✅ Reduces redundant OCR calls
- ✅ Improves FPS slightly
- ⚠️ May keep low-confidence results

### 4. Frame Sampling

**Usage:** In batch processing script

```bash
python scripts/process_video.py \
    --input video.mp4 \
    --config configs/pipeline_config.yaml \
    --sample-rate 2  # Process every 2nd frame
```

**Impact:**
- ✅ Doubles FPS
- ⚠️ May miss fast-moving plates

## Interpreting Results

### Example: Good Performance

```
FPS: 18.5
OCR per 100 Frames: 8.2
Avg GPU Memory: 1.2GB
Avg CPU Usage: 45%
```

- ✅ Exceeds 15 FPS target
- ✅ Low OCR frequency (tracking works well)
- ✅ Moderate resource usage
- **Recommendation:** Production-ready

### Example: Poor Performance

```
FPS: 6.3
OCR per 100 Frames: 95.0
Avg GPU Memory: 3.5GB
Avg CPU Usage: 85%
```

- ❌ Below 15 FPS target
- ❌ High OCR frequency (tracking not working)
- ⚠️ High resource usage
- **Recommendation:** Investigate tracking configuration, reduce preprocessing

## Performance Targets

For **RTX 3070Ti** (or equivalent):

| Metric | Target | Acceptable | Poor |
|--------|--------|------------|------|
| FPS | 15+ | 10-15 | <10 |
| OCR per 100 Frames | <10 | 10-20 | >20 |
| GPU Memory | <2GB | 2-4GB | >4GB |
| CPU Usage | <60% | 60-80% | >80% |

## Troubleshooting

### Low FPS (<10)

1. **Check preprocessing:** Disable `use_clahe` and `enable_enhancement`
2. **Check OCR frequency:** Should be <15 per 100 frames
3. **Reduce frame sampling:** Process fewer frames
4. **Check GPU utilization:** Use `nvidia-smi` to verify GPU usage

### High OCR Frequency (>20 per 100 frames)

1. **Increase ocr_interval:** Try 30, 60, or higher
2. **Check tracking stability:** Verify tracks aren't lost frequently
3. **Adjust confidence threshold:** Increase `ocr_confidence_threshold`

### High GPU Memory (>4GB)

1. **Use smaller model:** YOLOv8n instead of YOLOv8m/l/x
2. **Reduce image size:** Lower `detection.img_size`
3. **Check for memory leaks:** Monitor over time

### Tracking Not Working (OCR every frame)

1. **Check ByteTrack configuration:** Verify `tracker_type: bytetrack`
2. **Check should_run_ocr logic:** See `src/tracking/tracker.py`
3. **Verify ocr_interval setting:** Should be >1

## Advanced Usage

### Custom Test Configurations

Create a custom test by modifying the configuration programmatically:

```python
from profile_pipeline import run_profiling_test, load_config
from pathlib import Path

config = load_config('configs/pipeline_config.yaml')
config['tracking']['ocr_interval'] = 45
config['preprocessing']['use_clahe'] = True

report = run_profiling_test(
    video_path=Path('video.mp4'),
    config=config,
    num_frames=500,
    test_name='custom_test',
    output_dir=Path('outputs/profiling')
)
```

### Analyzing Results Programmatically

```python
import pandas as pd

# Load comparison CSV
df = pd.read_csv('outputs/profiling/optimization_comparison.csv')

# Find best FPS configuration
best_fps = df.loc[df['fps'].idxmax()]
print(f"Best FPS: {best_fps['test_name']} ({best_fps['fps']:.2f} FPS)")

# Find lowest OCR frequency
best_ocr = df.loc[df['ocr_calls_per_100_frames'].idxmin()]
print(f"Lowest OCR: {best_ocr['test_name']} ({best_ocr['ocr_calls_per_100_frames']:.2f}/100)")
```

## Contributing

When adding new profiling metrics:

1. Update `PipelineProfiler.profile_frame()` to collect the metric
2. Update `PipelineProfiler.generate_report()` to aggregate the metric
3. Update `save_comparison_csv()` to include the metric in CSV
4. Update `generate_markdown_report()` to display the metric
5. Update this README with the new metric description

## References

- [Pipeline Implementation](../../src/pipeline/alpr_pipeline.py)
- [Tracking Module](../../src/tracking/tracker.py)
- [Configuration Guide](../../configs/pipeline_config.yaml)
- [Batch Processing Script](../process_video.py)

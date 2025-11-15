# Frame Sampling Guide for Video Processing

## Overview

The `process_video.py` script includes a `--sample-rate` parameter that controls how many frames are processed. Understanding this parameter is critical for balancing processing speed vs. detection coverage.

## Frame Sampling Explained

### What is Frame Sampling?

Frame sampling skips frames to reduce processing time. For example:
- `--sample-rate 1`: Process every frame (100% coverage, slowest)
- `--sample-rate 5`: Process every 5th frame (20% coverage, 5x faster)
- `--sample-rate 30`: Process every 30th frame (3.3% coverage, 30x faster)

### Video Playback Speed

The output video FPS is automatically adjusted to maintain correct temporal relationships:
- **Input**: 30 FPS video, 2667 frames
- **sample-rate=30**: Processes 89 frames → Output at 1 FPS = 89 seconds (correct!)
- **sample-rate=1**: Processes 2667 frames → Output at 30 FPS = 89 seconds (same duration)

## Detection Coverage vs. Speed Trade-off

### Test Results (day_clear_airport_01.mp4)

| Sample Rate | Frames Processed | Coverage | Plates Detected | Recognition Rate | Processing Time |
|-------------|------------------|----------|-----------------|------------------|-----------------|
| 30          | 89 / 2667        | 3.3%     | 2               | 50% (1/2)        | ~25s |
| 5           | 534 / 2667       | 20%      | ~12-15 (est.)   | TBD              | ~150s (est.) |
| 1           | 2667 / 2667      | 100%     | ~30-40 (est.)   | TBD              | ~400s (est.) |

**Key Finding:** Frame sampling can miss plates! In our test:
- Frames 0, 30, 60, 90, 120: All had detectable plates
- Only 2 plates were recognized because only 89 frames were processed
- Most plates appeared in skipped frames

## Recommendations

### For Development/Testing
```bash
# Fast preview with high sampling
python scripts/process_video.py --input video.mp4 --sample-rate 30 --output preview.mp4
```

### For Production/Analysis
```bash
# Full coverage, process all frames
python scripts/process_video.py --input video.mp4 --output full_results.mp4 --export-json results.json
```

### Balanced Approach
```bash
# Good balance: 20% coverage, reasonable speed
python scripts/process_video.py --input video.mp4 --sample-rate 5 --output results.mp4 --export-json results.json
```

## Detection Performance

### Current Model Performance (yolov8n.pt)
- **Detection rate**: ~86% on processed frames
- **Confidence range**: 0.62 - 0.76
- **Recognition rate**: ~50% (OCR successful when plate is clear and large enough)

### Improving Detection Coverage

1. **Reduce sample-rate**: More frames = more plates detected
   ```bash
   --sample-rate 5  # Process every 5th frame
   ```

2. **Lower confidence threshold**: Catch more marginal detections
   ```yaml
   # configs/pipeline_config.yaml
   detection:
     confidence_threshold: 0.20  # Reduced from 0.25
   ```

3. **Process all frames**: Maximum coverage (slow but thorough)
   ```bash
   # No --sample-rate parameter
   ```

## Common Issues

### "Many plates are missed"
**Cause**: High sample-rate skipping frames with plates  
**Solution**: Reduce `--sample-rate` or remove it entirely

### "Video plays too fast"
**Cause**: Bug was fixed in latest version (output FPS now adjusted correctly)  
**Check**: Ensure you're using the latest code with FPS fix

### "OCR fails with 'tuple index out of range'"
**Cause**: Bug was fixed - grayscale images now converted to BGR  
**Solution**: Update to latest code

## Performance Optimization Tips

1. **Use GPU**: Ensure `device: cuda` in `configs/pipeline_config.yaml`
2. **Batch processing**: Process multiple videos in parallel
3. **Adjust tracking interval**: Reduce OCR frequency in config
4. **Skip video output**: Use `--no-video` flag if only need JSON/CSV

## Example Commands

```bash
# Quick test (3% coverage, ~25 seconds)
python scripts/process_video.py \
    --input raw_footage/video.mp4 \
    --output quick_preview.mp4 \
    --sample-rate 30 \
    --export-json quick_results.json

# Balanced (20% coverage, ~3 minutes)
python scripts/process_video.py \
    --input raw_footage/video.mp4 \
    --output balanced.mp4 \
    --sample-rate 5 \
    --export-json balanced_results.json \
    --export-csv balanced_data.csv

# Full analysis (100% coverage, ~7 minutes)
python scripts/process_video.py \
    --input raw_footage/video.mp4 \
    --output full_analysis.mp4 \
    --export-json full_results.json \
    --export-csv full_data.csv

# Data only, no video (fastest)
python scripts/process_video.py \
    --input raw_footage/video.mp4 \
    --sample-rate 5 \
    --no-video \
    --export-json results.json \
    --export-csv results.csv
```

## Expected Output

### JSON Format
```json
{
  "statistics": {
    "frames_processed": 534,
    "total_frames": 2667,
    "sample_rate": 5,
    "plates_detected": 15,
    "plates_recognized": 8,
    "unique_plates": 7
  },
  "frames": [...]
}
```

### CSV Format
```csv
frame,timestamp,track_id,text,ocr_confidence,detection_confidence,x1,y1,x2,y2,age
0,0.00,1,,0.0,0.616,880,450,934,477,0
150,5.00,39,07CSI699,0.967,0.764,1061,533,1177,600,0
```

## Conclusion

Frame sampling is a powerful tool for speeding up video processing, but it trades detection coverage for speed. For production use, process all frames or use a low sample-rate (5-10) to ensure comprehensive plate detection.

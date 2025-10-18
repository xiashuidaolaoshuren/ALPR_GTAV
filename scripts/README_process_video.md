# Batch Video Processing Script

## Overview

`process_video.py` is a command-line tool for processing video files through the complete ALPR pipeline. It performs frame-by-frame license plate detection, tracking, and recognition with real-time progress tracking and multiple export formats.

## Features

- **Frame-by-Frame Processing**: Processes entire videos through the ALPR pipeline
- **Progress Tracking**: Real-time progress bar with tqdm showing processing status
- **Frame Sampling**: Process every Nth frame for faster processing (--sample-rate)
- **Annotated Output**: Generates video with bounding boxes, track IDs, and recognized text
- **Multiple Export Formats**: Export results to JSON and CSV
- **Flexible Options**: Skip video output for faster data-only processing
- **Comprehensive Statistics**: Processing metrics including FPS, detection counts, unique plates
- **Error Resilience**: Gracefully handles errors and continues processing

## Installation

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

Required packages:
- opencv-python
- tqdm (progress bar)
- ultralytics (YOLOv8)
- paddleocr (text recognition)
- pyyaml

## Usage

### Basic Usage

Process a video with default settings:

```bash
python scripts/process_video.py --input video.mp4 --output output.mp4
```

### Advanced Usage

#### Process Every 5th Frame (Faster)

```bash
python scripts/process_video.py --input video.mp4 --output output.mp4 --sample-rate 5
```

#### Export to JSON and CSV

```bash
python scripts/process_video.py \
    --input video.mp4 \
    --output output.mp4 \
    --export-json results.json \
    --export-csv results.csv
```

#### Skip Video Output (Data Only, Faster)

```bash
python scripts/process_video.py \
    --input video.mp4 \
    --no-video \
    --export-json results.json
```

#### Custom Configuration

```bash
python scripts/process_video.py \
    --input video.mp4 \
    --output output.mp4 \
    --config custom_config.yaml
```

## Command-Line Arguments

### Required Arguments

- `--input`, `-i`: Path to input video file

### Optional Arguments

- `--output`, `-o`: Path to output video file (required unless --no-video)
- `--config`, `-c`: Pipeline configuration file (default: configs/pipeline_config.yaml)
- `--sample-rate`, `-s`: Process every Nth frame (default: 1)
- `--no-video`: Skip video output generation (faster)
- `--export-json`: Export results to JSON file
- `--export-csv`: Export results to CSV file
- `--show-track-id`: Show track IDs on output video (default: True)
- `--show-confidence`: Show confidence scores on output video (default: True)

## Output Formats

### JSON Output

Structure:
```json
{
  "statistics": {
    "frames_processed": 89,
    "total_frames": 2667,
    "sample_rate": 30,
    "elapsed_time": 52.94,
    "fps": 1.68,
    "plates_detected": 2,
    "plates_recognized": 0,
    "unique_plates": 0
  },
  "frames": [
    {
      "frame": 0,
      "timestamp": 0.0,
      "tracks": [
        {
          "id": 1,
          "text": "12ABC345",
          "ocr_confidence": 0.95,
          "detection_confidence": 0.87,
          "bbox": [100, 200, 300, 250],
          "age": 5
        }
      ]
    }
  ]
}
```

### CSV Output

Columns:
- `frame`: Frame number
- `timestamp`: Time in seconds
- `track_id`: Unique track identifier
- `text`: Recognized plate text (empty if not recognized)
- `ocr_confidence`: OCR confidence score
- `detection_confidence`: Detection confidence score
- `x1, y1, x2, y2`: Bounding box coordinates
- `age`: Track age in frames

### Video Output

Annotated video with:
- Bounding boxes around detected plates
- Track IDs (if --show-track-id)
- Recognized text below bounding boxes
- Confidence scores (if --show-confidence)

## Performance Tips

### Speed Optimization

1. **Use Frame Sampling**: Process every 5th or 10th frame for 5-10x speedup
   ```bash
   --sample-rate 5
   ```

2. **Skip Video Output**: 2-3x faster when only exporting data
   ```bash
   --no-video
   ```

3. **GPU Acceleration**: Ensure CUDA is available for detection and OCR models

### Quality Optimization

1. **Process All Frames**: Use default sample-rate=1 for best tracking
2. **Adjust Detection Threshold**: Edit `configs/pipeline_config.yaml`
3. **OCR Tuning**: Adjust OCR parameters in configuration

## Examples

### Example 1: Quick Preview (Every 10th Frame)

```bash
python scripts/process_video.py \
    --input outputs/raw_footage/day_clear/day_clear_airport_01.mp4 \
    --output outputs/preview.mp4 \
    --sample-rate 10
```

### Example 2: Full Analysis with Exports

```bash
python scripts/process_video.py \
    --input footage.mp4 \
    --output annotated.mp4 \
    --export-json results.json \
    --export-csv results.csv
```

### Example 3: Data Extraction Only (Fastest)

```bash
python scripts/process_video.py \
    --input footage.mp4 \
    --no-video \
    --sample-rate 5 \
    --export-json quick_results.json
```

## Processing Statistics

The script outputs comprehensive statistics:

- **Frames Processed**: Number of frames actually processed
- **Total Frames**: Total frames in video
- **Sample Rate**: Sampling ratio used
- **Processing Time**: Total elapsed time
- **Processing Speed**: FPS (frames per second)
- **Plates Detected**: Total detection count (may include duplicates)
- **Plates Recognized**: Detections with recognized text
- **Unique Plates**: Count of unique plate texts

## Error Handling

The script includes comprehensive error handling:

- **Video I/O Errors**: Validates input video exists and can be opened
- **Pipeline Errors**: Catches and logs OCR/detection failures, continues processing
- **Keyboard Interrupt**: Gracefully stops and saves progress
- **Resource Cleanup**: Ensures proper cleanup of video readers/writers

## Integration with Pipeline

The script integrates seamlessly with the ALPR pipeline:

1. **ALPRPipeline**: Main orchestration class (Task 19)
2. **VideoReader/VideoWriter**: Reuses existing utilities from src/utils/video_io.py
3. **Visualization**: Uses draw_tracks_on_frame() from src/pipeline/utils.py
4. **Configuration**: Loads settings from pipeline_config.yaml

## Troubleshooting

### Issue: "Failed to open video"

- Check that video file exists and path is correct
- Ensure video codec is supported by OpenCV
- Try re-encoding video with ffmpeg

### Issue: Slow processing

- Use `--sample-rate` to process fewer frames
- Use `--no-video` to skip video output
- Check GPU availability for acceleration

### Issue: OCR errors

- Verify PaddleOCR is correctly installed
- Check `configs/pipeline_config.yaml` recognition settings
- Ensure GPU drivers are up to date

## Related Files

- `src/utils/video_io.py`: VideoReader and VideoWriter classes
- `src/pipeline/alpr_pipeline.py`: Main ALPR pipeline
- `src/pipeline/utils.py`: Visualization utilities
- `configs/pipeline_config.yaml`: Pipeline configuration

## License

Part of the GTA V ALPR project.

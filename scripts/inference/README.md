# Inference Scripts

Image and video inference pipelines for plate detection and recognition.

## Quick Start

### Detect Plates in Single Image
```powershell
python scripts/inference/detect_image.py `
  --image photo.jpg `
  --output outputs/annotated.jpg
```

### Process Video with Full Pipeline
```powershell
python scripts/inference/process_video.py `
  --input video.mp4 `
  --output outputs/processed.mp4 `
  --export-json results.json
```

## Key Scripts

**detect_image.py** - Single-image detection with optional JSON/CSV export.

**detect_video.py** - Video inference with real-time visualization and tracking.

**process_video.py** - Full ALPR pipeline (detection + OCR + tracking).

**recognize_plates.py** - OCR on detected plate images.

## Performance Tips

1. **Use sample-rate:** `--sample-rate 2` doubles speed (skip every other frame)
2. **Use ocr-interval:** `--ocr-interval 15` reduces OCR calls by 15x via tracking
3. **Skip video output:** Only export JSON for 50% faster processing
4. **CPU fallback:** Use `--device cpu` if GPU unavailable

## Best Practices

1. **Validate configuration:** `python scripts/diagnostics/validate_config.py`
2. **Test on small sample:** Use `--max-frames 100` for quick testing
3. **Monitor GPU:** Watch `nvidia-smi` for memory/utilization
4. **Export JSON:** More useful than video for analysis

## Common Issues

**Slow processing:** Use `--sample-rate 2` and `--ocr-interval 15`

**GPU out of memory:** Use `--device cpu` or reduce batch size in config

**Missing video codec:** Install ffmpeg or use MP4 format

## References

- **[Configuration Guide](../../docs/configuration_guide.md)**
- **[Evaluation Scripts](../evaluation/README.md)**

---

*Last updated: November 15, 2025*

# Data Ingestion Scripts

Utilities for video processing, frame extraction, and dataset preparation.

## Quick Start

### Extract Frames from Video
```powershell
python scripts/data_ingestion/extract_frames.py `
  --input outputs/raw_footage/day_clear/video1.mp4 `
  --output outputs/frames/day_clear `
  --fps 5 `
  --quality 95
```

### Batch Process All Raw Footage
```powershell
python scripts/data_ingestion/batch_process_footage.py `
  --input-dir outputs/raw_footage `
  --output-dir outputs/frames `
  --fps 5
```

## Key Scripts

**extract_frames.py** - Core frame extraction with configurable FPS and quality control.

**batch_process_footage.py** - Batch process all raw footage by condition (day/night, clear/rain).

**generate_metadata.py** - Generate dataset metadata files with image information.

**clean_test_images.py** - Remove duplicates and low-quality frames automatically.

**generate_dataset_stats.py** - Compute dataset statistics and quality metrics.

## Best Practices

1. **Frame rate:** Use 5 FPS for general purpose, 2 FPS for speed, 10 FPS for detail
2. **Quality:** Use 95 for annotation, 85 for testing, 75 for training storage
3. **Organization:** Maintain condition structure (day_clear, day_rain, night_clear, night_rain)
4. **Cleanup:** Remove duplicates and low-quality frames before annotation
5. **Versioning:** Keep raw footage separate and version extracted datasets

## References

- **[Data Collection Guide](../../docs/data_collection_strategy.md)**
- **[Annotation Guide](../../docs/annotation_guide.md)**
- **[Annotation Scripts](../annotation/README.md)**

---

*Last updated: November 15, 2025*

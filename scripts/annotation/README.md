# Annotation Scripts

Label Studio integration and annotation format conversion utilities for the GTA V ALPR project.

## Quick Start

### Start Annotation Server
```powershell
python scripts/annotation/start_annotation.py
```

### Convert Label Studio Export to YOLO Format
```powershell
python scripts/annotation/convert_labelstudio_to_yolo.py `
  --input datasets/labelstudio_exports/project-1.json `
  --output datasets/lpr
```

## Key Scripts

**start_annotation.py** - Launch Label Studio with automatic server startup, log streaming, and helpful guidance.

**convert_labelstudio_to_yolo.py** - Convert Label Studio JSON exports to YOLOv8 format with automatic train/val/test splits.

**validate_annotations.py** - Verify annotation quality and completeness.

**merge_datasets.py** - Combine multiple Label Studio exports while handling duplicates.

## Workflow: Annotation to Training

1. Launch Label Studio: `python scripts/annotation/start_annotation.py`
2. Create project and import images
3. Export annotations as JSON
4. Convert to YOLO: `python scripts/annotation/convert_labelstudio_to_yolo.py`
5. Validate: `python scripts/annotation/validate_annotations.py`
6. Train model: `python scripts/training/train_detection.py`

## References

- **[Label Studio Documentation](https://labelstudio.io/docs/)**
- **[Annotation Guide](../../docs/annotation_guide.md)**
- **[Training Scripts](../training/README.md)**

---

*Last updated: November 15, 2025*

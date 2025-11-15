# Evaluation Scripts

Model evaluation and performance analysis tools.

## Quick Start

### Evaluate Detection Model
```powershell
python scripts/evaluation/evaluate_detection.py `
  --model models/detection/yolov8_finetuned_v2_best.pt `
  --data datasets/lpr/data.yaml
```

### Generate Report
```powershell
python scripts/evaluation/generate_evaluation_report.py `
  --detection-results outputs/evaluation/detection_results.json `
  --output outputs/evaluation_report.md
```

## Key Scripts

**evaluate_detection.py** - Compute precision, recall, F1-score, and mAP metrics on test set.

**evaluate_ocr.py** - Evaluate PaddleOCR recognition accuracy on ground truth.

**evaluate_tracking.py** - Analyze tracking performance across video frames.

**generate_evaluation_report.py** - Create comprehensive markdown/HTML evaluation reports.

**visualize_detection_results.py** - Generate precision-recall curves, confusion matrices, and example detections.

## Performance Targets

For GTA V ALPR:
- **Precision:** >95% (false positives are costly)
- **Recall:** >90% (acceptable to miss some plates)
- **F1-Score:** >92%
- **mAP@0.5:** >90%

## Best Practices

1. **Evaluate on test set only** - Not training/validation data
2. **Compare across conditions** - Day, night, clear, rain separately
3. **Use high confidence threshold** - 0.5+ for production accuracy
4. **Save annotated images** - Review false positives/negatives
5. **Generate HTML reports** - Share visualizations with stakeholders

## References

- **[Performance Report](../../outputs/evaluation/detection_evaluation_report.md)**
- **[Training Scripts](../training/README.md)**

---

*Last updated: November 15, 2025*

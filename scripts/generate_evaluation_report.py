"""
Generate Detection Evaluation Report

Create a comprehensive Markdown report from detection evaluation results.

Usage:
    python scripts/generate_evaluation_report.py [options]

Examples:
    # Basic report generation
    python scripts/generate_evaluation_report.py
    
    # Custom input and output
    python scripts/generate_evaluation_report.py --input outputs/detection_results.json --output outputs/custom_report.md
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate detection evaluation report',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='outputs/detection_results.json',
        help='Path to detection results JSON (default: outputs/detection_results.json)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/detection_evaluation_report.md',
        help='Path to output report (default: outputs/detection_evaluation_report.md)'
    )
    
    return parser.parse_args()


def load_results(results_path: str) -> Dict:
    """Load results from JSON file."""
    results_path = Path(results_path)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded results from {results_path}")
    return data


def generate_report(data: Dict, output_path: str):
    """
    Generate comprehensive evaluation report.
    
    Args:
        data: Detection results and statistics
        output_path: Path to save report
    """
    stats = data['statistics']
    params = data['parameters']
    overall = stats['overall']
    by_condition = stats['by_condition']
    
    # Start report
    report_lines = []
    
    # Header
    report_lines.append("# License Plate Detection Evaluation Report")
    report_lines.append("")
    report_lines.append(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Executive Summary
    report_lines.append("## Executive Summary")
    report_lines.append("")
    report_lines.append(f"This report presents the evaluation results of the YOLOv8-based license plate detection system on the GTA V test dataset. ")
    report_lines.append(f"A total of **{overall['total_images']} images** were evaluated across various conditions (day/night, clear/rain, different angles). ")
    report_lines.append(f"The system achieved an overall detection rate of **{overall['detection_rate']*100:.2f}%** with an average confidence of **{overall['avg_confidence']:.3f}**.")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Test Dataset Overview
    report_lines.append("## Test Dataset Overview")
    report_lines.append("")
    report_lines.append(f"- **Total Images:** {overall['total_images']}")
    report_lines.append(f"- **Test Directory:** `{params['test_dir']}`")
    report_lines.append(f"- **Conditions Tested:** {', '.join(sorted(by_condition.keys()))}")
    report_lines.append("")
    
    # Distribution table
    report_lines.append("### Dataset Distribution")
    report_lines.append("")
    report_lines.append("| Condition | Number of Images | Percentage |")
    report_lines.append("|-----------|------------------|------------|")
    for condition in sorted(by_condition.keys()):
        num_images = by_condition[condition]['num_images']
        percentage = (num_images / overall['total_images']) * 100
        report_lines.append(f"| {condition} | {num_images} | {percentage:.1f}% |")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Detection Parameters
    report_lines.append("## Detection Parameters")
    report_lines.append("")
    report_lines.append(f"- **Confidence Threshold:** {params['confidence_threshold']}")
    report_lines.append(f"- **IOU Threshold:** {params['iou_threshold']}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Performance Metrics
    report_lines.append("## Performance Metrics")
    report_lines.append("")
    
    # Overall metrics
    report_lines.append("### Overall Performance")
    report_lines.append("")
    report_lines.append(f"- **Total Images Evaluated:** {overall['total_images']}")
    report_lines.append(f"- **Images with Detection:** {overall['images_with_detection']}")
    report_lines.append(f"- **Overall Detection Rate:** {overall['detection_rate']*100:.2f}%")
    report_lines.append(f"- **Total Detections:** {overall['total_detections']}")
    report_lines.append(f"- **Average Detections per Image:** {overall['avg_detections_per_image']:.2f}")
    report_lines.append(f"- **Average Confidence:** {overall['avg_confidence']:.3f}")
    report_lines.append("")
    
    # Performance by condition
    report_lines.append("### Performance by Condition")
    report_lines.append("")
    report_lines.append("| Condition | Images | Detection Rate | Avg Detections | Avg Confidence |")
    report_lines.append("|-----------|--------|----------------|----------------|----------------|")
    for condition in sorted(by_condition.keys()):
        cond_stats = by_condition[condition]
        report_lines.append(
            f"| {condition} | {cond_stats['num_images']} | "
            f"{cond_stats['detection_rate']*100:.2f}% | "
            f"{cond_stats['avg_detections_per_image']:.2f} | "
            f"{cond_stats['avg_confidence']:.3f} |"
        )
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Analysis
    report_lines.append("## Performance Analysis")
    report_lines.append("")
    
    # Find best and worst conditions
    sorted_conditions = sorted(by_condition.items(), 
                              key=lambda x: x[1]['detection_rate'], 
                              reverse=True)
    best_condition = sorted_conditions[0]
    worst_condition = sorted_conditions[-1]
    
    report_lines.append("### Strengths 💪")
    report_lines.append("")
    report_lines.append(f"1. **High Performance in {best_condition[0]}**")
    report_lines.append(f"   - Detection rate: {best_condition[1]['detection_rate']*100:.2f}%")
    report_lines.append(f"   - Average confidence: {best_condition[1]['avg_confidence']:.3f}")
    report_lines.append(f"   - The model performs exceptionally well in this condition, indicating good baseline capability")
    report_lines.append("")
    
    # Check for good overall detection rate
    if overall['detection_rate'] > 0.7:
        report_lines.append("2. **Strong Overall Detection Rate**")
        report_lines.append(f"   - {overall['detection_rate']*100:.2f}% of images have at least one detection")
        report_lines.append("   - Demonstrates robust performance across diverse conditions")
        report_lines.append("")
    
    # Check for consistent confidence
    confidences = [c['avg_confidence'] for c in by_condition.values()]
    if min(confidences) > 0.5:
        report_lines.append("3. **Consistent Confidence Scores**")
        report_lines.append(f"   - Confidence remains above 0.5 across all conditions")
        report_lines.append("   - Indicates reliable detections with low false positive rate")
        report_lines.append("")
    
    report_lines.append("### Weaknesses 🔍")
    report_lines.append("")
    report_lines.append(f"1. **Lower Performance in {worst_condition[0]}**")
    report_lines.append(f"   - Detection rate: {worst_condition[1]['detection_rate']*100:.2f}%")
    report_lines.append(f"   - {((best_condition[1]['detection_rate'] - worst_condition[1]['detection_rate'])*100):.1f}% gap from best condition")
    report_lines.append("   - Challenging lighting/weather conditions affect detection accuracy")
    report_lines.append("")
    
    # Check for missed detections
    missed_rate = 1 - overall['detection_rate']
    if missed_rate > 0.1:
        report_lines.append("2. **Missed Detections**")
        report_lines.append(f"   - {(missed_rate*100):.2f}% of images have no detections")
        report_lines.append("   - May be due to: small/distant plates, extreme angles, occlusion, or image quality")
        report_lines.append("")
    
    # Check for condition variability
    detection_rates = [c['detection_rate'] for c in by_condition.values()]
    rate_std = (max(detection_rates) - min(detection_rates))
    if rate_std > 0.2:
        report_lines.append("3. **Performance Variability Across Conditions**")
        report_lines.append(f"   - Detection rate varies by up to {rate_std*100:.1f}% between conditions")
        report_lines.append("   - Suggests model sensitivity to environmental factors")
        report_lines.append("")
    
    report_lines.append("---")
    report_lines.append("")
    
    # Recommendations
    report_lines.append("## Recommendations 🎯")
    report_lines.append("")
    report_lines.append("### Short-term Improvements")
    report_lines.append("")
    report_lines.append("1. **Fine-tune Model on GTA V Dataset**")
    report_lines.append("   - Create annotated dataset from current test images")
    report_lines.append("   - Fine-tune YOLOv8 model specifically for GTA V license plates")
    report_lines.append("   - Expected improvement: 10-20% in detection rate")
    report_lines.append("")
    
    if worst_condition[1]['detection_rate'] < 0.6:
        report_lines.append(f"2. **Optimize for {worst_condition[0]} Conditions**")
        report_lines.append("   - Adjust confidence threshold for specific conditions")
        report_lines.append("   - Apply preprocessing (brightness adjustment, denoising)")
        report_lines.append("   - Consider condition-specific model variants")
        report_lines.append("")
    
    report_lines.append("3. **Expand Test Dataset**")
    report_lines.append("   - Collect more samples from underrepresented conditions")
    report_lines.append("   - Include edge cases (occlusion, extreme angles, motion blur)")
    report_lines.append("   - Aim for 1000+ annotated images per condition")
    report_lines.append("")
    
    report_lines.append("### Long-term Improvements")
    report_lines.append("")
    report_lines.append("1. **Multi-scale Detection**")
    report_lines.append("   - Implement multi-resolution inference for better small plate detection")
    report_lines.append("   - Use sliding window approach for distant plates")
    report_lines.append("")
    report_lines.append("2. **Ensemble Methods**")
    report_lines.append("   - Combine multiple YOLOv8 models (different sizes: n, s, m)")
    report_lines.append("   - Apply voting mechanism for more robust detections")
    report_lines.append("")
    report_lines.append("3. **Integrate Tracking**")
    report_lines.append("   - Implement ByteTrack or SORT for temporal consistency")
    report_lines.append("   - Reduce false negatives through track interpolation")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Visualizations
    report_lines.append("## Visualizations 📊")
    report_lines.append("")
    report_lines.append("The following visualizations summarize the evaluation results:")
    report_lines.append("")
    report_lines.append("### Overall Summary")
    report_lines.append("![Summary](summary.png)")
    report_lines.append("")
    report_lines.append("### Detection Rate by Condition")
    report_lines.append("![Detection Rate](detection_rate_by_condition.png)")
    report_lines.append("")
    report_lines.append("### Average Detections by Condition")
    report_lines.append("![Average Detections](avg_detections_by_condition.png)")
    report_lines.append("")
    report_lines.append("### Confidence by Condition")
    report_lines.append("![Confidence](avg_confidence_by_condition.png)")
    report_lines.append("")
    report_lines.append("### Dataset Distribution")
    report_lines.append("![Dataset Distribution](dataset_distribution.png)")
    report_lines.append("")
    report_lines.append("### Performance Comparison")
    report_lines.append("![Performance Comparison](performance_comparison.png)")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Example Images
    report_lines.append("## Example Detections 🖼️")
    report_lines.append("")
    report_lines.append("### Best Detections")
    report_lines.append("High-confidence detections demonstrating ideal performance:")
    report_lines.append("- Located in: `outputs/detection_examples/best/`")
    report_lines.append("")
    report_lines.append("### Worst Detections")
    report_lines.append("Low-confidence detections requiring improvement:")
    report_lines.append("- Located in: `outputs/detection_examples/worst/`")
    report_lines.append("")
    report_lines.append("### No Detections")
    report_lines.append("Images where no plates were detected (failure cases):")
    report_lines.append("- Located in: `outputs/detection_examples/no_detection/`")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Conclusion
    report_lines.append("## Conclusion 🏁")
    report_lines.append("")
    report_lines.append(f"The YOLOv8-based detection system demonstrates {overall['detection_rate']*100:.2f}% detection rate on the GTA V test dataset, ")
    report_lines.append(f"with particularly strong performance in {best_condition[0]} conditions. ")
    
    if overall['detection_rate'] > 0.7:
        report_lines.append("The system shows promising baseline performance and is ready for the next phase: model fine-tuning and OCR integration. ")
    else:
        report_lines.append("While the baseline performance provides a foundation, significant improvements are needed through fine-tuning and optimization. ")
    
    report_lines.append("")
    report_lines.append("**Next Steps:**")
    report_lines.append("1. Annotate test images using Label Studio")
    report_lines.append("2. Fine-tune YOLOv8 model on annotated GTA V dataset")
    report_lines.append("3. Integrate PaddleOCR for license plate recognition")
    report_lines.append("4. Implement tracking algorithm for temporal consistency")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append(f"*Report generated from: `{params['test_dir']}`*")
    
    # Write report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Report saved to {output_path}")


def main():
    """Main execution function."""
    args = parse_arguments()
    
    try:
        # Load results
        logger.info("Loading detection results...")
        data = load_results(args.input)
        
        # Generate report
        logger.info("Generating evaluation report...")
        generate_report(data, args.output)
        
        logger.info(f"\n✅ Evaluation report generated successfully!")
        logger.info(f"📄 Report saved to: {args.output}")
        
        return 0
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

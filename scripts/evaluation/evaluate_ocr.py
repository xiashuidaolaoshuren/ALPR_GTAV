"""
OCR Performance Evaluation Script

Evaluate OCR accuracy on test dataset by detecting plates, cropping them,
running OCR, and measuring Character Error Rate (CER), word-level accuracy,
and detection rate grouped by recording condition.

Usage:
    python scripts/evaluation/evaluate_ocr.py [options]

Examples:
    # Generate ground truth from test images
    python scripts/evaluation/evaluate_ocr.py --generate-ground-truth
    
    # Evaluate with existing ground truth
    python scripts/evaluation/evaluate_ocr.py --ground-truth datasets/ocr/ground_truth.txt
    
    # Evaluate with preprocessing enabled
    python scripts/evaluation/evaluate_ocr.py --ground-truth datasets/ocr/ground_truth.txt --use-preprocessing
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import cv2
import numpy as np
import yaml
from tqdm import tqdm

# Ensure project root is importable
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.detection.model import load_detection_model, detect_plates
from src.detection.utils import crop_detections
from src.recognition.model import load_ocr_model, recognize_text
from src.preprocessing.image_enhancement import preprocess_plate

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate OCR performance on test dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/pipeline_config.yaml',
        help='Path to configuration file (default: configs/pipeline_config.yaml)'
    )
    
    parser.add_argument(
        '--test-images',
        type=str,
        default='outputs/test_images',
        help='Directory containing test images (default: outputs/test_images)'
    )
    
    parser.add_argument(
        '--ground-truth',
        type=str,
        default='datasets/ocr/ground_truth.txt',
        help='Path to ground truth file (filename<TAB>text format)'
    )
    
    parser.add_argument(
        '--generate-ground-truth',
        action='store_true',
        help='Generate initial ground truth from OCR results for manual review'
    )
    
    parser.add_argument(
        '--use-preprocessing',
        action='store_true',
        help='Apply preprocessing pipeline before OCR'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/evaluation',
        help='Directory for evaluation outputs (default: outputs/evaluation)'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Number of images to sample (default: all images)'
    )
    
    return parser.parse_args()


def calculate_levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein (edit) distance between two strings.
    
    Args:
        s1: First string
        s2: Second string
    
    Returns:
        Edit distance (minimum number of insertions, deletions, substitutions)
    """
    if len(s1) < len(s2):
        return calculate_levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def calculate_cer(predicted: str, ground_truth: str) -> float:
    """
    Calculate Character Error Rate (CER).
    
    CER = edit_distance(predicted, ground_truth) / len(ground_truth)
    
    Args:
        predicted: Predicted text
        ground_truth: Ground truth text
    
    Returns:
        Character Error Rate [0.0 - infinity]
    """
    if not ground_truth:
        return 0.0 if not predicted else 1.0
    
    distance = calculate_levenshtein_distance(predicted or '', ground_truth)
    return distance / len(ground_truth)


def calculate_word_accuracy(predictions: List[Optional[str]], ground_truths: List[str]) -> float:
    """
    Calculate word-level (exact match) accuracy.
    
    Args:
        predictions: List of predicted texts (can contain None)
        ground_truths: List of ground truth texts
    
    Returns:
        Accuracy [0.0 - 1.0]
    """
    if not ground_truths:
        return 0.0
    
    correct = sum(1 for p, g in zip(predictions, ground_truths) 
                  if p is not None and p == g)
    return correct / len(ground_truths)


def calculate_detection_rate(predictions: List[Optional[str]]) -> float:
    """
    Calculate detection rate (percentage with valid text extracted).
    
    Args:
        predictions: List of predicted texts (can contain None)
    
    Returns:
        Detection rate [0.0 - 1.0]
    """
    if not predictions:
        return 0.0
    
    detected = sum(1 for p in predictions if p is not None and p != '')
    return detected / len(predictions)


def parse_condition_from_filename(filename: str) -> str:
    """
    Parse recording condition from filename.
    
    Expected format: {time}_{weather}_{angle}_{frame}.jpg
    Example: day_clear_front_00001.jpg -> day_clear
    
    Args:
        filename: Image filename
    
    Returns:
        Condition string (e.g., 'day_clear', 'night_rain')
    """
    parts = filename.split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"  # time_weather
    return 'unknown'


def load_ground_truth(ground_truth_path: Path) -> Dict[str, str]:
    """
    Load ground truth labels from file.
    
    Format: filename<TAB>plate_text
    Example: day_clear_front_00001.jpg\t12ABC345
    
    Args:
        ground_truth_path: Path to ground truth file
    
    Returns:
        Dictionary mapping filename to plate text
    """
    ground_truth = {}
    
    if not ground_truth_path.exists():
        logger.warning(f"Ground truth file not found: {ground_truth_path}")
        return ground_truth
    
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('\t')
            if len(parts) != 2:
                logger.warning(f"Skipping malformed line {line_num}: {line}")
                continue
            
            filename, plate_text = parts
            ground_truth[filename] = plate_text.upper()
    
    logger.info(f"Loaded {len(ground_truth)} ground truth labels")
    return ground_truth


def generate_ground_truth_from_ocr(
    test_images_dir: Path,
    detection_model,
    ocr_model,
    config: dict,
    output_path: Path,
    use_preprocessing: bool = False,
    sample_size: Optional[int] = None
) -> Dict[str, str]:
    """
    Generate initial ground truth by running OCR on all test images.
    Results should be manually reviewed and corrected.
    
    Args:
        test_images_dir: Directory with test images
        detection_model: Loaded detection model
        ocr_model: Loaded OCR model
        config: Configuration dictionary
        output_path: Where to save generated ground truth
        use_preprocessing: Whether to apply preprocessing
        sample_size: Number of images to process (None = all)
    
    Returns:
        Dictionary mapping filename to detected plate text
    """
    image_files = sorted(test_images_dir.glob('*.jpg'))
    if sample_size:
        image_files = image_files[:sample_size]
    
    logger.info(f"Generating ground truth from {len(image_files)} images...")
    
    generated_gt = {}
    crops_dir = output_path.parent / 'ocr_crops'
    crops_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning(f"Failed to load: {img_path.name}")
                continue
            
            # Detect plates
            detections = detect_plates(
                image, 
                detection_model, 
                conf_threshold=config['detection']['confidence_threshold'],
                iou_threshold=config['detection']['iou_threshold'],
                min_area=1000  # Filter small false positives
            )
            
            if not detections:
                logger.debug(f"No plates detected in {img_path.name}")
                continue
            
            # Process ALL detections (not just first one)
            detected_plates = []
            
            for det_idx, detection in enumerate(detections):
                # Crop plate
                crops = crop_detections(image, [detection])
                if not crops:
                    continue
                
                crop = crops[0]
                
                # Apply preprocessing if requested
                if use_preprocessing:
                    crop = preprocess_plate(crop, config.get('preprocessing', {}))
                
                # Run OCR
                text, confidence = recognize_text(crop, ocr_model, config['recognition'])
                
                if text:
                    detected_plates.append(text)
                    
                    # Save crop for manual review
                    # Add detection index to filename for multiple plates
                    if len(detections) > 1:
                        crop_path = crops_dir / f"{img_path.stem}_det{det_idx+1}_{text}.jpg"
                    else:
                        crop_path = crops_dir / f"{img_path.stem}_{text}.jpg"
                    cv2.imwrite(str(crop_path), crop)
            
            # Store all detected plates (comma-separated if multiple)
            if detected_plates:
                generated_gt[img_path.name] = ','.join(detected_plates)
        
        except Exception as e:
            logger.error(f"Error processing {img_path.name}: {e}")
            continue
    
    # Save generated ground truth
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Generated ground truth - PLEASE REVIEW AND CORRECT\n")
        f.write("# Format: filename<TAB>plate_text\n")
        f.write("# Remove or correct any incorrect detections\n\n")
        
        for filename in sorted(generated_gt.keys()):
            f.write(f"{filename}\t{generated_gt[filename]}\n")
    
    logger.info(f"Generated ground truth saved to: {output_path}")
    logger.info(f"Cropped plates saved to: {crops_dir}")
    logger.info(f"⚠️  IMPORTANT: Please manually review and correct {output_path}")
    
    return generated_gt


def evaluate_ocr(
    test_images_dir: Path,
    ground_truth: Dict[str, str],
    detection_model,
    ocr_model,
    config: dict,
    output_dir: Path,
    use_preprocessing: bool = False
) -> Dict:
    """
    Evaluate OCR performance on test images with ground truth.
    
    Args:
        test_images_dir: Directory with test images
        ground_truth: Dictionary mapping filename to correct plate text
        detection_model: Loaded detection model
        ocr_model: Loaded OCR model
        config: Configuration dictionary
        output_dir: Directory for evaluation outputs
        use_preprocessing: Whether to apply preprocessing
    
    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Evaluating OCR on {len(ground_truth)} images...")
    
    # Prepare output directories
    failure_cases_dir = output_dir / 'ocr_failure_cases'
    failure_cases_dir.mkdir(parents=True, exist_ok=True)
    
    # Track results
    results = {
        'predictions': [],
        'ground_truths': [],
        'confidences': [],
        'filenames': [],
        'conditions': [],
        'cer_scores': [],
        'correct': []
    }
    
    # Process each image
    for filename, gt_text in tqdm(ground_truth.items(), desc="Evaluating"):
        img_path = test_images_dir / filename
        
        if not img_path.exists():
            logger.warning(f"Image not found: {filename}")
            continue
        
        try:
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # Detect plates
            detections = detect_plates(
                image, 
                detection_model, 
                conf_threshold=config['detection']['confidence_threshold'],
                iou_threshold=config['detection']['iou_threshold'],
                min_area=1000  # Filter small false positives
            )
            
            # Split ground truth if multiple plates
            gt_plates = gt_text.split(',')
            
            # Collect all predicted plates
            predicted_plates = []
            confidences = []
            
            if detections:
                for detection in detections:
                    # Crop plate
                    crops = crop_detections(image, [detection])
                    if crops:
                        crop = crops[0]
                        
                        # Apply preprocessing if requested
                        if use_preprocessing:
                            crop = preprocess_plate(crop, config.get('preprocessing', {}))
                        
                        # Run OCR
                        pred_text, conf = recognize_text(
                            crop, ocr_model, config['recognition']
                        )
                        
                        if pred_text:
                            predicted_plates.append(pred_text)
                            confidences.append(conf)
            
            # Match predictions to ground truth
            # For each ground truth plate, find best matching prediction
            condition = parse_condition_from_filename(filename)
            
            for gt_plate in gt_plates:
                best_pred = None
                best_conf = 0.0
                best_cer = 1.0
                
                # Find best matching prediction for this ground truth
                if predicted_plates:
                    for pred, conf in zip(predicted_plates, confidences):
                        cer = calculate_cer(pred, gt_plate)
                        if cer < best_cer or (cer == best_cer and conf > best_conf):
                            best_pred = pred
                            best_conf = conf
                            best_cer = cer
                
                # Calculate metrics
                is_correct = (best_pred == gt_plate)
                
                # Store results
                results['predictions'].append(best_pred)
                results['ground_truths'].append(gt_plate)
                results['confidences'].append(best_conf)
                results['filenames'].append(filename)
                results['conditions'].append(condition)
                results['cer_scores'].append(best_cer)
                results['correct'].append(is_correct)
            
            # Save failure cases
            if not is_correct:
                failure_img = image.copy()
                
                # Add annotation
                text_display = f"GT: {gt_text} | Pred: {best_pred or 'NONE'}"
                cv2.putText(failure_img, text_display, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                failure_path = failure_cases_dir / f"fail_{filename}"
                cv2.imwrite(str(failure_path), failure_img)
        
        except Exception as e:
            logger.error(f"Error evaluating {filename}: {e}")
            # Add failed result
            results['predictions'].append(None)
            results['ground_truths'].append(gt_text)
            results['confidences'].append(0.0)
            results['filenames'].append(filename)
            results['conditions'].append(parse_condition_from_filename(filename))
            results['cer_scores'].append(1.0)
            results['correct'].append(False)
    
    return results


def generate_report(results: Dict, output_path: Path):
    """
    Generate comprehensive evaluation report.
    
    Args:
        results: Dictionary with evaluation results
        output_path: Where to save report (markdown)
    """
    # Calculate overall metrics
    word_accuracy = calculate_word_accuracy(results['predictions'], results['ground_truths'])
    detection_rate = calculate_detection_rate(results['predictions'])
    avg_cer = np.mean(results['cer_scores'])
    avg_confidence = np.mean([c for c in results['confidences'] if c > 0])
    
    # Group by condition
    condition_metrics = defaultdict(lambda: {
        'predictions': [],
        'ground_truths': [],
        'cer_scores': [],
        'correct': []
    })
    
    for i, condition in enumerate(results['conditions']):
        condition_metrics[condition]['predictions'].append(results['predictions'][i])
        condition_metrics[condition]['ground_truths'].append(results['ground_truths'][i])
        condition_metrics[condition]['cer_scores'].append(results['cer_scores'][i])
        condition_metrics[condition]['correct'].append(results['correct'][i])
    
    # Find top failure cases (highest CER)
    failures = [(results['filenames'][i], results['ground_truths'][i], 
                results['predictions'][i], results['cer_scores'][i])
               for i in range(len(results['filenames']))
               if not results['correct'][i]]
    failures.sort(key=lambda x: x[3], reverse=True)
    top_failures = failures[:10]
    
    # Generate report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# OCR Performance Evaluation Report\n\n")
        f.write(f"**Generated:** {Path(output_path).stat().st_mtime}\n\n")
        f.write(f"**Total Images Evaluated:** {len(results['filenames'])}\n\n")
        
        f.write("## Overall Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Word Accuracy (Exact Match) | {word_accuracy:.2%} |\n")
        f.write(f"| Detection Rate | {detection_rate:.2%} |\n")
        f.write(f"| Average CER | {avg_cer:.4f} |\n")
        f.write(f"| Average Confidence | {avg_confidence:.4f} |\n")
        f.write(f"| Correct Predictions | {sum(results['correct'])} / {len(results['correct'])} |\n\n")
        
        f.write("## Performance by Condition\n\n")
        f.write("| Condition | Word Accuracy | Detection Rate | Avg CER | Count |\n")
        f.write("|-----------|---------------|----------------|---------|-------|\n")
        
        for condition in sorted(condition_metrics.keys()):
            metrics = condition_metrics[condition]
            cond_word_acc = calculate_word_accuracy(metrics['predictions'], metrics['ground_truths'])
            cond_det_rate = calculate_detection_rate(metrics['predictions'])
            cond_avg_cer = np.mean(metrics['cer_scores'])
            count = len(metrics['predictions'])
            
            f.write(f"| {condition} | {cond_word_acc:.2%} | {cond_det_rate:.2%} | "
                   f"{cond_avg_cer:.4f} | {count} |\n")
        
        f.write("\n## Top 10 Failure Cases\n\n")
        f.write("| Filename | Ground Truth | Predicted | CER |\n")
        f.write("|----------|--------------|-----------|-----|\n")
        
        for filename, gt, pred, cer in top_failures:
            pred_display = pred if pred else "NONE"
            f.write(f"| {filename} | {gt} | {pred_display} | {cer:.4f} |\n")
        
        f.write("\n## Error Analysis\n\n")
        
        # Character-level confusion
        char_errors = defaultdict(int)
        for gt, pred in zip(results['ground_truths'], results['predictions']):
            if pred and gt != pred:
                for i, (g, p) in enumerate(zip(gt, pred)):
                    if g != p:
                        char_errors[f"{g}→{p}"] += 1
        
        if char_errors:
            f.write("### Most Common Character Substitutions\n\n")
            f.write("| Substitution | Count |\n")
            f.write("|--------------|-------|\n")
            
            for sub, count in sorted(char_errors.items(), key=lambda x: x[1], reverse=True)[:10]:
                f.write(f"| {sub} | {count} |\n")
        
        f.write("\n## Recommendations\n\n")
        
        # Generate recommendations based on metrics
        if detection_rate < 0.95:
            f.write("- **Low Detection Rate:** Consider improving detection model or lowering confidence threshold\n")
        
        if avg_cer > 0.1:
            f.write("- **High CER:** Enable preprocessing pipeline (CLAHE, sharpening) to improve image quality\n")
        
        worst_condition = min(condition_metrics.items(), 
                             key=lambda x: calculate_word_accuracy(x[1]['predictions'], x[1]['ground_truths']))
        f.write(f"- **Worst Performing Condition:** {worst_condition[0]} - "
               f"collect more training data or apply condition-specific preprocessing\n")
        
        if avg_confidence < 0.7:
            f.write("- **Low Confidence:** Model uncertainty is high - consider fine-tuning OCR model\n")
        
        f.write("\n---\n\n")
        f.write("**Failure case images saved to:** `outputs/evaluation/ocr_failure_cases/`\n")
    
    logger.info(f"Report saved to: {output_path}")


def main():
    """Main evaluation workflow."""
    args = parse_arguments()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup paths
    test_images_dir = Path(args.test_images)
    ground_truth_path = Path(args.ground_truth)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    logger.info("Loading models...")
    detection_model = load_detection_model(
        config['detection']['model_path'],
        config['detection'].get('device', 'auto')
    )
    ocr_model = load_ocr_model(config['recognition'])
    
    # Generate or load ground truth
    if args.generate_ground_truth:
        logger.info("Generating ground truth from OCR results...")
        ground_truth = generate_ground_truth_from_ocr(
            test_images_dir,
            detection_model,
            ocr_model,
            config,
            ground_truth_path,
            use_preprocessing=args.use_preprocessing,
            sample_size=args.sample_size
        )
        
        logger.info("\n" + "="*70)
        logger.info("Ground truth generation complete!")
        logger.info(f"Please review and correct: {ground_truth_path}")
        logger.info("Then run evaluation again without --generate-ground-truth flag")
        logger.info("="*70)
        return
    
    # Load ground truth
    ground_truth = load_ground_truth(ground_truth_path)
    
    if not ground_truth:
        logger.error("No ground truth available. Use --generate-ground-truth to create it.")
        return
    
    # Run evaluation
    results = evaluate_ocr(
        test_images_dir,
        ground_truth,
        detection_model,
        ocr_model,
        config,
        output_dir,
        use_preprocessing=args.use_preprocessing
    )
    
    # Generate report
    report_path = output_dir / 'ocr_report.md'
    generate_report(results, report_path)
    
    # Save raw results
    results_json = output_dir / 'ocr_results.json'
    with open(results_json, 'w') as f:
        # Convert to serializable format
        json_results = {
            'predictions': results['predictions'],
            'ground_truths': results['ground_truths'],
            'confidences': results['confidences'],
            'filenames': results['filenames'],
            'conditions': results['conditions'],
            'cer_scores': results['cer_scores'],
            'correct': results['correct']
        }
        json.dump(json_results, f, indent=2)
    
    logger.info(f"\n{'='*70}")
    logger.info("Evaluation complete!")
    logger.info(f"Report: {report_path}")
    logger.info(f"Raw results: {results_json}")
    logger.info(f"Failure cases: {output_dir / 'ocr_failure_cases'}")
    logger.info("="*70)


if __name__ == '__main__':
    main()

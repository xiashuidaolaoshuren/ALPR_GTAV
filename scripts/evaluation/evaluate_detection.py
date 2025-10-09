"""
Detection Performance Evaluation Script

Systematically evaluate detection performance on the test dataset and
summarize metrics by recording condition.

Usage:
    python scripts/evaluation/evaluate_detection.py [options]

Examples:
    # Basic evaluation
    python scripts/evaluation/evaluate_detection.py
    
    # Custom configuration and output
    python scripts/evaluation/evaluate_detection.py --config configs/pipeline_config.yaml --output outputs/results.json
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

import cv2
import yaml
from tqdm import tqdm

# Ensure project root is importable
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.detection.model import load_detection_model, detect_plates
from src.detection.utils import draw_bounding_boxes

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate detection performance on test dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--test_dir',
        type=str,
        default='outputs/test_images',
        help='Directory containing test images (default: outputs/test_images)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/pipeline_config.yaml',
        help='Path to configuration file (default: configs/pipeline_config.yaml)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/detection_results.json',
        help='Path to output results JSON (default: outputs/detection_results.json)'
    )
    
    parser.add_argument(
        '--examples_dir',
        type=str,
        default='outputs/detection_examples',
        help='Directory to save best/worst examples (default: outputs/detection_examples)'
    )
    
    parser.add_argument(
        '--num_examples',
        type=int,
        default=10,
        help='Number of best/worst examples to save (default: 10)'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=None,
        help='Confidence threshold (overrides config)'
    )
    
    parser.add_argument(
        '--iou',
        type=float,
        default=None,
        help='IOU threshold (overrides config)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to run inference on (cpu/cuda/auto, overrides config)'
    )
    
    return parser.parse_args()


def load_configuration(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def parse_filename(filename: str) -> Dict[str, str]:
    """Parse image filename to extract metadata."""
    stem = Path(filename).stem
    parts = stem.split('_')
    
    if len(parts) >= 4:
        return {
            'time': parts[0],
            'weather': parts[1],
            'angle': parts[2],
            'id': parts[3],
            'condition': f"{parts[0]}_{parts[1]}"
        }
    
    return {
        'time': 'unknown',
        'weather': 'unknown',
        'angle': 'unknown',
        'id': stem,
        'condition': 'unknown'
    }


def evaluate_detection(
    test_dir: str,
    model,
    confidence_threshold: float,
    iou_threshold: float
) -> Dict:
    """Run detection on all test images and collect results."""
    test_dir = Path(test_dir)
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [
        f for f in test_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    logger.info(f"Found {len(image_files)} test images")
    
    results = defaultdict(list)
    all_results = []
    
    for image_path in tqdm(image_files, desc='Evaluating detection'):
        frame = cv2.imread(str(image_path))
        if frame is None:
            logger.warning(f"Failed to load image: {image_path}")
            continue
        
        metadata = parse_filename(image_path.name)
        detections = detect_plates(
            frame,
            model,
            confidence_threshold,
            iou_threshold
        )
        
        num_detections = len(detections)
        confidences = [d['confidence'] for d in detections]
        max_confidence = max(confidences, default=0.0)
        avg_confidence = sum(confidences) / num_detections if num_detections else 0.0
        
        result = {
            'filename': image_path.name,
            'num_detections': num_detections,
            'max_confidence': float(max_confidence),
            'avg_confidence': float(avg_confidence),
            'detections': [
                {
                    'bbox': d['bbox'],
                    'confidence': float(d['confidence'])
                }
                for d in detections
            ],
            'metadata': metadata
        }
        
        condition = metadata['condition']
        results[condition].append(result)
        all_results.append(result)
    
    logger.info(f"Processed {len(all_results)} images")
    
    return {
        'by_condition': dict(results),
        'all_images': all_results
    }


def calculate_statistics(results: Dict) -> Dict:
    """Calculate detection statistics from evaluation results."""
    all_images = results['all_images']
    by_condition = results['by_condition']
    
    total_images = len(all_images)
    images_with_detection = sum(1 for r in all_images if r['num_detections'] > 0)
    total_detections = sum(r['num_detections'] for r in all_images)
    
    overall_stats = {
        'total_images': total_images,
        'images_with_detection': images_with_detection,
        'detection_rate': images_with_detection / total_images if total_images else 0,
        'total_detections': total_detections,
        'avg_detections_per_image': total_detections / total_images if total_images else 0,
        'avg_confidence': sum(r['avg_confidence'] for r in all_images) / total_images if total_images else 0
    }
    
    condition_stats = {}
    for condition, images in by_condition.items():
        num_images = len(images)
        with_detection = sum(1 for r in images if r['num_detections'] > 0)
        num_detections = sum(r['num_detections'] for r in images)
        
        condition_stats[condition] = {
            'num_images': num_images,
            'images_with_detection': with_detection,
            'detection_rate': with_detection / num_images if num_images else 0,
            'total_detections': num_detections,
            'avg_detections_per_image': num_detections / num_images if num_images else 0,
            'avg_confidence': sum(r['avg_confidence'] for r in images) / num_images if num_images else 0
        }
    
    return {
        'overall': overall_stats,
        'by_condition': condition_stats
    }


def save_examples(
    results: Dict,
    test_dir: str,
    examples_dir: str,
    num_examples: int = 10
):
    """Save best/worst detection examples for qualitative review."""
    test_dir = Path(test_dir)
    examples_dir = Path(examples_dir)
    best_dir = examples_dir / 'best'
    worst_dir = examples_dir / 'worst'
    no_detection_dir = examples_dir / 'no_detection'
    
    best_dir.mkdir(parents=True, exist_ok=True)
    worst_dir.mkdir(parents=True, exist_ok=True)
    no_detection_dir.mkdir(parents=True, exist_ok=True)
    
    all_images = results['all_images']
    with_detections = [r for r in all_images if r['num_detections'] > 0]
    without_detections = [r for r in all_images if r['num_detections'] == 0]
    with_detections.sort(key=lambda x: x['max_confidence'], reverse=True)
    
    for result in with_detections[:num_examples]:
        image_path = test_dir / result['filename']
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        annotated = draw_bounding_boxes(frame, result['detections'])
        cv2.imwrite(str(best_dir / result['filename']), annotated)
    
    for result in with_detections[-num_examples:]:
        image_path = test_dir / result['filename']
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        annotated = draw_bounding_boxes(frame, result['detections'])
        cv2.imwrite(str(worst_dir / result['filename']), annotated)
    
    for result in without_detections[:num_examples]:
        image_path = test_dir / result['filename']
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        cv2.imwrite(str(no_detection_dir / result['filename']), frame)
    
    logger.info(f"Saved qualitative examples to {examples_dir}")


def print_statistics(stats: Dict):
    """Log evaluation statistics in a readable format."""
    overall = stats['overall']
    by_condition = stats['by_condition']
    
    logger.info("=" * 80)
    logger.info("DETECTION EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info("\n📊 Overall Statistics:")
    logger.info(f"  Total images: {overall['total_images']}")
    logger.info(f"  Images with detection: {overall['images_with_detection']}")
    logger.info(f"  Detection rate: {overall['detection_rate']:.2%}")
    logger.info(f"  Total detections: {overall['total_detections']}")
    logger.info(f"  Avg detections per image: {overall['avg_detections_per_image']:.2f}")
    logger.info(f"  Avg confidence: {overall['avg_confidence']:.3f}")
    logger.info("\n📈 Performance by Condition:")
    for condition, cond_stats in sorted(by_condition.items()):
        logger.info(f"\n  {condition}:")
        logger.info(f"    Images: {cond_stats['num_images']}")
        logger.info(f"    Detection rate: {cond_stats['detection_rate']:.2%}")
        logger.info(f"    Avg detections: {cond_stats['avg_detections_per_image']:.2f}")
        logger.info(f"    Avg confidence: {cond_stats['avg_confidence']:.3f}")
    logger.info("\n" + "=" * 80)


def main():
    """Entry point."""
    args = parse_arguments()
    
    try:
        logger.info("Loading configuration...")
        config = load_configuration(args.config)
        detection_config = config['detection']
        
        confidence_threshold = args.conf if args.conf is not None else detection_config['confidence_threshold']
        iou_threshold = args.iou if args.iou is not None else detection_config['iou_threshold']
        device = args.device if args.device is not None else detection_config.get('device', 'auto')
        
        logger.info("Detection parameters:")
        logger.info(f"  Confidence threshold: {confidence_threshold}")
        logger.info(f"  IOU threshold: {iou_threshold}")
        logger.info(f"  Device: {device}")
        
        logger.info("Loading detection model...")
        model = load_detection_model(
            model_path=detection_config['model_path'],
            device=device
        )
        logger.info("Model loaded successfully")
        
        logger.info(f"Evaluating detection on: {args.test_dir}")
        results = evaluate_detection(
            test_dir=args.test_dir,
            model=model,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold
        )
        
        stats = calculate_statistics(results)
        print_statistics(stats)
        
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                'results': results,
                'statistics': stats,
                'parameters': {
                    'confidence_threshold': confidence_threshold,
                    'iou_threshold': iou_threshold,
                    'test_dir': args.test_dir
                }
            }, f, indent=2)
        logger.info(f"\n💾 Results saved to: {output_path}")
        
        save_examples(
            results=results,
            test_dir=args.test_dir,
            examples_dir=args.examples_dir,
            num_examples=args.num_examples
        )
        
        logger.info("\n✅ Evaluation completed successfully!")
        return 0
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

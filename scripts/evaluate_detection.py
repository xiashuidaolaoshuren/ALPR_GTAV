"""
Detection Performance Evaluation Script

Systematically evaluate detection performance on test dataset.
Analyze performance by condition (day/night/weather) and generate detailed metrics.

Usage:
    python scripts/evaluate_detection.py [options]

Examples:
    # Basic evaluation
    python scripts/evaluate_detection.py
    
    # Custom configuration and output
    python scripts/evaluate_detection.py --config configs/pipeline_config.yaml --output outputs/results.json
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import yaml
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    """
    Parse image filename to extract metadata.
    
    Expected format: {time}_{weather}_{angle}_{id}.jpg
    Example: day_clear_front_00001.jpg
    
    Args:
        filename: Image filename
        
    Returns:
        Dictionary with time, weather, angle, and id
    """
    stem = Path(filename).stem
    parts = stem.split('_')
    
    if len(parts) >= 4:
        return {
            'time': parts[0],
            'weather': parts[1],
            'angle': parts[2],
            'id': parts[3],
            'condition': f"{parts[0]}_{parts[1]}"  # e.g., day_clear
        }
    else:
        # Fallback for unexpected format
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
    """
    Run detection on all test images and collect results.
    
    Args:
        test_dir: Directory containing test images
        model: Loaded detection model
        confidence_threshold: Confidence threshold for detections
        iou_threshold: IOU threshold for NMS
        
    Returns:
        Dictionary with evaluation results organized by condition
    """
    test_dir = Path(test_dir)
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [
        f for f in test_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    logger.info(f"Found {len(image_files)} test images")
    
    # Process each image
    results = defaultdict(list)
    all_results = []
    
    for image_path in tqdm(image_files, desc='Evaluating detection'):
        # Load image
        frame = cv2.imread(str(image_path))
        if frame is None:
            logger.warning(f"Failed to load image: {image_path}")
            continue
        
        # Parse metadata from filename
        metadata = parse_filename(image_path.name)
        
        # Run detection
        detections = detect_plates(
            frame,
            model,
            confidence_threshold,
            iou_threshold
        )
        
        # Calculate statistics
        num_detections = len(detections)
        max_confidence = max([d['confidence'] for d in detections], default=0.0)
        avg_confidence = sum([d['confidence'] for d in detections]) / num_detections if num_detections > 0 else 0.0
        
        # Store result
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
        
        # Organize by condition
        condition = metadata['condition']
        results[condition].append(result)
        all_results.append(result)
    
    logger.info(f"Processed {len(all_results)} images")
    
    return {
        'by_condition': dict(results),
        'all_images': all_results
    }


def calculate_statistics(results: Dict) -> Dict:
    """
    Calculate detection statistics from results.
    
    Args:
        results: Evaluation results dictionary
        
    Returns:
        Dictionary with calculated statistics
    """
    all_images = results['all_images']
    by_condition = results['by_condition']
    
    # Overall statistics
    total_images = len(all_images)
    images_with_detection = sum(1 for r in all_images if r['num_detections'] > 0)
    total_detections = sum(r['num_detections'] for r in all_images)
    
    overall_stats = {
        'total_images': total_images,
        'images_with_detection': images_with_detection,
        'detection_rate': images_with_detection / total_images if total_images > 0 else 0,
        'total_detections': total_detections,
        'avg_detections_per_image': total_detections / total_images if total_images > 0 else 0,
        'avg_confidence': sum(r['avg_confidence'] for r in all_images) / total_images if total_images > 0 else 0
    }
    
    # Statistics by condition
    condition_stats = {}
    for condition, images in by_condition.items():
        num_images = len(images)
        with_detection = sum(1 for r in images if r['num_detections'] > 0)
        num_detections = sum(r['num_detections'] for r in images)
        
        condition_stats[condition] = {
            'num_images': num_images,
            'images_with_detection': with_detection,
            'detection_rate': with_detection / num_images if num_images > 0 else 0,
            'total_detections': num_detections,
            'avg_detections_per_image': num_detections / num_images if num_images > 0 else 0,
            'avg_confidence': sum(r['avg_confidence'] for r in images) / num_images if num_images > 0 else 0
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
    """
    Save best and worst detection examples.
    
    Args:
        results: Evaluation results dictionary
        test_dir: Directory containing test images
        examples_dir: Directory to save examples
        num_examples: Number of examples to save for each category
    """
    test_dir = Path(test_dir)
    examples_dir = Path(examples_dir)
    
    # Create subdirectories
    best_dir = examples_dir / 'best'
    worst_dir = examples_dir / 'worst'
    no_detection_dir = examples_dir / 'no_detection'
    
    best_dir.mkdir(parents=True, exist_ok=True)
    worst_dir.mkdir(parents=True, exist_ok=True)
    no_detection_dir.mkdir(parents=True, exist_ok=True)
    
    all_images = results['all_images']
    
    # Separate images with and without detections
    with_detections = [r for r in all_images if r['num_detections'] > 0]
    without_detections = [r for r in all_images if r['num_detections'] == 0]
    
    # Sort by confidence
    with_detections.sort(key=lambda x: x['max_confidence'], reverse=True)
    
    # Save best examples (highest confidence)
    logger.info(f"Saving {min(num_examples, len(with_detections))} best detection examples...")
    for result in with_detections[:num_examples]:
        image_path = test_dir / result['filename']
        frame = cv2.imread(str(image_path))
        
        if frame is not None:
            # Draw bounding boxes
            annotated = draw_bounding_boxes(frame, result['detections'])
            
            # Save annotated image
            output_path = best_dir / result['filename']
            cv2.imwrite(str(output_path), annotated)
    
    # Save worst examples (lowest confidence, but still detected)
    logger.info(f"Saving {min(num_examples, len(with_detections))} worst detection examples...")
    for result in with_detections[-num_examples:]:
        image_path = test_dir / result['filename']
        frame = cv2.imread(str(image_path))
        
        if frame is not None:
            # Draw bounding boxes
            annotated = draw_bounding_boxes(frame, result['detections'])
            
            # Save annotated image
            output_path = worst_dir / result['filename']
            cv2.imwrite(str(output_path), annotated)
    
    # Save no detection examples
    logger.info(f"Saving {min(num_examples, len(without_detections))} no detection examples...")
    for result in without_detections[:num_examples]:
        image_path = test_dir / result['filename']
        frame = cv2.imread(str(image_path))
        
        if frame is not None:
            output_path = no_detection_dir / result['filename']
            cv2.imwrite(str(output_path), frame)
    
    logger.info(f"Saved examples to {examples_dir}")


def print_statistics(stats: Dict):
    """Print evaluation statistics."""
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
    """Main execution function."""
    args = parse_arguments()
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_configuration(args.config)
        detection_config = config['detection']
        
        # Override config with command line arguments
        confidence_threshold = args.conf if args.conf is not None else detection_config['confidence_threshold']
        iou_threshold = args.iou if args.iou is not None else detection_config['iou_threshold']
        device = args.device if args.device is not None else detection_config.get('device', 'auto')
        
        logger.info(f"Detection parameters:")
        logger.info(f"  Confidence threshold: {confidence_threshold}")
        logger.info(f"  IOU threshold: {iou_threshold}")
        logger.info(f"  Device: {device}")
        
        # Load detection model
        logger.info("Loading detection model...")
        model = load_detection_model(
            model_path=detection_config['model_path'],
            device=device
        )
        logger.info("Model loaded successfully")
        
        # Run evaluation
        logger.info(f"Evaluating detection on: {args.test_dir}")
        results = evaluate_detection(
            test_dir=args.test_dir,
            model=model,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold
        )
        
        # Calculate statistics
        stats = calculate_statistics(results)
        
        # Print statistics
        print_statistics(stats)
        
        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            'results': results,
            'statistics': stats,
            'parameters': {
                'confidence_threshold': confidence_threshold,
                'iou_threshold': iou_threshold,
                'test_dir': args.test_dir
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"\n💾 Results saved to: {output_path}")
        
        # Save best/worst examples
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

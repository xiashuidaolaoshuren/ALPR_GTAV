"""
Single Image Detection Script

This script runs license plate detection on a single image and saves the annotated result.
It's useful for testing and visualizing detection performance.

Usage:
    python scripts/detect_image.py --image path/to/image.jpg
    python scripts/detect_image.py --image outputs/test_images/day_clear_front_00001.jpg --output result.jpg
    python scripts/detect_image.py --image test.jpg --conf 0.3 --no-save-viz

Author: GTA V ALPR Development Team
Version: 1.0
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import yaml
from src.detection.model import load_detection_model, detect_plates
from src.detection.utils import draw_bounding_boxes
from src.detection.config import DetectionConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run license plate detection on a single image'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input image'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save annotated output image (default: outputs/detection_result.jpg)'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=None,
        help='Confidence threshold (overrides config file)'
    )
    
    parser.add_argument(
        '--iou',
        type=float,
        default=None,
        help='IOU threshold for NMS (overrides config file)'
    )
    
    parser.add_argument(
        '--no-save-viz',
        action='store_true',
        help='Do not save visualization (just print detections)'
    )
    
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display result in window (requires GUI)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu', 'auto'],
        help='Device to run inference on (overrides config)'
    )
    
    return parser.parse_args()


def load_configuration():
    """Load configuration from YAML file."""
    config_path = project_root / 'configs' / 'pipeline_config.yaml'
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return None
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return config_dict
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return None


def main():
    """Main entry point."""
    args = parse_arguments()
    
    logger.info("="*70)
    logger.info("License Plate Detection - Single Image")
    logger.info("="*70)
    
    # Validate input image
    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"Image file not found: {image_path}")
        return 1
    
    logger.info(f"Input image: {image_path}")
    
    # Load configuration
    logger.info("\n1. Loading configuration...")
    config_dict = load_configuration()
    if config_dict is None:
        return 1
    
    detection_config = DetectionConfig(config_dict)
    
    # Override config with command line arguments
    if args.conf is not None:
        detection_config.confidence_threshold = args.conf
        logger.info(f"   Using custom confidence threshold: {args.conf}")
    
    if args.iou is not None:
        detection_config.iou_threshold = args.iou
        logger.info(f"   Using custom IOU threshold: {args.iou}")
    
    if args.device is not None:
        detection_config.device = args.device
        logger.info(f"   Using custom device: {args.device}")
    
    # Validate configuration
    try:
        detection_config.validate()
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        logger.info("\nPlease run: python models/detection/download_model.py")
        return 1
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return 1
    
    # Load model
    logger.info("\n2. Loading YOLOv8 model...")
    try:
        model = load_detection_model(
            detection_config.model_path,
            device=detection_config.device
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1
    
    # Read image
    logger.info("\n3. Reading input image...")
    frame = cv2.imread(str(image_path))
    if frame is None:
        logger.error(f"Failed to read image: {image_path}")
        return 1
    
    logger.info(f"   Image shape: {frame.shape}")
    
    # Run detection
    logger.info("\n4. Running detection...")
    try:
        detections = detect_plates(
            frame,
            model,
            conf_threshold=detection_config.confidence_threshold,
            iou_threshold=detection_config.iou_threshold
        )
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return 1
    
    # Print detection results
    logger.info("\n" + "="*70)
    logger.info("DETECTION RESULTS")
    logger.info("="*70)
    logger.info(f"Total detections: {len(detections)}")
    
    if detections:
        logger.info("\nDetailed results:")
        for i, (x1, y1, x2, y2, conf) in enumerate(detections, 1):
            width = x2 - x1
            height = y2 - y1
            logger.info(f"  {i}. Position: ({x1}, {y1}) to ({x2}, {y2})")
            logger.info(f"     Size: {width}x{height} pixels")
            logger.info(f"     Confidence: {conf:.4f}")
    else:
        logger.info("No license plates detected.")
        logger.info(f"Try lowering the confidence threshold (current: {detection_config.confidence_threshold})")
    
    # Visualize and save
    if not args.no_save_viz:
        logger.info("\n5. Creating visualization...")
        annotated = draw_bounding_boxes(
            frame,
            detections,
            color=(0, 255, 0),  # Green
            thickness=2,
            show_confidence=True
        )
        
        # Determine output path
        if args.output is None:
            output_dir = project_root / 'outputs'
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / 'detection_result.jpg'
        else:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save annotated image
        cv2.imwrite(str(output_path), annotated)
        logger.info(f"   Saved annotated image to: {output_path}")
        
        # Show image if requested
        if args.show:
            cv2.imshow('Detection Result', annotated)
            logger.info("\nPress any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    logger.info("\n" + "="*70)
    logger.info("Detection completed successfully!")
    logger.info("="*70)
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)

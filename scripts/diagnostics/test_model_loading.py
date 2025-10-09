"""
Test Model Loading Script

This script tests the YOLOv8 model loading functionality.
It verifies that the model can be loaded and is ready for inference.

Usage:
    python scripts/test_model_loading.py
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.detection.model import load_detection_model, validate_model
from src.detection.config import DetectionConfig
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
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


def test_model_loading():
    """Test loading the detection model."""
    logger.info("="*70)
    logger.info("YOLOv8 Model Loading Test")
    logger.info("="*70)
    
    # Load configuration
    logger.info("\n1. Loading configuration...")
    config_dict = load_config()
    if config_dict is None:
        logger.error("✗ Failed to load configuration")
        return False
    
    # Create detection config
    try:
        detection_config = DetectionConfig(config_dict)
        logger.info(f"✓ Configuration loaded: {detection_config}")
    except Exception as e:
        logger.error(f"✗ Failed to create detection config: {e}")
        return False
    
    # Validate configuration
    logger.info("\n2. Validating configuration...")
    try:
        detection_config.validate()
        logger.info("✓ Configuration validated")
    except FileNotFoundError as e:
        logger.error(f"✗ Model file not found: {e}")
        logger.info("\nPlease run: python models/detection/download_model.py")
        return False
    except Exception as e:
        logger.error(f"✗ Configuration validation failed: {e}")
        return False
    
    # Load model
    logger.info("\n3. Loading YOLOv8 model...")
    try:
        model = load_detection_model(
            detection_config.model_path,
            device=detection_config.device
        )
        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Model type: {type(model)}")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        return False
    
    # Validate model
    logger.info("\n4. Validating model inference...")
    try:
        if validate_model(model):
            logger.info("✓ Model validation passed")
        else:
            logger.error("✗ Model validation failed")
            return False
    except Exception as e:
        logger.error(f"✗ Model validation error: {e}")
        return False
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)
    logger.info("✓ All tests passed!")
    logger.info(f"\nModel Details:")
    logger.info(f"  Path: {detection_config.model_path}")
    logger.info(f"  Device: {detection_config.device}")
    logger.info(f"  Confidence Threshold: {detection_config.confidence_threshold}")
    logger.info(f"  IOU Threshold: {detection_config.iou_threshold}")
    logger.info(f"  Image Size: {detection_config.img_size}")
    logger.info("\nModel is ready for inference!")
    logger.info("="*70)
    
    return True


def main():
    """Main entry point."""
    try:
        success = test_model_loading()
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

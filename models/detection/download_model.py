"""
YOLOv8 License Plate Detection Model Download Script

This script helps download or verify the pre-trained YOLOv8 model for license plate detection.
Model source: yasirfaizahmed/license-plate-object-detection from Hugging Face

Usage:
    python models/detection/download_model.py
"""

import os
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def download_yolov8_lpr_model():
    """
    Download or verify YOLOv8 license plate detection model.
    
    The model should be placed in models/detection/ directory.
    If the model is not found, provides instructions for manual download.
    """
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    model_dir = project_root / 'models' / 'detection'
    model_path = model_dir / 'yolov8n.pt'
    
    # Ensure model directory exists
    model_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("YOLOv8 License Plate Detection Model Setup")
    logger.info("="*70)
    
    # Check if model already exists
    if model_path.exists():
        logger.info(f"✓ Model found at: {model_path}")
        logger.info(f"  File size: {model_path.stat().st_size / (1024*1024):.2f} MB")
        logger.info("\nModel is ready for use!")
        return True
    
    # Model not found - provide download instructions
    logger.warning("✗ Model not found!")
    logger.info("\n" + "="*70)
    logger.info("DOWNLOAD INSTRUCTIONS")
    logger.info("="*70)
    
    print(f"""
The pre-trained YOLOv8 license plate detection model is not found.

Please download the model manually using one of the following methods:

=== METHOD 1: Hugging Face Hub (Recommended) ===

1. Visit: https://huggingface.co/yasirfaizahmed/license-plate-object-detection

2. Download the model weights file (usually named 'best.pt' or similar)

3. Rename the downloaded file to 'yolov8n.pt'

4. Place the file in: {model_dir.absolute()}

=== METHOD 2: Use Ultralytics Pre-trained Model (Alternative) ===

If the Hugging Face model is not available, you can use a standard YOLOv8 model
trained on COCO dataset (not optimal but may work for initial testing):

    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

Then move the downloaded yolov8n.pt to: {model_dir.absolute()}

Note: For best results, use a model specifically trained on license plates.

=== METHOD 3: Train Your Own Model ===

See the training documentation in docs/ for instructions on training
a custom YOLOv8 model on your GTA V dataset.

=== VERIFICATION ===

After downloading, run this script again to verify:
    python models/detection/download_model.py

Or test model loading:
    python scripts/test_model_loading.py

=== IMPORTANT NOTES ===

- Model file size is typically 10-100 MB
- DO NOT commit model files to Git (already in .gitignore)
- Model path can be changed in configs/pipeline_config.yaml
- GPU highly recommended for inference (CUDA compatible)

    """)
    
    logger.info("="*70)
    return False


def verify_model_compatibility():
    """
    Verify that the required dependencies are installed.
    """
    try:
        from ultralytics import YOLO
        import torch
        logger.info("✓ Ultralytics YOLO library found")
        logger.info(f"✓ PyTorch version: {torch.__version__}")
        logger.info(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"  CUDA version: {torch.version.cuda}")
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        return True
    except ImportError as e:
        logger.error("✗ Required dependencies not found!")
        logger.error(f"  Error: {e}")
        logger.error("\nPlease install required packages:")
        logger.error("  pip install ultralytics torch")
        return False


def main():
    """Main entry point."""
    logger.info("Checking model and dependencies...\n")
    
    # Check dependencies
    if not verify_model_compatibility():
        return 1
    
    print()
    
    # Check/download model
    if download_yolov8_lpr_model():
        logger.info("\n✓ Setup complete! Model is ready for use.")
        return 0
    else:
        logger.warning("\n✗ Model not found. Please follow the download instructions above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

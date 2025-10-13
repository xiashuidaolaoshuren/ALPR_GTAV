"""
YOLO Dataset Validation Script

Validate YOLO format dataset for license plate detection training.
Checks for common errors and ensures data quality.

Usage:
    python scripts/diagnostics/validate_yolo_dataset.py --dataset datasets/lpr
    
Checks performed:
- YOLO format compliance (normalized coordinates [0, 1])
- Image-label correspondence
- Empty label files
- Missing image files
- Bounding box validity
- Visual spot-check of random samples

"""

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YOLODatasetValidator:
    """Validate YOLO format dataset."""
    
    def __init__(self, dataset_path: Path):
        """
        Initialize validator.
        
        Args:
            dataset_path: Path to YOLO dataset root directory
        """
        self.dataset_path = Path(dataset_path)
        self.issues = []
        self.warnings = []
        self.stats = {
            'train': {'images': 0, 'labels': 0, 'annotations': 0},
            'valid': {'images': 0, 'labels': 0, 'annotations': 0},
            'test': {'images': 0, 'labels': 0, 'annotations': 0}
        }
    
    def check_dataset_structure(self) -> bool:
        """Check if dataset has proper directory structure."""
        logger.info("Checking dataset structure...")
        
        if not self.dataset_path.exists():
            self.issues.append(f"Dataset path does not exist: {self.dataset_path}")
            return False
        
        # Check for required directories
        required_dirs = ['train/images', 'train/labels',
                        'valid/images', 'valid/labels',
                        'test/images', 'test/labels']
        
        for dir_path in required_dirs:
            full_path = self.dataset_path / dir_path
            if not full_path.exists():
                self.warnings.append(f"Missing directory: {dir_path}")
        
        # Check for data.yaml
        yaml_path = self.dataset_path / 'data.yaml'
        if not yaml_path.exists():
            self.issues.append("Missing data.yaml configuration file")
        else:
            logger.info("✓ Found data.yaml")
        
        # Check for classes.txt
        classes_path = self.dataset_path / 'classes.txt'
        if not classes_path.exists():
            self.warnings.append("Missing classes.txt (optional but recommended)")
        else:
            logger.info("✓ Found classes.txt")
        
        return len(self.issues) == 0
    
    def check_yolo_format(self, label_path: Path) -> Tuple[bool, List[str], int]:
        """
        Check if label file is in valid YOLO format.
        
        Args:
            label_path: Path to label file
            
        Returns:
            Tuple of (is_valid, errors, num_annotations)
        """
        errors = []
        num_annotations = 0
        
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                return True, [], 0  # Empty file is valid (no objects)
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                num_annotations += 1
                parts = line.split()
                
                # Check format: class_id x_center y_center width height
                if len(parts) != 5:
                    errors.append(
                        f"Line {line_num}: Invalid format (expected 5 values, got {len(parts)})"
                    )
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                except ValueError as e:
                    errors.append(f"Line {line_num}: Invalid numeric values: {e}")
                    continue
                
                # Check class_id (should be 0 for license_plate)
                if class_id != 0:
                    errors.append(f"Line {line_num}: Invalid class_id {class_id} (expected 0)")
                
                # Check normalized coordinates [0, 1]
                if not (0 <= x_center <= 1):
                    errors.append(f"Line {line_num}: x_center {x_center} out of range [0, 1]")
                if not (0 <= y_center <= 1):
                    errors.append(f"Line {line_num}: y_center {y_center} out of range [0, 1]")
                if not (0 < width <= 1):
                    errors.append(f"Line {line_num}: width {width} out of range (0, 1]")
                if not (0 < height <= 1):
                    errors.append(f"Line {line_num}: height {height} out of range (0, 1]")
                
                # Check if box is within image bounds
                x_min = x_center - width / 2
                x_max = x_center + width / 2
                y_min = y_center - height / 2
                y_max = y_center + height / 2
                
                if x_min < 0 or x_max > 1 or y_min < 0 or y_max > 1:
                    errors.append(
                        f"Line {line_num}: Bounding box extends outside image bounds "
                        f"(x: {x_min:.3f}-{x_max:.3f}, y: {y_min:.3f}-{y_max:.3f})"
                    )
        
        except Exception as e:
            errors.append(f"Error reading file: {e}")
            return False, errors, 0
        
        return len(errors) == 0, errors, num_annotations
    
    def validate_split(self, split_name: str) -> Dict:
        """
        Validate a dataset split (train/valid/test).
        
        Args:
            split_name: Name of split to validate
            
        Returns:
            Dictionary with validation results
        """
        logger.info(f"\nValidating {split_name} split...")
        
        images_dir = self.dataset_path / split_name / 'images'
        labels_dir = self.dataset_path / split_name / 'labels'
        
        if not images_dir.exists():
            self.warnings.append(f"{split_name}/images directory not found")
            return self.stats[split_name]
        
        if not labels_dir.exists():
            self.warnings.append(f"{split_name}/labels directory not found")
            return self.stats[split_name]
        
        # Get all images and labels
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f'*{ext}'))
            image_files.extend(images_dir.glob(f'*{ext.upper()}'))
        
        label_files = list(labels_dir.glob('*.txt'))
        
        self.stats[split_name]['images'] = len(image_files)
        self.stats[split_name]['labels'] = len(label_files)
        
        logger.info(f"  Images: {len(image_files)}")
        logger.info(f"  Labels: {len(label_files)}")
        
        # Check image-label correspondence
        image_stems = {img.stem for img in image_files}
        label_stems = {lbl.stem for lbl in label_files}
        
        missing_labels = image_stems - label_stems
        missing_images = label_stems - image_stems
        
        if missing_labels:
            self.issues.append(
                f"{split_name}: {len(missing_labels)} images without labels"
            )
            logger.warning(f"  Missing labels for: {list(missing_labels)[:5]}...")
        
        if missing_images:
            self.warnings.append(
                f"{split_name}: {len(missing_images)} labels without images"
            )
        
        if not missing_labels and not missing_images:
            logger.info("  ✓ All images have corresponding labels")
        
        # Validate label files
        empty_labels = []
        invalid_labels = []
        total_annotations = 0
        
        for label_file in label_files:
            is_valid, errors, num_annotations = self.check_yolo_format(label_file)
            total_annotations += num_annotations
            
            if num_annotations == 0:
                empty_labels.append(label_file.name)
            
            if not is_valid:
                invalid_labels.append((label_file.name, errors))
        
        self.stats[split_name]['annotations'] = total_annotations
        
        if empty_labels:
            self.warnings.append(
                f"{split_name}: {len(empty_labels)} empty label files (no objects)"
            )
        
        if invalid_labels:
            self.issues.append(f"{split_name}: {len(invalid_labels)} invalid label files")
            for label_name, errors in invalid_labels[:3]:  # Show first 3
                logger.error(f"  {label_name}:")
                for error in errors[:5]:  # Show first 5 errors per file
                    logger.error(f"    - {error}")
        
        if not invalid_labels:
            logger.info(f"  ✓ All label files valid YOLO format")
        
        logger.info(f"  Total annotations: {total_annotations}")
        
        return self.stats[split_name]
    
    def visualize_samples(self, split_name: str, num_samples: int = 5):
        """
        Visualize random samples with bounding boxes.
        
        Args:
            split_name: Name of split to visualize
            num_samples: Number of samples to visualize
        """
        logger.info(f"\nVisualizing {num_samples} random samples from {split_name}...")
        
        images_dir = self.dataset_path / split_name / 'images'
        labels_dir = self.dataset_path / split_name / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            logger.warning(f"Cannot visualize {split_name}: directories not found")
            return
        
        # Get all images with labels
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f'*{ext}'))
        
        # Filter to only images that have labels
        images_with_labels = []
        for img_file in image_files:
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                # Check if label file is not empty
                with open(label_file, 'r') as f:
                    if f.read().strip():
                        images_with_labels.append(img_file)
        
        if not images_with_labels:
            logger.warning(f"No images with annotations found in {split_name}")
            return
        
        # Random sample
        samples = random.sample(
            images_with_labels,
            min(num_samples, len(images_with_labels))
        )
        
        output_dir = self.dataset_path / 'validation_samples'
        output_dir.mkdir(exist_ok=True)
        
        for img_file in samples:
            # Load image
            image = cv2.imread(str(img_file))
            if image is None:
                logger.warning(f"Failed to load image: {img_file.name}")
                continue
            
            height, width = image.shape[:2]
            
            # Load labels
            label_file = labels_dir / f"{img_file.stem}.txt"
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            # Draw bounding boxes
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
                try:
                    class_id, x_center, y_center, box_width, box_height = map(float, parts)
                    
                    # Convert to pixel coordinates
                    x_center_px = int(x_center * width)
                    y_center_px = int(y_center * height)
                    box_width_px = int(box_width * width)
                    box_height_px = int(box_height * height)
                    
                    # Calculate corner points
                    x1 = int(x_center_px - box_width_px / 2)
                    y1 = int(y_center_px - box_height_px / 2)
                    x2 = int(x_center_px + box_width_px / 2)
                    y2 = int(y_center_px + box_height_px / 2)
                    
                    # Draw rectangle
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label
                    cv2.putText(
                        image,
                        'license_plate',
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )
                except Exception as e:
                    logger.error(f"Error drawing box for {img_file.name}: {e}")
            
            # Save annotated image
            output_path = output_dir / f"{split_name}_{img_file.name}"
            cv2.imwrite(str(output_path), image)
            logger.info(f"  Saved: {output_path}")
        
        logger.info(f"✓ Visualizations saved to {output_dir}")
    
    def generate_report(self):
        """Generate validation report."""
        logger.info("\n" + "=" * 80)
        logger.info("YOLO DATASET VALIDATION REPORT")
        logger.info("=" * 80)
        logger.info(f"\nDataset: {self.dataset_path}")
        
        # Summary statistics
        logger.info("\n📊 Dataset Statistics:")
        total_images = sum(split['images'] for split in self.stats.values())
        total_labels = sum(split['labels'] for split in self.stats.values())
        total_annotations = sum(split['annotations'] for split in self.stats.values())
        
        for split_name in ['train', 'valid', 'test']:
            stats = self.stats[split_name]
            logger.info(f"  {split_name.capitalize()}:")
            logger.info(f"    Images: {stats['images']}")
            logger.info(f"    Labels: {stats['labels']}")
            logger.info(f"    Annotations: {stats['annotations']}")
        
        logger.info(f"\n  Total:")
        logger.info(f"    Images: {total_images}")
        logger.info(f"    Labels: {total_labels}")
        logger.info(f"    Annotations: {total_annotations}")
        
        # Issues and warnings
        if self.issues:
            logger.error(f"\n❌ CRITICAL ISSUES ({len(self.issues)}):")
            for issue in self.issues:
                logger.error(f"  • {issue}")
        else:
            logger.info("\n✓ No critical issues found")
        
        if self.warnings:
            logger.warning(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                logger.warning(f"  • {warning}")
        else:
            logger.info("✓ No warnings")
        
        # Final verdict
        logger.info("\n" + "=" * 80)
        if self.issues:
            logger.error("❌ VALIDATION FAILED")
            logger.error("Dataset has critical issues. Fix them before training.")
        elif self.warnings:
            logger.info("✓ VALIDATION PASSED WITH WARNINGS")
            logger.info("Dataset is usable but has minor issues. Review warnings.")
        else:
            logger.info("✅ VALIDATION PASSED")
            logger.info("Dataset is ready for training!")
        logger.info("=" * 80)
    
    def validate(self, visualize: bool = True, num_samples: int = 5):
        """
        Run complete validation.
        
        Args:
            visualize: Whether to generate visualization samples
            num_samples: Number of samples to visualize per split
        """
        logger.info("Starting YOLO Dataset Validation...\n")
        
        # Check structure
        if not self.check_dataset_structure():
            self.generate_report()
            return
        
        # Validate splits
        for split_name in ['train', 'valid', 'test']:
            self.validate_split(split_name)
        
        # Visualize samples
        if visualize:
            for split_name in ['train', 'valid', 'test']:
                if self.stats[split_name]['images'] > 0:
                    self.visualize_samples(split_name, num_samples)
        
        # Generate report
        self.generate_report()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Validate YOLO format dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='datasets/lpr',
        help='Path to YOLO dataset directory (default: datasets/lpr)'
    )
    
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Skip visualization of sample images'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=5,
        help='Number of samples to visualize per split (default: 5)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    dataset_path = Path(args.dataset)
    validator = YOLODatasetValidator(dataset_path)
    
    validator.validate(
        visualize=not args.no_visualize,
        num_samples=args.samples
    )
    
    # Return exit code based on validation result
    return 1 if validator.issues else 0


if __name__ == '__main__':
    sys.exit(main())

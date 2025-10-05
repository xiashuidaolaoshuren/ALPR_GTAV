"""
Dataset Quality Check Tool

This script validates the test dataset to ensure it meets the requirements
for Task 5: Initial Test Dataset Collection.

Checks:
- Image count (50-100 target)
- Condition diversity
- Angle diversity
- Metadata completeness
- Image quality (file size, dimensions)

Usage:
    python scripts/check_dataset_quality.py

Author: GTA V ALPR Development Team
Version: 1.0
"""

import os
import argparse
import logging
from pathlib import Path
from collections import Counter
import cv2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetQualityChecker:
    """Check quality and completeness of test dataset."""
    
    def __init__(self, images_dir, metadata_file):
        """
        Initialize checker.
        
        Args:
            images_dir: Path to directory containing test images
            metadata_file: Path to metadata.txt file
        """
        self.images_dir = Path(images_dir)
        self.metadata_file = Path(metadata_file)
        self.image_files = []
        self.metadata_entries = []
        self.issues = []
        self.warnings = []
    
    def scan_images(self):
        """Scan images directory for image files."""
        if not self.images_dir.exists():
            self.issues.append(f"Images directory not found: {self.images_dir}")
            return
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        for ext in image_extensions:
            self.image_files.extend(self.images_dir.glob(f'*{ext}'))
            self.image_files.extend(self.images_dir.glob(f'*{ext.upper()}'))
        
        self.image_files = sorted(self.image_files)
        logger.info(f"Found {len(self.image_files)} image files")
    
    def read_metadata(self):
        """Read and parse metadata file."""
        if not self.metadata_file.exists():
            self.issues.append(f"Metadata file not found: {self.metadata_file}")
            return
        
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and header
                if line and not line.startswith('#') and line != 'filename,condition,time_of_day,weather,angle,notes':
                    parts = line.split(',')
                    if len(parts) >= 6:
                        entry = {
                            'filename': parts[0],
                            'condition': parts[1],
                            'time_of_day': parts[2],
                            'weather': parts[3],
                            'angle': parts[4],
                            'notes': ','.join(parts[5:])  # Notes may contain commas
                        }
                        self.metadata_entries.append(entry)
        
        logger.info(f"Read {len(self.metadata_entries)} metadata entries")
    
    def check_image_count(self):
        """Check if image count meets requirements."""
        count = len(self.image_files)
        
        if count < 50:
            self.issues.append(f"Insufficient images: {count} (minimum: 50)")
        elif count > 100:
            self.warnings.append(f"More than target: {count} images (recommended: 50-100)")
            self.warnings.append("  Consider curating to best 100 images")
        else:
            logger.info(f"✓ Image count: {count} (within target range)")
        
        return count
    
    def check_condition_diversity(self):
        """Check diversity of conditions in metadata."""
        if not self.metadata_entries:
            self.issues.append("No metadata entries to check condition diversity")
            return
        
        # Count conditions
        time_counts = Counter(e['time_of_day'] for e in self.metadata_entries)
        weather_counts = Counter(e['weather'] for e in self.metadata_entries)
        condition_counts = Counter(e['condition'] for e in self.metadata_entries)
        
        logger.info("\nCondition Diversity:")
        logger.info(f"  Time of Day: {dict(time_counts)}")
        logger.info(f"  Weather: {dict(weather_counts)}")
        logger.info(f"  Combined: {dict(condition_counts)}")
        
        # Check minimums
        target_conditions = {
            'day_clear': (10, 15),
            'day_rain': (5, 10),
            'night_clear': (10, 15),
            'night_rain': (5, 10)
        }
        
        all_good = True
        for condition, (min_count, max_count) in target_conditions.items():
            actual = condition_counts.get(condition, 0)
            if actual < min_count:
                self.warnings.append(f"Low count for {condition}: {actual} (target: {min_count}-{max_count})")
                all_good = False
            elif actual >= min_count and actual <= max_count:
                logger.info(f"  ✓ {condition}: {actual} images (target: {min_count}-{max_count})")
            else:
                logger.info(f"  ✓ {condition}: {actual} images (exceeds target)")
        
        if all_good:
            logger.info("✓ Condition diversity meets requirements")
        else:
            self.warnings.append("Some conditions below target - consider collecting more")
    
    def check_angle_diversity(self):
        """Check diversity of camera angles."""
        if not self.metadata_entries:
            return
        
        angle_counts = Counter(e['angle'] for e in self.metadata_entries)
        
        logger.info("\nAngle Diversity:")
        for angle, count in angle_counts.items():
            percentage = (count / len(self.metadata_entries)) * 100
            logger.info(f"  {angle}: {count} ({percentage:.1f}%)")
        
        # Check for 'to_be_determined' or 'unknown'
        tbd = angle_counts.get('to_be_determined', 0) + angle_counts.get('unknown', 0)
        if tbd > 0:
            self.warnings.append(f"{tbd} images with undetermined angles - review metadata")
        
        # Rough distribution check (Front ~30%, Rear ~30%, Others ~40%)
        front_count = angle_counts.get('front', 0)
        rear_count = angle_counts.get('rear', 0)
        other_count = len(self.metadata_entries) - front_count - rear_count
        
        if front_count > 0 and rear_count > 0:
            logger.info("✓ Multiple angle types present")
        else:
            self.warnings.append("Limited angle diversity - try to capture front, rear, and side views")
    
    def check_metadata_completeness(self):
        """Check metadata completeness and consistency."""
        # Check if all images have metadata
        image_names = {f.name for f in self.image_files}
        metadata_names = {e['filename'] for e in self.metadata_entries}
        
        missing_metadata = image_names - metadata_names
        missing_images = metadata_names - image_names
        
        if missing_metadata:
            self.warnings.append(f"{len(missing_metadata)} images without metadata entries")
            logger.warning(f"  Images without metadata: {list(missing_metadata)[:5]}")
        
        if missing_images:
            self.warnings.append(f"{len(missing_images)} metadata entries without corresponding images")
        
        if not missing_metadata and not missing_images:
            logger.info("\n✓ Metadata completeness: All images have metadata entries")
        
        # Check for 'unknown' or incomplete fields
        unknown_count = 0
        needs_review_count = 0
        
        for entry in self.metadata_entries:
            if 'unknown' in [entry['condition'], entry['time_of_day'], entry['weather'], entry['angle']]:
                unknown_count += 1
            if 'needs_review' in entry['notes']:
                needs_review_count += 1
        
        if unknown_count > 0:
            self.warnings.append(f"{unknown_count} entries with 'unknown' fields - review and update")
        
        if needs_review_count > 0:
            self.warnings.append(f"{needs_review_count} entries marked 'needs_review' - review and update notes")
    
    def check_image_quality(self):
        """Check basic image quality metrics."""
        logger.info("\nImage Quality Check:")
        
        small_images = []
        large_images = []
        corrupt_images = []
        
        for img_file in self.image_files[:10]:  # Sample first 10 images
            try:
                # Check file size
                file_size = img_file.stat().st_size
                
                if file_size < 50_000:  # Less than 50KB
                    small_images.append(img_file.name)
                elif file_size > 5_000_000:  # More than 5MB
                    large_images.append(img_file.name)
                
                # Try to load image
                img = cv2.imread(str(img_file))
                if img is None:
                    corrupt_images.append(img_file.name)
                    continue
                
                height, width = img.shape[:2]
                
                # Check dimensions
                if width < 640 or height < 480:
                    self.warnings.append(f"Low resolution: {img_file.name} ({width}x{height})")
            
            except Exception as e:
                corrupt_images.append(img_file.name)
                logger.error(f"Error checking {img_file.name}: {e}")
        
        if corrupt_images:
            self.issues.append(f"Corrupt images detected: {corrupt_images}")
        
        if small_images:
            self.warnings.append(f"Very small files (possible low quality): {small_images}")
        
        if not corrupt_images and not small_images:
            logger.info("✓ Sample images passed quality check")
    
    def generate_report(self):
        """Generate and display quality check report."""
        logger.info("\n" + "="*60)
        logger.info("DATASET QUALITY REPORT")
        logger.info("="*60)
        
        # Summary
        logger.info(f"\nDataset Location: {self.images_dir}")
        logger.info(f"Total Images: {len(self.image_files)}")
        logger.info(f"Metadata Entries: {len(self.metadata_entries)}")
        
        # Issues (blocking)
        if self.issues:
            logger.error(f"\n❌ CRITICAL ISSUES ({len(self.issues)}):")
            for issue in self.issues:
                logger.error(f"  • {issue}")
        else:
            logger.info("\n✓ No critical issues found")
        
        # Warnings (non-blocking)
        if self.warnings:
            logger.warning(f"\n⚠ WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                logger.warning(f"  • {warning}")
        else:
            logger.info("✓ No warnings")
        
        # Overall assessment
        logger.info("\n" + "="*60)
        if not self.issues:
            if not self.warnings:
                logger.info("✅ DATASET QUALITY: EXCELLENT")
                logger.info("Dataset meets all requirements and is ready for annotation.")
            else:
                logger.info("✅ DATASET QUALITY: GOOD")
                logger.info("Dataset meets minimum requirements.")
                logger.info("Address warnings to improve quality.")
        else:
            logger.info("❌ DATASET QUALITY: NEEDS IMPROVEMENT")
            logger.info("Critical issues must be resolved before proceeding.")
        
        logger.info("="*60)
    
    def run_all_checks(self):
        """Run all quality checks."""
        logger.info("="*60)
        logger.info("Starting Dataset Quality Check")
        logger.info("="*60 + "\n")
        
        self.scan_images()
        self.read_metadata()
        
        if self.issues:
            # Can't proceed with further checks
            self.generate_report()
            return
        
        self.check_image_count()
        self.check_condition_diversity()
        self.check_angle_diversity()
        self.check_metadata_completeness()
        self.check_image_quality()
        
        self.generate_report()


def main():
    parser = argparse.ArgumentParser(
        description='Check quality and completeness of test dataset'
    )
    parser.add_argument('--images_dir', type=str,
                       default='outputs/test_images',
                       help='Directory containing test images')
    parser.add_argument('--metadata', type=str,
                       default='outputs/test_images/metadata.txt',
                       help='Path to metadata file')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    project_root = Path(__file__).parent.parent
    images_dir = project_root / args.images_dir
    metadata_file = project_root / args.metadata
    
    # Run checks
    checker = DatasetQualityChecker(str(images_dir), str(metadata_file))
    checker.run_all_checks()
    
    # Exit with appropriate code
    return 0 if not checker.issues else 1


if __name__ == "__main__":
    exit(main())

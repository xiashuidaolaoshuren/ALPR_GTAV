"""
Generate Metadata Template for Test Images

This script scans the outputs/test_images/ directory and creates metadata 
template entries for all images, which can then be reviewed and filled in.

Usage:
    python scripts/data_ingestion/generate_metadata.py [--output outputs/test_images/metadata.txt]

"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_condition_from_filename(filename):
    """
    Extract condition information from filename.
    
    Expected patterns:
    - day_clear_01_00123.jpg
    - night_rain_03_00456.jpg
    
    Returns:
        tuple: (condition, time_of_day, weather, estimated_angle)
    """
    filename = filename.lower()

    if 'day' in filename or 'daytime' in filename:
        time_of_day = 'day'
    elif 'night' in filename or 'nighttime' in filename:
        time_of_day = 'night'
    elif 'dawn' in filename or 'sunrise' in filename:
        time_of_day = 'dawn'
    elif 'dusk' in filename or 'sunset' in filename:
        time_of_day = 'dusk'
    else:
        time_of_day = 'unknown'

    if 'rain' in filename or 'rainy' in filename or 'wet' in filename:
        weather = 'rain'
    elif 'fog' in filename or 'foggy' in filename:
        weather = 'fog'
    elif 'overcast' in filename or 'cloudy' in filename:
        weather = 'overcast'
    elif 'clear' in filename or 'sunny' in filename:
        weather = 'clear'
    else:
        weather = 'unknown'

    if 'front' in filename:
        angle = 'front'
    elif 'rear' in filename or 'back' in filename:
        angle = 'rear'
    elif 'side' in filename:
        angle = 'side'
    elif '45' in filename or 'angle' in filename:
        angle = 'angled'
    else:
        angle = 'to_be_determined'

    if time_of_day != 'unknown' and weather != 'unknown':
        condition = f"{time_of_day}_{weather}"
    else:
        condition = 'unknown'

    return condition, time_of_day, weather, angle


def generate_metadata(images_dir, output_file, overwrite=False):
    """
    Generate metadata template for all images in directory.
    
    Args:
        images_dir: Path to directory containing images
        output_file: Path to output metadata.txt file
        overwrite: If True, overwrite existing file. If False, append.
    """
    images_path = Path(images_dir)

    if not images_path.exists():
        logger.error(f"Images directory not found: {images_dir}")
        return

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = sorted(
        {
            path
            for path in images_path.iterdir()
            if path.is_file() and path.suffix.lower() in image_extensions
        }
    )

    if not image_files:
        logger.warning(f"No image files found in {images_dir}")
        return

    logger.info(f"Found {len(image_files)} image files")

    existing_files = set()
    if not overwrite and os.path.exists(output_file):
        logger.info("Reading existing metadata entries...")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and line != 'filename,condition,time_of_day,weather,angle,notes':
                    parts = line.split(',')
                    if parts:
                        existing_files.add(parts[0])
        logger.info(f"Found {len(existing_files)} existing entries")

    new_entries = []
    skipped = 0

    for img_file in image_files:
        filename = img_file.name

        if filename in existing_files:
            skipped += 1
            continue

        condition, time_of_day, weather, angle = extract_condition_from_filename(filename)
        notes = "needs_review"
        entry = f"{filename},{condition},{time_of_day},{weather},{angle},{notes}"
        new_entries.append(entry)

    logger.info(f"Generated {len(new_entries)} new metadata entries")
    logger.info(f"Skipped {skipped} existing entries")

    mode = 'w' if overwrite else 'a'
    with open(output_file, mode, encoding='utf-8') as f:
        if overwrite:
            f.write("# GTA V ALPR Test Dataset Metadata\n")
            f.write("# \n")
            f.write("# Format: filename,condition,time_of_day,weather,angle,notes\n")
            f.write("# \n")
            f.write("# Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write("# \n")
            f.write("filename,condition,time_of_day,weather,angle,notes\n")

        for entry in new_entries:
            f.write(entry + "\n")

    logger.info(f"Metadata written to: {output_file}")

    if new_entries:
        logger.info("\nSample entries generated:")
        for entry in new_entries[:5]:
            logger.info(f"  {entry}")
        if len(new_entries) > 5:
            logger.info(f"  ... and {len(new_entries) - 5} more")

    logger.info("\nNEXT STEPS:")
    logger.info("1. Review metadata.txt and correct any 'unknown' or 'to_be_determined' values")
    logger.info("2. Update 'notes' column with specific observations:")
    logger.info("   - Lighting quality (good_lighting, shadow, glare)")
    logger.info("   - Distance (close_range, medium_distance, far_distance)")
    logger.info("   - Vehicle type (sedan, truck, suv, sports_car)")
    logger.info("   - Any issues (blurry, occlusion, low_resolution)")
    logger.info("3. Remove rows for images you decide to exclude")


def main():
    parser = argparse.ArgumentParser(
        description='Generate metadata template for test images'
    )
    parser.add_argument('--images_dir', type=str,
                        default='outputs/test_images',
                        help='Directory containing test images')
    parser.add_argument('--output', type=str,
                        default='outputs/test_images/metadata.txt',
                        help='Output metadata file path')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing metadata file')

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    images_dir = project_root / args.images_dir
    output_file = project_root / args.output

    logger.info("=" * 60)
    logger.info("Metadata Generation Tool")
    logger.info("=" * 60)
    logger.info(f"Images directory: {images_dir}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Mode: {'OVERWRITE' if args.overwrite else 'APPEND'}")
    logger.info("=" * 60)

    generate_metadata(str(images_dir), str(output_file), args.overwrite)

    return 0


if __name__ == "__main__":
    exit(main())

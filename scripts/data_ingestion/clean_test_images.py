"""
Interactive Test Image Cleaning Script

This script helps organize and clean test images by:
1. Displaying images one by one
2. Allowing user to keep/skip/delete each image
3. For kept images, prompting for angle classification (front/rear/angle)
4. Renaming files from {time}_{weather}_{location}_{id} to {time}_{weather}_{angle}_{id}
5. Handling duplicate filenames automatically

Usage:
    python scripts/data_ingestion/clean_test_images.py [--images_dir outputs/test_images]

Controls:
    - k/K: Keep image (will prompt for angle)
    - s/S: Skip image (move to next without changes)
    - d/D: Delete image
    - q/Q: Quit script
    - ESC: Close current image window

"""

import argparse
import logging
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImageCleaner:
    """Interactive image cleaning and organization tool"""

    def __init__(self, images_dir):
        self.images_dir = Path(images_dir)
        self.deleted_dir = self.images_dir / 'deleted'
        self.deleted_dir.mkdir(exist_ok=True)

        self.processed_files = set()
        self.angle_counters = defaultdict(lambda: defaultdict(int))

        self.stats = {
            'kept': 0,
            'skipped': 0,
            'deleted': 0,
            'renamed': 0,
            'total': 0
        }

    def parse_filename(self, filename):
        name_without_ext = filename.rsplit('.', 1)[0]
        extension = filename.rsplit('.', 1)[1] if '.' in filename else 'jpg'
        parts = name_without_ext.split('_')

        if len(parts) < 4:
            logger.warning(f"Filename format not recognized: {filename}")
            return None

        time_of_day = parts[0]
        weather = parts[1]
        location_parts = parts[2:-2]
        location = '_'.join(location_parts)
        session = parts[-2]
        image_id = parts[-1]

        return {
            'time_of_day': time_of_day,
            'weather': weather,
            'location': location,
            'session': session,
            'id': image_id,
            'extension': extension
        }

    def get_new_filename(self, parsed, angle):
        base_name = f"{parsed['time_of_day']}_{parsed['weather']}_{angle}"
        original_id = int(parsed['id'])
        new_id = original_id

        while True:
            new_filename = f"{base_name}_{new_id:05d}.{parsed['extension']}"
            new_path = self.images_dir / new_filename

            if not new_path.exists() and new_filename not in self.processed_files:
                break

            new_id += 1
            if new_id > original_id + 10000:
                logger.error(f"Could not find unique filename after {new_id - original_id} attempts")
                return None

        return new_filename

    def display_image(self, image_path):
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                logger.error(f"Failed to load image: {image_path}")
                return False

            max_height = 900
            max_width = 1600
            height, width = img.shape[:2]

            if height > max_height or width > max_width:
                scale = min(max_height / height, max_width / width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

            filename = image_path.name
            cv2.putText(img, filename, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Image Cleaner - Press K(eep), S(kip), D(elete), Q(uit)', img)
            return True

        except Exception as e:
            logger.error(f"Error displaying image {image_path}: {e}")
            return False

    def get_angle_input(self):
        print("\n" + "=" * 60)
        print("Select angle classification:")
        print("  1 - Front")
        print("  2 - Rear")
        print("  3 - Angle")
        print("  0 - Cancel (skip this image)")
        print("=" * 60)

        while True:
            choice = input("Enter choice (1/2/3/0): ").strip()

            if choice == '1':
                return 'front'
            if choice == '2':
                return 'rear'
            if choice == '3':
                return 'angle'
            if choice == '0':
                return None
            print("Invalid choice. Please enter 1, 2, 3, or 0.")

    def rename_image(self, image_path, angle):
        parsed = self.parse_filename(image_path.name)
        if parsed is None:
            logger.error(f"Could not parse filename: {image_path.name}")
            return False

        new_filename = self.get_new_filename(parsed, angle)
        if new_filename is None:
            logger.error(f"Could not generate new filename for: {image_path.name}")
            return False

        new_path = self.images_dir / new_filename

        try:
            image_path.rename(new_path)
            self.processed_files.add(new_filename)
            logger.info(f"Renamed: {image_path.name} -> {new_filename}")
            return True
        except Exception as e:
            logger.error(f"Error renaming {image_path.name}: {e}")
            return False

    def delete_image(self, image_path):
        try:
            dest_path = self.deleted_dir / image_path.name
            counter = 1
            while dest_path.exists():
                name_parts = image_path.stem, counter, image_path.suffix
                dest_path = self.deleted_dir / f"{name_parts[0]}_{name_parts[1]}{name_parts[2]}"
                counter += 1

            image_path.rename(dest_path)
            logger.info(f"Deleted: {image_path.name} -> {dest_path.relative_to(self.images_dir)}")
            return True

        except Exception as e:
            logger.error(f"Error deleting {image_path.name}: {e}")
            return False

    def process_images(self):
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.images_dir.glob(f'*{ext}'))
            image_files.extend(self.images_dir.glob(f'*{ext.upper()}'))

        image_files = sorted(image_files)

        if not image_files:
            logger.warning(f"No image files found in {self.images_dir}")
            return

        self.stats['total'] = len(image_files)
        logger.info(f"Found {len(image_files)} images to process")
        print("\n" + "=" * 60)
        print("CONTROLS:")
        print("  K - Keep image (will prompt for angle)")
        print("  S - Skip image (no changes)")
        print("  D - Delete image (move to deleted folder)")
        print("  Q - Quit script")
        print("=" * 60 + "\n")

        for idx, image_path in enumerate(image_files, 1):
            print(f"\n[{idx}/{len(image_files)}] Processing: {image_path.name}")

            if not self.display_image(image_path):
                print("Failed to display image. Skipping...")
                self.stats['skipped'] += 1
                continue

            while True:
                key = cv2.waitKey(0) & 0xFF

                if key in [ord('k'), ord('K')]:
                    cv2.destroyAllWindows()
                    angle = self.get_angle_input()

                    if angle is None:
                        print("Cancelled. Skipping image.")
                        self.stats['skipped'] += 1
                        break

                    if self.rename_image(image_path, angle):
                        self.stats['kept'] += 1
                        self.stats['renamed'] += 1
                        print(f"✓ Image kept and renamed with angle: {angle}")
                    else:
                        print("✗ Failed to rename image. Skipping.")
                        self.stats['skipped'] += 1
                    break

                if key in [ord('s'), ord('S')]:
                    cv2.destroyAllWindows()
                    print("Skipped.")
                    self.stats['skipped'] += 1
                    break

                if key in [ord('d'), ord('D')]:
                    cv2.destroyAllWindows()
                    confirm = input("Are you sure you want to delete this image? (y/n): ").strip().lower()
                    if confirm == 'y':
                        if self.delete_image(image_path):
                            self.stats['deleted'] += 1
                            print("✓ Image deleted.")
                        else:
                            print("✗ Failed to delete image.")
                            self.stats['skipped'] += 1
                    else:
                        print("Deletion cancelled. Skipping.")
                        self.stats['skipped'] += 1
                    break

                if key in [ord('q'), ord('Q')]:
                    cv2.destroyAllWindows()
                    print("\nQuitting early...")
                    return

                if key == 27:
                    cv2.destroyAllWindows()
                    print("Window closed. Skipping.")
                    self.stats['skipped'] += 1
                    break

        cv2.destroyAllWindows()

    def print_summary(self):
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total images processed: {self.stats['total']}")
        print(f"  - Kept & Renamed: {self.stats['kept']}")
        print(f"  - Skipped: {self.stats['skipped']}")
        print(f"  - Deleted: {self.stats['deleted']}")
        print("=" * 60)

        if self.stats['deleted'] > 0:
            print(f"\nDeleted images moved to: {self.deleted_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Interactive tool for cleaning and organizing test images'
    )
    parser.add_argument('--images_dir', type=str,
                        default='outputs/test_images',
                        help='Directory containing test images')

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    images_dir = project_root / args.images_dir

    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        return 1

    logger.info("=" * 60)
    logger.info("Interactive Test Image Cleaning Tool")
    logger.info("=" * 60)
    logger.info(f"Images directory: {images_dir}")
    logger.info("=" * 60)

    cleaner = ImageCleaner(images_dir)
    cleaner.process_images()
    cleaner.print_summary()

    return 0


if __name__ == "__main__":
    exit(main())

"""
Dataset Quality Validation Script

Validates YOLO format dataset for correctness and quality issues.

Usage:
    python scripts/annotation/validate_dataset.py --dataset datasets/lpr
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Validate YOLO format dataset")

    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset root directory")

    parser.add_argument(
        "--visualize", action="store_true", help="Show sample images with bounding boxes"
    )

    parser.add_argument(
        "--num-samples", type=int, default=10, help="Number of samples to visualize (default: 10)"
    )

    return parser.parse_args()


def load_yolo_label(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    Load YOLO format label file.

    Args:
        label_path: Path to label file

    Returns:
        List of (class_id, x_center, y_center, width, height)
    """
    if not label_path.exists():
        return []

    if label_path.stat().st_size == 0:
        return []

    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                boxes.append((class_id, x_center, y_center, width, height))

    return boxes


def validate_coordinates(boxes: List[Tuple]) -> Tuple[bool, List[str]]:
    """
    Validate that coordinates are in [0, 1] range.

    Args:
        boxes: List of bounding boxes

    Returns:
        (is_valid, error_messages)
    """
    errors = []

    for i, (class_id, x, y, w, h) in enumerate(boxes):
        if not (0 <= x <= 1):
            errors.append(f"Box {i}: x_center {x} out of range [0, 1]")
        if not (0 <= y <= 1):
            errors.append(f"Box {i}: y_center {y} out of range [0, 1]")
        if not (0 < w <= 1):
            errors.append(f"Box {i}: width {w} out of range (0, 1]")
        if not (0 < h <= 1):
            errors.append(f"Box {i}: height {h} out of range (0, 1]")

    return len(errors) == 0, errors


def validate_split(dataset_path: Path, split_name: str) -> Dict:
    """
    Validate a dataset split.

    Args:
        dataset_path: Root dataset directory
        split_name: Name of split (train/valid/test)

    Returns:
        Dictionary with validation statistics
    """
    split_dir = dataset_path / split_name
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    if not images_dir.exists():
        logger.error(f"{split_name}/images directory not found")
        return {}

    if not labels_dir.exists():
        logger.error(f"{split_name}/labels directory not found")
        return {}

    image_files = list(images_dir.glob("*.jpg"))
    label_files = list(labels_dir.glob("*.txt"))

    stats = {
        "num_images": len(image_files),
        "num_labels": len(label_files),
        "missing_labels": [],
        "missing_images": [],
        "empty_labels": [],
        "invalid_coordinates": [],
        "total_boxes": 0,
        "multi_box_images": 0,
        "valid": True,
    }

    # Check for missing labels
    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            stats["missing_labels"].append(img_path.name)
            stats["valid"] = False

    # Check for missing images
    for label_path in label_files:
        img_path = images_dir / f"{label_path.stem}.jpg"
        if not img_path.exists():
            stats["missing_images"].append(label_path.name)
            stats["valid"] = False

    # Validate labels
    for label_path in label_files:
        boxes = load_yolo_label(label_path)

        if len(boxes) == 0:
            stats["empty_labels"].append(label_path.name)
        else:
            stats["total_boxes"] += len(boxes)

            if len(boxes) > 1:
                stats["multi_box_images"] += 1

            is_valid, errors = validate_coordinates(boxes)
            if not is_valid:
                stats["invalid_coordinates"].append({"file": label_path.name, "errors": errors})
                stats["valid"] = False

    return stats


def draw_boxes_on_image(image_path: Path, label_path: Path) -> np.ndarray:
    """
    Draw bounding boxes on image.

    Args:
        image_path: Path to image
        label_path: Path to label file

    Returns:
        Image with boxes drawn
    """
    img = cv2.imread(str(image_path))
    if img is None:
        logger.error(f"Could not load image: {image_path}")
        return None

    height, width = img.shape[:2]
    boxes = load_yolo_label(label_path)

    for class_id, x_center, y_center, w, h in boxes:
        # Convert normalized to pixel coordinates
        x1 = int((x_center - w / 2) * width)
        y1 = int((y_center - h / 2) * height)
        x2 = int((x_center + w / 2) * width)
        y2 = int((y_center + h / 2) * height)

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        label = f"plate ({w * width:.0f}x{h * height:.0f})"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img


def visualize_samples(dataset_path: Path, split_name: str, num_samples: int):
    """
    Visualize random samples from split.

    Args:
        dataset_path: Root dataset directory
        split_name: Name of split
        num_samples: Number of samples to show
    """
    images_dir = dataset_path / split_name / "images"
    labels_dir = dataset_path / split_name / "labels"

    image_files = list(images_dir.glob("*.jpg"))

    if len(image_files) == 0:
        logger.warning(f"No images found in {split_name}")
        return

    # Select random samples
    np.random.seed(42)
    samples = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)

    for img_path in samples:
        label_path = labels_dir / f"{img_path.stem}.txt"

        img = draw_boxes_on_image(img_path, label_path)
        if img is not None:
            # Resize for display
            display_height = 600
            aspect = img.shape[1] / img.shape[0]
            display_width = int(display_height * aspect)
            img_resized = cv2.resize(img, (display_width, display_height))

            cv2.imshow(f"{split_name}: {img_path.name}", img_resized)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()

            if key == ord("q"):
                break


def print_stats(split_name: str, stats: Dict):
    """Print validation statistics."""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"{split_name.upper()} SET VALIDATION")
    logger.info(f"{'=' * 80}")

    logger.info(f"Images: {stats['num_images']}")
    logger.info(f"Labels: {stats['num_labels']}")
    logger.info(f"Total bounding boxes: {stats['total_boxes']}")
    logger.info(
        f"Average boxes per image: {stats['total_boxes'] / max(stats['num_images'], 1):.2f}"
    )
    logger.info(f"Images with multiple boxes: {stats['multi_box_images']}")
    logger.info(f"Empty labels: {len(stats['empty_labels'])}")

    if stats["missing_labels"]:
        logger.warning(f"Missing labels: {len(stats['missing_labels'])}")
        for name in stats["missing_labels"][:5]:
            logger.warning(f"  - {name}")
        if len(stats["missing_labels"]) > 5:
            logger.warning(f"  ... and {len(stats['missing_labels']) - 5} more")

    if stats["missing_images"]:
        logger.error(f"Missing images: {len(stats['missing_images'])}")
        for name in stats["missing_images"][:5]:
            logger.error(f"  - {name}")
        if len(stats["missing_images"]) > 5:
            logger.error(f"  ... and {len(stats['missing_images']) - 5} more")

    if stats["invalid_coordinates"]:
        logger.error(f"Invalid coordinates: {len(stats['invalid_coordinates'])}")
        for item in stats["invalid_coordinates"][:3]:
            logger.error(f"  {item['file']}:")
            for error in item["errors"]:
                logger.error(f"    - {error}")

    if stats["valid"]:
        logger.info("✅ Validation PASSED")
    else:
        logger.error("❌ Validation FAILED")


def main():
    """Main execution function."""
    args = parse_arguments()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return 1

    logger.info(f"Validating dataset: {dataset_path}")

    all_valid = True
    total_stats = {"images": 0, "labels": 0, "boxes": 0, "empty": 0}

    # Validate each split
    for split_name in ["train", "valid", "test"]:
        stats = validate_split(dataset_path, split_name)
        if stats:
            print_stats(split_name, stats)
            all_valid = all_valid and stats["valid"]

            total_stats["images"] += stats["num_images"]
            total_stats["labels"] += stats["num_labels"]
            total_stats["boxes"] += stats["total_boxes"]
            total_stats["empty"] += len(stats["empty_labels"])

    # Overall summary
    logger.info(f"\n{'=' * 80}")
    logger.info("OVERALL SUMMARY")
    logger.info(f"{'=' * 80}")
    logger.info(f"Total images: {total_stats['images']}")
    logger.info(f"Total labels: {total_stats['labels']}")
    logger.info(f"Total bounding boxes: {total_stats['boxes']}")
    logger.info(
        f"Empty labels: {
            total_stats['empty']} ({
            total_stats['empty'] / max(
                total_stats['labels'],
                1) * 100:.1f}%)"
    )
    logger.info(
        f"Average boxes per image: {total_stats['boxes'] / max(total_stats['images'], 1):.2f}"
    )

    if all_valid:
        logger.info("\n✅ DATASET VALIDATION PASSED")
    else:
        logger.error("\n❌ DATASET VALIDATION FAILED")

    # Visualization
    if args.visualize:
        logger.info("\nVisualizing samples...")
        for split_name in ["train", "valid", "test"]:
            visualize_samples(dataset_path, split_name, args.num_samples)

    return 0 if all_valid else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())

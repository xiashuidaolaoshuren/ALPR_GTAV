"""
Label Studio to YOLO Format Converter

Convert Label Studio JSON export to YOLO format for training.

Usage:
    python scripts/annotation/convert_labelstudio_to_yolo.py --input export.json --output datasets/lpr

The script will:
1. Parse Label Studio JSON export
2. Convert annotations to YOLO format (normalized coordinates)
3. Organize images and labels into train/valid/test splits
4. Generate classes.txt and data.yaml files

Format:
    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    All coordinates normalized to [0, 1]
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Convert Label Studio export to YOLO format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to Label Studio JSON export file'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='datasets/lpr',
        help='Output directory for YOLO dataset (default: datasets/lpr)'
    )

    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='Ratio of images for training set (default: 0.7)'
    )

    parser.add_argument(
        '--valid_ratio',
        type=float,
        default=0.2,
        help='Ratio of images for validation set (default: 0.2)'
    )

    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.1,
        help='Ratio of images for test set (default: 0.1)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for split reproducibility (default: 42)'
    )

    return parser.parse_args()


def load_labelstudio_export(export_path: str) -> List[Dict]:
    """
    Load Label Studio JSON export.

    Args:
        export_path: Path to JSON export file

    Returns:
        List of annotation tasks
    """
    export_path = Path(export_path)
    if not export_path.exists():
        raise FileNotFoundError(f"Export file not found: {export_path}")

    with open(export_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} tasks from {export_path}")
    return data


def convert_labelstudio_to_yolo(annotation: Dict, image_width: int, image_height: int) -> List[str]:
    """
    Convert Label Studio annotation to YOLO format.

    Args:
        annotation: Label Studio annotation dict
        image_width: Original image width
        image_height: Original image height

    Returns:
        List of YOLO format strings
    """
    yolo_lines = []

    for result in annotation.get('value', {}).get('rectanglelabels', []):
        x_percent = annotation['value']['x']
        y_percent = annotation['value']['y']
        width_percent = annotation['value']['width']
        height_percent = annotation['value']['height']

        x_center = (x_percent + width_percent / 2) / 100.0
        y_center = (y_percent + height_percent / 2) / 100.0
        width = width_percent / 100.0
        height = height_percent / 100.0

        class_id = 0
        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        yolo_lines.append(yolo_line)

    return yolo_lines


def split_dataset(tasks: List[Dict], train_ratio: float, valid_ratio: float,
                  test_ratio: float, seed: int) -> Tuple[List, List, List]:
    """
    Split dataset into train/valid/test sets.

    Args:
        tasks: List of annotation tasks
        train_ratio: Training set ratio
        valid_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed

    Returns:
        Tuple of (train_tasks, valid_tasks, test_tasks)
    """
    total_ratio = train_ratio + valid_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    random.seed(seed)
    shuffled_tasks = tasks.copy()
    random.shuffle(shuffled_tasks)

    n_total = len(shuffled_tasks)
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)

    train_tasks = shuffled_tasks[:n_train]
    valid_tasks = shuffled_tasks[n_train:n_train + n_valid]
    test_tasks = shuffled_tasks[n_train + n_valid:]

    logger.info(f"Dataset split: {len(train_tasks)} train, {len(valid_tasks)} valid, {len(test_tasks)} test")

    return train_tasks, valid_tasks, test_tasks


def create_yolo_dataset(tasks: List[Dict], output_dir: Path, split_name: str):
    """Create YOLO format dataset from tasks."""
    images_dir = output_dir / split_name / 'images'
    labels_dir = output_dir / split_name / 'labels'

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating {split_name} set...")

    for task in tasks:
        image_path = task.get('data', {}).get('image')
        if not image_path:
            logger.warning(f"No image path in task {task.get('id')}")
            continue

        image_filename = Path(image_path).name
        annotations = task.get('annotations', [])
        if not annotations:
            logger.warning(f"No annotations for {image_filename}, skipping")
            continue

        annotation_data = annotations[0]
        image_width = annotation_data.get('original_width', 1920)
        image_height = annotation_data.get('original_height', 1080)

        yolo_lines = []
        for result in annotation_data.get('result', []):
            if result.get('type') == 'rectanglelabels':
                value = result['value']
                x_percent = value['x']
                y_percent = value['y']
                width_percent = value['width']
                height_percent = value['height']

                x_center = (x_percent + width_percent / 2) / 100.0
                y_center = (y_percent + height_percent / 2) / 100.0
                width = width_percent / 100.0
                height = height_percent / 100.0

                yolo_line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                yolo_lines.append(yolo_line)

        if not yolo_lines:
            logger.warning(f"No valid annotations for {image_filename}")
            continue

        label_filename = Path(image_filename).stem + '.txt'
        label_path = labels_dir / label_filename

        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

        logger.debug(f"Processed {image_filename} -> {len(yolo_lines)} annotations")

    logger.info(f"Created {len(list(labels_dir.glob('*.txt')))} label files in {split_name} set")


def create_dataset_yaml(output_dir: Path, class_names: List[str]):
    """Create data.yaml file for YOLO training."""
    yaml_content = f"""# GTA V License Plate Detection Dataset
# Generated by convert_labelstudio_to_yolo.py

path: {output_dir.absolute()}  # dataset root dir
train: train/images  # train images (relative to 'path')
val: valid/images  # val images (relative to 'path')
test: test/images  # test images (relative to 'path')

# Classes
nc: {len(class_names)}  # number of classes
names: {class_names}  # class names
"""

    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    logger.info(f"Created data.yaml at {yaml_path}")


def create_classes_txt(output_dir: Path, class_names: List[str]):
    """Create classes.txt file."""
    classes_path = output_dir / 'classes.txt'
    with open(classes_path, 'w') as f:
        f.write('\n'.join(class_names))

    logger.info(f"Created classes.txt at {classes_path}")


def main():
    """Main execution function."""
    args = parse_arguments()

    try:
        logger.info("Loading Label Studio export...")
        tasks = load_labelstudio_export(args.input)

        annotated_tasks = [t for t in tasks if t.get('annotations')]
        logger.info(f"Found {len(annotated_tasks)} annotated tasks")

        if not annotated_tasks:
            logger.error("No annotated tasks found in export!")
            return 1

        logger.info("Splitting dataset...")
        train_tasks, valid_tasks, test_tasks = split_dataset(
            annotated_tasks,
            args.train_ratio,
            args.valid_ratio,
            args.test_ratio,
            args.seed
        )

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        create_yolo_dataset(train_tasks, output_dir, 'train')
        create_yolo_dataset(valid_tasks, output_dir, 'valid')
        create_yolo_dataset(test_tasks, output_dir, 'test')

        class_names = ['license_plate']
        create_classes_txt(output_dir, class_names)
        create_dataset_yaml(output_dir, class_names)

        logger.info("=" * 80)
        logger.info("✅ Conversion completed successfully!")
        logger.info("=" * 80)
        logger.info(f"\nDataset created at: {output_dir.absolute()}")
        logger.info("\nStructure:")
        logger.info(f"  {output_dir}/")
        logger.info(f"  ├── train/")
        logger.info(f"  │   ├── images/")
        logger.info(f"  │   └── labels/")
        logger.info(f"  ├── valid/")
        logger.info(f"  │   ├── images/")
        logger.info(f"  │   └── labels/")
        logger.info(f"  ├── test/")
        logger.info(f"  │   ├── images/")
        logger.info(f"  │   └── labels/")
        logger.info(f"  ├── data.yaml")
        logger.info(f"  └── classes.txt")
        logger.info("\n⚠️  Note: You need to copy the actual image files to the images/ directories")

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

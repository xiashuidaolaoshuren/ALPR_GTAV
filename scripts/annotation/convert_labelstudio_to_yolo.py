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
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

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
    
    parser.add_argument(
        '--filter-readability',
        type=str,
        default=None,
        help='Comma-separated list of readability values to include (e.g., "clear,blurred"). '
             'If not specified, all annotations are included. '
             'Valid values: clear, blurred, occluded'
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


def sanitize_image_filename(filename: str) -> str:
    """Remove internal ID prefixes from Label Studio filenames."""
    name = Path(filename).name
    if '-' in name:
        prefix, remainder = name.split('-', 1)
        if remainder:
            return remainder
    return name


def create_yolo_dataset(tasks: List[Dict], output_dir: Path, split_name: str,
                        readability_filter: List[str] = None) -> Set[str]:
    """
    Create YOLO format dataset from tasks.
    
    Args:
        tasks: List of annotation tasks
        output_dir: Output directory
        split_name: Split name (train/valid/test)
        readability_filter: List of readability values to include (e.g., ['clear', 'blurred'])
                          If None, all annotations are included
    """
    images_dir = output_dir / split_name / 'images'
    labels_dir = output_dir / split_name / 'labels'

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating {split_name} set...")
    if readability_filter:
        logger.info(f"  Filtering readability: {', '.join(readability_filter)}")

    processed_filenames: Set[str] = set()

    for task in tasks:
        image_path = task.get('data', {}).get('image')
        if not image_path:
            logger.warning(f"No image path in task {task.get('id')}")
            continue

        image_filename = Path(image_path).name
        sanitized_image_filename = sanitize_image_filename(image_filename)
        annotations = task.get('annotations') or []
        annotation_data = annotations[0] if annotations else None
        results = annotation_data.get('result', []) if annotation_data else []

        readability_value = None
        if readability_filter and annotation_data:
            for res in results:
                if res.get('type') == 'choices' and res.get('from_name') == 'readability':
                    choices = res['value'].get('choices', [])
                    if choices:
                        readability_value = choices[0]
                    break

        skip_boxes_for_readability = (
            readability_filter
            and readability_value is not None
            and readability_value not in readability_filter
        )

        if skip_boxes_for_readability:
            logger.debug(
                "Skipping bounding boxes for %s due to readability '%s' not in filter",
                image_filename,
                readability_value,
            )

        yolo_lines: List[str] = []
        filtered_count = 0
        for result in results:
            if result.get('type') != 'rectanglelabels':
                continue

            if skip_boxes_for_readability:
                filtered_count += 1
                continue

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

        if filtered_count > 0 and readability_filter:
            logger.debug(
                "Filtered out %d bounding box(es) from %s based on readability",
                filtered_count,
                image_filename,
            )

        label_stem = Path(sanitized_image_filename).stem
        label_path = labels_dir / f"{label_stem}.txt"

        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

        if yolo_lines:
            logger.debug(f"Processed {image_filename} -> {len(yolo_lines)} annotations")
        else:
            logger.info(
                "No bounding boxes for %s (or filtered out); wrote empty label file.",
                image_filename,
            )
        processed_filenames.add(sanitized_image_filename)

    logger.info(f"Created {len(list(labels_dir.glob('*.txt')))} label files in {split_name} set")
    return processed_filenames


def find_source_image(source_dir: Path, filename: str) -> Optional[Path]:
    """Locate the source image within the outputs/test_images directory."""
    direct_path = source_dir / filename
    if direct_path.exists():
        return direct_path

    matches = list(source_dir.rglob(filename))
    if matches:
        return matches[0]
    return None


def copy_images_for_split(filenames: Iterable[str], source_dir: Path, destination_dir: Path):
    """Copy matching images into the split's images directory."""
    copied = 0
    missing = []
    destination_dir.mkdir(parents=True, exist_ok=True)

    for filename in sorted(set(filenames)):
        source_path = find_source_image(source_dir, filename)
        if source_path is None:
            missing.append(filename)
            continue

        destination_path = destination_dir / filename
        try:
            shutil.copy2(source_path, destination_path)
            copied += 1
        except Exception as err:
            logger.error(f"Failed to copy {source_path} -> {destination_path}: {err}")

    logger.info(f"Copied {copied} images into {destination_dir}")
    if missing:
        logger.warning(
            "Missing source images for %d file(s) in %s: %s",
            len(missing), destination_dir, ', '.join(missing)
        )


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

        image_tasks = [t for t in tasks if t.get('data', {}).get('image')]
        annotated_count = sum(1 for t in image_tasks if t.get('annotations'))
        logger.info(
            "Found %d tasks with image references (%d annotated)",
            len(image_tasks),
            annotated_count,
        )

        if not image_tasks:
            logger.error("No tasks with image data found in export!")
            return 1

        if annotated_count == 0:
            logger.warning(
                "No annotations present; empty label files will be generated for all images."
            )

        logger.info("Splitting dataset...")
        train_tasks, valid_tasks, test_tasks = split_dataset(
            image_tasks,
            args.train_ratio,
            args.valid_ratio,
            args.test_ratio,
            args.seed
        )

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parse readability filter if provided
        readability_filter = None
        if args.filter_readability:
            readability_filter = [r.strip() for r in args.filter_readability.split(',')]
            logger.info(f"Applying readability filter: {readability_filter}")

        train_files = create_yolo_dataset(train_tasks, output_dir, 'train', readability_filter)
        valid_files = create_yolo_dataset(valid_tasks, output_dir, 'valid', readability_filter)
        test_files = create_yolo_dataset(test_tasks, output_dir, 'test', readability_filter)

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
        source_images_dir = Path(__file__).resolve().parents[2] / 'outputs' / 'test_images'
        if source_images_dir.exists():
            logger.info("\nCopying images from %s into dataset structure...", source_images_dir)
            copy_images_for_split(train_files, source_images_dir, output_dir / 'train' / 'images')
            copy_images_for_split(valid_files, source_images_dir, output_dir / 'valid' / 'images')
            copy_images_for_split(test_files, source_images_dir, output_dir / 'test' / 'images')
        else:
            logger.warning(
                "Source image directory not found at %s; skipping image copy step.",
                source_images_dir
            )
        
        if args.filter_readability:
            logger.info(f"\n📊 Readability filter applied: {', '.join(readability_filter)}")
            logger.info("   Use --filter-readability 'clear' to create a recognition-only dataset")
            logger.info("   Omit the flag to include all annotations for detection")

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

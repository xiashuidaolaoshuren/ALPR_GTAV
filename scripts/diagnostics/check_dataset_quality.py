"""
Dataset Quality Check Tool

Validate the curated test image set to ensure it meets the targets for
Task 5: Initial Test Dataset Collection.

Checks performed:
- Image count (50-100 target)
- Condition diversity
- Angle diversity
- Metadata completeness
- Basic image quality (file size, dimensions)

Usage:
    python scripts/diagnostics/check_dataset_quality.py

"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import cv2

# Configure logging once per process (compatible with wrapper invocations)
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DatasetQualityChecker:
    """Check quality and completeness of test dataset."""

    def __init__(self, images_dir: Path, metadata_file: Path) -> None:
        self.images_dir = Path(images_dir)
        self.metadata_file = Path(metadata_file)
        self.image_files: list[Path] = []
        self.metadata_entries: list[dict[str, str]] = []
        self.issues: list[str] = []
        self.warnings: list[str] = []

    def scan_images(self) -> None:
        """Scan images directory for image files."""
        if not self.images_dir.exists():
            self.issues.append(f"Images directory not found: {self.images_dir}")
            return

        image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
        for ext in image_extensions:
            self.image_files.extend(self.images_dir.glob(f"*{ext}"))
            self.image_files.extend(self.images_dir.glob(f"*{ext.upper()}"))

        self.image_files = sorted(self.image_files)
        logger.info("Found %s image files", len(self.image_files))

    def read_metadata(self) -> None:
        """Read and parse metadata file."""
        if not self.metadata_file.exists():
            self.issues.append(f"Metadata file not found: {self.metadata_file}")
            return

        with self.metadata_file.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line == "filename,condition,time_of_day,weather,angle,notes":
                    continue

                parts = line.split(",")
                if len(parts) < 6:
                    continue

                self.metadata_entries.append(
                    {
                        "filename": parts[0],
                        "condition": parts[1],
                        "time_of_day": parts[2],
                        "weather": parts[3],
                        "angle": parts[4],
                        "notes": ",".join(parts[5:]),
                    }
                )

        logger.info("Read %s metadata entries", len(self.metadata_entries))

    def check_image_count(self) -> int:
        """Check if image count meets requirements."""
        count = len(self.image_files)

        if count < 50:
            self.issues.append(f"Insufficient images: {count} (minimum: 50)")
        elif count > 100:
            self.warnings.append(f"More than target: {count} images (recommended: 50-100)")
            self.warnings.append("  Consider curating to the best 100 images")
        else:
            logger.info("\u2713 Image count: %s (within target range)", count)

        return count

    def check_condition_diversity(self) -> None:
        """Check diversity of conditions in metadata."""
        if not self.metadata_entries:
            self.issues.append("No metadata entries to check condition diversity")
            return

        time_counts = Counter(entry["time_of_day"] for entry in self.metadata_entries)
        weather_counts = Counter(entry["weather"] for entry in self.metadata_entries)
        condition_counts = Counter(entry["condition"] for entry in self.metadata_entries)

        logger.info("\nCondition Diversity:")
        logger.info("  Time of Day: %s", dict(time_counts))
        logger.info("  Weather: %s", dict(weather_counts))
        logger.info("  Combined: %s", dict(condition_counts))

        target_conditions = {
            "day_clear": (10, 15),
            "day_rain": (5, 10),
            "night_clear": (10, 15),
            "night_rain": (5, 10),
        }

        all_good = True
        for condition, (min_count, max_count) in target_conditions.items():
            actual = condition_counts.get(condition, 0)
            if actual < min_count:
                self.warnings.append(
                    f"Low count for {condition}: {actual} (target: {min_count}-{max_count})"
                )
                all_good = False
            elif actual <= max_count:
                logger.info(
                    "  \u2713 %s: %s images (target: %s-%s)",
                    condition,
                    actual,
                    min_count,
                    max_count,
                )
            else:
                logger.info("  \u2713 %s: %s images (exceeds target)", condition, actual)

        if all_good:
            logger.info("\u2713 Condition diversity meets requirements")
        else:
            self.warnings.append("Some conditions below target - consider collecting more")

    def check_angle_diversity(self) -> None:
        """Check diversity of camera angles."""
        if not self.metadata_entries:
            return

        angle_counts = Counter(entry["angle"] for entry in self.metadata_entries)

        logger.info("\nAngle Diversity:")
        for angle, count in angle_counts.items():
            percentage = (count / len(self.metadata_entries)) * 100
            logger.info("  %s: %s (%.1f%%)", angle, count, percentage)

        tbd_count = angle_counts.get("to_be_determined", 0) + angle_counts.get("unknown", 0)
        if tbd_count:
            self.warnings.append(f"{tbd_count} images with undetermined angles - review metadata")

        front = angle_counts.get("front", 0)
        rear = angle_counts.get("rear", 0)
        if front == 0 or rear == 0:
            self.warnings.append("Limited angle diversity - target front, rear, and side views")
        else:
            logger.info("\u2713 Multiple angle types present")

    def check_metadata_completeness(self) -> None:
        """Check metadata completeness and consistency."""
        image_names = {file.name for file in self.image_files}
        metadata_names = {entry["filename"] for entry in self.metadata_entries}

        missing_metadata = image_names - metadata_names
        missing_images = metadata_names - image_names

        if missing_metadata:
            self.warnings.append(f"{len(missing_metadata)} images without metadata entries")
            logger.warning("  Images without metadata: %s", list(missing_metadata)[:5])

        if missing_images:
            self.warnings.append(
                f"{len(missing_images)} metadata entries without corresponding images"
            )

        if not missing_metadata and not missing_images:
            logger.info("\n\u2713 Metadata completeness: all images accounted for")

        unknown_fields = sum(
            "unknown"
            in {entry["condition"], entry["time_of_day"], entry["weather"], entry["angle"]}
            for entry in self.metadata_entries
        )
        if unknown_fields:
            self.warnings.append(
                f"{unknown_fields} entries contain 'unknown' fields - update metadata"
            )

        needs_review = sum("needs_review" in entry["notes"] for entry in self.metadata_entries)
        if needs_review:
            self.warnings.append(
                f"{needs_review} entries marked 'needs_review' - resolve before training"
            )

    def check_image_quality(self) -> None:
        """Check basic image quality metrics."""
        logger.info("\nImage Quality Check:")

        small_files: list[str] = []
        corrupt_files: list[str] = []

        for img_file in self.image_files[:10]:
            try:
                file_size = img_file.stat().st_size
                if file_size < 50_000:
                    small_files.append(img_file.name)

                image = cv2.imread(str(img_file))
                if image is None:
                    corrupt_files.append(img_file.name)
                    continue

                height, width = image.shape[:2]
                if width < 640 or height < 480:
                    self.warnings.append(
                        f"Low resolution: {img_file.name} ({width}x{height}) - target >= 640x480"
                    )
            except Exception as exc:
                corrupt_files.append(img_file.name)
                logger.error("Error checking %s: %s", img_file.name, exc)

        if corrupt_files:
            self.issues.append(f"Corrupt images detected: {corrupt_files}")

        if small_files:
            self.warnings.append(f"Very small files (potential low quality): {small_files}")

        if not corrupt_files and not small_files:
            logger.info("\u2713 Sample images passed quality check")

    def generate_report(self) -> None:
        """Generate and display quality check report."""
        logger.info("\n" + "=" * 60)
        logger.info("DATASET QUALITY REPORT")
        logger.info("=" * 60)

        logger.info("\nDataset Location: %s", self.images_dir)
        logger.info("Total Images: %s", len(self.image_files))
        logger.info("Metadata Entries: %s", len(self.metadata_entries))

        if self.issues:
            logger.error("\n\u274c CRITICAL ISSUES (%s):", len(self.issues))
            for issue in self.issues:
                logger.error("  • %s", issue)
        else:
            logger.info("\n\u2713 No critical issues found")

        if self.warnings:
            logger.warning("\n\u26a0 WARNINGS (%s):", len(self.warnings))
            for warning in self.warnings:
                logger.warning("  • %s", warning)
        else:
            logger.info("\u2713 No warnings")

        logger.info("\n" + "=" * 60)
        if self.issues:
            logger.info("\u274c DATASET QUALITY: NEEDS IMPROVEMENT")
            logger.info("Resolve critical issues before proceeding with annotation.")
        elif self.warnings:
            logger.info("\u2713 DATASET QUALITY: GOOD")
            logger.info("Dataset meets minimum requirements. Address warnings to improve quality.")
        else:
            logger.info("\u2705 DATASET QUALITY: EXCELLENT")
            logger.info("Dataset meets all requirements and is ready for annotation.")
        logger.info("=" * 60)

    def run_all_checks(self) -> None:
        """Run all quality checks."""
        logger.info("=" * 60)
        logger.info("Starting Dataset Quality Check")
        logger.info("=" * 60 + "\n")

        self.scan_images()
        self.read_metadata()

        if self.issues:
            self.generate_report()
            return

        self.check_image_count()
        self.check_condition_diversity()
        self.check_angle_diversity()
        self.check_metadata_completeness()
        self.check_image_quality()
        self.generate_report()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check quality and completeness of test dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("outputs/test_images"),
        help="Directory containing curated test images",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("outputs/test_images/metadata.txt"),
        help="Path to metadata file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[2]
    images_dir = project_root / args.images_dir
    metadata_file = project_root / args.metadata

    checker = DatasetQualityChecker(images_dir, metadata_file)
    checker.run_all_checks()
    return 0 if not checker.issues else 1


if __name__ == "__main__":
    raise SystemExit(main())

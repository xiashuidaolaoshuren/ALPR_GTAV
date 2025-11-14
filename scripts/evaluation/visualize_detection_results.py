"""
Visualization Script for Detection Results

Create visualizations from detection evaluation results and export
publication-ready charts.

Usage:
    python scripts/evaluation/visualize_detection_results.py [options]

Examples:
    # Basic visualization
    python scripts/evaluation/visualize_detection_results.py

    # Custom input and output directory
    python scripts/evaluation/visualize_detection_results.py --input outputs/detection_results.json --output_dir outputs/visualizations  # noqa: E501
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

# Ensure matplotlib does not require an interactive backend for headless runs
plt.switch_backend("Agg")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize detection evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/detection_results.json",
        help="Path to detection results JSON (default: outputs/detection_results.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save visualizations (default: outputs)",
    )
    return parser.parse_args()


def load_results(results_path: str) -> Dict:
    """Load results from JSON file."""
    path = Path(results_path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    with open(path, "r") as f:
        data = json.load(f)
    logger.info("Loaded results from %s", path)
    return data


def _sorted_conditions(stats: Dict) -> list[str]:
    return sorted(stats["by_condition"].keys())


def plot_detection_rate_by_condition(stats: Dict, output_dir: Path):
    conditions = _sorted_conditions(stats)
    detection_rates = [stats["by_condition"][c]["detection_rate"] * 100 for c in conditions]
    colors = [
        "#2ecc71" if rate >= 70 else "#f39c12" if rate >= 50 else "#e74c3c"
        for rate in detection_rates
    ]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(conditions, detection_rates, color=colors, alpha=0.85, edgecolor="black")
    for bar, rate in zip(bars, detection_rates):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.xlabel("Condition", fontsize=12, fontweight="bold")
    plt.ylabel("Detection Rate (%)", fontsize=12, fontweight="bold")
    plt.title("License Plate Detection Rate by Condition", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 110)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = output_dir / "detection_rate_by_condition.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved detection rate plot to %s", path)


def plot_avg_detections_by_condition(stats: Dict, output_dir: Path):
    conditions = _sorted_conditions(stats)
    avg_detections = [stats["by_condition"][c]["avg_detections_per_image"] for c in conditions]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(conditions, avg_detections, color="#3498db", alpha=0.85, edgecolor="black")
    for bar, avg in zip(bars, avg_detections):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{avg:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.xlabel("Condition", fontsize=12, fontweight="bold")
    plt.ylabel("Average Detections per Image", fontsize=12, fontweight="bold")
    plt.title(
        "Average License Plate Detections per Image by Condition", fontsize=14, fontweight="bold"
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, max(avg_detections) * 1.2 if avg_detections else 1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = output_dir / "avg_detections_by_condition.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved average detections plot to %s", path)


def plot_confidence_by_condition(stats: Dict, output_dir: Path):
    conditions = _sorted_conditions(stats)
    avg_confidences = [stats["by_condition"][c]["avg_confidence"] for c in conditions]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(conditions, avg_confidences, color="#9b59b6", alpha=0.85, edgecolor="black")
    for bar, conf in zip(bars, avg_confidences):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{conf:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.xlabel("Condition", fontsize=12, fontweight="bold")
    plt.ylabel("Average Confidence", fontsize=12, fontweight="bold")
    plt.title("Average Detection Confidence by Condition", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = output_dir / "avg_confidence_by_condition.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved confidence plot to %s", path)


def plot_dataset_distribution(stats: Dict, output_dir: Path):
    conditions = _sorted_conditions(stats)
    num_images = [stats["by_condition"][c]["num_images"] for c in conditions]

    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set3(range(len(conditions)))
    wedges, texts, autotexts = plt.pie(
        num_images,
        labels=conditions,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 10, "fontweight": "bold"},
    )
    for i, (condition, count) in enumerate(zip(conditions, num_images)):
        texts[i].set_text(f"{condition}\n({count} images)")

    plt.title("Test Dataset Distribution by Condition", fontsize=14, fontweight="bold", pad=20)
    plt.axis("equal")
    plt.tight_layout()

    path = output_dir / "dataset_distribution.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved dataset distribution plot to %s", path)


def plot_comparison_chart(stats: Dict, output_dir: Path):
    conditions = _sorted_conditions(stats)
    detection_rates = [stats["by_condition"][c]["detection_rate"] * 100 for c in conditions]
    avg_detections = [stats["by_condition"][c]["avg_detections_per_image"] * 10 for c in conditions]
    avg_confidences = [stats["by_condition"][c]["avg_confidence"] * 100 for c in conditions]

    x = np.arange(len(conditions))
    width = 0.25

    plt.figure(figsize=(14, 7))
    plt.bar(
        x - width, detection_rates, width, label="Detection Rate (%)", color="#2ecc71", alpha=0.85
    )
    plt.bar(x, avg_detections, width, label="Avg Detections (×10)", color="#3498db", alpha=0.85)
    plt.bar(
        x + width,
        avg_confidences,
        width,
        label="Avg Confidence (×100)",
        color="#9b59b6",
        alpha=0.85,
    )

    plt.xlabel("Condition", fontsize=12, fontweight="bold")
    plt.ylabel("Value", fontsize=12, fontweight="bold")
    plt.title("Detection Performance Comparison by Condition", fontsize=14, fontweight="bold")
    plt.xticks(x, conditions, rotation=45, ha="right")
    plt.legend(fontsize=10)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = output_dir / "performance_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved comparison chart to %s", path)


def plot_overall_summary(stats: Dict, output_dir: Path):
    overall = stats["overall"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("of")
    fig.suptitle("Detection Performance Summary", fontsize=18, fontweight="bold", y=0.95)

    metrics = [
        ("Total Images", overall["total_images"], ""),
        ("Images with Detection", overall["images_with_detection"], ""),
        ("Detection Rate", overall["detection_rate"], "%"),
        ("Total Detections", overall["total_detections"], ""),
        ("Avg Detections per Image", overall["avg_detections_per_image"], ""),
        ("Avg Confidence", overall["avg_confidence"], ""),
    ]

    y_pos = 0.8
    for label, value, suffix in metrics:
        if suffix == "%":
            formatted = f"{value * 100:.2f}%"
        elif isinstance(value, float):
            formatted = f"{value:.3f}"
        else:
            formatted = str(value)

        ax.text(0.1, y_pos, f"{label}:", fontsize=14, fontweight="bold", ha="left")
        ax.text(0.9, y_pos, formatted, fontsize=14, ha="right")
        y_pos -= 0.12

    plt.tight_layout()

    path = output_dir / "summary.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved summary figure to %s", path)


def main():
    args = parse_arguments()
    try:
        logger.info("Loading detection results...")
        data = load_results(args.input)
        stats = data["statistics"]

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating visualizations...")
        plot_detection_rate_by_condition(stats, output_dir)
        plot_avg_detections_by_condition(stats, output_dir)
        plot_confidence_by_condition(stats, output_dir)
        plot_dataset_distribution(stats, output_dir)
        plot_comparison_chart(stats, output_dir)
        plot_overall_summary(stats, output_dir)

        logger.info("\n✅ Visualizations saved to: %s", output_dir)
        logger.info("Generated files:")
        for filename in (
            "detection_rate_by_condition.png",
            "avg_detections_by_condition.png",
            "avg_confidence_by_condition.png",
            "dataset_distribution.png",
            "performance_comparison.png",
            "summary.png",
        ):
            logger.info("  - %s", filename)
        return 0
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        return 1
    except Exception as exc:  # pragma: no cover - diagnostic logging
        logger.error("An error occurred: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

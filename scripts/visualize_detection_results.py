"""
Visualization Script for Detection Results

Create visualizations from detection evaluation results.
Generate bar charts, pie charts, and analysis plots.

Usage:
    python scripts/visualize_detection_results.py [options]

Examples:
    # Basic visualization
    python scripts/visualize_detection_results.py
    
    # Custom input and output
    python scripts/visualize_detection_results.py --input outputs/detection_results.json --output_dir outputs/visualizations
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize detection evaluation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='outputs/detection_results.json',
        help='Path to detection results JSON (default: outputs/detection_results.json)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs',
        help='Directory to save visualizations (default: outputs)'
    )
    
    return parser.parse_args()


def load_results(results_path: str) -> Dict:
    """Load results from JSON file."""
    results_path = Path(results_path)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded results from {results_path}")
    return data


def plot_detection_rate_by_condition(stats: Dict, output_dir: Path):
    """
    Create bar chart showing detection rate by condition.
    
    Args:
        stats: Statistics dictionary
        output_dir: Directory to save plot
    """
    by_condition = stats['by_condition']
    
    # Sort conditions for consistent display
    conditions = sorted(by_condition.keys())
    detection_rates = [by_condition[c]['detection_rate'] * 100 for c in conditions]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create bars with colors
    colors = ['#2ecc71' if rate >= 70 else '#f39c12' if rate >= 50 else '#e74c3c' 
              for rate in detection_rates]
    bars = plt.bar(conditions, detection_rates, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, rate in zip(bars, detection_rates):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xlabel('Condition', fontsize=12, fontweight='bold')
    plt.ylabel('Detection Rate (%)', fontsize=12, fontweight='bold')
    plt.title('License Plate Detection Rate by Condition', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 110)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / 'detection_rate_by_condition.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved detection rate plot to {output_path}")


def plot_avg_detections_by_condition(stats: Dict, output_dir: Path):
    """
    Create bar chart showing average detections per image by condition.
    
    Args:
        stats: Statistics dictionary
        output_dir: Directory to save plot
    """
    by_condition = stats['by_condition']
    
    conditions = sorted(by_condition.keys())
    avg_detections = [by_condition[c]['avg_detections_per_image'] for c in conditions]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(conditions, avg_detections, color='#3498db', alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bar, avg in zip(bars, avg_detections):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{avg:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xlabel('Condition', fontsize=12, fontweight='bold')
    plt.ylabel('Average Detections per Image', fontsize=12, fontweight='bold')
    plt.title('Average License Plate Detections per Image by Condition', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(avg_detections) * 1.2)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_path = output_dir / 'avg_detections_by_condition.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved average detections plot to {output_path}")


def plot_confidence_by_condition(stats: Dict, output_dir: Path):
    """
    Create bar chart showing average confidence by condition.
    
    Args:
        stats: Statistics dictionary
        output_dir: Directory to save plot
    """
    by_condition = stats['by_condition']
    
    conditions = sorted(by_condition.keys())
    avg_confidences = [by_condition[c]['avg_confidence'] for c in conditions]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(conditions, avg_confidences, color='#9b59b6', alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bar, conf in zip(bars, avg_confidences):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{conf:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xlabel('Condition', fontsize=12, fontweight='bold')
    plt.ylabel('Average Confidence', fontsize=12, fontweight='bold')
    plt.title('Average Detection Confidence by Condition', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_path = output_dir / 'avg_confidence_by_condition.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved confidence plot to {output_path}")


def plot_dataset_distribution(stats: Dict, output_dir: Path):
    """
    Create pie chart showing dataset distribution by condition.
    
    Args:
        stats: Statistics dictionary
        output_dir: Directory to save plot
    """
    by_condition = stats['by_condition']
    
    conditions = sorted(by_condition.keys())
    num_images = [by_condition[c]['num_images'] for c in conditions]
    
    plt.figure(figsize=(10, 8))
    
    # Create pie chart
    colors = plt.cm.Set3(range(len(conditions)))
    wedges, texts, autotexts = plt.pie(
        num_images,
        labels=conditions,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 10, 'fontweight': 'bold'}
    )
    
    # Add count labels
    for i, (condition, count) in enumerate(zip(conditions, num_images)):
        texts[i].set_text(f'{condition}\n({count} images)')
    
    plt.title('Test Dataset Distribution by Condition', fontsize=14, fontweight='bold', pad=20)
    plt.axis('equal')
    
    output_path = output_dir / 'dataset_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved dataset distribution plot to {output_path}")


def plot_comparison_chart(stats: Dict, output_dir: Path):
    """
    Create grouped bar chart comparing all metrics by condition.
    
    Args:
        stats: Statistics dictionary
        output_dir: Directory to save plot
    """
    by_condition = stats['by_condition']
    
    conditions = sorted(by_condition.keys())
    detection_rates = [by_condition[c]['detection_rate'] * 100 for c in conditions]
    avg_detections = [by_condition[c]['avg_detections_per_image'] * 10 for c in conditions]  # Scale for visibility
    avg_confidences = [by_condition[c]['avg_confidence'] * 100 for c in conditions]
    
    x = np.arange(len(conditions))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    bars1 = ax.bar(x - width, detection_rates, width, label='Detection Rate (%)', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x, avg_detections, width, label='Avg Detections (×10)', color='#3498db', alpha=0.8)
    bars3 = ax.bar(x + width, avg_confidences, width, label='Avg Confidence (×100)', color='#9b59b6', alpha=0.8)
    
    ax.set_xlabel('Condition', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('Detection Performance Comparison by Condition', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / 'performance_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved comparison chart to {output_path}")


def plot_overall_summary(stats: Dict, output_dir: Path):
    """
    Create summary figure with key metrics.
    
    Args:
        stats: Statistics dictionary
        output_dir: Directory to save plot
    """
    overall = stats['overall']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # Title
    fig.suptitle('Detection Performance Summary', fontsize=18, fontweight='bold', y=0.95)
    
    # Metrics
    metrics = [
        ('Total Images', overall['total_images'], ''),
        ('Images with Detection', overall['images_with_detection'], ''),
        ('Detection Rate', overall['detection_rate'], '%'),
        ('Total Detections', overall['total_detections'], ''),
        ('Avg Detections per Image', overall['avg_detections_per_image'], ''),
        ('Avg Confidence', overall['avg_confidence'], ''),
    ]
    
    y_pos = 0.8
    for label, value, suffix in metrics:
        # Format value
        if suffix == '%':
            formatted_value = f"{value * 100:.2f}%"
        elif isinstance(value, float):
            formatted_value = f"{value:.3f}"
        else:
            formatted_value = f"{value}"
        
        # Display metric
        ax.text(0.1, y_pos, f"{label}:", fontsize=14, fontweight='bold', ha='left')
        ax.text(0.9, y_pos, formatted_value, fontsize=14, ha='right')
        
        y_pos -= 0.12
    
    plt.tight_layout()
    
    output_path = output_dir / 'summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved summary figure to {output_path}")


def main():
    """Main execution function."""
    args = parse_arguments()
    
    try:
        # Load results
        logger.info("Loading detection results...")
        data = load_results(args.input)
        
        stats = data['statistics']
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        
        plot_detection_rate_by_condition(stats, output_dir)
        plot_avg_detections_by_condition(stats, output_dir)
        plot_confidence_by_condition(stats, output_dir)
        plot_dataset_distribution(stats, output_dir)
        plot_comparison_chart(stats, output_dir)
        plot_overall_summary(stats, output_dir)
        
        logger.info(f"\n✅ Visualizations saved to: {output_dir}")
        logger.info("\nGenerated files:")
        logger.info("  - detection_rate_by_condition.png")
        logger.info("  - avg_detections_by_condition.png")
        logger.info("  - avg_confidence_by_condition.png")
        logger.info("  - dataset_distribution.png")
        logger.info("  - performance_comparison.png")
        logger.info("  - summary.png")
        
        return 0
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

"""
Pipeline Performance Profiling Tool

This script profiles the ALPR pipeline to measure performance metrics including:
- FPS (Frames Per Second)
- OCR call frequency
- GPU memory utilization
- CPU usage
- Latency breakdown

Usage:
    python scripts/profiling/profile_pipeline.py \\
        --video outputs/raw_footage/day_clear/day_clear_airport_01.mp4 \\
        --config configs/pipeline_config.yaml \\
        --output-dir outputs/profiling \\
        --num-frames 500 \\
        --ocr-intervals 1,15,30,60

Author: GTA V ALPR Project
Date: 2025-10-21
"""

import argparse
import csv
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import psutil
import torch
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.alpr_pipeline import ALPRPipeline
from src.utils.video_io import VideoReader

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PipelineProfiler:
    """
    Profiler for ALPR pipeline performance measurement.

    Collects timing, resource usage, and operational metrics during
    frame processing to identify bottlenecks and optimization opportunities.

    Attributes:
        pipeline (ALPRPipeline): The ALPR pipeline to profile
        metrics (defaultdict): Collected metrics per frame
        ocr_call_count (int): Total OCR calls made
        frame_count (int): Total frames processed
    """

    def __init__(self, pipeline: ALPRPipeline):
        """
        Initialize profiler with pipeline instance.

        Args:
            pipeline: Initialized ALPRPipeline to profile
        """
        self.pipeline = pipeline
        self.metrics = defaultdict(list)
        self.ocr_call_count = 0
        self.frame_count = 0

        logger.info("PipelineProfiler initialized")

    def profile_frame(self, frame: np.ndarray) -> Dict:
        """
        Profile a single frame through the pipeline.

        Measures:
        - Total processing time
        - OCR calls made
        - GPU memory usage (if available)
        - CPU usage percentage
        - Active track count

        Args:
            frame: Input frame in BGR format

        Returns:
            Dictionary with frame metrics:
                - frame_time_ms: Total processing time in milliseconds
                - ocr_calls: Number of OCR calls in this frame
                - gpu_memory_gb: GPU memory usage in GB
                - cpu_percent: CPU usage percentage
                - active_tracks: Number of active tracks
                - total_tracks: Total tracks (including lost)
        """
        # Overall timing
        start = time.perf_counter()

        # Store track states before processing (for potential debugging)
        _ = {tid: track.frames_since_last_ocr for tid, track in self.pipeline.tracks.items()}

        # Process frame through pipeline
        tracks = self.pipeline.process_frame(frame)

        # Total timing
        total_time = time.perf_counter() - start

        # Count OCR calls by checking which tracks just ran OCR
        # A track ran OCR if: frames_since_last_ocr == 0 AND age > 0 (not first frame)
        ocr_calls = 0
        for tid, track in tracks.items():
            # Check if this track just ran OCR
            if track.frames_since_last_ocr == 0 and track.age > 0:
                ocr_calls += 1

        self.ocr_call_count += ocr_calls

        # GPU memory usage
        gpu_memory = 0.0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB

        # CPU usage (interval=0 gives instant reading)
        cpu_percent = psutil.cpu_percent(interval=0)

        # Track counts
        active_tracks = sum(1 for t in tracks.values() if t.is_active)
        total_tracks = len(tracks)

        # Store metrics
        frame_metrics = {
            "frame_time_ms": total_time * 1000,
            "ocr_calls": ocr_calls,
            "gpu_memory_gb": gpu_memory,
            "cpu_percent": cpu_percent,
            "active_tracks": active_tracks,
            "total_tracks": total_tracks,
        }

        for key, value in frame_metrics.items():
            self.metrics[key].append(value)

        self.frame_count += 1

        return frame_metrics

    def generate_report(self) -> Dict:
        """
        Generate aggregate performance report from collected metrics.

        Returns:
            Dictionary with aggregate statistics:
                - fps: Average frames per second
                - avg_frame_time_ms: Average frame processing time
                - min_frame_time_ms: Minimum frame processing time
                - max_frame_time_ms: Maximum frame processing time
                - std_frame_time_ms: Standard deviation of frame time
                - ocr_calls_per_100_frames: OCR call frequency
                - total_ocr_calls: Total OCR calls made
                - avg_gpu_memory_gb: Average GPU memory usage
                - max_gpu_memory_gb: Peak GPU memory usage
                - avg_cpu_percent: Average CPU usage
                - max_cpu_percent: Peak CPU usage
                - frames_processed: Total frames processed
                - avg_active_tracks: Average active tracks per frame
                - avg_total_tracks: Average total tracks per frame
        """
        if self.frame_count == 0:
            logger.warning("No frames processed, returning empty report")
            return {}

        # Convert metrics to numpy arrays for statistics
        frame_times = np.array(self.metrics["frame_time_ms"])
        gpu_memory = np.array(self.metrics["gpu_memory_gb"])
        cpu_percent = np.array(self.metrics["cpu_percent"])
        active_tracks = np.array(self.metrics["active_tracks"])
        total_tracks = np.array(self.metrics["total_tracks"])

        report = {
            # Timing metrics
            "fps": 1000.0 / np.mean(frame_times),  # Convert ms to FPS
            "avg_frame_time_ms": float(np.mean(frame_times)),
            "min_frame_time_ms": float(np.min(frame_times)),
            "max_frame_time_ms": float(np.max(frame_times)),
            "std_frame_time_ms": float(np.std(frame_times)),
            # OCR metrics
            "ocr_calls_per_100_frames": (self.ocr_call_count / self.frame_count) * 100,
            "total_ocr_calls": self.ocr_call_count,
            # Resource metrics
            "avg_gpu_memory_gb": float(np.mean(gpu_memory)),
            "max_gpu_memory_gb": float(np.max(gpu_memory)),
            "avg_cpu_percent": float(np.mean(cpu_percent)),
            "max_cpu_percent": float(np.max(cpu_percent)),
            # General metrics
            "frames_processed": self.frame_count,
            "avg_active_tracks": float(np.mean(active_tracks)),
            "avg_total_tracks": float(np.mean(total_tracks)),
        }

        logger.info("Performance report generated")
        return report

    def reset(self):
        """Reset profiler metrics for next test run."""
        self.metrics = defaultdict(list)
        self.ocr_call_count = 0
        self.frame_count = 0
        logger.info("Profiler metrics reset")


def load_config(config_path: Path) -> dict:
    """
    Load pipeline configuration from YAML file.

    Args:
        config_path: Path to configuration YAML

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    logger.info(f"Loading configuration from: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: dict, output_path: Path):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        output_path: Output path for YAML file
    """
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Configuration saved to: {output_path}")


def run_profiling_test(
    video_path: Path, config: dict, num_frames: int, test_name: str, output_dir: Path
) -> Dict:
    """
    Run a single profiling test with given configuration.

    Args:
        video_path: Path to input video
        config: Pipeline configuration
        num_frames: Number of frames to process
        test_name: Name of this test (for logging/output)
        output_dir: Directory for output files

    Returns:
        Performance report dictionary
    """
    logger.info("=" * 70)
    logger.info(f"Running Profiling Test: {test_name}")
    logger.info("=" * 70)

    # Save configuration for this test
    config_output = output_dir / f"config_{test_name}.yaml"
    save_config(config, config_output)

    # Initialize pipeline with test configuration
    # Create temporary config file
    temp_config_path = output_dir / f"temp_config_{test_name}.yaml"
    save_config(config, temp_config_path)

    try:
        pipeline = ALPRPipeline(str(temp_config_path))
        profiler = PipelineProfiler(pipeline)

        # Open video
        logger.info(f"Opening video: {video_path}")
        reader = VideoReader(str(video_path))

        logger.info(f"Processing {num_frames} frames...")

        # Process frames using generator
        frames_processed = 0
        for frame_idx, frame in reader.read_frames(sample_rate=1):
            if frames_processed >= num_frames:
                break

            # Profile frame
            frame_metrics = profiler.profile_frame(frame)
            frames_processed += 1

            # Log progress every 50 frames
            if frames_processed % 50 == 0:
                logger.info(
                    f"Frame {frames_processed}/{num_frames}: "
                    f'{frame_metrics["frame_time_ms"]:.2f}ms, '
                    f'{frame_metrics["ocr_calls"]} OCR calls, '
                    f'{frame_metrics["active_tracks"]} active tracks'
                )

        reader.release()

        # Generate report
        report = profiler.generate_report()
        report["test_name"] = test_name
        report["video_path"] = str(video_path)
        report["config_path"] = str(config_output)

        # Log summary
        logger.info("-" * 70)
        logger.info("Test Summary:")
        logger.info(f'  FPS: {report["fps"]:.2f}')
        logger.info(f'  Average Frame Time: {report["avg_frame_time_ms"]:.2f}ms')
        logger.info(f'  OCR Calls per 100 Frames: {report["ocr_calls_per_100_frames"]:.2f}')
        logger.info(f'  Total OCR Calls: {report["total_ocr_calls"]}')
        logger.info(f'  Average GPU Memory: {report["avg_gpu_memory_gb"]:.3f}GB')
        logger.info(f'  Average CPU Usage: {report["avg_cpu_percent"]:.1f}%')
        logger.info("-" * 70)

        return report

    finally:
        # Clean up temporary config
        if temp_config_path.exists():
            temp_config_path.unlink()


def save_comparison_csv(reports: List[Dict], output_path: Path):
    """
    Save comparison of all test results to CSV.

    Args:
        reports: List of performance reports
        output_path: Output CSV file path
    """
    if not reports:
        logger.warning("No reports to save")
        return

    # Define CSV columns
    columns = [
        "test_name",
        "fps",
        "avg_frame_time_ms",
        "min_frame_time_ms",
        "max_frame_time_ms",
        "std_frame_time_ms",
        "ocr_calls_per_100_frames",
        "total_ocr_calls",
        "avg_gpu_memory_gb",
        "max_gpu_memory_gb",
        "avg_cpu_percent",
        "max_cpu_percent",
        "frames_processed",
        "avg_active_tracks",
        "avg_total_tracks",
    ]

    logger.info(f"Saving comparison CSV to: {output_path}")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(reports)

    logger.info("Comparison CSV saved")


def generate_markdown_report(reports: List[Dict], output_path: Path):
    """
    Generate comprehensive Markdown performance report.

    Args:
        reports: List of performance reports
        output_path: Output Markdown file path
    """
    if not reports:
        logger.warning("No reports to generate Markdown from")
        return

    logger.info(f"Generating Markdown report: {output_path}")

    # Sort reports by ocr_interval if available
    sorted_reports = sorted(reports, key=lambda r: r.get("ocr_calls_per_100_frames", 0))

    # Find best performing configuration
    best_fps = max(reports, key=lambda r: r["fps"])
    lowest_ocr = min(reports, key=lambda r: r["ocr_calls_per_100_frames"])

    # Generate Markdown content
    md_lines = [
        "# ALPR Pipeline Performance Report",
        "",
        f'**Generated:** {time.strftime("%Y-%m-%d %H:%M:%S")}',
        "",
        "## Executive Summary",
        "",
        f'Tested {len(reports)} configurations on {reports[0]["frames_processed"]} frames.',
        "",
        "### Best Configurations",
        "",
        f'- **Highest FPS:** {best_fps["test_name"]} ({best_fps["fps"]:.2f} FPS)',
        f'- **Lowest OCR Frequency:** {lowest_ocr["test_name"]} ({lowest_ocr["ocr_calls_per_100_frames"]:.2f} calls/100 frames)',  # noqa: E501
        "",
        "---",
        "",
        "## Detailed Results",
        "",
    ]

    # Add table for each report
    for report in sorted_reports:
        md_lines.extend(
            [
                f'### {report["test_name"]}',
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f'| **FPS** | {report["fps"]:.2f} |',
                f'| Average Frame Time | {report["avg_frame_time_ms"]:.2f}ms |',
                f'| Min Frame Time | {report["min_frame_time_ms"]:.2f}ms |',
                f'| Max Frame Time | {report["max_frame_time_ms"]:.2f}ms |',
                f'| Std Dev Frame Time | {report["std_frame_time_ms"]:.2f}ms |',
                f'| **OCR Calls per 100 Frames** | {report["ocr_calls_per_100_frames"]:.2f} |',
                f'| Total OCR Calls | {report["total_ocr_calls"]} |',
                f'| Average GPU Memory | {report["avg_gpu_memory_gb"]:.3f}GB |',
                f'| Peak GPU Memory | {report["max_gpu_memory_gb"]:.3f}GB |',
                f'| Average CPU Usage | {report["avg_cpu_percent"]:.1f}% |',
                f'| Peak CPU Usage | {report["max_cpu_percent"]:.1f}% |',
                f'| Average Active Tracks | {report["avg_active_tracks"]:.1f} |',
                f'| Average Total Tracks | {report["avg_total_tracks"]:.1f} |',
                f'| Frames Processed | {report["frames_processed"]} |',
                "",
            ]
        )

    # Add comparison table
    md_lines.extend(
        [
            "---",
            "",
            "## Configuration Comparison",
            "",
            "| Test | FPS | OCR per 100 Frames | Avg Frame Time (ms) | Avg GPU Memory (GB) |",
            "|------|-----|-------------------|---------------------|---------------------|",
        ]
    )

    for report in sorted_reports:
        md_lines.append(
            f'| {report["test_name"]} | '
            f'{report["fps"]:.2f} | '
            f'{report["ocr_calls_per_100_frames"]:.2f} | '
            f'{report["avg_frame_time_ms"]:.2f} | '
            f'{report["avg_gpu_memory_gb"]:.3f} |'
        )

    md_lines.extend(["", "---", "", "## Optimization Recommendations", ""])

    # Generate recommendations
    if lowest_ocr["ocr_calls_per_100_frames"] < 15:
        md_lines.extend(
            [
                "### ✅ OCR Optimization Success",
                "",
                f'The tracking module successfully reduced OCR calls to **{lowest_ocr["ocr_calls_per_100_frames"]:.2f} per 100 frames**, '  # noqa: E501
                "which is well below the target of 10-15 calls. This represents significant computational savings.",  # noqa: E501
                "",
            ]
        )
    else:
        md_lines.extend(
            [
                "### ⚠️ OCR Frequency Still High",
                "",
                f'The lowest OCR frequency achieved was **{lowest_ocr["ocr_calls_per_100_frames"]:.2f} per 100 frames**. '  # noqa: E501
                "Consider increasing `ocr_interval` further or adjusting `ocr_confidence_threshold` to reduce redundant OCR calls.",  # noqa: E501
                "",
            ]
        )

    if best_fps["fps"] >= 15:
        md_lines.extend(
            [
                "### ✅ Real-Time Performance Achieved",
                "",
                f'The pipeline achieved **{best_fps["fps"]:.2f} FPS**, exceeding the 15 FPS target for real-time processing.',  # noqa: E501
                "",
            ]
        )
    else:
        md_lines.extend(
            [
                "### ⚠️ Below Real-Time Target",
                "",
                f'The maximum FPS achieved was **{best_fps["fps"]:.2f}**, which is below the 15 FPS real-time target. '  # noqa: E501
                "Consider optimizing preprocessing (disable CLAHE if enabled) or increasing frame sampling rate.",  # noqa: E501
                "",
            ]
        )

    # GPU memory recommendations
    max_gpu = max(r["max_gpu_memory_gb"] for r in reports)
    if max_gpu < 4.0:
        md_lines.extend(
            [
                "### ✅ GPU Memory Within Limits",
                "",
                f"Peak GPU memory usage was **{max_gpu:.3f}GB**, well within the 4GB target for RTX 3070Ti.",  # noqa: E501
                "",
            ]
        )
    else:
        md_lines.extend(
            [
                "### ⚠️ High GPU Memory Usage",
                "",
                f"Peak GPU memory usage was **{max_gpu:.3f}GB**, approaching or exceeding the 4GB target. "  # noqa: E501
                "Consider using a smaller model or reducing batch sizes if applicable.",
                "",
            ]
        )

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    logger.info("Markdown report generated")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Profile ALPR pipeline performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test single configuration
  python scripts/profiling/profile_pipeline.py \\
      --video outputs/raw_footage/day_clear/day_clear_airport_01.mp4 \\
      --config configs/pipeline_config.yaml \\
      --num-frames 500

  # Test multiple OCR intervals
  python scripts/profiling/profile_pipeline.py \\
      --video outputs/raw_footage/day_clear/day_clear_airport_01.mp4 \\
      --config configs/pipeline_config.yaml \\
      --num-frames 500 \\
      --ocr-intervals 1,15,30,60

  # Test preprocessing variations
  python scripts/profiling/profile_pipeline.py \\
      --video outputs/raw_footage/day_clear/day_clear_airport_01.mp4 \\
      --config configs/pipeline_config.yaml \\
      --num-frames 500 \\
      --test-preprocessing
        """,
    )

    parser.add_argument("--video", type=str, required=True, help="Path to input video file")

    parser.add_argument(
        "--config", type=str, required=True, help="Path to pipeline configuration YAML"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/profiling",
        help="Output directory for profiling results (default: outputs/profiling)",
    )

    parser.add_argument(
        "--num-frames", type=int, default=500, help="Number of frames to process (default: 500)"
    )

    parser.add_argument(
        "--ocr-intervals",
        type=str,
        default=None,
        help='Comma-separated list of ocr_interval values to test (e.g., "1,15,30,60")',
    )

    parser.add_argument(
        "--test-preprocessing",
        action="store_true",
        help="Test preprocessing variations (with/without enhancement)",
    )

    return parser.parse_args()


def main():
    """Main profiling script."""
    args = parse_args()

    # Convert paths to Path objects
    video_path = Path(args.video)
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)

    # Validate inputs
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        return 1

    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return 1

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base configuration
    base_config = load_config(config_path)

    # Collect all test configurations
    test_configs = []

    # OCR interval tests
    if args.ocr_intervals:
        intervals = [int(x.strip()) for x in args.ocr_intervals.split(",")]
        for interval in intervals:
            config = yaml.safe_load(yaml.dump(base_config))  # Deep copy
            config["tracking"]["ocr_interval"] = interval
            test_configs.append((f"ocr_interval_{interval}", config))

    # Preprocessing tests
    if args.test_preprocessing:
        # Test with enhancement disabled (default)
        config = yaml.safe_load(yaml.dump(base_config))
        config["preprocessing"]["enable_enhancement"] = False
        config["preprocessing"]["use_clahe"] = False
        test_configs.append(("preprocessing_disabled", config))

        # Test with CLAHE enabled
        config = yaml.safe_load(yaml.dump(base_config))
        config["preprocessing"]["enable_enhancement"] = True
        config["preprocessing"]["use_clahe"] = True
        test_configs.append(("preprocessing_clahe", config))

    # If no specific tests, use base config
    if not test_configs:
        test_configs.append(("baseline", base_config))

    # Run all tests
    logger.info(f"Running {len(test_configs)} profiling tests")
    reports = []

    for test_name, config in test_configs:
        report = run_profiling_test(
            video_path=video_path,
            config=config,
            num_frames=args.num_frames,
            test_name=test_name,
            output_dir=output_dir,
        )
        reports.append(report)

    # Save comparison CSV
    csv_path = output_dir / "optimization_comparison.csv"
    save_comparison_csv(reports, csv_path)

    # Generate Markdown report
    md_path = output_dir / "performance_report.md"
    generate_markdown_report(reports, md_path)

    logger.info("=" * 70)
    logger.info("Profiling Complete!")
    logger.info(f"  Comparison CSV: {csv_path}")
    logger.info(f"  Performance Report: {md_path}")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

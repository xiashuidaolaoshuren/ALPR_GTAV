"""
Video Detection Script

Process video files frame-by-frame, detecting license plates and saving
an annotated output video with bounding boxes.

Usage:
    python scripts/inference/detect_video.py --video path/to/video.mp4 [options]

Examples:
    # Basic usage
    python scripts/inference/detect_video.py --video outputs/raw_footage/day_clear/video1.mp4

    # Custom output path and confidence threshold
    python scripts/inference/detect_video.py --video input.mp4 --output results/output.mp4 --conf 0.5  # noqa: E501

    # Process every 2nd frame for faster processing
    python scripts/inference/detect_video.py --video input.mp4 --sample_rate 2
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict

import yaml
from tqdm import tqdm

# Ensure project root is on the Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.detection.model import load_detection_model, detect_plates
from src.detection.utils import draw_bounding_boxes
from src.utils.video_io import VideoReader, VideoWriter

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process video with license plate detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--video", type=str, required=True, help="Path to input video file")

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/detection_video.mp4",
        help="Path to output video file (default: outputs/detection_video.mp4)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline_config.yaml",
        help="Path to configuration file (default: configs/pipeline_config.yaml)",
    )

    parser.add_argument(
        "--con", type=float, default=None, help="Confidence threshold (overrides config)"
    )

    parser.add_argument("--iou", type=float, default=None, help="IOU threshold (overrides config)")

    parser.add_argument(
        "--sample_rate",
        type=int,
        default=1,
        help="Process every Nth frame (1=all frames, 2=every other frame, etc.)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (cpu/cuda/auto, overrides config)",
    )

    parser.add_argument(
        "--codec", type=str, default="mp4v", help="Video codec to use (default: mp4v)"
    )

    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")

    return parser.parse_args()


def load_configuration(config_path: str) -> Dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary containing configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")
    return config


def process_video(
    video_path: str,
    output_path: str,
    model,
    confidence_threshold: float,
    iou_threshold: float,
    sample_rate: int = 1,
    codec: str = "mp4v",
    show_progress: bool = True,
) -> Dict:
    """
    Process video with license plate detection.

    Args:
        video_path: Path to input video
        output_path: Path to output video
        model: Loaded YOLOv8 model
        confidence_threshold: Confidence threshold for detections
        iou_threshold: IOU threshold for NMS
        sample_rate: Process every Nth frame
        codec: Video codec to use
        show_progress: Whether to show progress bar

    Returns:
        Dictionary with processing statistics
    """
    # Initialize video reader and writer
    reader = VideoReader(video_path)
    writer = VideoWriter(
        output_path,
        reader.fps / sample_rate,  # Adjust FPS for sampling
        reader.width,
        reader.height,
        codec=codec,
    )

    # Statistics tracking
    stats = {
        "total_frames": 0,
        "frames_processed": 0,
        "total_detections": 0,
        "frames_with_detections": 0,
        "processing_time": 0.0,
    }

    start_time = time.time()

    try:
        # Process frames with optional progress bar
        frame_generator = reader.read_frames(sample_rate=sample_rate)

        if show_progress:
            # Calculate expected number of frames after sampling
            expected_frames = reader.total_frames // sample_rate
            frame_generator = tqdm(
                frame_generator, total=expected_frames, desc="Processing video", unit="frames"
            )

        for frame_idx, frame in frame_generator:
            stats["frames_processed"] += 1

            # Detect plates
            detections = detect_plates(frame, model, confidence_threshold, iou_threshold)

            # Update statistics
            num_detections = len(detections)
            stats["total_detections"] += num_detections
            if num_detections > 0:
                stats["frames_with_detections"] += 1

            # Annotate frame with bounding boxes
            annotated = draw_bounding_boxes(frame, detections)

            # Write to output video
            writer.write_frame(annotated)

            # Update progress bar with detection info
            if show_progress and hasattr(frame_generator, "set_postfix"):
                frame_generator.set_postfix(
                    {"detections": num_detections, "total": stats["total_detections"]}
                )

    finally:
        # Cleanup
        reader.release()
        writer.release()

    stats["processing_time"] = time.time() - start_time
    stats["total_frames"] = reader.total_frames

    return stats


def print_statistics(stats: Dict, output_path: str):
    """
    Print processing statistics.

    Args:
        stats: Dictionary containing processing statistics
        output_path: Path to output video
    """
    logger.info("=" * 60)
    logger.info("Processing Complete")
    logger.info("=" * 60)
    logger.info(f"Total frames in video: {stats['total_frames']}")
    logger.info(f"Frames processed: {stats['frames_processed']}")
    logger.info(f"Total detections: {stats['total_detections']}")
    logger.info(f"Frames with detections: {stats['frames_with_detections']}")

    if stats["frames_processed"] > 0:
        detection_rate = (stats["frames_with_detections"] / stats["frames_processed"]) * 100
        avg_detections = stats["total_detections"] / stats["frames_processed"]
        logger.info(f"Detection rate: {detection_rate:.2f}%")
        logger.info(f"Average detections per frame: {avg_detections:.2f}")

    logger.info(f"Processing time: {stats['processing_time']:.2f} seconds")

    if stats["processing_time"] > 0:
        fps = stats["frames_processed"] / stats["processing_time"]
        logger.info(f"Processing speed: {fps:.2f} fps")

    logger.info(f"Output saved to: {output_path}")
    logger.info("=" * 60)


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()

    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_configuration(args.config)
        detection_config = config["detection"]

        # Override config with command line arguments if provided
        confidence_threshold = (
            args.conf if args.conf is not None else detection_config["confidence_threshold"]
        )
        iou_threshold = args.iou if args.iou is not None else detection_config["iou_threshold"]
        device = args.device if args.device is not None else detection_config.get("device", "auto")

        logger.info("Detection parameters:")
        logger.info(f"  Confidence threshold: {confidence_threshold}")
        logger.info(f"  IOU threshold: {iou_threshold}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Sample rate: {args.sample_rate}")

        # Load detection model
        logger.info("Loading detection model...")
        model = load_detection_model(model_path=detection_config["model_path"], device=device)
        logger.info("Model loaded successfully")

        # Process video
        logger.info(f"Processing video: {args.video}")
        stats = process_video(
            video_path=args.video,
            output_path=args.output,
            model=model,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            sample_rate=args.sample_rate,
            codec=args.codec,
            show_progress=not args.no_progress,
        )

        # Print statistics
        print_statistics(stats, args.output)

        logger.info("Video processing completed successfully!")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid value: {e}")
        return 1
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

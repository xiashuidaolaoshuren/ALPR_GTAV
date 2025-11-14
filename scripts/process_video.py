"""
Batch Video Processing Script

Processes video files through the complete ALPR pipeline with frame-by-frame
processing, progress tracking, and multiple export formats.

Usage:
    python scripts/process_video.py --input video.mp4 --output output.mp4 --config configs/pipeline_config.yaml  # noqa: E501
    python scripts/process_video.py --input video.mp4 --output output.mp4 --sample-rate 5 --export-json results.json  # noqa: E501
    python scripts/process_video.py --input video.mp4 --no-video --export-csv results.csv
"""

import argparse
import sys
import logging
from pathlib import Path
import json
import csv
from typing import Dict, List
import time

from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.utils import draw_tracks_on_frame
from src.pipeline.alpr_pipeline import ALPRPipeline
from src.utils.video_io import VideoReader, VideoWriter


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Process video files with ALPR pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video with default settings
  python scripts/process_video.py --input video.mp4 --output output.mp4

  # Process every 5th frame and export to JSON
  python scripts/process_video.py --input video.mp4 --output output.mp4 \\
      --sample-rate 5 --export-json results.json

  # Skip video output (faster) and export to CSV
  python scripts/process_video.py --input video.mp4 --no-video \\
      --export-csv results.csv

  # Process with custom config
  python scripts/process_video.py --input video.mp4 --output output.mp4 \\
      --config my_config.yaml
        """,
    )

    # Required arguments
    parser.add_argument("--input", "-i", required=True, type=str, help="Path to input video file")

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Path to output video file (required unless --no-video is set)",
    )

    # Configuration
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/pipeline_config.yaml",
        help="Path to pipeline configuration file (default: configs/pipeline_config.yaml)",
    )

    # Processing options
    parser.add_argument(
        "--sample-rate",
        "-s",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1 = all frames, 2 = every other frame, etc.)",
    )

    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip video output generation (faster processing, only exports data)",
    )

    # Export options
    parser.add_argument("--export-json", type=str, help="Export results to JSON file")

    parser.add_argument("--export-csv", type=str, help="Export results to CSV file")

    # Visualization options
    parser.add_argument(
        "--show-track-id",
        action="store_true",
        default=True,
        help="Show track IDs on output video (default: True)",
    )

    parser.add_argument(
        "--show-confidence",
        action="store_true",
        default=True,
        help="Show confidence scores on output video (default: True)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.no_video and not args.output:
        parser.error("--output is required unless --no-video is set")

    if args.sample_rate < 1:
        parser.error("--sample-rate must be at least 1")

    return args


def process_video(args):
    """
    Process video through ALPR pipeline.

    Args:
        args: Parsed command-line arguments

    Returns:
        Dict with processing statistics
    """
    logger.info("=" * 80)
    logger.info("ALPR Video Processing Started")
    logger.info("=" * 80)
    logger.info(f"Input video: {args.input}")
    logger.info(f'Output video: {args.output if not args.no_video else "None (--no-video)"}')
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Sample rate: {args.sample_rate}")
    logger.info(f'Export JSON: {args.export_json or "None"}')
    logger.info(f'Export CSV: {args.export_csv or "None"}')
    logger.info("=" * 80)

    start_time = time.time()

    # Initialize pipeline
    logger.info("Initializing ALPR pipeline...")
    try:
        pipeline = ALPRPipeline(args.config)
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise

    # Open video
    logger.info("Opening input video...")
    try:
        reader = VideoReader(args.input)
    except Exception as e:
        logger.error(f"Failed to open video: {e}")
        raise

    # Initialize video writer if needed
    writer = None
    if not args.no_video:
        logger.info("Initializing output video writer...")
        try:
            # Adjust output FPS to account for frame sampling
            # If we sample every Nth frame, output should play at fps/N to maintain correct timing
            output_fps = reader.fps / args.sample_rate
            logger.info(
                f"Output video FPS: {
                    output_fps:.2f} (original: {
                    reader.fps}, sample_rate: {
                    args.sample_rate})"
            )

            writer = VideoWriter(args.output, output_fps, reader.width, reader.height)
        except Exception as e:
            logger.error(f"Failed to initialize video writer: {e}")
            reader.release()
            raise

    # Storage for results
    all_results = []
    frames_processed = 0
    plates_detected = 0
    plates_recognized = 0

    # Calculate total frames to process
    total_frames_to_process = reader.total_frames // args.sample_rate

    logger.info("=" * 80)
    logger.info("Processing video frames...")
    logger.info("=" * 80)

    try:
        # Process frames with progress bar
        with tqdm(total=total_frames_to_process, desc="Processing", unit="frames") as pbar:
            for frame_idx, frame in reader.read_frames(args.sample_rate):
                # Run pipeline
                tracks = pipeline.process_frame(frame)

                # Count statistics
                active_tracks = [t for t in tracks.values() if t.is_active]
                recognized_tracks = [t for t in active_tracks if t.text is not None]

                plates_detected += len(active_tracks)
                plates_recognized += len(recognized_tracks)

                # Collect results for this frame
                frame_result = {
                    "frame": frame_idx,
                    "timestamp": frame_idx / reader.fps,
                    "tracks": [],
                }

                for track in active_tracks:
                    track_data = {
                        "id": track.id,
                        "text": track.text,
                        "ocr_confidence": round(track.ocr_confidence, 3) if track.text else 0.0,
                        "detection_confidence": round(track.detection_confidence, 3),
                        "bbox": list(track.bbox),
                        "age": track.age,
                    }
                    frame_result["tracks"].append(track_data)

                all_results.append(frame_result)

                # Annotate and write frame if video output enabled
                if writer:
                    # Use draw_tracks_on_frame utility for consistent visualization
                    annotated = draw_tracks_on_frame(
                        frame,
                        tracks,
                        show_text=True,
                        show_track_id=args.show_track_id,
                        show_confidence=args.show_confidence,
                    )
                    writer.write_frame(annotated)

                frames_processed += 1

                # Update progress bar with current stats
                pbar.set_postfix(
                    {"tracks": len(active_tracks), "recognized": len(recognized_tracks)}
                )
                pbar.update(1)

    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Cleaning up resources...")
        reader.release()
        if writer:
            writer.release()

    # Calculate statistics
    elapsed_time = time.time() - start_time
    fps = frames_processed / elapsed_time if elapsed_time > 0 else 0

    stats = {
        "frames_processed": frames_processed,
        "total_frames": reader.total_frames,
        "sample_rate": args.sample_rate,
        "elapsed_time": elapsed_time,
        "fps": fps,
        "plates_detected": plates_detected,
        "plates_recognized": plates_recognized,
        "unique_plates": len(
            set(
                track["text"]
                for result in all_results
                for track in result["tracks"]
                if track["text"]
            )
        ),
    }

    # Export results
    if args.export_json:
        logger.info(f"Exporting results to JSON: {args.export_json}")
        export_json(all_results, args.export_json, stats)

    if args.export_csv:
        logger.info(f"Exporting results to CSV: {args.export_csv}")
        export_csv(all_results, args.export_csv)

    # Print summary
    logger.info("=" * 80)
    logger.info("Processing Complete")
    logger.info("=" * 80)
    logger.info(f'Frames processed: {stats["frames_processed"]} / {stats["total_frames"]}')
    logger.info(f'Sample rate: 1 in {stats["sample_rate"]} frames')
    logger.info(f'Processing time: {stats["elapsed_time"]:.2f}s')
    logger.info(f'Processing speed: {stats["fps"]:.2f} FPS')
    logger.info(f'Plates detected: {stats["plates_detected"]} (total detections)')
    logger.info(f'Plates recognized: {stats["plates_recognized"]} (with text)')
    logger.info(f'Unique plates: {stats["unique_plates"]}')
    logger.info("=" * 80)

    return stats


def export_json(results: List[Dict], output_path: str, stats: Dict):
    """
    Export results to JSON file.

    Args:
        results: List of frame results
        output_path: Path to output JSON file
        stats: Processing statistics
    """
    output_data = {"statistics": stats, "frames": results}

    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Results exported to JSON: {output_path}")
    except Exception as e:
        logger.error(f"Failed to export JSON: {e}")


def export_csv(results: List[Dict], output_path: str):
    """
    Export results to CSV file.

    Args:
        results: List of frame results
        output_path: Path to output CSV file
    """
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(
                [
                    "frame",
                    "timestamp",
                    "track_id",
                    "text",
                    "ocr_confidence",
                    "detection_confidence",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "age",
                ]
            )

            # Write data
            for frame_result in results:
                for track in frame_result["tracks"]:
                    writer.writerow(
                        [
                            frame_result["frame"],
                            f"{frame_result['timestamp']:.2f}",
                            track["id"],
                            track["text"] or "",
                            track["ocr_confidence"],
                            track["detection_confidence"],
                            track["bbox"][0],
                            track["bbox"][1],
                            track["bbox"][2],
                            track["bbox"][3],
                            track["age"],
                        ]
                    )

        logger.info(f"✓ Results exported to CSV: {output_path}")
    except Exception as e:
        logger.error(f"Failed to export CSV: {e}")


def main():
    """Main entry point."""
    try:
        args = parse_args()
        process_video(args)
        return 0
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

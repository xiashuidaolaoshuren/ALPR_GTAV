#!/usr/bin/env python3
"""
End-to-End ALPR Pipeline Evaluation Script

This script evaluates the complete ALPR pipeline on test videos across
diverse scenarios (day/night × clear/rain). It collects comprehensive
metrics for detection, recognition, tracking, and overall performance.

Usage:
    python scripts/evaluation/evaluate_pipeline.py --videos <video_paths> --output-dir <output_path>

    # Evaluate specific videos
    python scripts/evaluation/evaluate_pipeline.py \
        --videos outputs/raw_footage/day_clear/*.mp4 \
        --output-dir outputs/evaluation

    # Evaluate with custom config
    python scripts/evaluation/evaluate_pipeline.py \
        --videos outputs/raw_footage/**/*.mp4 \
        --config configs/pipeline_config.yaml \
        --output-dir outputs/evaluation \
        --max-frames 500
"""

import os
import sys
import argparse
import json
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict
import cv2
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Pre-import torch to avoid PaddleOCR DLL load issues on Windows
import torch  # noqa: F401, E402

from src.pipeline.alpr_pipeline import ALPRPipeline
from src.utils.video_io import VideoReader


class PipelineEvaluator:
    """Evaluate ALPR pipeline performance on test videos."""

    def __init__(self, config_path: str, output_dir: str):
        """
        Initialize evaluator.

        Args:
            config_path: Path to pipeline configuration file
            output_dir: Directory to save evaluation results
        """
        self.config_path = config_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Initialize pipeline
        print("Initializing ALPR pipeline...")
        self.pipeline = ALPRPipeline(config_path)

        # Metrics storage
        self.video_results = {}
        self.condition_results = defaultdict(
            lambda: {
                "videos": [],
                "total_frames": 0,
                "total_detections": 0,
                "total_tracks": 0,
                "total_recognized": 0,
                "avg_confidence": [],
                "processing_times": [],
            }
        )

    def evaluate_video(
        self, video_path: str, max_frames: Optional[int] = None, save_samples: bool = True
    ) -> Dict:
        """
        Evaluate pipeline on a single video.

        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process (None = all)
            save_samples: Whether to save sample annotated frames

        Returns:
            Dictionary with evaluation metrics
        """
        video_name = Path(video_path).stem
        print(f"\nEvaluating: {video_name}")

        # Determine condition from filename
        condition = self._parse_condition(video_name)

        # Reset pipeline state
        self.pipeline.reset()

        # Metrics collection
        metrics = {
            "video_name": video_name,
            "video_path": video_path,
            "condition": condition,
            "frames_processed": 0,
            "total_detections": 0,
            "total_tracks": 0,
            "unique_plates_recognized": set(),
            "recognition_attempts": 0,
            "successful_recognitions": 0,
            "confidence_scores": [],
            "processing_times": [],
            "detection_counts_per_frame": [],
            "track_ages": [],
            "sample_frames": [],
        }

        # Initialize video reader
        reader = VideoReader(video_path)
        total_frames = min(max_frames, reader.total_frames) if max_frames else reader.total_frames

        # Process frames
        processed_count = 0
        with tqdm(total=total_frames, desc=f"Processing {video_name}") as pbar:
            for frame_idx, frame in reader.read_frames(sample_rate=1):
                if max_frames and processed_count >= max_frames:
                    break

                # Process frame
                start_time = cv2.getTickCount()
                tracks = self.pipeline.process_frame(frame)
                end_time = cv2.getTickCount()

                processing_time = (end_time - start_time) / cv2.getTickFrequency() * 1000  # ms
                metrics["processing_times"].append(processing_time)

                # Collect metrics
                detections_this_frame = len(tracks)
                metrics["detection_counts_per_frame"].append(detections_this_frame)
                metrics["total_detections"] += detections_this_frame

                for track_id, track in tracks.items():
                    metrics["track_ages"].append(track.age)

                    if track.text:
                        metrics["unique_plates_recognized"].add(track.text)
                        metrics["successful_recognitions"] += 1
                        if track.ocr_confidence:
                            metrics["confidence_scores"].append(track.ocr_confidence)

                # Save sample frames (first, middle, last with detections)
                if save_samples and detections_this_frame > 0:
                    if len(metrics["sample_frames"]) < 3:
                        annotated = self._annotate_frame(frame.copy(), tracks)
                        metrics["sample_frames"].append(
                            {
                                "frame_idx": frame_idx,
                                "image": annotated,
                                "num_detections": detections_this_frame,
                            }
                        )
                    elif frame_idx == total_frames // 2:  # Middle frame
                        annotated = self._annotate_frame(frame.copy(), tracks)
                        metrics["sample_frames"].insert(
                            1,
                            {
                                "frame_idx": frame_idx,
                                "image": annotated,
                                "num_detections": detections_this_frame,
                            },
                        )

                processed_count += 1
                pbar.update(1)

        # Final statistics
        metrics["frames_processed"] = processed_count
        metrics["total_tracks"] = len(self.pipeline.tracks)
        metrics["recognition_attempts"] = sum(
            1
            for track in self.pipeline.tracks.values()
            if hasattr(track, "ocr_attempts") and track.ocr_attempts > 0
        )

        # Convert set to count
        metrics["unique_plates_recognized"] = len(metrics["unique_plates_recognized"])

        # Calculate averages
        metrics["avg_processing_time_ms"] = (
            np.mean(metrics["processing_times"]) if metrics["processing_times"] else 0
        )
        metrics["avg_confidence"] = (
            np.mean(metrics["confidence_scores"]) if metrics["confidence_scores"] else 0
        )
        metrics["avg_detections_per_frame"] = (
            np.mean(metrics["detection_counts_per_frame"])
            if metrics["detection_counts_per_frame"]
            else 0
        )
        metrics["avg_track_age"] = np.mean(metrics["track_ages"]) if metrics["track_ages"] else 0
        metrics["fps"] = (
            1000 / metrics["avg_processing_time_ms"] if metrics["avg_processing_time_ms"] > 0 else 0
        )

        # Save sample frames
        if save_samples and metrics["sample_frames"]:
            self._save_sample_frames(video_name, metrics["sample_frames"])

        # Clean up large data before storing
        del metrics["processing_times"]
        del metrics["confidence_scores"]
        del metrics["detection_counts_per_frame"]
        del metrics["track_ages"]
        del metrics["sample_frames"]

        # Store results
        self.video_results[video_name] = metrics

        # Update condition aggregates
        self.condition_results[condition]["videos"].append(video_name)
        self.condition_results[condition]["total_frames"] += metrics["frames_processed"]
        self.condition_results[condition]["total_detections"] += metrics["total_detections"]
        self.condition_results[condition]["total_tracks"] += metrics["total_tracks"]
        self.condition_results[condition]["total_recognized"] += metrics["unique_plates_recognized"]
        self.condition_results[condition]["avg_confidence"].append(metrics["avg_confidence"])

        return metrics

    def evaluate_multiple_videos(
        self, video_paths: List[str], max_frames: Optional[int] = None, save_samples: bool = True
    ) -> Dict:
        """
        Evaluate pipeline on multiple videos.

        Args:
            video_paths: List of video file paths
            max_frames: Maximum frames per video (None = all)
            save_samples: Whether to save sample frames

        Returns:
            Dictionary with aggregated results
        """
        print(f"\n{'=' * 80}")
        print(f"Starting evaluation on {len(video_paths)} videos")
        print(f"{'=' * 80}")

        for video_path in video_paths:
            try:
                self.evaluate_video(video_path, max_frames, save_samples)
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                import traceback

                traceback.print_exc()
                continue

        # Calculate condition averages
        for condition in self.condition_results:
            data = self.condition_results[condition]
            data["avg_confidence"] = (
                np.mean(data["avg_confidence"]) if data["avg_confidence"] else 0
            )
            data["avg_detections_per_frame"] = (
                data["total_detections"] / data["total_frames"] if data["total_frames"] > 0 else 0
            )
            data["avg_tracks_per_video"] = (
                data["total_tracks"] / len(data["videos"]) if data["videos"] else 0
            )
            data["avg_recognized_per_video"] = (
                data["total_recognized"] / len(data["videos"]) if data["videos"] else 0
            )

        return {
            "video_results": self.video_results,
            "condition_results": dict(self.condition_results),
        }

    def generate_report(self, results: Dict) -> str:
        """
        Generate comprehensive evaluation report.

        Args:
            results: Evaluation results dictionary

        Returns:
            Markdown-formatted report string
        """
        report = []
        report.append("# ALPR Pipeline End-to-End Evaluation Report\n")
        report.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**Configuration:** {self.config_path}\n")
        report.append(f"**Total Videos Evaluated:** {len(results['video_results'])}\n")
        report.append("\n---\n")

        # Executive Summary
        report.append("## Executive Summary\n")

        total_frames = sum(v["frames_processed"] for v in results["video_results"].values())
        total_detections = sum(v["total_detections"] for v in results["video_results"].values())
        total_recognized = sum(
            v["unique_plates_recognized"] for v in results["video_results"].values()
        )
        avg_fps = np.mean([v["fps"] for v in results["video_results"].values()])
        avg_confidence = np.mean(
            [
                v["avg_confidence"]
                for v in results["video_results"].values()
                if v["avg_confidence"] > 0
            ]
        )

        report.append(f"- **Total Frames Processed:** {total_frames:,}\n")
        report.append(f"- **Total Detections:** {total_detections:,}\n")
        report.append(f"- **Unique Plates Recognized:** {total_recognized}\n")
        report.append(f"- **Average Processing Speed:** {avg_fps:.2f} FPS\n")
        report.append(f"- **Average Recognition Confidence:** {avg_confidence:.2%}\n")
        report.append("\n")

        # Performance by Condition
        report.append("## Performance by Condition\n")
        report.append(
            "| Condition | Videos | Frames | Detections | Tracks | Recognized | Avg Confidence |\n"
        )
        report.append(
            "|-----------|--------|--------|------------|--------|------------|----------------|\n"
        )

        for condition in sorted(results["condition_results"].keys()):
            data = results["condition_results"][condition]
            report.append(
                f"| {condition} | {len(data['videos'])} | "
                f"{data['total_frames']:,} | {data['total_detections']:,} | "
                f"{data['total_tracks']} | {data['total_recognized']} | "
                f"{data['avg_confidence']:.2%} |\n"
            )

        report.append("\n")

        # Per-Video Results
        report.append("## Individual Video Results\n")
        report.append(
            "| Video | Condition | Frames | Detections | Tracks | Recognized | FPS | Avg Conf |\n"
        )
        report.append(
            "|-------|-----------|--------|------------|--------|------------|-----|----------|\n"
        )

        for video_name in sorted(results["video_results"].keys()):
            metrics = results["video_results"][video_name]
            report.append(
                f"| {video_name} | {metrics['condition']} | "
                f"{metrics['frames_processed']} | {metrics['total_detections']} | "
                f"{metrics['total_tracks']} | {metrics['unique_plates_recognized']} | "
                f"{metrics['fps']:.1f} | {metrics['avg_confidence']:.2%} |\n"
            )

        report.append("\n")

        # Detailed Analysis
        report.append("## Detailed Analysis\n")

        # Best/Worst performing videos
        report.append("### Detection Performance\n")
        videos_by_detection_rate = sorted(
            results["video_results"].items(),
            key=lambda x: x[1]["avg_detections_per_frame"],
            reverse=True,
        )

        report.append("**Top 3 Videos by Detection Rate:**\n")
        for video_name, metrics in videos_by_detection_rate[:3]:
            report.append(
                f"- {video_name}: {metrics['avg_detections_per_frame']:.2f} detections/frame "
                f"({metrics['condition']})\n"
            )

        report.append("\n**Bottom 3 Videos by Detection Rate:**\n")
        for video_name, metrics in videos_by_detection_rate[-3:]:
            report.append(
                f"- {video_name}: {metrics['avg_detections_per_frame']:.2f} detections/frame "
                f"({metrics['condition']})\n"
            )

        report.append("\n### Recognition Performance\n")
        videos_with_recognition = {
            k: v for k, v in results["video_results"].items() if v["unique_plates_recognized"] > 0
        }

        report.append(
            f"- **Videos with Successful Recognition:** {len(videos_with_recognition)}/{len(results['video_results'])}\n"  # noqa: E501
        )

        if videos_with_recognition:
            videos_by_confidence = sorted(
                videos_with_recognition.items(), key=lambda x: x[1]["avg_confidence"], reverse=True
            )

            report.append("\n**Top 3 by Recognition Confidence:**\n")
            for video_name, metrics in videos_by_confidence[:3]:
                report.append(
                    f"- {video_name}: {metrics['avg_confidence']:.2%} avg confidence "
                    f"({metrics['unique_plates_recognized']} plates recognized)\n"
                )

        report.append("\n")

        # Recommendations
        report.append("## Recommendations for Optimization\n")

        # Analyze conditions
        condition_performance = {}
        for condition, data in results["condition_results"].items():
            if data["total_frames"] > 0:
                detection_rate = data["total_detections"] / data["total_frames"]
                recognition_rate = (
                    data["total_recognized"] / len(data["videos"]) if data["videos"] else 0
                )
                condition_performance[condition] = {
                    "detection_rate": detection_rate,
                    "recognition_rate": recognition_rate,
                    "confidence": data["avg_confidence"],
                }

        # Find weakest condition
        if condition_performance:
            weakest = min(condition_performance.items(), key=lambda x: x[1]["detection_rate"])
            report.append(f"1. **Focus on {weakest[0]} condition:** ")
            report.append(
                f"Lowest detection rate ({
                    weakest[1]['detection_rate']:.2f} detections/frame).\n"
            )
            report.append("   - Consider augmenting training data for this scenario\n")
            report.append("   - Review detection threshold settings\n\n")

        if avg_confidence < 0.7:
            report.append(
                f"2. **Improve OCR accuracy:** Average confidence is {avg_confidence:.2%}.\n"
            )
            report.append("   - Fine-tune OCR model on GTA V plate images\n")
            report.append("   - Adjust preprocessing steps\n\n")

        report.append("3. **Track optimization:**\n")
        report.append("   - Monitor track persistence across frames\n")
        report.append("   - Optimize OCR trigger intervals based on condition\n\n")

        report.append("\n---\n")
        report.append("*Report generated by evaluate_pipeline.py*\n")

        return "".join(report)

    def _parse_condition(self, video_name: str) -> str:
        """Parse condition from video filename."""
        parts = video_name.split("_")
        if len(parts) >= 2:
            time_of_day = parts[0]  # day/night
            weather = parts[1]  # clear/rain
            return f"{time_of_day}_{weather}"
        return "unknown"

    def _annotate_frame(self, frame: np.ndarray, tracks: Dict) -> np.ndarray:
        """Draw bounding boxes and plate numbers on frame."""
        for track_id, track in tracks.items():
            # Draw bounding box
            x1, y1, x2, y2 = map(int, track.bbox)
            color = (0, 255, 0) if track.text else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw track ID and plate number
            label = f"ID:{track_id}"
            if track.text:
                label += f" {track.text}"
                if track.ocr_confidence:
                    label += f" ({track.ocr_confidence:.2f})"

            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    def _save_sample_frames(self, video_name: str, samples: List[Dict]):
        """Save sample annotated frames."""
        sample_dir = os.path.join(self.output_dir, "samples", video_name)
        os.makedirs(sample_dir, exist_ok=True)

        for i, sample in enumerate(samples):
            filename = f"frame_{sample['frame_idx']:06d}_{sample['num_detections']}det.jpg"
            filepath = os.path.join(sample_dir, filename)
            cv2.imwrite(filepath, sample["image"])

    def save_results(self, results: Dict, report: str):
        """Save evaluation results and report to files."""
        # Save JSON results
        json_path = os.path.join(self.output_dir, "evaluation_results.json")

        # Convert results to JSON-serializable format
        json_results = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config_path,
            "video_results": results["video_results"],
            "condition_results": {
                k: {key: val for key, val in v.items() if key != "videos"}
                for k, v in results["condition_results"].items()
            },
        }

        with open(json_path, "w") as f:
            json.dump(json_results, f, indent=2)

        print(f"\n✓ Results saved to: {json_path}")

        # Save Markdown report
        report_path = os.path.join(self.output_dir, "evaluation_report.md")
        with open(report_path, "w") as f:
            f.write(report)

        print(f"✓ Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ALPR pipeline on test videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--videos", nargs="+", required=True, help="Video file paths or glob patterns"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline_config.yaml",
        help="Path to pipeline configuration file",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="Directory to save evaluation results",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process per video (default: all)",
    )

    parser.add_argument(
        "--no-samples", action="store_true", help="Do not save sample annotated frames"
    )

    args = parser.parse_args()

    # Expand glob patterns
    video_paths = []
    for pattern in args.videos:
        matches = glob.glob(pattern, recursive=True)
        video_paths.extend([p for p in matches if p.endswith((".mp4", ".avi", ".mov"))])

    if not video_paths:
        print("Error: No video files found matching the specified patterns.")
        sys.exit(1)

    print(f"Found {len(video_paths)} videos to evaluate")

    # Initialize evaluator
    evaluator = PipelineEvaluator(config_path=args.config, output_dir=args.output_dir)

    # Run evaluation
    results = evaluator.evaluate_multiple_videos(
        video_paths=video_paths, max_frames=args.max_frames, save_samples=not args.no_samples
    )

    # Generate report
    report = evaluator.generate_report(results)

    # Save results
    evaluator.save_results(results, report)

    print(f"\n{'=' * 80}")
    print("Evaluation complete!")
    print(f"{'=' * 80}")
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

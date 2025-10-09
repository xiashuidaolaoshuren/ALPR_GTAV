"""
Frame Extraction Script for GTA V ALPR Dataset Collection

This script extracts frames from recorded GTA V gameplay videos at specified intervals,
preparing them for license plate annotation and model training.

Usage:
    Single video:
        python scripts/data_ingestion/extract_frames.py --input video.mp4 --output datasets/lpr/train/images/
    
    Batch processing:
        python scripts/data_ingestion/extract_frames.py --batch --input_dir raw_footage/ --output_dir datasets/lpr/train/images/
    
    Custom frame rate:
        python scripts/data_ingestion/extract_frames.py --input video.mp4 --output datasets/lpr/train/images/ --fps 10

Author: GTA V ALPR Development Team
Version: 1.0
"""

import os
import argparse
import cv2
from pathlib import Path
from typing import Tuple, Optional
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FrameExtractor:
    """
    Extract frames from video files for dataset preparation.
    
    Attributes:
        output_fps (int): Target frame rate for extraction
        quality (int): JPEG quality (0-100)
        format (str): Output image format ('jpg' or 'png')
    """
    
    def __init__(self, output_fps: int = 5, quality: int = 95, format: str = 'jpg'):
        """
        Initialize FrameExtractor.
        
        Args:
            output_fps: Frames per second to extract (e.g., 5 means 1 frame every 200ms)
            quality: JPEG quality percentage (higher = better quality but larger files)
            format: Output format - 'jpg' (recommended) or 'png' (lossless but larger)
        """
        self.output_fps = output_fps
        self.quality = quality
        self.format = format
        
        logger.info(f"FrameExtractor initialized: {output_fps} FPS, Quality: {quality}%, Format: {format}")
    
    def extract_from_video(self, video_path: str, output_dir: str, prefix: Optional[str] = None) -> int:
        """
        Extract frames from a single video file.
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save extracted frames
            prefix: Optional prefix for output filenames (defaults to video name)
        
        Returns:
            Number of frames successfully extracted
        
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video cannot be opened or is invalid
        """
        # Validate input
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Open video
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Get video properties
        source_fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / source_fps if source_fps > 0 else 0
        
        logger.info(f"Processing: {os.path.basename(video_path)}")
        logger.info(f"  Source FPS: {source_fps:.2f}, Total frames: {total_frames}, Duration: {duration:.2f}s")
        
        # Calculate frame interval for extraction
        frame_interval = int(source_fps / self.output_fps) if source_fps > 0 else 1
        frame_interval = max(1, frame_interval)  # Ensure at least 1
        
        logger.info(f"  Extracting every {frame_interval} frames (target: {self.output_fps} FPS)")
        
        # Generate prefix from video filename if not provided
        if prefix is None:
            video_name = Path(video_path).stem  # filename without extension
            prefix = video_name
        
        # Extract frames
        frame_count = 0
        extracted_count = 0
        
        # Progress bar
        pbar = tqdm(total=total_frames, desc="Extracting frames", unit="frame")
        
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            # Extract every Nth frame
            if frame_count % frame_interval == 0:
                # Generate filename
                filename = f"{prefix}_{extracted_count:05d}.{self.format}"
                filepath = os.path.join(output_dir, filename)
                
                # Save frame
                if self.format == 'jpg':
                    cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                elif self.format == 'png':
                    cv2.imwrite(filepath, frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                else:
                    cv2.imwrite(filepath, frame)
                
                extracted_count += 1
            
            frame_count += 1
            pbar.update(1)
        
        pbar.close()
        video.release()
        
        logger.info(f"✓ Extracted {extracted_count} frames from {total_frames} total frames")
        return extracted_count
    
    def batch_extract(self, input_dir: str, output_dir: str, video_extensions: Tuple[str, ...] = ('.mp4', '.avi', '.mov')) -> dict:
        """
        Extract frames from all videos in a directory (recursively).
        
        Args:
            input_dir: Directory containing video files
            output_dir: Directory to save all extracted frames
            video_extensions: Tuple of video file extensions to process
        
        Returns:
            Dictionary mapping video paths to number of frames extracted
        """
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Find all video files recursively
        video_files = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(video_extensions):
                    video_files.append(os.path.join(root, file))
        
        if not video_files:
            logger.warning(f"No video files found in {input_dir} with extensions {video_extensions}")
            return {}
        
        logger.info(f"Found {len(video_files)} video files to process")
        
        # Process each video
        results = {}
        for i, video_path in enumerate(video_files, 1):
            logger.info(f"\n[{i}/{len(video_files)}] Processing: {os.path.basename(video_path)}")
            
            try:
                # Generate unique prefix from relative path
                rel_path = os.path.relpath(video_path, input_dir)
                prefix = os.path.splitext(rel_path.replace(os.sep, '_'))[0]
                
                extracted = self.extract_from_video(video_path, output_dir, prefix)
                results[video_path] = extracted
                
            except Exception as e:
                logger.error(f"Failed to process {video_path}: {e}")
                results[video_path] = 0
        
        # Summary
        total_extracted = sum(results.values())
        successful = sum(1 for count in results.values() if count > 0)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Batch extraction complete!")
        logger.info(f"  Videos processed: {successful}/{len(video_files)}")
        logger.info(f"  Total frames extracted: {total_extracted}")
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"{'='*60}")
        
        return results


def main():
    """Command-line interface for frame extraction."""
    parser = argparse.ArgumentParser(
        description='Extract frames from GTA V gameplay videos for ALPR dataset collection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from single video at 5 FPS
  python scripts/data_ingestion/extract_frames.py --input footage/day_clear_01.mp4 --output datasets/lpr/train/images/
  
  # Extract from single video at 10 FPS with custom quality
  python scripts/data_ingestion/extract_frames.py --input footage/night.mp4 --output datasets/lpr/valid/images/ --fps 10 --quality 90
  
  # Batch process all videos in directory
  python scripts/data_ingestion/extract_frames.py --batch --input_dir raw_footage/ --output_dir datasets/lpr/train/images/
  
  # Extract as PNG (lossless but larger files)
  python scripts/data_ingestion/extract_frames.py --input video.mp4 --output images/ --format png
        """
    )
    
    # Input arguments
    parser.add_argument('--input', type=str, help='Path to input video file (for single file mode)')
    parser.add_argument('--input_dir', type=str, help='Path to directory containing videos (for batch mode)')
    parser.add_argument('--batch', action='store_true', help='Enable batch processing mode')
    
    # Output arguments
    parser.add_argument('--output', '--output_dir', type=str, required=True, 
                       help='Output directory for extracted frames')
    
    # Extraction parameters
    parser.add_argument('--fps', type=int, default=5,
                       help='Target frame rate for extraction (default: 5 FPS = 1 frame every 200ms)')
    parser.add_argument('--quality', type=int, default=95,
                       help='JPEG quality percentage (0-100, default: 95)')
    parser.add_argument('--format', type=str, choices=['jpg', 'png'], default='jpg',
                       help='Output image format (default: jpg)')
    parser.add_argument('--prefix', type=str, default=None,
                       help='Custom prefix for output filenames (default: use video name)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.batch and not args.input_dir:
        parser.error("--batch mode requires --input_dir")
    if not args.batch and not args.input:
        parser.error("Single file mode requires --input")
    
    # Initialize extractor
    extractor = FrameExtractor(
        output_fps=args.fps,
        quality=args.quality,
        format=args.format
    )
    
    try:
        if args.batch:
            # Batch mode
            results = extractor.batch_extract(args.input_dir, args.output)
        else:
            # Single file mode
            extracted = extractor.extract_from_video(args.input, args.output, args.prefix)
            logger.info(f"\n✓ Successfully extracted {extracted} frames to {args.output}")
    
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

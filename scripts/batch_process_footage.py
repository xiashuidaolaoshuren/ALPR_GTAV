"""
Batch Frame Extraction Helper Script

This script automates the extraction of frames from all video files in the 
outputs/raw_footage/ directory, processing each condition folder separately.

Usage:
    python scripts/batch_process_footage.py [--fps 5] [--quality 95]

Author: GTA V ALPR Development Team
Version: 1.0
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path to import extract_frames module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import FrameExtractor from extract_frames.py
try:
    from scripts.extract_frames import FrameExtractor
except ImportError:
    print("ERROR: Could not import FrameExtractor. Ensure extract_frames.py is in scripts/ directory.")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Process all video files in raw_footage directory."""
    parser = argparse.ArgumentParser(
        description='Batch process all GTA V footage for frame extraction'
    )
    parser.add_argument('--fps', type=int, default=5, 
                       help='Frame extraction rate (default: 5)')
    parser.add_argument('--quality', type=int, default=95,
                       help='JPEG quality (default: 95)')
    parser.add_argument('--format', type=str, choices=['jpg', 'png'], default='jpg',
                       help='Output format (default: jpg)')
    
    args = parser.parse_args()
    
    # Define paths
    project_root = Path(__file__).parent.parent
    raw_footage_dir = project_root / 'outputs' / 'raw_footage'
    output_dir = project_root / 'outputs' / 'test_images'
    
    # Check if raw_footage directory exists
    if not raw_footage_dir.exists():
        logger.error(f"Raw footage directory not found: {raw_footage_dir}")
        logger.info("Please create the directory and add video files, or record gameplay first.")
        return 1
    
    # Define condition folders
    condition_folders = ['day_clear', 'day_rain', 'night_clear', 'night_rain']
    
    # Check which folders have videos
    folders_with_videos = []
    for folder in condition_folders:
        folder_path = raw_footage_dir / folder
        if folder_path.exists():
            # Check for video files
            video_files = list(folder_path.glob('*.mp4')) + \
                         list(folder_path.glob('*.avi')) + \
                         list(folder_path.glob('*.mov'))
            if video_files:
                folders_with_videos.append((folder, len(video_files)))
    
    if not folders_with_videos:
        logger.warning("No video files found in any condition folders.")
        logger.info("Expected locations:")
        for folder in condition_folders:
            logger.info(f"  - {raw_footage_dir / folder}")
        logger.info("\nPlease add video files (.mp4, .avi, .mov) to these folders.")
        return 1
    
    # Display found videos
    logger.info("="*60)
    logger.info("Found videos in the following folders:")
    for folder, count in folders_with_videos:
        logger.info(f"  {folder}: {count} video(s)")
    logger.info("="*60)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Initialize extractor
    extractor = FrameExtractor(
        output_fps=args.fps,
        quality=args.quality,
        format=args.format
    )
    
    # Process each condition folder
    total_frames = 0
    results_by_condition = {}
    
    for folder_name, video_count in folders_with_videos:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing folder: {folder_name}")
        logger.info(f"{'='*60}")
        
        folder_path = raw_footage_dir / folder_name
        
        try:
            # Batch extract frames
            results = extractor.batch_extract(
                input_dir=str(folder_path),
                output_dir=str(output_dir)
            )
            
            folder_total = sum(results.values())
            results_by_condition[folder_name] = folder_total
            total_frames += folder_total
            
            logger.info(f"✓ {folder_name}: Extracted {folder_total} frames")
            
        except Exception as e:
            logger.error(f"Failed to process {folder_name}: {e}")
            results_by_condition[folder_name] = 0
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total frames extracted: {total_frames}")
    logger.info("\nBreakdown by condition:")
    for condition, count in results_by_condition.items():
        logger.info(f"  {condition}: {count} frames")
    logger.info(f"\nAll frames saved to: {output_dir}")
    logger.info(f"{'='*60}")
    
    # Next steps guidance
    logger.info("\nNEXT STEPS:")
    logger.info("1. Review extracted images in outputs/test_images/")
    logger.info("2. Remove any low-quality or duplicate images")
    logger.info("3. Update metadata.txt with image details")
    logger.info("4. Run quality check to verify dataset completeness")
    
    if total_frames < 50:
        logger.warning(f"\n⚠ WARNING: Only {total_frames} frames extracted. Target is 50-100.")
        logger.info("Consider:")
        logger.info("  - Recording more footage")
        logger.info("  - Reducing FPS (more frames per video): --fps 3")
        logger.info("  - Adding more videos to condition folders")
    elif total_frames > 200:
        logger.info(f"\n✓ Great! {total_frames} frames extracted (exceeds minimum).")
        logger.info("You can curate to the best 100-150 images for quality.")
    else:
        logger.info(f"\n✓ Excellent! {total_frames} frames extracted (within target range).")
    
    return 0


if __name__ == "__main__":
    exit(main())

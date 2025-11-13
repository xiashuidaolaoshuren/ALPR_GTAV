"""
Pipeline Wrapper for GUI Integration

Provides thread-safe wrapper for ALPR pipeline integration with Streamlit GUI.
Handles background video processing, result queuing, and lifecycle management.

Author: Felix (xiashuidaolaoshuren)
Date: 2025-11-13
"""

import streamlit as st
import threading
import queue
import time
import logging
import subprocess
from typing import Dict, Optional, Any
from pathlib import Path
import cv2
import numpy as np
import yaml
import tempfile

from src.pipeline.alpr_pipeline import ALPRPipeline
from src.pipeline.utils import draw_tracks_on_frame

logger = logging.getLogger(__name__)


def is_ffmpeg_available() -> bool:
    """Check if FFmpeg is available in the system."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@st.cache_resource
def load_cached_pipeline(config_str: str, device: str):
    """
    Load and cache ALPR pipeline models globally.
    
    This function is cached using @st.cache_resource to ensure models
    are loaded only once and shared across all sessions. The pipeline
    is cached based on configuration hash to support different configs.
    
    Args:
        config_str: YAML configuration as string (for cache key)
        device: Device to use ('cuda' or 'cpu')
    
    Returns:
        ALPRPipeline: Initialized pipeline with loaded models
    
    Note:
        - Models are loaded once per config/device combination
        - Shared across all user sessions (memory efficient)
        - Cache persists until Streamlit server restarts
    """
    logger.info(f"Loading cached pipeline (device={device})")
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
        f.write(config_str)
        temp_config_path = f.name
    
    try:
        # Load pipeline with config
        pipeline = ALPRPipeline(temp_config_path)
        logger.info("✓ Pipeline loaded and cached successfully")
        return pipeline
    finally:
        # Cleanup temp file
        Path(temp_config_path).unlink(missing_ok=True)


class GUIPipelineWrapper:
    """
    Thread-safe wrapper for ALPR pipeline GUI integration.
    
    Manages background video processing in a separate thread, communicates
    results via queue, and provides lifecycle control (start/stop/pause).
    
    Features:
    - Non-blocking background processing
    - Thread-safe result queue
    - Pause/resume support
    - Clean shutdown handling
    - Progress tracking
    - FPS monitoring
    
    Attributes:
        pipeline: ALPRPipeline instance
        result_queue: Thread-safe queue for results
        stop_event: Event to signal stop
        pause_event: Event to signal pause
        processing_thread: Background thread reference
        stats: Processing statistics
    """
    
    def __init__(self, config_dict: Dict):
        """
        Initialize pipeline wrapper.
        
        Args:
            config_dict: Pipeline configuration dictionary with keys:
                - detection: detection config (conf_threshold, iou_threshold, device)
                - recognition: OCR config (min_conf)
                - tracking: tracking config (ocr_interval)
        """
        logger.info("Initializing GUIPipelineWrapper")
        
        # Build config YAML string for caching
        config_yaml = yaml.dump(config_dict)
        device = config_dict.get('detection', {}).get('device', 'cpu')
        
        # Load cached pipeline
        self.pipeline = load_cached_pipeline(config_yaml, device)
        
        # Threading components
        self.result_queue = queue.Queue(maxsize=100)  # Bounded queue
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.processing_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'frames_displayed': 0,
            'start_time': None,
            'total_frames': 0,
            'fps': 0.0
        }
        
        logger.info("✓ GUIPipelineWrapper initialized")
    
    def process_video_background(self, video_path: str, display_fps: int = 15):
        """
        Background thread target for video processing.
        
        Processes video frames through ALPR pipeline, draws overlays,
        and writes output video. Sends progress updates to GUI.
        
        Args:
            video_path: Path to video file
            display_fps: Target FPS for progress updates (not output FPS)
        """
        logger.info(f"Starting background processing: {video_path}")
        
        cap = None
        out = None
        output_path = None
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                self.result_queue.put({'error': 'Failed to open video'})
                return
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create output video file
            output_dir = Path("outputs/processed_videos")
            output_dir.mkdir(parents=True, exist_ok=True)
            input_name = Path(video_path).stem
            output_path = str(output_dir / f"{input_name}_processed.mp4")
            temp_output_path = str(output_dir / f"{input_name}_processed_temp.avi")
            
            # Try to use H.264 codec for browser compatibility
            # On Windows, try 'H264' or 'avc1' codecs
            use_temp_file = False
            fourcc = None
            
            # Try H264 codec variants for browser compatibility
            for codec in ['avc1', 'H264', 'X264']:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    out = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
                    if out.isOpened():
                        logger.info(f"Using {codec} codec for browser-compatible output")
                        break
                    out.release()
                except:
                    pass
            
            # If H.264 variants fail, use temp file with mp4v and convert later
            if fourcc is None or not out.isOpened():
                logger.warning("H.264 codec not available, will use mp4v with FFmpeg conversion")
                use_temp_file = True
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(temp_output_path, fourcc, video_fps, (width, height))
                
                if not out.isOpened():
                    # Last resort: XVID
                    logger.warning("mp4v failed, trying XVID...")
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out = cv2.VideoWriter(temp_output_path, fourcc, video_fps, (width, height))
                    
                    if not out.isOpened():
                        logger.error("Failed to initialize VideoWriter with any codec")
                        self.result_queue.put({'error': 'Failed to create output video'})
                        return
            
            # Calculate frame skip for progress updates
            if video_fps > display_fps:
                frame_skip = int(video_fps / display_fps)
            else:
                frame_skip = 1
            
            self.stats['total_frames'] = total_frames
            self.stats['start_time'] = time.time()
            
            logger.info(f"Video properties: {total_frames} frames at {video_fps} FPS ({width}x{height})")
            logger.info(f"Output: {output_path}")
            logger.info(f"Progress update interval: every {frame_skip} frames")
            
            frame_idx = 0
            
            while not self.stop_event.is_set():
                # Check pause
                if self.pause_event.is_set():
                    time.sleep(0.1)
                    continue
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video reached")
                    break
                
                # Process frame through pipeline
                start_time = time.time()
                tracks = self.pipeline.process_frame(frame)
                process_time = time.time() - start_time
                
                # Draw detections on frame using the same utility as CLI
                frame_with_overlays = draw_tracks_on_frame(
                    frame,
                    tracks,
                    show_text=True,
                    show_track_id=True,
                    show_confidence=True
                )
                
                # Write frame to output video
                out.write(frame_with_overlays)
                
                # Update stats
                self.stats['frames_processed'] = frame_idx + 1
                elapsed = time.time() - self.stats['start_time']
                self.stats['fps'] = self.stats['frames_processed'] / elapsed if elapsed > 0 else 0
                
                # Send progress update to GUI (reduce frequency for better performance)
                # Only send updates every 3rd frame or at key intervals
                if frame_idx % 3 == 0:
                    result = {
                        'progress': True,
                        'tracks': tracks,
                        'frame_idx': frame_idx,
                        'total_frames': total_frames,
                        'processing_fps': self.stats['fps']
                    }
                    
                    try:
                        self.result_queue.put(result, timeout=0.1)
                        self.stats['frames_displayed'] += 1
                    except queue.Full:
                        logger.warning("Result queue full, dropping progress update")
                
                frame_idx += 1
            
            # Close the output video
            out.release()
            
            # Convert to H.264 if we used a temp file
            if use_temp_file:
                if is_ffmpeg_available():
                    logger.info("Converting video to H.264 for browser compatibility...")
                    try:
                        # Use FFmpeg to convert to H.264
                        conversion_cmd = [
                            'ffmpeg',
                            '-i', temp_output_path,
                            '-c:v', 'libx264',
                            '-preset', 'fast',
                            '-crf', '23',
                            '-c:a', 'aac',
                            '-y',  # Overwrite output file
                            output_path
                        ]
                        
                        result = subprocess.run(
                            conversion_cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            timeout=300  # 5 minute timeout
                        )
                        
                        if result.returncode == 0:
                            logger.info("Video conversion successful")
                            # Remove temp file
                            try:
                                Path(temp_output_path).unlink()
                            except:
                                pass
                        else:
                            logger.error(f"FFmpeg conversion failed: {result.stderr.decode()}")
                            # Use the temp file as final output
                            output_path = temp_output_path
                            logger.warning("Using non-H.264 video - may not play in browser")
                    
                    except subprocess.TimeoutExpired:
                        logger.error("FFmpeg conversion timed out")
                        output_path = temp_output_path
                        logger.warning("Using non-H.264 video - may not play in browser")
                    
                    except Exception as e:
                        logger.error(f"FFmpeg conversion error: {e}")
                        output_path = temp_output_path
                        logger.warning("Using non-H.264 video - may not play in browser")
                else:
                    logger.warning("FFmpeg not available - video may not play in browser")
                    logger.warning("Install FFmpeg for automatic H.264 conversion")
                    output_path = temp_output_path
            
            # Send completion signal with output path
            completion_result = {
                'done': True,
                'output_video': output_path,
                'total_frames': frame_idx,
                'processing_time': time.time() - self.stats['start_time']
            }
            self.result_queue.put(completion_result)
            logger.info(f"Processing complete: {frame_idx} frames processed -> {output_path}")
            
        except Exception as e:
            logger.error(f"Error in background processing: {e}", exc_info=True)
            self.result_queue.put({'error': str(e)})
        finally:
            if cap and cap.isOpened():
                cap.release()
            if out and out.isOpened():
                out.release()
            logger.info("Background processing thread finished")
    
    def start_processing(self, video_path: str, display_fps: int = 15) -> threading.Thread:
        """
        Start background video processing.
        
        Args:
            video_path: Path to video file
            display_fps: Target display FPS
        
        Returns:
            threading.Thread: Reference to processing thread
        
        Raises:
            RuntimeError: If thread is already running
        """
        if self.processing_thread and self.processing_thread.is_alive():
            raise RuntimeError("Processing thread already running")
        
        # Clear events
        self.stop_event.clear()
        self.pause_event.clear()
        
        # Clear queue
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
        
        # Reset stats
        self.stats = {
            'frames_processed': 0,
            'frames_displayed': 0,
            'start_time': None,
            'total_frames': 0,
            'fps': 0.0
        }
        
        # Start thread
        self.processing_thread = threading.Thread(
            target=self.process_video_background,
            args=(video_path, display_fps),
            daemon=True
        )
        self.processing_thread.start()
        
        logger.info("Background processing thread started")
        return self.processing_thread
    
    def stop_processing(self):
        """Stop background processing and wait for cleanup."""
        if self.processing_thread and self.processing_thread.is_alive():
            logger.info("Stopping background processing...")
            self.stop_event.set()
            self.processing_thread.join(timeout=5.0)
            
            if self.processing_thread.is_alive():
                logger.warning("Processing thread did not stop gracefully")
            else:
                logger.info("✓ Background processing stopped")
    
    def pause_processing(self):
        """Pause background processing."""
        self.pause_event.set()
        logger.info("Processing paused")
    
    def resume_processing(self):
        """Resume paused processing."""
        self.pause_event.clear()
        logger.info("Processing resumed")
    
    def get_result(self, timeout: float = 0.001) -> Optional[Dict]:
        """
        Get next result from queue (non-blocking).
        
        Args:
            timeout: Max time to wait for result (seconds)
        
        Returns:
            Dict with result data, None if queue empty, or {'error': msg} on error
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def is_processing(self) -> bool:
        """Check if processing thread is active."""
        return self.processing_thread is not None and self.processing_thread.is_alive()
    
    def get_stats(self) -> Dict:
        """Get current processing statistics."""
        return self.stats.copy()

"""
End-to-End ALPR Pipeline Module

This module provides the main ALPRPipeline class that orchestrates all components:
detection, tracking, preprocessing, and recognition to perform complete ALPR
on video frames.
"""

import logging
from typing import Dict, List, Optional
import numpy as np
import yaml

from src.detection.model import load_detection_model
from src.detection.utils import crop_detections
from src.recognition.model import load_ocr_model, recognize_text
from src.preprocessing.image_enhancement import preprocess_plate
from src.tracking.tracker import PlateTrack
from src.tracking.utils import cleanup_lost_tracks, get_track_summary

logger = logging.getLogger(__name__)


class ALPRPipeline:
    """
    End-to-end Automatic License Plate Recognition pipeline.
    
    This class coordinates all ALPR components to process video frames:
    1. Detection + Tracking: Uses YOLOv8 with ByteTrack for plate detection and tracking
    2. Track Management: Creates and updates PlateTrack objects for each detection
    3. Conditional OCR: Runs OCR only when needed based on track state
    4. Cleanup: Removes lost tracks to prevent memory leaks
    
    Attributes:
        config (dict): Full pipeline configuration loaded from YAML
        detection_model: Loaded YOLOv8 model for plate detection
        ocr_model: Loaded PaddleOCR model for text recognition
        tracks (Dict[int, PlateTrack]): Active tracks indexed by track ID
        frame_count (int): Number of frames processed
    
    Example:
        >>> pipeline = ALPRPipeline('configs/pipeline_config.yaml')
        >>> import cv2
        >>> frame = cv2.imread('test_frame.jpg')
        >>> tracks = pipeline.process_frame(frame)
        >>> for track_id, track in tracks.items():
        ...     if track.text:
        ...         print(f"Track {track_id}: {track.text} ({track.ocr_confidence:.2f})")
    """
    
    def __init__(self, config_path: str):
        """
        Initialize ALPR pipeline with configuration.
        
        Loads configuration from YAML file and initializes all models.
        
        Args:
            config_path: Path to pipeline configuration YAML file
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            RuntimeError: If model initialization fails
        
        Note:
            - Model loading may take several seconds on first run
            - GPU will be used if available for both detection and recognition
            - Logs detailed initialization information
        """
        logger.info('=' * 60)
        logger.info('Initializing ALPR Pipeline')
        logger.info('=' * 60)
        
        # Load configuration
        logger.info(f'Loading configuration from: {config_path}')
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            logger.info('✓ Configuration loaded successfully')
        except FileNotFoundError:
            logger.error(f'Configuration file not found: {config_path}')
            raise
        except Exception as e:
            logger.error(f'Failed to load configuration: {e}')
            raise RuntimeError(f'Configuration loading failed: {e}')
        
        # Initialize detection model
        logger.info('Loading detection model...')
        try:
            self.detection_model = load_detection_model(
                model_path=self.config['detection']['model_path'],
                device=self.config['detection']['device']
            )
            logger.info('✓ Detection model loaded')
        except Exception as e:
            logger.error(f'Failed to load detection model: {e}')
            raise RuntimeError(f'Detection model initialization failed: {e}')
        
        # Initialize OCR model
        logger.info('Loading OCR model...')
        try:
            self.ocr_model = load_ocr_model(self.config['recognition'])
            logger.info('✓ OCR model loaded')
        except Exception as e:
            logger.error(f'Failed to load OCR model: {e}')
            raise RuntimeError(f'OCR model initialization failed: {e}')
        
        # Initialize track storage
        self.tracks: Dict[int, PlateTrack] = {}
        self.frame_count = 0
        
        logger.info('=' * 60)
        logger.info('ALPR Pipeline Initialization Complete')
        logger.info('=' * 60)
    
    def process_frame(self, frame: np.ndarray) -> Dict[int, PlateTrack]:
        """
        Process a single video frame through the complete ALPR pipeline.
        
        Executes the 4-stage pipeline:
        1. Detection + Tracking: Run YOLOv8 with ByteTrack
        2. Track Management: Update track states
        3. Conditional OCR: Run OCR only when needed
        4. Cleanup: Remove lost tracks
        
        Args:
            frame: Input frame in BGR format (OpenCV convention)
                  Shape: (height, width, 3)
        
        Returns:
            Dict[int, PlateTrack]: Dictionary of active tracks indexed by track ID.
                                  Each PlateTrack contains current state including
                                  bbox, text, confidence, and age.
        
        Raises:
            ValueError: If frame is invalid
            RuntimeError: If processing fails
        
        Example:
            >>> tracks = pipeline.process_frame(frame)
            >>> print(f"Active tracks: {len(tracks)}")
            >>> for track_id, track in tracks.items():
            ...     print(f"  {track_id}: {track.text or 'No text'} "
            ...           f"(age={track.age}, conf={track.ocr_confidence:.2f})")
        
        Note:
            - Track IDs persist across frames (ByteTrack maintains identity)
            - OCR is only run when should_run_ocr() returns True
            - Lost tracks are kept for max_age frames before removal
        """
        # Validate input
        if not isinstance(frame, np.ndarray):
            raise ValueError(f"Frame must be numpy array, got {type(frame)}")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Frame must be BGR image with shape (H, W, 3), got {frame.shape}")
        
        logger.debug(f'Processing frame {self.frame_count} (shape={frame.shape})')
        
        # ===== Stage 1: Detection + Tracking =====
        logger.debug('Stage 1: Running detection + tracking')
        
        try:
            # Run YOLOv8 tracking (ByteTrack integration)
            results = self.detection_model.track(
                source=frame,
                conf=self.config['detection']['confidence_threshold'],
                iou=self.config['detection']['iou_threshold'],
                tracker='bytetrack.yaml',
                persist=True,  # Crucial: maintains track IDs across frames
                verbose=False
            )
        except Exception as e:
            logger.error(f'Detection + tracking failed: {e}')
            raise RuntimeError(f'Detection stage failed: {e}')
        
        # ===== Stage 2: Update Track States =====
        logger.debug('Stage 2: Updating track states')
        
        current_track_ids = set()
        ocr_count = 0
        
        # Parse detection results
        for result in results:
            # Check if any detections exist
            if not hasattr(result, 'boxes') or result.boxes is None:
                logger.debug('No detections in this frame')
                continue
            
            boxes = result.boxes
            
            # Check if tracks exist
            if boxes.id is None:
                logger.debug('No tracks assigned by ByteTrack')
                continue
            
            # Process each tracked detection
            for box in boxes:
                # Extract track information
                track_id = int(box.id.item())
                bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                confidence = float(box.conf.item())
                
                # Convert bbox to tuple of integers
                bbox_tuple = tuple(map(int, bbox))
                
                # Create new track or update existing
                if track_id not in self.tracks:
                    self.tracks[track_id] = PlateTrack(
                        track_id=track_id,
                        bbox=bbox_tuple,
                        confidence=confidence
                    )
                    logger.info(f'New track detected: ID={track_id}, bbox={bbox_tuple}, conf={confidence:.3f}')
                else:
                    self.tracks[track_id].update(bbox=bbox_tuple, confidence=confidence)
                    logger.debug(f'Track updated: ID={track_id}, age={self.tracks[track_id].age}')
                
                current_track_ids.add(track_id)
                
                # ===== Stage 3: Conditional OCR =====
                track = self.tracks[track_id]
                
                # Check if OCR should be run for this track
                if track.should_run_ocr(self.config['tracking']):
                    logger.debug(f'OCR triggered for track {track_id} '
                               f'(age={track.age}, frames_since_last_ocr={track.frames_since_last_ocr})')
                    
                    try:
                        # Crop plate from frame
                        x1, y1, x2, y2 = bbox_tuple
                        cropped = frame[y1:y2, x1:x2]
                        
                        if cropped.size == 0:
                            logger.warning(f'Empty crop for track {track_id}, skipping OCR')
                            continue
                        
                        # Preprocess plate image
                        preprocessed = preprocess_plate(
                            cropped_image=cropped,
                            config=self.config['preprocessing']
                        )
                        
                        # Run OCR
                        text, ocr_conf = recognize_text(
                            preprocessed_image=preprocessed,
                            ocr_model=self.ocr_model,
                            config=self.config['recognition']
                        )
                        
                        # Update track with OCR results
                        track.update_text(text=text, confidence=ocr_conf)
                        ocr_count += 1
                        
                        if text:
                            logger.info(f'Track {track_id} recognized: "{text}" (confidence={ocr_conf:.3f})')
                        else:
                            logger.debug(f'Track {track_id}: No valid text recognized')
                    
                    except Exception as e:
                        logger.error(f'OCR failed for track {track_id}: {e}')
                        # Continue processing other tracks
                        continue
        
        # ===== Stage 4: Cleanup Lost Tracks =====
        logger.debug('Stage 4: Cleaning up lost tracks')
        
        # Mark tracks that disappeared as lost
        lost_track_ids = set(self.tracks.keys()) - current_track_ids
        
        for track_id in lost_track_ids:
            self.tracks[track_id].mark_lost()
            logger.debug(f'Track {track_id} marked as lost')
        
        # Remove inactive tracks older than max_age
        max_age = self.config['tracking'].get('max_age', 30)
        initial_count = len(self.tracks)
        
        self.tracks = {
            tid: track
            for tid, track in self.tracks.items()
            if track.is_active or track.age < max_age
        }
        
        removed_count = initial_count - len(self.tracks)
        if removed_count > 0:
            logger.info(f'Removed {removed_count} old lost tracks')
        
        # Increment frame counter
        self.frame_count += 1
        
        # Log summary
        summary = get_track_summary(self.tracks)
        logger.info(f'Frame {self.frame_count}: {summary["active"]} active tracks, '
                   f'{summary["recognized"]} recognized, {ocr_count} OCR runs')
        
        return self.tracks
    
    def reset(self) -> None:
        """
        Reset pipeline state (clear all tracks and counters).
        
        Useful when starting to process a new video or after processing is complete.
        
        Example:
            >>> pipeline.reset()
            >>> print(f"Tracks cleared: {len(pipeline.tracks)}")
            Tracks cleared: 0
        """
        logger.info('Resetting pipeline state')
        self.tracks.clear()
        self.frame_count = 0
        logger.info('Pipeline reset complete')
    
    def get_statistics(self) -> dict:
        """
        Get current pipeline statistics.
        
        Returns:
            dict: Statistics including:
                - frame_count: Total frames processed
                - track_count: Current active track count
                - recognized_count: Tracks with recognized text
                - avg_track_age: Average track age
                - avg_ocr_confidence: Average OCR confidence
        
        Example:
            >>> stats = pipeline.get_statistics()
            >>> print(f"Processed {stats['frame_count']} frames, "
            ...       f"{stats['recognized_count']} plates recognized")
        """
        summary = get_track_summary(self.tracks)
        
        return {
            'frame_count': self.frame_count,
            'track_count': summary['total'],
            'active_count': summary['active'],
            'recognized_count': summary['recognized'],
            'avg_track_age': summary['avg_age'],
            'avg_ocr_confidence': summary['avg_ocr_confidence']
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        stats = self.get_statistics()
        return (f"ALPRPipeline(frames={stats['frame_count']}, "
                f"tracks={stats['track_count']}, "
                f"recognized={stats['recognized_count']})")

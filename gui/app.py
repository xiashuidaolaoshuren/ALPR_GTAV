"""
GTA V ALPR System - Streamlit GUI Application

Main entry point for the interactive license plate recognition demonstration interface.
This application provides a user-friendly way to upload videos, configure pipeline
parameters, and visualize real-time ALPR processing results.

Author: Felix (xiashuidaolaoshuren)
Date: 2025-11-12
"""

import streamlit as st
import logging
import cv2
import time
from pathlib import Path

# Import GUI components
from gui.utils.video_handler import VideoHandler
from gui.components.video_display import VideoDisplay
from gui.components.control_panel import ControlPanel
from gui.components.info_panel import InfoPanel
from gui.utils.logging_handler import StreamlitLogHandler
from gui.utils.pipeline_wrapper import GUIPipelineWrapper
from gui.utils.performance import UpdateBatcher

# IMPORTANT: st.set_page_config must be the first Streamlit command
st.set_page_config(
    page_title="GTA V ALPR System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state():
    """
    Initialize all session state variables used throughout the application.
    
    This function should be called once at the start of the app to ensure
    all required state variables exist before they're accessed.
    """
    # Pipeline state
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    
    # Processing control
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    
    # Configuration state - initialized by ControlPanel
    # Don't initialize here to avoid conflicts
    
    # Results storage
    if 'results' not in st.session_state:
        st.session_state.results = []
    
    # Video handling - video_handler stores the VideoHandler instance
    if 'video_handler' not in st.session_state:
        st.session_state.video_handler = None
    
    if 'video_info' not in st.session_state:
        st.session_state.video_info = None
    
    # Video display component
    if 'video_display' not in st.session_state:
        st.session_state.video_display = None
    
    # GUI components
    if 'control_panel' not in st.session_state:
        st.session_state.control_panel = None
    
    if 'info_panel' not in st.session_state:
        st.session_state.info_panel = None
    
    if 'log_handler' not in st.session_state:
        # Create logging handler
        log_handler = StreamlitLogHandler(max_logs=100)
        st.session_state.log_handler = log_handler
        
        # Add handler to root logger
        logger = logging.getLogger()
        logger.addHandler(log_handler)
        logger.setLevel(logging.INFO)
    
    # Detection and recognition results
    if 'unique_plates' not in st.session_state:
        st.session_state.unique_plates = {}
    
    if 'latest_recognitions' not in st.session_state:
        st.session_state.latest_recognitions = []
    
    if 'all_detections' not in st.session_state:
        st.session_state.all_detections = []
    
    # Tracking state
    if 'active_tracks' not in st.session_state:
        st.session_state.active_tracks = {}
    
    # Performance metrics
    if 'current_fps' not in st.session_state:
        st.session_state.current_fps = 0.0
    
    # Pipeline wrapper
    if 'pipeline_wrapper' not in st.session_state:
        st.session_state.pipeline_wrapper = None
    
    if 'processing_thread' not in st.session_state:
        st.session_state.processing_thread = None
    
    # Performance settings
    if 'display_fps' not in st.session_state:
        st.session_state.display_fps = 15
    
    if 'update_interval' not in st.session_state:
        st.session_state.update_interval = 30
    
    # Update batcher
    if 'update_batcher' not in st.session_state:
        st.session_state.update_batcher = UpdateBatcher(update_interval=30)
    
    # Pause state
    if 'paused' not in st.session_state:
        st.session_state.paused = False
    
    # Output video path
    if 'output_video_path' not in st.session_state:
        st.session_state.output_video_path = None


def cleanup_previous_video():
    """
    Clean up previous video resources when a new video is uploaded.
    """
    if st.session_state.pipeline_wrapper is not None:
        st.session_state.pipeline_wrapper.stop_processing()
        st.session_state.pipeline_wrapper = None
    
    if st.session_state.video_handler is not None:
        st.session_state.video_handler.cleanup()
        st.session_state.video_handler = None
        st.session_state.video_info = None
        st.session_state.results = []
        st.session_state.unique_plates = {}
        st.session_state.latest_recognitions = []
        st.session_state.all_detections = []
        st.session_state.active_tracks = {}
        st.session_state.processing = False
        st.session_state.paused = False
        st.session_state.current_fps = 0.0
        st.session_state.processing_thread = None
        st.session_state.output_video_path = None


def build_pipeline_config() -> dict:
    """
    Build pipeline configuration dict from GUI session state.
    
    Returns:
        dict: Pipeline configuration in expected format
    """
    gui_config = st.session_state.pipeline_config
    
    return {
        'detection': {
            'model_path': 'models/detection/yolov8_finetuned_v2_best.pt',
            'confidence_threshold': gui_config['conf_threshold'],
            'iou_threshold': gui_config['iou_threshold'],
            'device': gui_config['device']
        },
        'recognition': {
            'use_gpu': gui_config['device'] == 'cuda',
            'lang': 'en',
            'det': False,
            'rec': True,
            'show_log': False,
            'min_conf': gui_config['min_ocr_conf']
        },
        'tracking': {
            'max_age': 30,
            'min_hits': 3,
            'iou_threshold': 0.3,
            'ocr_interval': gui_config['ocr_interval']
        },
        'preprocessing': {
            'enabled': False
        }
    }


def handle_start_processing():
    """Handle the Start Processing button click."""
    try:
        # Build configuration
        config = build_pipeline_config()
        
        logging.info("Starting video processing...")
        logging.info(f"Configuration: {config}")
        
        # Reset output video path
        st.session_state.output_video_path = None
        
        # Create pipeline wrapper
        st.session_state.pipeline_wrapper = GUIPipelineWrapper(config)
        
        # Start processing
        video_path = st.session_state.video_handler.temp_path
        display_fps = st.session_state.display_fps
        
        thread = st.session_state.pipeline_wrapper.start_processing(
            video_path,
            display_fps=display_fps
        )
        st.session_state.processing_thread = thread
        st.session_state.processing = True
        st.session_state.paused = False
        
        # Reset update batcher
        st.session_state.update_batcher.reset()
        
        logging.info("✓ Video processing started")
        
    except Exception as e:
        logging.error(f"Failed to start processing: {e}", exc_info=True)
        st.session_state.processing = False
        st.error(f"❌ Error starting processing: {e}")


def handle_stop_processing():
    """Handle the Stop Processing button click."""
    try:
        logging.info("Stopping video processing...")
        
        if st.session_state.pipeline_wrapper:
            st.session_state.pipeline_wrapper.stop_processing()
        
        st.session_state.processing = False
        st.session_state.paused = False
        logging.info("✓ Video processing stopped")
        
    except Exception as e:
        logging.error(f"Error stopping processing: {e}", exc_info=True)


def handle_pause_processing():
    """Handle the Pause/Resume button click."""
    if st.session_state.pipeline_wrapper:
        if st.session_state.paused:
            st.session_state.pipeline_wrapper.resume_processing()
            st.session_state.paused = False
            logging.info("✓ Processing resumed")
        else:
            st.session_state.pipeline_wrapper.pause_processing()
            st.session_state.paused = True
            logging.info("Processing paused")


def main():
    """Main application entry point."""
    
    # Initialize session state
    initialize_session_state()
    
    # Application header
    st.title("🚗 GTA V License Plate Recognition System")
    st.markdown(
        """
        **Interactive demonstration of ALPR pipeline**
        
        This application allows you to upload GTA V gameplay videos and process them 
        through the complete ALPR pipeline with real-time visualization of detection, 
        recognition, and tracking results.
        """
    )
    
    st.divider()
    
    # Sidebar: Video Upload and Configuration
    with st.sidebar:
        st.header("📹 Video Input")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Select a GTA V gameplay video for processing",
            key="video_uploader"
        )
        
        # Handle video upload
        if uploaded_file is not None:
            # Check if this is a new file (different from previously uploaded)
            current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            previous_file_id = st.session_state.get('previous_file_id', None)
            
            if current_file_id != previous_file_id:
                # New file uploaded, clean up previous resources
                cleanup_previous_video()
                st.session_state.previous_file_id = current_file_id
                
                # Process new video
                try:
                    with st.spinner("Loading video..."):
                        # Create VideoHandler instance
                        video_handler = VideoHandler(uploaded_file)
                        
                        # Save to temporary file
                        temp_path = video_handler.save_temp_file()
                        
                        # Extract video metadata
                        video_info = video_handler.get_video_info()
                        
                        # Store in session state
                        st.session_state.video_handler = video_handler
                        st.session_state.video_info = video_info
                    
                    # Display success message and metadata
                    st.success(f"✅ Video loaded: {uploaded_file.name}")
                    
                    # Display video information
                    st.markdown("### 📊 Video Information")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Duration", f"{video_info['duration']:.2f}s")
                        st.metric("FPS", f"{video_info['fps']}")
                    with col2:
                        st.metric("Frames", f"{video_info['frame_count']:,}")
                        st.metric("Resolution", f"{video_info['width']}×{video_info['height']}")
                    
                except Exception as e:
                    st.error(f"❌ Error loading video: {e}")
                    st.session_state.video_handler = None
                    st.session_state.video_info = None
            
            else:
                # Same file, just display info
                if st.session_state.video_info:
                    video_info = st.session_state.video_info
                    st.success(f"✅ Video loaded: {uploaded_file.name}")
                    
                    st.markdown("### 📊 Video Information")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Duration", f"{video_info['duration']:.2f}s")
                        st.metric("FPS", f"{video_info['fps']}")
                    with col2:
                        st.metric("Frames", f"{video_info['frame_count']:,}")
                        st.metric("Resolution", f"{video_info['width']}×{video_info['height']}")
        
        else:
            # No file uploaded
            st.info("👆 Please upload a video file to begin")
        
        st.divider()
        
        # Configuration and Control Panel (Task 22)
        st.header("⚙️ Configuration & Control")
        
        # Initialize control panel if not already done
        if st.session_state.control_panel is None:
            st.session_state.control_panel = ControlPanel()
        
        # Render configuration section
        current_config = st.session_state.control_panel.render_configuration()
        
        st.divider()
        
        # Render control buttons
        start_btn, stop_btn, pause_btn = st.session_state.control_panel.render_controls()
        
        # Handle button clicks
        if start_btn:
            # Apply configuration changes when starting
            if st.session_state.config_changed:
                st.session_state.pipeline_config = current_config.copy()
                st.session_state.config_changed = False
                logging.info(f"Configuration applied: {current_config}")
            handle_start_processing()
        
        if stop_btn:
            handle_stop_processing()
        
        if pause_btn:
            handle_pause_processing()
    
    # Main content area with columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Video Processing")
        
        # Create video display placeholder
        video_placeholder = st.empty()
        
        # Initialize VideoDisplay component if not already done
        if st.session_state.video_display is None:
            st.session_state.video_display = VideoDisplay(video_placeholder)
        else:
            # Update placeholder reference (required for Streamlit's layout system)
            st.session_state.video_display.placeholder = video_placeholder
        
        # Display status based on whether video is uploaded and processing state
        if st.session_state.video_handler is None:
            st.session_state.video_display.display_message(
                "📤 Upload a video file to start processing",
                message_type="info"
            )
        elif not st.session_state.processing and not hasattr(st.session_state, 'output_video_path'):
            # Video uploaded but not yet processed - display first frame
            try:
                cap = st.session_state.video_handler.get_capture()
                ret, first_frame = cap.read()
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame
                
                if ret and first_frame is not None:
                    st.session_state.video_display.update_frame(first_frame)
                else:
                    st.session_state.video_display.display_message(
                        "✅ Video loaded and ready for processing",
                        message_type="success"
                    )
            except Exception as e:
                logging.error(f"Error displaying first frame: {e}")
                st.session_state.video_display.display_message(
                    "✅ Video loaded and ready for processing",
                    message_type="success"
                )
        elif hasattr(st.session_state, 'output_video_path') and st.session_state.output_video_path:
            # Processing complete - display output video
            video_placeholder.video(st.session_state.output_video_path)
            
            # Show download button
            try:
                with open(st.session_state.output_video_path, 'rb') as f:
                    video_bytes = f.read()
                st.download_button(
                    label="📥 Download Processed Video",
                    data=video_bytes,
                    file_name=Path(st.session_state.output_video_path).name,
                    mime="video/mp4"
                )
            except Exception as e:
                logging.error(f"Error creating download button: {e}")
        
        # Progress indicator placeholders (for future use)
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
    
    with col2:
        # Information Panel (Task 23)
        # Initialize info panel if not already done
        if st.session_state.info_panel is None:
            st.session_state.info_panel = InfoPanel(
                log_handler=st.session_state.log_handler
            )
        
        # Render the information panel with tabs
        st.session_state.info_panel.render()
    
    # Background processing loop
    if st.session_state.processing and st.session_state.pipeline_wrapper:
        # Poll for results from processing thread
        result = st.session_state.pipeline_wrapper.get_result(timeout=0.05)
        
        if result is not None:
            if 'error' in result:
                # Error occurred
                logging.error(f"Processing error: {result['error']}")
                st.error(f"❌ Processing error: {result['error']}")
                st.session_state.processing = False
                
            elif 'done' in result and result['done']:
                # Processing complete
                st.session_state.processing = False
                st.session_state.paused = False
                st.session_state.output_video_path = result['output_video']
                status_placeholder.success(f"✅ Processing complete! Processed {result['total_frames']} frames in {result['processing_time']:.1f}s")
                logging.info(f"Video processing completed: {result['output_video']}")
                st.rerun()  # Rerun to display the video
                
            elif 'progress' in result:
                # Progress update
                tracks = result['tracks']
                frame_idx = result['frame_idx']
                total_frames = result['total_frames']
                processing_fps = result['processing_fps']
                
                # Update progress bar
                progress = int((frame_idx / total_frames) * 100)
                progress_placeholder.progress(
                    progress / 100.0,
                    text=f"Processing: Frame {frame_idx}/{total_frames} ({progress}%) | {processing_fps:.1f} FPS"
                )
                
                # Update session state (batch updates)
                if st.session_state.update_batcher.should_update():
                    # Update FPS
                    st.session_state.current_fps = processing_fps
                    
                    # Update detections and recognitions
                    for track_id, track in tracks.items():
                        # Add to active tracks
                        st.session_state.active_tracks[track_id] = {
                            'bbox': track.bbox.tolist() if hasattr(track.bbox, 'tolist') else list(track.bbox) if track.bbox is not None else None,
                            'text': track.text,
                            'confidence': track.ocr_confidence if track.text else track.detection_confidence,
                            'age': track.age,
                            'is_active': track.is_active
                        }
                        
                        # Add to recognitions if text detected
                        if track.text and track.text not in st.session_state.unique_plates:
                            st.session_state.unique_plates[track.text] = {
                                'confidence': track.ocr_confidence,
                                'first_seen': frame_idx
                            }
                            st.session_state.latest_recognitions.append({
                                'text': track.text,
                                'confidence': track.ocr_confidence,
                                'frame': frame_idx
                            })
                    
                    # Limit list sizes to prevent memory issues
                    if len(st.session_state.latest_recognitions) > 100:
                        st.session_state.latest_recognitions = st.session_state.latest_recognitions[-100:]
                
                # Rerun to update progress
                st.rerun()
        else:
            # No result yet, but still processing - rerun after a delay
            time.sleep(0.05)  # Wait 50ms before next poll
            st.rerun()
    
    # Footer
    st.divider()
    st.caption(
        "GTA V ALPR System v0.1.0 | "
        "Powered by YOLOv8 + PaddleOCR + ByteTrack | "
        "Built with Streamlit"
    )


if __name__ == "__main__":
    main()

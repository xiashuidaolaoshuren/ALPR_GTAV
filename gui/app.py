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
from pathlib import Path

# Import GUI components
from gui.utils.video_handler import VideoHandler
from gui.components.video_display import VideoDisplay
from gui.components.control_panel import ControlPanel
from gui.components.info_panel import InfoPanel
from gui.utils.logging_handler import StreamlitLogHandler

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


def cleanup_previous_video():
    """
    Clean up previous video resources when a new video is uploaded.
    """
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
        st.session_state.current_fps = 0.0


def handle_start_processing():
    """Handle the Start Processing button click."""
    logging.info("Starting video processing...")
    st.session_state.processing = True
    # TODO: Task 24 will implement actual pipeline processing


def handle_stop_processing():
    """Handle the Stop Processing button click."""
    logging.info("Stopping video processing...")
    st.session_state.processing = False
    # TODO: Task 24 will implement cleanup/stop logic


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
        st.session_state.control_panel.render_controls()
        
        # Handle button clicks
        # Check if Start button was clicked (processing changed to True)
        if st.session_state.processing and not st.session_state.get('was_processing', False):
            # Apply configuration changes when starting
            if st.session_state.config_changed:
                st.session_state.pipeline_config = current_config.copy()
                st.session_state.config_changed = False
                logging.info(f"Configuration applied: {current_config}")
            handle_start_processing()
            st.session_state.was_processing = True
        # Check if Stop button was clicked (processing changed to False)
        elif not st.session_state.processing and st.session_state.get('was_processing', False):
            handle_stop_processing()
            st.session_state.was_processing = False
    
    # Main content area with columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Video Processing")
        
        # Create video display placeholder
        video_placeholder = st.empty()
        
        # Initialize VideoDisplay component if not already done
        if st.session_state.video_display is None:
            st.session_state.video_display = VideoDisplay(video_placeholder)
        
        # Display status based on whether video is uploaded
        if st.session_state.video_handler is None:
            st.session_state.video_display.display_message(
                "📤 Upload a video file to start processing",
                message_type="info"
            )
        else:
            # Video uploaded - display first frame
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
    
    # Footer
    st.divider()
    st.caption(
        "GTA V ALPR System v0.1.0 | "
        "Powered by YOLOv8 + PaddleOCR + ByteTrack | "
        "Built with Streamlit"
    )


if __name__ == "__main__":
    main()

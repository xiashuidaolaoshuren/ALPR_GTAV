"""
GTA V ALPR System - Streamlit GUI Application

Main entry point for the interactive license plate recognition demonstration interface.
This application provides a user-friendly way to upload videos, configure pipeline
parameters, and visualize real-time ALPR processing results.

Author: Felix (xiashuidaolaoshuren)
Date: 2025-11-12
"""

import streamlit as st
from pathlib import Path

# Import GUI components
from gui.utils.video_handler import VideoHandler
from gui.components.video_display import VideoDisplay

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
    
    # Detection and recognition results
    if 'unique_plates' not in st.session_state:
        st.session_state.unique_plates = set()
    
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
        st.session_state.unique_plates = set()
        st.session_state.latest_recognitions = []
        st.session_state.all_detections = []


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
        
        # Configuration section (placeholder for Task 22)
        st.header("⚙️ Configuration")
        st.info("Pipeline configuration will be available in Task 22.")
    
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
            # Video uploaded, ready for processing (Task 22 will add controls)
            st.session_state.video_display.display_message(
                "✅ Video loaded and ready for processing\n\n"
                "Control panel will be available in Task 22 (Build Interactive Control Panel)",
                message_type="success"
            )
        
        # Progress indicator placeholders (for future use)
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
    
    with col2:
        st.subheader("📊 Information Panel")
        st.info(
            "**Information Panel** will be implemented in Task 23.\n\n"
            "This panel will display:\n"
            "- Processing statistics\n"
            "- Latest plate recognitions\n"
            "- Active tracking information\n"
            "- System logs"
        )
    
    # Footer
    st.divider()
    st.caption(
        "GTA V ALPR System v0.1.0 | "
        "Powered by YOLOv8 + PaddleOCR + ByteTrack | "
        "Built with Streamlit"
    )


if __name__ == "__main__":
    main()

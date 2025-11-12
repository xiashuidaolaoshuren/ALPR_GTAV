"""
GTA V ALPR System - Streamlit GUI Application

Main entry point for the interactive license plate recognition demonstration interface.
This application provides a user-friendly way to upload videos, configure pipeline
parameters, and visualize real-time ALPR processing results.

Author: Felix (xiashuidaolaoshuren)
Date: 2025-11-12
"""

import streamlit as st

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
    
    # Video handling
    if 'video_handler' not in st.session_state:
        st.session_state.video_handler = None
    
    if 'video_info' not in st.session_state:
        st.session_state.video_info = None
    
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
    
    # Sidebar placeholder
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info("GUI components will be loaded in subsequent tasks.")
        st.markdown("**Current Status:** Environment Setup Complete ✓")
    
    # Main content area with columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Video Processing")
        st.info(
            "**Video Display Component** will be implemented in Task 21.\n\n"
            "This area will show the processed video frames with bounding boxes "
            "and recognized license plate text overlaid in real-time."
        )
        
        # Placeholder for video display
        st.empty()
    
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

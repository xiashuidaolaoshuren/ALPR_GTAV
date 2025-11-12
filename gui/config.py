"""
GUI Configuration

This module contains configuration constants for the Streamlit GUI application.

TODO: Expand configuration as needed during implementation of Tasks 20-24
"""

# Page Configuration
PAGE_TITLE = "🚗 GTA V License Plate Recognition System"
PAGE_ICON = "🚗"
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"

# Default Pipeline Configuration
DEFAULT_CONFIG = {
    'confidence_threshold': 0.25,
    'iou_threshold': 0.7,
    'ocr_interval': 30,
    'device': 'cuda',
    'tracker_type': 'bytetrack'
}

# Video Processing
MAX_VIDEO_SIZE_MB = 500  # Maximum upload size in MB
SUPPORTED_FORMATS = ['mp4', 'avi', 'mov', 'mkv']

# Display Settings
FRAME_WIDTH = 640  # Display width for video frames
FPS_UPDATE_INTERVAL = 0.5  # Seconds between FPS updates

# Logging Configuration
MAX_LOG_ENTRIES = 100  # Maximum log entries to display
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Status Panel Configuration
STATUS_UPDATE_INTERVAL = 1  # Seconds between status updates
MAX_RECENT_PLATES = 5  # Number of recent plates to display

# Colors (for visualizations)
DETECTION_COLOR = (0, 255, 0)  # Green for detection boxes
TEXT_COLOR = (255, 255, 255)  # White for text
BACKGROUND_COLOR = (0, 0, 0)  # Black for text background

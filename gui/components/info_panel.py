"""
Information Panel Component

This module provides the information display panel showing processing status,
detected plates, and system logs.

TODO: Implementation scheduled for Task 23 (Develop Information Panel)
"""

import streamlit as st


class InfoPanel:
    """
    Component for displaying processing information and logs.
    
    This class will provide:
    - Status tab: detected count, latest recognitions, active tracks
    - Log tab: real-time system logs
    - Performance metrics display
    
    Attributes:
        detected_plates: List of all detected plates
    """
    
    def __init__(self):
        """Initialize the information panel component."""
        self.detected_plates = []
    
    def render_status_tab(self):
        """
        Render the status information tab.
        
        Displays:
        - Total detected plates
        - Unique plates recognized
        - Current FPS
        - Latest recognitions list
        - Active tracks summary
        
        TODO: Implement in Task 23
        """
        # Placeholder implementation
        st.info("Status panel will be implemented in Task 23")
    
    def render_log_tab(self, log_handler):
        """
        Render the system logs tab.
        
        Args:
            log_handler: Custom logging handler for Streamlit
        
        TODO: Implement in Task 23
        """
        # Placeholder implementation
        st.info("Log panel will be implemented in Task 23")

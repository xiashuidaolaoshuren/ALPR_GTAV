"""
Information Panel Component

Displays status information, metrics, recognitions, and logs using tabs.

Author: Felix (xiashuidaolaoshuren)
Date: 2025-11-13
"""

import streamlit as st
from typing import Dict, List, Optional, Any
from gui.utils.logging_handler import StreamlitLogHandler


class InfoPanel:
    """
    Information panel component with tabbed interface.
    
    Provides two main tabs:
    1. Status Tab: Shows metrics, latest recognitions, and active tracks
    2. Logs Tab: Displays application logs with clear functionality
    
    Features:
    - Real-time status updates
    - Latest plate recognitions
    - Active track monitoring
    - Log viewing with clear button
    """
    
    def __init__(self, log_handler: Optional[StreamlitLogHandler] = None):
        """
        Initialize the information panel.
        
        Args:
            log_handler: StreamlitLogHandler instance for log display
        """
        self.log_handler = log_handler
    
    def render(self):
        """
        Render the complete information panel with tabs.
        """
        st.markdown("### 📊 Information Panel")
        
        # Create tabs - Status first, Logs second (per user preference)
        tab1, tab2 = st.tabs(["📈 Status", "📝 Logs"])
        
        with tab1:
            self.render_status_tab()
        
        with tab2:
            self.render_log_tab()
    
    def render_status_tab(self):
        """
        Render the status tab with simplified metrics and unique plates.
        """
        # Processing metrics - simplified
        st.markdown("#### Processing Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            total_detections = st.session_state.get('total_detections', 0)
            st.metric("Total Detections", total_detections)
        
        with col2:
            total_recognitions = st.session_state.get('total_recognitions', 0)
            st.metric("Total Recognitions", total_recognitions)
        
        st.divider()
        
        # All Unique Plates with details
        st.markdown("#### 🚗 All Unique Plates")
        unique_plates = st.session_state.get('unique_plates', {})
        
        if unique_plates:
            # Sort by first seen frame (newest first)
            sorted_plates = sorted(
                unique_plates.items(),
                key=lambda x: x[1].get('first_seen', 0),
                reverse=True
            )
            
            # Display each unique plate with number, confidence, and frame
            for idx, (plate_text, plate_data) in enumerate(sorted_plates, 1):
                confidence = plate_data.get('confidence', 0)
                frame_num = plate_data.get('first_seen', 'N/A')
                
                st.text(f"{idx}. {plate_text} | Conf: {confidence:.2%} | Frame: {frame_num}")
        else:
            st.info("No plates recognized yet. Start processing to detect plates.")
    
    def render_log_tab(self):
        """
        Render the logs tab with log display and clear button.
        
        Logs stay at the top (newest first) per user preference.
        """
        st.markdown("#### Application Logs")
        
        # Log controls
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if st.button("🗑️ Clear Logs", use_container_width=True):
                if self.log_handler:
                    self.log_handler.clear_logs()
                    st.rerun()
        
        # Display logs
        if self.log_handler:
            log_count = self.log_handler.get_log_count()
            
            with col1:
                st.caption(f"Showing {log_count} log entries (max 100)")
            
            logs = self.log_handler.get_logs()
            
            # Use text_area for log display (stays at top, newest first, copyable)
            st.text_area(
                "Logs",
                value=logs,
                height=400,
                disabled=False,  # Enabled so users can copy logs
                label_visibility="collapsed",
                key="log_display_area"
            )
        else:
            st.warning("Log handler not initialized.")
            st.text_area(
                "Logs",
                value="Log handler not available.",
                height=400,
                disabled=False,
                label_visibility="collapsed",
                key="log_display_placeholder"
            )


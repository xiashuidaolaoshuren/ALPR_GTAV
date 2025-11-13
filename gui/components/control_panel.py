"""
Control Panel Component

Interactive control panel for configuring ALPR pipeline parameters
and controlling video processing execution.

Author: Felix (xiashuidaolaoshuren)
Date: 2025-11-13
"""

import streamlit as st
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class ControlPanel:
    """
    Control panel component for ALPR pipeline configuration and control.
    
    Provides:
    - Configuration sliders for detection and OCR parameters
    - Device selection (CPU/GPU)
    - Start/Stop buttons for processing control
    - Save configuration button with change detection
    """
    
    def __init__(self):
        """Initialize control panel with default configuration."""
        self.config: Dict = {}
        self._initialize_default_config()
    
    def _initialize_default_config(self):
        """Set default configuration values matching pipeline_config.yaml."""
        # Initialize or reset pipeline_config if it's None or doesn't exist
        if 'pipeline_config' not in st.session_state or st.session_state.pipeline_config is None:
            st.session_state.pipeline_config = {
                'conf_threshold': 0.25,
                'iou_threshold': 0.45,
                'ocr_interval': 30,
                'min_ocr_conf': 0.3,
                'device': 'cpu'  # Will be updated based on availability
            }
        
        if 'config_changed' not in st.session_state:
            st.session_state.config_changed = False
    
    def _check_cuda_availability(self) -> bool:
        """
        Check if CUDA is available for GPU acceleration.
        
        Returns:
            bool: True if CUDA is available, False otherwise
        """
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            logger.warning("PyTorch not installed, CUDA not available")
            return False
    
    def _config_has_changed(self, new_config: Dict) -> bool:
        """
        Check if configuration has changed from saved values.
        
        Args:
            new_config: New configuration dictionary
            
        Returns:
            bool: True if configuration changed
        """
        saved_config = st.session_state.pipeline_config
        
        for key in new_config:
            if new_config[key] != saved_config.get(key):
                return True
        
        return False
    
    def render_configuration(self) -> Dict:
        """
        Render configuration controls in sidebar.
        
        Returns:
            Dict: Current configuration values
        """
        st.sidebar.header("⚙️ Pipeline Configuration")
        
        # Detection settings section
        st.sidebar.subheader("🔍 Detection Settings")
        
        conf_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.pipeline_config['conf_threshold'],
            step=0.05,
            help="Minimum confidence score for plate detection (0.0-1.0). "
                 "Lower values detect more plates but may increase false positives.",
            key="conf_slider"
        )
        
        iou_threshold = st.sidebar.slider(
            "IOU Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.pipeline_config['iou_threshold'],
            step=0.05,
            help="Intersection over Union threshold for Non-Maximum Suppression. "
                 "Higher values allow more overlapping detections.",
            key="iou_slider"
        )
        
        # OCR settings section
        st.sidebar.subheader("📝 OCR Settings")
        
        ocr_interval = st.sidebar.number_input(
            "OCR Interval (frames)",
            min_value=1,
            max_value=120,
            value=st.session_state.pipeline_config['ocr_interval'],
            step=5,
            help="Run OCR every N frames to optimize performance. "
                 "Higher values = faster processing, lower accuracy.",
            key="ocr_interval_input"
        )
        
        min_ocr_conf = st.sidebar.slider(
            "Min OCR Confidence",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.pipeline_config['min_ocr_conf'],
            step=0.05,
            help="Minimum confidence for accepting OCR results. "
                 "Higher values reject more uncertain recognitions.",
            key="ocr_conf_slider"
        )
        
        # System settings section
        st.sidebar.subheader("💻 System Settings")
        
        # Check CUDA availability
        cuda_available = self._check_cuda_availability()
        device_options = ['cuda', 'cpu'] if cuda_available else ['cpu']
        device_index = device_options.index(st.session_state.pipeline_config['device']) \
                      if st.session_state.pipeline_config['device'] in device_options else 0
        
        device = st.sidebar.selectbox(
            "Computation Device",
            options=device_options,
            index=device_index,
            help="Select GPU (cuda) for faster processing or CPU for compatibility. "
                 f"CUDA {'is' if cuda_available else 'is NOT'} available on this system.",
            key="device_select"
        )
        
        if not cuda_available and device == 'cuda':
            st.sidebar.warning("⚠️ CUDA not available, using CPU instead")
            device = 'cpu'
        
        # Build current configuration
        current_config = {
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold,
            'ocr_interval': ocr_interval,
            'min_ocr_conf': min_ocr_conf,
            'device': device
        }
        
        # Check if configuration changed
        config_changed = self._config_has_changed(current_config)
        st.session_state.config_changed = config_changed
        
        # Show change indicator
        if config_changed:
            st.sidebar.info("ℹ️ Configuration changed. Changes will apply when you start processing.")
        
        return current_config
    
    def render_controls(self) -> Tuple[bool, bool, bool]:
        """
        Render Start/Stop/Pause control buttons with Reset functionality.
        
        Returns:
            Tuple[bool, bool, bool]: (start_pressed, stop_pressed, pause_pressed)
        """
        st.sidebar.divider()
        st.sidebar.header("🎮 Processing Controls")
        
        # Check if video is uploaded
        video_uploaded = st.session_state.video_handler is not None
        processing = st.session_state.processing
        
        # Start button only
        start_disabled = processing or not video_uploaded
        start_btn = st.sidebar.button(
            "▶️ Start Processing",
            use_container_width=True,
            disabled=start_disabled,
            help="Start video processing with current configuration" if video_uploaded 
                 else "⚠️ Please upload a video first",
            key="start_btn"
        )
        
        # Placeholder for removed buttons
        stop_btn = False
        pause_btn = False
        
        # Reset button - Row 2
        reset_btn = st.sidebar.button(
            "🔄 Reset Config",
            use_container_width=True,
            help="Reset configuration to default values",
            key="reset_config_btn"
        )
        
        # Handle reset button
        if reset_btn:
            cuda_available = self._check_cuda_availability()
            st.session_state.pipeline_config = {
                'conf_threshold': 0.25,
                'iou_threshold': 0.45,
                'ocr_interval': 30,
                'min_ocr_conf': 0.3,
                'device': 'cuda' if cuda_available else 'cpu'
            }
            st.session_state.config_changed = False
            st.sidebar.success("✅ Configuration reset to defaults!")
            logger.info("Configuration reset to defaults")
            st.rerun()
        
        # Status indicator
        if processing:
            st.sidebar.success("🟢 Processing active")
        elif video_uploaded:
            st.sidebar.info("⚪ Ready to process")
        else:
            st.sidebar.warning("🟡 No video uploaded")
        
        # Processing progress bar (shown only during processing)
        if processing:
            st.sidebar.divider()
            st.sidebar.markdown("#### 📊 Processing Status")
            
            current_frame = st.session_state.get('current_frame', 0)
            total_frames = st.session_state.get('total_frames', 1)
            
            if total_frames > 0:
                progress_pct = (current_frame / total_frames) * 100
                st.sidebar.progress(progress_pct / 100.0, text=f"Frame {current_frame}/{total_frames} ({progress_pct:.1f}%)")
            else:
                st.sidebar.info("Initializing...")
        
        return start_btn, stop_btn, pause_btn
    
    def get_pipeline_config_dict(self) -> Dict:
        """
        Get configuration in pipeline-compatible format.
        
        Returns:
            Dict: Configuration structured for pipeline consumption
        """
        config = st.session_state.pipeline_config
        
        return {
            'detection': {
                'confidence_threshold': config['conf_threshold'],
                'iou_threshold': config['iou_threshold'],
                'device': config['device']
            },
            'recognition': {
                'min_conf': config['min_ocr_conf']
            },
            'tracking': {
                'ocr_interval': config['ocr_interval']
            },
            'device': config['device']
        }

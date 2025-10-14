"""
Recognition Configuration Module

Handles loading and validation of OCR-specific configuration parameters.
Loads settings from pipeline_config.yaml recognition section.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

logger = logging.getLogger(__name__)


class RecognitionConfig:
    """
    Configuration class for OCR recognition module.
    
    Loads and validates parameters from pipeline_config.yaml recognition section.
    Provides default values and parameter validation.
    
    Attributes:
        use_gpu (bool): Enable GPU acceleration for PaddleOCR.
        use_angle_cls (bool): Enable text angle classification.
        lang (str): Language model ('en' for English).
        show_log (bool): Display PaddleOCR verbose logs.
        use_rec (bool): Enable text recognition.
        regex (str): Regex pattern for plate validation.
        min_conf (float): Minimum confidence threshold.
        prefer_largest_box (bool): Prefer text from largest bounding box.
    
    Example:
        >>> config = RecognitionConfig.from_yaml('configs/pipeline_config.yaml')
        >>> print(f"GPU enabled: {config.use_gpu}")
        >>> config_dict = config.to_dict()
    
    Note:
        - Provides sensible defaults if config file missing parameters
        - Validates parameter types and ranges
        - Used by load_ocr_model() and recognize_text()
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration with dict or defaults.
        
        Args:
            config_dict: Configuration dictionary from YAML, or None for defaults.
        
        Raises:
            ValueError: If config_dict contains invalid parameters.
        """
        pass
    
    @classmethod
    def from_yaml(cls, yaml_path: str, section: str = 'recognition') -> 'RecognitionConfig':
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to pipeline_config.yaml file.
            section: Section name to load (default: 'recognition').
        
        Returns:
            RecognitionConfig: Loaded configuration instance.
        
        Raises:
            FileNotFoundError: If yaml_path does not exist.
            yaml.YAMLError: If YAML file is malformed.
            KeyError: If section not found in YAML.
        
        Example:
            >>> config = RecognitionConfig.from_yaml('configs/pipeline_config.yaml')
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            dict: Configuration as dictionary, compatible with load_ocr_model().
        
        Example:
            >>> config = RecognitionConfig()
            >>> config_dict = config.to_dict()
        """
        pass
    
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            bool: True if configuration is valid.
        
        Raises:
            ValueError: If any parameter is invalid (type, range, etc.).
        
        Note:
            - Checks parameter types match expected
            - Validates confidence thresholds in [0, 1]
            - Validates regex pattern is compilable
        """
        pass

"""
Configuration Loading Utility

Provides ConfigLoader class for loading and validating YAML configuration files.
"""
import os
import yaml
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Utility class for loading and validating YAML configuration files.
    
    This class provides methods to load YAML files, validate their structure,
    and ensure all required keys are present according to a schema.
    """
    
    @staticmethod
    def load_yaml(filepath: str) -> Dict[str, Any]:
        """
        Load a YAML configuration file.
        
        Args:
            filepath: Path to the YAML configuration file
            
        Returns:
            Dictionary containing the configuration data
            
        Raises:
            FileNotFoundError: If the configuration file does not exist
            yaml.YAMLError: If the file cannot be parsed as valid YAML
            ValueError: If the file is empty or invalid
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            if config is None:
                raise ValueError(f"Configuration file is empty: {filepath}")
            
            if not isinstance(config, dict):
                raise ValueError(f"Configuration file must contain a dictionary: {filepath}")
            
            logger.info(f"Successfully loaded configuration from: {filepath}")
            return config
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse YAML file {filepath}: {e}")
    
    @staticmethod
    def validate_config(config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """
        Validate configuration against an expected schema.
        
        Args:
            config: Configuration dictionary to validate
            schema: Schema dictionary defining required keys and their types
                   Format: {'key_name': type, ...}
                   
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If validation fails (missing keys or wrong types)
        """
        missing_keys = []
        type_errors = []
        
        for key, expected_type in schema.items():
            # Check if required key exists
            if key not in config:
                missing_keys.append(key)
                continue
            
            # Check if value type matches expected type
            if expected_type is not None and not isinstance(config[key], expected_type):
                type_errors.append(
                    f"Key '{key}' expected type {expected_type.__name__}, "
                    f"got {type(config[key]).__name__}"
                )
        
        # Raise error if validation fails
        if missing_keys or type_errors:
            error_msg = []
            if missing_keys:
                error_msg.append(f"Missing required keys: {', '.join(missing_keys)}")
            if type_errors:
                error_msg.append(f"Type errors: {'; '.join(type_errors)}")
            raise ValueError("Configuration validation failed:\n" + "\n".join(error_msg))
        
        logger.info("Configuration validation passed")
        return True
    
    @staticmethod
    def load_and_validate(filepath: str, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load and optionally validate a YAML configuration file.
        
        Args:
            filepath: Path to the YAML configuration file
            schema: Optional schema for validation
            
        Returns:
            Dictionary containing the validated configuration data
        """
        config = ConfigLoader.load_yaml(filepath)
        
        if schema is not None:
            ConfigLoader.validate_config(config, schema)
        
        return config

"""Configuration validation utility for ALPR pipeline.

This module provides validation functions for YAML configuration files,
ensuring all required fields are present with correct types and valid values.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    pass


class ConfigValidator:
    """Validates ALPR pipeline configuration files.

    Checks required fields, data types, value ranges, and file paths.
    Provides clear error messages for debugging configuration issues.
    """

    # Required fields with their expected types
    REQUIRED_FIELDS = {
        "detection": {
            "model_path": str,
            "confidence_threshold": (int, float),
            "iou_threshold": (int, float),
            "img_size": int,
            "device": str,
            "max_det": int,
        },
        "recognition": {
            "use_gpu": bool,
            "use_textline_orientation": bool,
            "lang": str,
            "rec_threshold": (int, float),
            "show_log": bool,
            "use_rec": bool,
            "regex": str,
            "prefer_largest_box": bool,
            "mask_top_ratio": (int, float),
            "min_conf": (int, float),
        },
        "tracking": {
            "tracker_type": str,
            "max_age": int,
            "min_hits": int,
            "iou_threshold": (int, float),
            "ocr_interval": int,
            "ocr_confidence_threshold": (int, float),
        },
        "preprocessing": {
            "enable_enhancement": bool,
            "min_width": int,
            "use_clahe": bool,
            "clahe_clip_limit": (int, float),
            "clahe_tile_grid_size": list,
            "use_gaussian_blur": bool,
            "gaussian_kernel_size": list,
            "use_sharpening": bool,
            "sharpen_strength": (int, float),
        },
        "pipeline": {
            "enable_full_pipeline": bool,
            "save_intermediate": bool,
            "output_dir": str,
            "batch_size": int,
            "enable_logging": bool,
            "log_level": str,
            "log_file": str,
        },
    }

    # Valid value ranges for numeric parameters
    VALUE_RANGES = {
        "detection.confidence_threshold": (0.0, 1.0),
        "detection.iou_threshold": (0.0, 1.0),
        "detection.img_size": (320, 1920),
        "detection.max_det": (1, 300),
        "recognition.rec_threshold": (0.0, 1.0),
        "recognition.mask_top_ratio": (0.0, 0.5),
        "recognition.min_conf": (0.0, 1.0),
        "tracking.max_age": (1, 300),
        "tracking.min_hits": (1, 10),
        "tracking.iou_threshold": (0.0, 1.0),
        "tracking.ocr_interval": (1, 300),
        "tracking.ocr_confidence_threshold": (0.0, 1.0),
        "preprocessing.min_width": (50, 1000),
        "preprocessing.clahe_clip_limit": (0.1, 10.0),
        "preprocessing.sharpen_strength": (0.0, 3.0),
        "pipeline.batch_size": (1, 128),
    }

    # Valid string enumerations
    VALID_ENUMS = {
        "detection.device": ["cuda", "cpu"],
        "recognition.lang": ["en", "ch", "fr", "german", "korean", "japan"],
        "tracking.tracker_type": ["bytetrack", "botsort", "iou"],
        "pipeline.log_level": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    }

    def __init__(self, config_path: Path):
        """Initialize validator with configuration file path.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.errors: List[str] = []

    def load_config(self) -> bool:
        """Load YAML configuration file.

        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
            return True
        except FileNotFoundError:
            self.errors.append(f"Configuration file not found: {self.config_path}")
            return False
        except yaml.YAMLError as e:
            self.errors.append(f"YAML parsing error: {e}")
            return False
        except Exception as e:
            self.errors.append(f"Unexpected error loading config: {e}")
            return False

    def validate_required_fields(self) -> bool:
        """Check that all required fields are present with correct types.

        Returns:
            True if all required fields are valid
        """
        valid = True

        for section, fields in self.REQUIRED_FIELDS.items():
            if section not in self.config:
                self.errors.append(f"Missing required section: {section}")
                valid = False
                continue

            for field, expected_type in fields.items():
                if field not in self.config[section]:
                    self.errors.append(f"Missing required field: {section}.{field}")
                    valid = False
                    continue

                value = self.config[section][field]
                if not isinstance(value, expected_type):
                    self.errors.append(
                        f"Invalid type for {section}.{field}: "
                        f"expected {expected_type}, got {type(value).__name__}"
                    )
                    valid = False

        return valid

    def validate_value_ranges(self) -> bool:
        """Check that numeric values are within valid ranges.

        Returns:
            True if all values are within valid ranges
        """
        valid = True

        for field_path, (min_val, max_val) in self.VALUE_RANGES.items():
            section, field = field_path.split(".")
            if section not in self.config or field not in self.config[section]:
                continue

            value = self.config[section][field]
            if not isinstance(value, (int, float)):
                continue

            if not (min_val <= value <= max_val):
                self.errors.append(
                    f"Value out of range for {field_path}: "
                    f"{value} not in [{min_val}, {max_val}]"
                )
                valid = False

        return valid

    def validate_enumerations(self) -> bool:
        """Check that string values match valid enumerations.

        Returns:
            True if all enum values are valid
        """
        valid = True

        for field_path, valid_values in self.VALID_ENUMS.items():
            section, field = field_path.split(".")
            if section not in self.config or field not in self.config[section]:
                continue

            value = self.config[section][field]
            if value not in valid_values:
                self.errors.append(
                    f"Invalid value for {field_path}: "
                    f"'{value}' not in {valid_values}"
                )
                valid = False

        return valid

    def validate_file_paths(self) -> bool:
        """Check that referenced file paths exist.

        Returns:
            True if all file paths are valid
        """
        valid = True

        # Check detection model path
        if "detection" in self.config and "model_path" in self.config["detection"]:
            model_path = Path(self.config["detection"]["model_path"])
            if not model_path.is_absolute():
                # Get project root (script is in scripts/utils/, so go up 2 levels)
                script_dir = Path(__file__).resolve().parent
                project_root = script_dir.parent.parent
                model_path = project_root / model_path

            if not model_path.exists():
                self.errors.append(
                    f"Detection model not found: {self.config['detection']['model_path']}"
                )
                valid = False

        return valid

    def validate_special_fields(self) -> bool:
        """Validate special fields with custom logic.

        Returns:
            True if all special validations pass
        """
        valid = True

        # Validate CLAHE tile grid size
        if "preprocessing" in self.config:
            tile_size = self.config["preprocessing"].get("clahe_tile_grid_size")
            if tile_size and isinstance(tile_size, list):
                if len(tile_size) != 2:
                    self.errors.append(
                        "clahe_tile_grid_size must have exactly 2 elements [width, height]"
                    )
                    valid = False
                elif not all(isinstance(x, int) and x > 0 for x in tile_size):
                    self.errors.append(
                        "clahe_tile_grid_size elements must be positive integers"
                    )
                    valid = False

            # Validate Gaussian kernel size
            kernel_size = self.config["preprocessing"].get("gaussian_kernel_size")
            if kernel_size and isinstance(kernel_size, list):
                if len(kernel_size) != 2:
                    self.errors.append(
                        "gaussian_kernel_size must have exactly 2 elements [width, height]"
                    )
                    valid = False
                elif not all(isinstance(x, int) and x > 0 and x % 2 == 1 for x in kernel_size):
                    self.errors.append(
                        "gaussian_kernel_size elements must be positive odd integers"
                    )
                    valid = False

        return valid

    def validate(self) -> bool:
        """Run all validation checks.

        Returns:
            True if configuration is valid, False otherwise
        """
        if not self.load_config():
            return False

        validations = [
            self.validate_required_fields(),
            self.validate_value_ranges(),
            self.validate_enumerations(),
            self.validate_file_paths(),
            self.validate_special_fields(),
        ]

        return all(validations)

    def get_errors(self) -> List[str]:
        """Get list of validation error messages.

        Returns:
            List of error messages
        """
        return self.errors

    def print_errors(self) -> None:
        """Print validation errors to stderr."""
        if self.errors:
            print("Configuration Validation Errors:", file=sys.stderr)
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}", file=sys.stderr)


def validate_config(config_path: Path) -> bool:
    """Validate a configuration file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        True if valid, False otherwise

    Raises:
        ConfigValidationError: If configuration is invalid
    """
    validator = ConfigValidator(config_path)

    if not validator.validate():
        validator.print_errors()
        raise ConfigValidationError(
            f"Configuration validation failed with {len(validator.get_errors())} errors"
        )

    return True


def main():
    """Command-line interface for config validation."""
    if len(sys.argv) < 2:
        print("Usage: python validate_config.py <config_file.yaml>")
        sys.exit(1)

    config_path = Path(sys.argv[1])

    print(f"Validating configuration: {config_path}")

    try:
        if validate_config(config_path):
            print("✓ Configuration is valid!")
            sys.exit(0)
    except ConfigValidationError as e:
        print(f"\n✗ {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

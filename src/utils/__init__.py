"""
Utility Functions Module

Provides common utilities for configuration loading, logging, file operations, and video I/O.
"""

import logging

# Setup module logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Import utility modules
from . import video_io

__all__ = ["video_io"]

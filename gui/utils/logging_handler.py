"""
Logging Handler Utility

Custom logging handler for displaying logs in Streamlit interface.
Uses deque for efficient log buffer management.

Author: Felix (xiashuidaolaoshuren)
Date: 2025-11-13
"""

import logging
from collections import deque
from datetime import datetime
from typing import List


class StreamlitLogHandler(logging.Handler):
    """
    Custom logging handler for Streamlit display.
    
    This handler captures log messages and stores them in an efficient
    deque buffer that can be displayed in the Streamlit UI.
    
    Features:
    - Automatic log rotation (keeps last N logs)
    - Thread-safe log collection
    - Formatted output with timestamps and levels
    """
    
    def __init__(self, max_logs: int = 100):
        """
        Initialize the logging handler.
        
        Args:
            max_logs: Maximum number of log entries to keep (default: 100)
        """
        super().__init__()
        # Use deque for efficient FIFO operations
        self.logs = deque(maxlen=max_logs)
        self.max_logs = max_logs
        
        # Set handler level to DEBUG to capture all messages
        self.setLevel(logging.DEBUG)
        
        # Set default formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        self.setFormatter(formatter)
    
    def emit(self, record: logging.LogRecord):
        """
        Process a log record and add to buffer.
        
        Args:
            record: LogRecord object to process
        """
        try:
            # Format the log message
            msg = self.format(record)
            
            # Add to deque (automatically removes oldest if at maxlen)
            self.logs.append(msg)
            
        except Exception:
            # Don't let logging errors crash the application
            self.handleError(record)
    
    def get_logs(self) -> str:
        """
        Get all stored log messages as a single string.
        
        Returns:
            str: Formatted log messages joined with newlines (newest at bottom)
        """
        if not self.logs:
            return "No logs available yet."
        
        # Return logs with newest at top (reversed order)
        return "\n".join(self.logs)
    
    def get_logs_list(self) -> List[str]:
        """
        Get log messages as a list.
        
        Returns:
            List[str]: List of log messages (newest at top)
        """
        return list(self.logs)
    
    def clear_logs(self):
        """Clear all stored log messages."""
        self.logs.clear()
    
    def get_log_count(self) -> int:
        """
        Get the number of logs currently stored.
        
        Returns:
            int: Number of log entries
        """
        return len(self.logs)

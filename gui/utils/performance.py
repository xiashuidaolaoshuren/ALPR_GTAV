"""
Performance Monitoring and Optimization Utilities

Provides tools for GUI performance optimization including FPS tracking,
frame skipping, and memory management.

Author: Felix (xiashuidaolaoshuren)
Date: 2025-11-13
"""

import time
from collections import deque
from typing import Optional


class PerformanceMonitor:
    """
    Monitor and track processing performance with rolling window.
    
    Tracks frame processing times and calculates real-time FPS
    using a sliding window approach for smooth metrics.
    
    Attributes:
        window_size: Number of frames to average over
        frame_times: Deque of recent frame processing times
        last_update: Timestamp of last update
    """
    
    def __init__(self, window_size: int = 30):
        """
        Initialize performance monitor.
        
        Args:
            window_size: Number of frames for rolling average (default: 30)
        """
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.last_update = time.time()
        self.frame_count = 0
    
    def record_frame(self, start_time: Optional[float] = None):
        """
        Record a processed frame.
        
        Args:
            start_time: Frame processing start time. If None, uses time since last record.
        """
        current_time = time.time()
        
        if start_time is not None:
            frame_time = current_time - start_time
        else:
            frame_time = current_time - self.last_update
        
        self.frame_times.append(frame_time)
        self.last_update = current_time
        self.frame_count += 1
    
    def get_fps(self) -> float:
        """
        Get current FPS based on rolling average.
        
        Returns:
            float: Current FPS, 0 if no frames recorded
        """
        if not self.frame_times:
            return 0.0
        
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def get_avg_frame_time(self) -> float:
        """
        Get average frame processing time in milliseconds.
        
        Returns:
            float: Average time in ms, 0 if no frames recorded
        """
        if not self.frame_times:
            return 0.0
        
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return avg_time * 1000  # Convert to ms
    
    def reset(self):
        """Reset all statistics."""
        self.frame_times.clear()
        self.last_update = time.time()
        self.frame_count = 0


class FrameSkipper:
    """
    Optimize display performance by skipping frames.
    
    Calculates which frames to display based on target FPS
    to reduce CPU usage while maintaining smooth visualization.
    
    Attributes:
        target_fps: Target display FPS
        source_fps: Source video FPS
        frame_skip: Calculated skip interval
        frame_counter: Current frame number
    """
    
    def __init__(self, target_fps: int = 15, source_fps: Optional[float] = None):
        """
        Initialize frame skipper.
        
        Args:
            target_fps: Target display FPS (default: 15)
            source_fps: Source video FPS (optional, for optimization)
        """
        self.target_fps = target_fps
        self.source_fps = source_fps
        self.frame_skip = 1
        self.frame_counter = 0
        
        if source_fps and source_fps > target_fps:
            self.frame_skip = max(1, int(source_fps / target_fps))
    
    def should_display_frame(self, frame_number: Optional[int] = None) -> bool:
        """
        Determine if frame should be displayed.
        
        Args:
            frame_number: Specific frame number to check. If None, uses internal counter.
        
        Returns:
            bool: True if frame should be displayed
        """
        if frame_number is not None:
            return frame_number % self.frame_skip == 0
        
        self.frame_counter += 1
        return self.frame_counter % self.frame_skip == 0
    
    def update_source_fps(self, fps: float):
        """
        Update source FPS and recalculate skip interval.
        
        Args:
            fps: New source FPS value
        """
        self.source_fps = fps
        if fps > self.target_fps:
            self.frame_skip = max(1, int(fps / self.target_fps))
        else:
            self.frame_skip = 1
    
    def reset(self):
        """Reset frame counter."""
        self.frame_counter = 0


class UpdateBatcher:
    """
    Batch UI updates to reduce rerun frequency.
    
    Controls how often the UI should be updated based on frame count
    or time elapsed, improving responsiveness while reducing overhead.
    
    Attributes:
        update_interval: Frames between updates
        frame_counter: Current frame count
        last_update_time: Timestamp of last update
    """
    
    def __init__(self, update_interval: int = 30):
        """
        Initialize update batcher.
        
        Args:
            update_interval: Number of frames between UI updates (default: 30)
        """
        self.update_interval = update_interval
        self.frame_counter = 0
        self.last_update_time = time.time()
    
    def should_update(self, force_time_based: bool = False, time_threshold: float = 1.0) -> bool:
        """
        Determine if UI should be updated.
        
        Args:
            force_time_based: Also consider time-based updates
            time_threshold: Time threshold in seconds for forced update
        
        Returns:
            bool: True if UI should update
        """
        self.frame_counter += 1
        
        # Check frame-based update
        if self.frame_counter >= self.update_interval:
            self.frame_counter = 0
            self.last_update_time = time.time()
            return True
        
        # Check time-based update (optional)
        if force_time_based:
            elapsed = time.time() - self.last_update_time
            if elapsed >= time_threshold:
                self.last_update_time = time.time()
                return True
        
        return False
    
    def force_update(self):
        """Force an immediate update."""
        self.frame_counter = 0
        self.last_update_time = time.time()
    
    def reset(self):
        """Reset counters."""
        self.frame_counter = 0
        self.last_update_time = time.time()

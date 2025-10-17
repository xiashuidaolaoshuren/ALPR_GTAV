"""
License Plate Tracking Module

Maintains plate identity across video frames using ByteTrack or IOU tracking.
"""

from .tracker import PlateTrack
from .utils import (
    cleanup_lost_tracks,
    get_track_summary,
    get_tracks_needing_ocr
)

__all__ = [
    'PlateTrack',
    'cleanup_lost_tracks',
    'get_track_summary',
    'get_tracks_needing_ocr'
]

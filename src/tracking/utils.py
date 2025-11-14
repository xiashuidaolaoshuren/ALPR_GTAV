"""
Utility functions for managing plate tracks.

This module provides helper functions for track lifecycle management,
cleanup, and statistics generation.
"""

from typing import Dict, List
from .tracker import PlateTrack


def cleanup_lost_tracks(tracks: Dict[int, PlateTrack], max_age: int) -> Dict[int, PlateTrack]:
    """
    Remove inactive tracks from the tracking dictionary.

    This function filters out tracks that have been marked as lost
    to prevent memory buildup and maintain a clean active track set.

    Args:
        tracks: Dictionary of track_id -> PlateTrack objects
        max_age: Maximum number of frames a track can be lost before removal
                 (currently unused, but reserved for future lost track grace period)

    Returns:
        Dict[int, PlateTrack]: Dictionary containing only active tracks

    Example:
        >>> tracks = {1: active_track, 2: lost_track}
        >>> active_only = cleanup_lost_tracks(tracks, max_age=30)
        >>> len(active_only)  # Only contains active tracks
        1
    """
    active_tracks = {tid: track for tid, track in tracks.items() if track.is_active}
    return active_tracks


def get_track_summary(tracks: Dict[int, PlateTrack]) -> dict:
    """
    Generate statistics summary for current tracks.

    This function provides a quick overview of tracking status, useful for
    logging and monitoring pipeline performance.

    Args:
        tracks: Dictionary of track_id -> PlateTrack objects

    Returns:
        dict: Statistics dictionary containing:
            - total: Total number of tracks
            - active: Number of active tracks
            - recognized: Number of tracks with recognized text
            - avg_age: Average track age in frames
            - avg_ocr_confidence: Average OCR confidence for recognized tracks

    Example:
        >>> summary = get_track_summary(tracks)
        >>> print(f"Tracking {summary['total']} plates, "
        ...       f"{summary['recognized']} recognized")
        Tracking 5 plates, 3 recognized
    """
    if not tracks:
        return {"total": 0, "active": 0, "recognized": 0, "avg_age": 0.0, "avg_ocr_confidence": 0.0}

    total = len(tracks)
    active = sum(1 for t in tracks.values() if t.is_active)
    recognized = sum(1 for t in tracks.values() if t.text is not None)

    # Calculate average age
    avg_age = sum(t.age for t in tracks.values()) / total if total > 0 else 0.0

    # Calculate average OCR confidence for recognized tracks
    recognized_tracks = [t for t in tracks.values() if t.text is not None]
    avg_ocr_confidence = (
        sum(t.ocr_confidence for t in recognized_tracks) / len(recognized_tracks)
        if recognized_tracks
        else 0.0
    )

    return {
        "total": total,
        "active": active,
        "recognized": recognized,
        "avg_age": round(avg_age, 1),
        "avg_ocr_confidence": round(avg_ocr_confidence, 3),
    }


def get_tracks_needing_ocr(tracks: Dict[int, PlateTrack], config: dict) -> List[int]:
    """
    Get list of track IDs that need OCR processing.

    This is a convenience function that applies the should_run_ocr() decision
    logic to all active tracks and returns the IDs of tracks requiring OCR.

    Args:
        tracks: Dictionary of track_id -> PlateTrack objects
        config: Tracking configuration with OCR trigger parameters

    Returns:
        List[int]: List of track IDs that should have OCR performed

    Example:
        >>> track_ids = get_tracks_needing_ocr(tracks, config)
        >>> for tid in track_ids:
        ...     crop = crop_plate(frame, tracks[tid].bbox)
        ...     text, conf = ocr_model.recognize(crop)
        ...     tracks[tid].update_text(text, conf)
    """
    ocr_needed = [
        tid for tid, track in tracks.items() if track.is_active and track.should_run_ocr(config)
    ]
    return ocr_needed

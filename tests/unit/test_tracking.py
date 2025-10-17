"""
Unit tests for tracking module components.

Tests the PlateTrack class and utility functions to ensure correct behavior
of OCR trigger logic and track management.
"""

import pytest
from src.tracking import PlateTrack, cleanup_lost_tracks, get_track_summary, get_tracks_needing_ocr


class TestPlateTrack:
    """Test cases for PlateTrack class."""
    
    def test_initialization(self):
        """Test PlateTrack initialization with default values."""
        track = PlateTrack(track_id=1, bbox=(100, 200, 300, 400), confidence=0.85)
        
        assert track.id == 1
        assert track.bbox == (100, 200, 300, 400)
        assert track.detection_confidence == 0.85
        assert track.text is None
        assert track.ocr_confidence == 0.0
        assert track.age == 0
        assert track.frames_since_last_ocr == 0
        assert track.is_active is True
    
    def test_update(self):
        """Test track update increments counters correctly."""
        track = PlateTrack(track_id=1, bbox=(100, 200, 300, 400), confidence=0.85)
        
        # Update with new detection
        track.update(bbox=(105, 205, 305, 405), confidence=0.87)
        
        assert track.bbox == (105, 205, 305, 405)
        assert track.detection_confidence == 0.87
        assert track.age == 1
        assert track.frames_since_last_ocr == 1
        
        # Update again
        track.update(bbox=(110, 210, 310, 410), confidence=0.88)
        assert track.age == 2
        assert track.frames_since_last_ocr == 2
    
    def test_update_text(self):
        """Test update_text stores OCR results and resets timer."""
        track = PlateTrack(track_id=1, bbox=(100, 200, 300, 400), confidence=0.85)
        track.update(bbox=(100, 200, 300, 400), confidence=0.85)
        track.update(bbox=(100, 200, 300, 400), confidence=0.85)
        
        assert track.frames_since_last_ocr == 2
        
        # Update with OCR results
        track.update_text(text="ABC123", confidence=0.95)
        
        assert track.text == "ABC123"
        assert track.ocr_confidence == 0.95
        assert track.frames_since_last_ocr == 0
    
    def test_mark_lost(self):
        """Test marking track as lost."""
        track = PlateTrack(track_id=1, bbox=(100, 200, 300, 400), confidence=0.85)
        assert track.is_active is True
        
        track.mark_lost()
        assert track.is_active is False
    
    def test_should_run_ocr_condition1_new_track(self):
        """Test Condition 1: New track (age == 0) should always run OCR."""
        track = PlateTrack(track_id=1, bbox=(100, 200, 300, 400), confidence=0.85)
        config = {'ocr_interval': 30, 'ocr_confidence_threshold': 0.7}
        
        # New track (age 0)
        assert track.age == 0
        assert track.should_run_ocr(config) is True
    
    def test_should_run_ocr_condition2_stale_recognition(self):
        """Test Condition 2: Stale recognition (frames >= ocr_interval) should run OCR."""
        track = PlateTrack(track_id=1, bbox=(100, 200, 300, 400), confidence=0.85)
        config = {'ocr_interval': 30, 'ocr_confidence_threshold': 0.7}
        
        # Age > 0 (not new track)
        track.update(bbox=(100, 200, 300, 400), confidence=0.85)
        assert track.age == 1
        
        # Set text with good confidence
        track.update_text(text="ABC123", confidence=0.95)
        
        # Initially should not run OCR (frames_since_last_ocr == 0)
        assert track.should_run_ocr(config) is False
        
        # Fast-forward to ocr_interval
        for _ in range(30):
            track.update(bbox=(100, 200, 300, 400), confidence=0.85)
        
        # Now should run OCR (stale recognition)
        assert track.frames_since_last_ocr == 30
        assert track.should_run_ocr(config) is True
    
    def test_should_run_ocr_condition3_low_confidence_retry(self):
        """Test Condition 3: Low OCR confidence should trigger retry."""
        track = PlateTrack(track_id=1, bbox=(100, 200, 300, 400), confidence=0.85)
        config = {'ocr_interval': 30, 'ocr_confidence_threshold': 0.7}
        
        # Age > 0 (not new track)
        track.update(bbox=(100, 200, 300, 400), confidence=0.85)
        
        # Set text with LOW confidence
        track.update_text(text="ABC123", confidence=0.5)
        
        # Should run OCR immediately (low confidence)
        assert track.ocr_confidence < 0.7
        assert track.should_run_ocr(config) is True
        
        # Update with HIGH confidence
        track.update_text(text="ABC123", confidence=0.95)
        
        # Should not run OCR now (high confidence + recent)
        assert track.should_run_ocr(config) is False
    
    def test_should_run_ocr_condition4_high_detection_no_text(self):
        """Test Condition 4: High detection confidence + no text should run OCR."""
        track = PlateTrack(track_id=1, bbox=(100, 200, 300, 400), confidence=0.92)
        config = {'ocr_interval': 30, 'ocr_confidence_threshold': 0.7}
        
        # Age > 0 (not new track)
        track.update(bbox=(100, 200, 300, 400), confidence=0.92)
        
        # High detection confidence but no text yet
        assert track.detection_confidence > 0.9
        assert track.text is None
        assert track.should_run_ocr(config) is True
        
        # After recognizing text, should not run OCR
        track.update_text(text="ABC123", confidence=0.95)
        assert track.should_run_ocr(config) is False
    
    def test_should_run_ocr_no_conditions_met(self):
        """Test that OCR is NOT run when no conditions are met."""
        track = PlateTrack(track_id=1, bbox=(100, 200, 300, 400), confidence=0.85)
        config = {'ocr_interval': 30, 'ocr_confidence_threshold': 0.7}
        
        # Age > 0 (not new)
        track.update(bbox=(100, 200, 300, 400), confidence=0.85)
        
        # Set text with good confidence
        track.update_text(text="ABC123", confidence=0.95)
        
        # Advance a few frames (but not to ocr_interval)
        for _ in range(10):
            track.update(bbox=(100, 200, 300, 400), confidence=0.85)
        
        # Should NOT run OCR:
        # - Not new track (age > 0)
        # - Not stale (frames_since_last_ocr < ocr_interval)
        # - Good confidence (ocr_confidence >= threshold)
        # - Already has text
        assert track.should_run_ocr(config) is False
    
    def test_repr(self):
        """Test string representation."""
        track = PlateTrack(track_id=5, bbox=(100, 200, 300, 400), confidence=0.85)
        track.update(bbox=(100, 200, 300, 400), confidence=0.85)
        track.update_text(text="ABC123", confidence=0.95)
        
        repr_str = repr(track)
        assert "PlateTrack" in repr_str
        assert "id=5" in repr_str
        assert "ABC123" in repr_str
        assert "active" in repr_str


class TestTrackingUtils:
    """Test cases for tracking utility functions."""
    
    def test_cleanup_lost_tracks(self):
        """Test cleanup removes only lost tracks."""
        track1 = PlateTrack(track_id=1, bbox=(100, 200, 300, 400), confidence=0.85)
        track2 = PlateTrack(track_id=2, bbox=(100, 200, 300, 400), confidence=0.85)
        track3 = PlateTrack(track_id=3, bbox=(100, 200, 300, 400), confidence=0.85)
        
        track2.mark_lost()  # Mark track2 as lost
        
        tracks = {1: track1, 2: track2, 3: track3}
        
        cleaned = cleanup_lost_tracks(tracks, max_age=30)
        
        assert len(cleaned) == 2
        assert 1 in cleaned
        assert 2 not in cleaned  # Lost track removed
        assert 3 in cleaned
    
    def test_get_track_summary_empty(self):
        """Test summary with no tracks."""
        summary = get_track_summary({})
        
        assert summary['total'] == 0
        assert summary['active'] == 0
        assert summary['recognized'] == 0
        assert summary['avg_age'] == 0.0
        assert summary['avg_ocr_confidence'] == 0.0
    
    def test_get_track_summary_with_tracks(self):
        """Test summary with multiple tracks."""
        track1 = PlateTrack(track_id=1, bbox=(100, 200, 300, 400), confidence=0.85)
        track1.update(bbox=(100, 200, 300, 400), confidence=0.85)  # age=1
        track1.update_text(text="ABC123", confidence=0.95)
        
        track2 = PlateTrack(track_id=2, bbox=(100, 200, 300, 400), confidence=0.85)
        track2.update(bbox=(100, 200, 300, 400), confidence=0.85)  # age=1
        track2.update(bbox=(100, 200, 300, 400), confidence=0.85)  # age=2
        track2.update_text(text="XYZ789", confidence=0.85)
        
        track3 = PlateTrack(track_id=3, bbox=(100, 200, 300, 400), confidence=0.85)
        track3.mark_lost()  # Lost track
        
        tracks = {1: track1, 2: track2, 3: track3}
        
        summary = get_track_summary(tracks)
        
        assert summary['total'] == 3
        assert summary['active'] == 2  # track1, track2 active; track3 lost
        assert summary['recognized'] == 2  # track1, track2 have text
        assert summary['avg_age'] == 1.0  # (1 + 2 + 0) / 3 = 1.0
        assert summary['avg_ocr_confidence'] == 0.9  # (0.95 + 0.85) / 2 = 0.9
    
    def test_get_tracks_needing_ocr(self):
        """Test getting list of tracks needing OCR."""
        config = {'ocr_interval': 30, 'ocr_confidence_threshold': 0.7}
        
        # New track (age 0) - should need OCR
        track1 = PlateTrack(track_id=1, bbox=(100, 200, 300, 400), confidence=0.85)
        
        # Old track with good text - should NOT need OCR
        track2 = PlateTrack(track_id=2, bbox=(100, 200, 300, 400), confidence=0.85)
        track2.update(bbox=(100, 200, 300, 400), confidence=0.85)
        track2.update_text(text="ABC123", confidence=0.95)
        
        # Track with low confidence - should need OCR
        track3 = PlateTrack(track_id=3, bbox=(100, 200, 300, 400), confidence=0.85)
        track3.update(bbox=(100, 200, 300, 400), confidence=0.85)
        track3.update_text(text="XYZ789", confidence=0.5)  # Low confidence
        
        # Lost track - should NOT need OCR
        track4 = PlateTrack(track_id=4, bbox=(100, 200, 300, 400), confidence=0.85)
        track4.mark_lost()
        
        tracks = {1: track1, 2: track2, 3: track3, 4: track4}
        
        ocr_needed = get_tracks_needing_ocr(tracks, config)
        
        assert 1 in ocr_needed  # New track
        assert 2 not in ocr_needed  # Good text, recent
        assert 3 in ocr_needed  # Low confidence
        assert 4 not in ocr_needed  # Lost track


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

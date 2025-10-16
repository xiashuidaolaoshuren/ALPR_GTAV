"""
Recognition Utility Functions

Helper functions for text post-processing, filtering, and candidate scoring.
Used by recognize_text() for selecting best OCR result.
"""

import re
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


def filter_by_regex(text: str, pattern: str) -> bool:
    """
    Validate text against regex pattern for GTA V license plate format.
    
    Args:
        text: Recognized text string to validate (should be uppercase alphanumeric).
        pattern: Regex pattern string (e.g., '^\\d{2}[A-Z]{3}\\d{3}$' for GTA V format).
    
    Returns:
        bool: True if text matches pattern, False otherwise.
    
    Raises:
        re.error: If pattern is invalid regex.
    
    Example:
        >>> filter_by_regex('12ABC345', '^\\d{2}[A-Z]{3}\\d{3}$')
        True
        >>> filter_by_regex('ABC-123', '^\\d{2}[A-Z]{3}\\d{3}$')
        False
        >>> filter_by_regex('12AB34', '^\\d{2}[A-Z]{3}\\d{3}$')
        False
    
    Note:
        - Used to filter out invalid OCR results (headers, small text, etc.)
        - GTA V plate format: 2 digits, 3 letters, 3 digits (e.g., 12ABC345)
        - Pattern loaded from config for flexibility
    """
    if not text or not pattern:
        return False
    
    try:
        return bool(re.match(pattern, text))
    except re.error as e:
        logger.error(f"Invalid regex pattern '{pattern}': {e}")
        raise


def score_candidate(
    text: str,
    confidence: float,
    bbox_height: float,
    image_height: float
) -> float:
    """
    Calculate score for OCR text candidate using multi-factor formula.
    
    Scoring formula from shrimp-rules.md:
    score = p * h * min(L/8, 1)
    where:
    - p = OCR confidence [0.0-1.0]
    - h = normalized bbox height (bbox_height / image_height)
    - L = text length (number of characters)
    
    Args:
        text: Recognized text string (candidate).
        confidence: OCR confidence score for this text [0.0-1.0].
        bbox_height: Height of text bounding box in pixels.
        image_height: Total height of plate image in pixels.
    
    Returns:
        float: Calculated score [0.0-1.0], higher is better.
    
    Raises:
        ValueError: If confidence not in [0,1] or heights are non-positive.
    
    Example:
        >>> score = score_candidate('12ABC345', 0.95, 50, 100)
        >>> print(f"Score: {score:.3f}")
        Score: 0.475
    
    Note:
        - Balances confidence, size, and length for robust selection
        - Larger text (higher h) gets higher score (likely main plate number)
        - Length capped at 8 to avoid bias toward long text (GTA V plates are 8 chars)
        - Used when OCR returns multiple text lines per plate
    """
    # Validate inputs
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"Confidence must be in [0,1], got {confidence}")
    if bbox_height <= 0 or image_height <= 0:
        raise ValueError(f"Heights must be positive, got bbox_height={bbox_height}, image_height={image_height}")
    if not text:
        return 0.0
    
    # Calculate score components
    p = confidence  # OCR confidence
    h = bbox_height / image_height  # Normalized bbox height
    L = len(text)  # Text length
    
    # Calculate final score: p * h * min(L/8, 1)
    # For GTA V plates (8 characters), valid plates get full length contribution
    score = p * h * min(L / 8.0, 1.0)
    
    logger.debug(f"Scored candidate '{text}': p={p:.3f}, h={h:.3f}, L={L}, score={score:.3f}")
    
    return score


def select_best_candidate(candidates: List[Dict]) -> Tuple[Optional[str], float]:
    """
    Select best text candidate from list based on scores.
    
    Args:
        candidates: List of candidate dictionaries, each containing:
                   - 'text' (str): Recognized text
                   - 'confidence' (float): OCR confidence [0.0-1.0]
                   - 'score' (float): Calculated score from score_candidate()
                   - 'bbox' (list): Bounding box coordinates
    
    Returns:
        Tuple containing:
        - best_text (str or None): Text of highest-scoring candidate,
                                   or None if candidates list is empty
        - best_confidence (float): Confidence of best candidate,
                                  or 0.0 if no candidates
    
    Raises:
        KeyError: If candidate dict missing required keys.
        ValueError: If candidates list contains invalid data.
    
    Example:
        >>> candidates = [
        ...     {'text': '12ABC345', 'confidence': 0.95, 'score': 0.475},
        ...     {'text': 'HEADER', 'confidence': 0.80, 'score': 0.200}
        ... ]
        >>> text, conf = select_best_candidate(candidates)
        >>> print(f"Best: {text} ({conf:.2f})")
        Best: 12ABC345 (0.95)
    
    Note:
        - Selects candidate with highest 'score' value
        - Returns None if candidates list is empty (no valid text found)
        - Used as final step in recognize_text() post-processing
        - Handles ties by selecting first candidate (stable sort)
    """
    if not candidates:
        logger.debug("No candidates to select from")
        return None, 0.0
    
    # Validate candidates have required keys
    required_keys = {'text', 'confidence', 'score'}
    for i, candidate in enumerate(candidates):
        missing_keys = required_keys - set(candidate.keys())
        if missing_keys:
            raise KeyError(f"Candidate {i} missing required keys: {missing_keys}")
    
    # Select candidate with highest score
    best = max(candidates, key=lambda x: x['score'])
    
    logger.info(f"Selected best candidate: '{best['text']}' "
                f"(confidence={best['confidence']:.3f}, score={best['score']:.3f})")
    
    return best['text'], best['confidence']

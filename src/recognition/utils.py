"""
Recognition Utility Functions

Helper functions for text post-processing, filtering, and candidate scoring.
Used by recognize_text() for selecting best OCR result.
"""

import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


def filter_by_regex(text: str, pattern: str) -> bool:
    """
    Validate text against regex pattern for GTA V license plate format.
    
    Args:
        text: Recognized text string to validate (should be uppercase alphanumeric).
        pattern: Regex pattern string (e.g., '^[A-Z0-9]{6,8}$' for 6-8 character plates).
    
    Returns:
        bool: True if text matches pattern, False otherwise.
    
    Raises:
        re.error: If pattern is invalid regex.
    
    Example:
        >>> filter_by_regex('SA8821A', '^[A-Z0-9]{6,8}$')
        True
        >>> filter_by_regex('ABC-123', '^[A-Z0-9]{6,8}$')
        False
        >>> filter_by_regex('12', '^[A-Z0-9]{6,8}$')
        False
    
    Note:
        - Used to filter out invalid OCR results (headers, small text, etc.)
        - Common GTA V plate format: 6-8 uppercase letters and numbers
        - Pattern loaded from config for flexibility
    """
    pass


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
        >>> score = score_candidate('SA8821A', 0.95, 50, 100)
        >>> print(f"Score: {score:.3f}")
        Score: 0.475
    
    Note:
        - Balances confidence, size, and length for robust selection
        - Larger text (higher h) gets higher score (likely main plate number)
        - Length capped at 8 to avoid bias toward long text
        - Used when OCR returns multiple text lines per plate
    """
    pass


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
        ...     {'text': 'SA8821A', 'confidence': 0.95, 'score': 0.475},
        ...     {'text': 'HEADER', 'confidence': 0.80, 'score': 0.200}
        ... ]
        >>> text, conf = select_best_candidate(candidates)
        >>> print(f"Best: {text} ({conf:.2f})")
        Best: SA8821A (0.95)
    
    Note:
        - Selects candidate with highest 'score' value
        - Returns None if candidates list is empty (no valid text found)
        - Used as final step in recognize_text() post-processing
        - Handles ties by selecting first candidate (stable sort)
    """
    pass

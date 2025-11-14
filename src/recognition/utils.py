"""
Recognition Utility Functions

Helper functions for text post-processing, filtering, and candidate scoring.
Used by recognize_text() for selecting best OCR result.

Key Features:
- OCR confusion correction (O↔0, I/L↔1, S↔5, B↔8, Z↔2, G↔6)
- Regex-based format validation (GTA V: 2 digits + 3 letters + 3 digits)
- Multi-factor candidate scoring (confidence × bbox_height × length)
- Best candidate selection based on calculated scores

OCR Confusion Correction:
The correct_ocr_confusions() function handles systematic OCR errors by mapping
lookalike characters to the expected type (digit or letter) at each position:
  - If position expects digit: O→0, Q→0, I→1, L→1, S→5, B→8, Z→2, G→6
  - If position expects letter: 0→O, 1→I, 5→S, 2→Z, 6→G, 8→B

This postprocessor is applied AFTER OCR inference but BEFORE regex validation,
significantly improving recognition success rate (estimated +10-20% for GTA V plates).

Example Pipeline:
1. PaddleOCR inference → raw text (e.g., "I2ABC34S")
2. Normalize to uppercase → "I2ABC34S"
3. Apply confusion correction → "12ABC345" ✓
4. Regex validation → passes (^\\d{2}[A-Z]{3}\\d{3}$)
5. Scoring and selection → final result
"""

import re
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


def correct_ocr_confusions(text: str, plate_format: str = r"^\d{2}[A-Z]{3}\d{3}$") -> str:
    """
    Correct common OCR character confusions based on expected position type.

    Handles classic OCR confusion pairs by mapping lookalike characters to the
    expected type (digit or letter) at each position according to the plate format.

    Args:
        text: Raw OCR text to correct (typically uppercase alphanumeric).
        plate_format: Regex pattern defining expected character types at each position.
                     Default: r'^\\d{2}[A-Z]{3}\\d{3}$' (GTA V: 2 digits, 3 letters, 3 digits)

    Returns:
        str: Corrected text with confusions resolved based on expected types.

    Confusion Mappings:
        If position expects DIGIT (\\d):
        - O → 0, Q → 0, I → 1, L → 1, S → 5, B → 8, Z → 2, G → 6

        If position expects LETTER ([A-Z]):
        - 0 → O, 1 → I, 5 → S, 2 → Z, 6 → G, 8 → B

    Example:
        >>> # GTA V format: ^^\\d{2}[A-Z]{3}\\d{3}$
        >>> correct_ocr_confusions('I2ABC34S')  # I→1, S→5
        '12ABC345'
        >>> correct_ocr_confusions('12AB0345')  # 0→O in letter position
        '12ABO345'
        >>> correct_ocr_confusions('OZ4BC123')  # O→0, Z→2
        '024BC123'

    Note:
        - Only corrects characters that violate the expected type
        - Preserves already-correct characters
        - Uses conservative 1→I mapping (could be 1→L, but I is more common)
        - Should be called AFTER OCR but BEFORE regex validation
        - Improves recognition rate by handling systematic OCR errors
    """
    if not text:
        return text

    # Define confusion mappings
    DIGIT_CONFUSIONS = {
        "O": "0",
        "Q": "0",  # O and Q look like 0
        "I": "1",
        "L": "1",  # I and L look like 1
        "S": "5",  # S looks like 5
        "B": "8",  # B looks like 8
        "Z": "2",  # Z looks like 2
        "G": "6",  # G looks like 6
    }

    LETTER_CONFUSIONS = {
        "0": "O",  # 0 looks like O
        "1": "I",  # 1 looks like I (could also be L, but I is more common)
        "5": "S",  # 5 looks like S
        "2": "Z",  # 2 looks like Z
        "6": "G",  # 6 looks like G
        "8": "B",  # 8 looks like B
    }

    # Parse plate format to determine expected type at each position
    # For GTA V format: r'^\d{2}[A-Z]{3}\d{3}$'
    # Expected: [digit, digit, letter, letter, letter, digit, digit, digit]

    expected_types = []

    # Simple parser for common regex patterns
    # Handles: \d{n}, [A-Z]{n}, \d, [A-Z]
    pattern = plate_format.strip("^$")  # Remove anchors
    i = 0
    while i < len(pattern):
        if pattern[i : i + 2] == r"\d":
            # Check for {n} quantifier
            if i + 2 < len(pattern) and pattern[i + 2] == "{":
                # Extract count
                end = pattern.find("}", i + 2)
                if end != -1:
                    count = int(pattern[i + 3 : end])
                    expected_types.extend(["digit"] * count)
                    i = end + 1
                    continue
            # Single \d
            expected_types.append("digit")
            i += 2
        elif pattern[i : i + 5] == "[A-Z]":
            # Check for {n} quantifier
            if i + 5 < len(pattern) and pattern[i + 5] == "{":
                # Extract count
                end = pattern.find("}", i + 5)
                if end != -1:
                    count = int(pattern[i + 6 : end])
                    expected_types.extend(["letter"] * count)
                    i = end + 1
                    continue
            # Single [A-Z]
            expected_types.append("letter")
            i += 5
        else:
            i += 1

    # Correct text based on expected types
    corrected = []
    for idx, char in enumerate(text):
        if idx >= len(expected_types):
            # Beyond expected length, keep as-is
            corrected.append(char)
            continue

        expected_type = expected_types[idx]

        if expected_type == "digit":
            # Position expects digit: apply digit confusion mapping
            if char in DIGIT_CONFUSIONS:
                corrected_char = DIGIT_CONFUSIONS[char]
                if corrected_char != char:
                    logger.debug(
                        f"Position {idx}: corrected '{char}' → '{corrected_char}' (expected digit)"
                    )
                corrected.append(corrected_char)
            else:
                corrected.append(char)

        elif expected_type == "letter":
            # Position expects letter: apply letter confusion mapping
            if char in LETTER_CONFUSIONS:
                corrected_char = LETTER_CONFUSIONS[char]
                if corrected_char != char:
                    logger.debug(
                        f"Position {idx}: corrected '{char}' → '{corrected_char}' (expected letter)"
                    )
                corrected.append(corrected_char)
            else:
                corrected.append(char)

        else:
            # Unknown expected type, keep as-is
            corrected.append(char)

    corrected_text = "".join(corrected)

    if corrected_text != text:
        logger.info(f"OCR confusion correction: '{text}' → '{corrected_text}'")

    return corrected_text


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
        logger.debug(f"Regex validation skipped: text='{text}' or pattern='{pattern}' is empty")
        return False

    try:
        matches = bool(re.match(pattern, text))
        if not matches:
            logger.warning(f"Text '{text}' failed regex validation against pattern '{pattern}'")
        else:
            logger.debug(f"Text '{text}' passed regex validation")
        return matches
    except re.error as e:
        logger.error(f"Invalid regex pattern '{pattern}': {e}")
        raise


def score_candidate(text: str, confidence: float, bbox_height: float, image_height: float) -> float:
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
        raise ValueError(
            f"Heights must be positive, got bbox_height={bbox_height}, image_height={image_height}"
        )
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
    required_keys = {"text", "confidence", "score"}
    for i, candidate in enumerate(candidates):
        missing_keys = required_keys - set(candidate.keys())
        if missing_keys:
            raise KeyError(f"Candidate {i} missing required keys: {missing_keys}")

    # Select candidate with highest score
    best = max(candidates, key=lambda x: x["score"])

    logger.info(
        f"Selected best candidate: '{best['text']}' "
        f"(confidence={best['confidence']:.3f}, score={best['score']:.3f})"
    )

    return best["text"], best["confidence"]

"""
Test OCR Confusion Correction

Validates the correct_ocr_confusions() function with various test cases
covering common OCR confusion patterns.
"""

import sys
sys.path.insert(0, 'd:/Felix_stuff/ALPR_GTA5')

from src.recognition.utils import correct_ocr_confusions


def test_ocr_corrections():
    """
    Test OCR confusion correction with various realistic cases.
    """
    # Define test cases: (input, expected_output, description)
    test_cases = [
        # Digit confusions (positions 0-1, 5-7)
        ("I2ABC34S", "12ABC345", "I→1, S→5 in digit positions"),
        ("OZ4BC123", "024BC123", "O→0, Z→2 in digit positions"),
        ("L2ABC3G5", "12ABC365", "L→1, G→6 in digit positions"),
        ("QB8YC1BB", "08BYC188", "Q→0 in pos 0, 8→B in pos 2, B→8 in pos 6-7"),
        # Letter confusions (positions 2-4)
        ("12AB0345", "12ABO345", "0→O in letter position"),
        ("12A1C345", "12AIC345", "1→I in letter position"),
        ("125BC345", "12SBC345", "5→S in letter position"),
        ("122BC345", "12ZBC345", "2→Z in letter position"),
        ("126BC345", "12GBC345", "6→G in letter position"),
        ("128BC345", "12BBC345", "8→B in letter position"),
        # Mixed confusions
        ("I2A1C34S", "12AIC345", "Mixed I→1, 1→I, S→5"),
        ("OZ50C1BB", "02SOC188", "O→0, Z→2, 5→S in pos 2, 0→O in pos 3, B→8"),
        ("12012345", "12OIZ345", "Multiple letter confusions"),
        # Already correct plates (no changes)
        ("12ABC345", "12ABC345", "Already correct - no changes"),
        ("00XYZ999", "00XYZ999", "All correct - no changes"),
        # Edge cases
        ("", "", "Empty string"),
        ("12ABC", "12ABC", "Incomplete plate (length < 8)"),
        ("12ABC3456789", "12ABC3456789", "Longer than expected"),
        # Real OCR errors from dataset
        ("28FJN643", "28FJN643", "Real example - no correction needed"),
        ("491IV281", "49IIV281", "Real example - 1→I in position 3"),
        ("23W8E599", "23WBE599", "Real example - 8→B in position 4"),
    ]

    print("=" * 80)
    print("OCR CONFUSION CORRECTION TEST")
    print("=" * 80)

    passed = 0
    failed = 0

    for input_text, expected, description in test_cases:
        result = correct_ocr_confusions(input_text)

        if result == expected:
            status = "✓ PASS"
            passed += 1
        else:
            status = "✗ FAIL"
            failed += 1

        print(f"\n{status}: {description}")
        print(f"  Input:    '{input_text}'")
        print(f"  Expected: '{expected}'")
        print(f"  Got:      '{result}'")

        if result != expected:
            print("  ERROR: Mismatch!")

    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 80)

    return failed == 0


def test_position_mapping():
    """
    Test the position type detection for GTA V format.
    """
    print("\n" + "=" * 80)
    print("POSITION TYPE MAPPING TEST (GTA V Format: ^^\\d{2}[A-Z]{3}\\d{3}$)")
    print("=" * 80)

    # Expected: [digit, digit, letter, letter, letter, digit, digit, digit]
    # Position:   0      1      2       3       4       5      6      7

    # Apply correction with all confusions
    confused_text = "OILBBBSSG"  # 9 characters, we only process first 8
    # O→0 (pos 0: digit), I→1 (pos 1: digit), L→L (pos 2: letter), B→B (pos 3: letter),
    # B→B (pos 4: letter), B→8 (pos 5: digit), S→5 (pos 6: digit), S→5 (pos 7: digit)

    result = correct_ocr_confusions(confused_text)
    expected = "01LBB855G"  # Last G is beyond position 7, kept as-is

    print(f"\nInput:    '{confused_text}'")
    print(f"Expected: '{expected}'")
    print(f"Got:      '{result}'")
    print("\nPosition mapping:")
    print("  Pos 0 (digit):  O → 0 ✓")
    print("  Pos 1 (digit):  I → 1 ✓")
    print("  Pos 2 (letter): L → L (no change)")
    print("  Pos 3 (letter): B → B (no change)")
    print("  Pos 4 (letter): B → B (no change)")
    print("  Pos 5 (digit):  B → 8 ✓")
    print("  Pos 6 (digit):  S → 5 ✓")
    print("  Pos 7 (digit):  S → 5 ✓")
    print("  Pos 8 (beyond): G → G (kept as-is)")

    if result == expected:
        print("\n✓ Position mapping works correctly!")
        return True
    else:
        print("\n✗ Position mapping failed!")
        return False


if __name__ == "__main__":
    print("\nTesting OCR Confusion Correction Postprocessor\n")

    test1_passed = test_ocr_corrections()
    test2_passed = test_position_mapping()

    print("\n" + "=" * 80)
    if test1_passed and test2_passed:
        print("ALL TESTS PASSED ✓")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED ✗")
        sys.exit(1)

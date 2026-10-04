"""Unit tests for offline typo correction (spelling.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spelling import correct_text


def test_correct_text_fixes_obvious_typos():
    text, corrections = correct_text("How can I improve my Engish speeking skils?")
    assert "english" in text.lower()
    assert "speaking" in text.lower()
    assert "skills" in text.lower()
    assert len(corrections) == 3


def test_correct_text_leaves_correct_text_unchanged():
    text, corrections = correct_text("What is the capital of France?")
    assert text == "What is the capital of France?"
    assert corrections == []


def test_correct_text_handles_empty_string():
    text, corrections = correct_text("")
    assert text == ""
    assert corrections == []


def test_correct_text_preserves_capitalization_of_corrected_word():
    text, corrections = correct_text("Wat is the best way to learn?")
    # whatever correction is made (if any) for the capitalized typo should
    # itself stay capitalized in the output
    if corrections:
        for c in corrections:
            if c["original"][0].isupper():
                assert text.split()[0][0].isupper()

"""Unit tests for offline language detection (langid.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langid import detect_language


def test_detect_language_english():
    result = detect_language("What is the capital of France and why is it important?")
    assert result["lang"] == "en"
    assert result["is_english"] is True


def test_detect_language_french_flagged_non_english():
    result = detect_language("Quelle est la capitale de la France et pourquoi est-elle importante?")
    assert result["lang"] == "fr"
    assert result["is_english"] is False


def test_detect_language_short_text_treated_as_unknown_but_not_flagged():
    result = detect_language("hi")
    assert result["lang"] == "unknown"
    assert result["is_english"] is True  # benefit of the doubt on short text


def test_detect_language_empty_string():
    result = detect_language("")
    assert result["lang"] == "unknown"
    assert result["is_english"] is True


def test_detect_language_returns_confidence_in_valid_range():
    result = detect_language("What is the best way to learn a new language quickly?")
    assert 0.0 <= result["confidence"] <= 1.0

"""
Lightweight, fully-offline spelling correction for question text.

Uses `pyspellchecker`, which ships its own word-frequency dictionary (no
network access needed). Only corrects words it's confident are misspelled
(not in its dictionary) and leaves everything else — including names,
domain jargon, and correctly-spelled uncommon words — untouched.
"""
import re
from spellchecker import SpellChecker

_spell = SpellChecker(distance=1)  # distance=1 keeps suggestions conservative

_WORD_RE = re.compile(r"[A-Za-z']+")


def correct_text(text):
    """
    Return (corrected_text, corrections) where corrections is a list of
    {"original": ..., "corrected": ...} for each word that was changed.
    Non-word tokens (numbers, punctuation, whitespace) pass through as-is.
    """
    if not text:
        return text, []

    corrections = []

    def _fix(match):
        word = match.group(0)
        lower = word.lower()
        if lower not in _spell.unknown([lower]):
            return word  # already a known word
        suggestion = _spell.correction(lower)
        if not suggestion or suggestion == lower:
            return word  # no confident correction available
        corrections.append({"original": word, "corrected": suggestion})
        # Preserve capitalization style of the original word
        if word[0].isupper():
            suggestion = suggestion.capitalize()
        return suggestion

    corrected = _WORD_RE.sub(_fix, text)
    return corrected, corrections

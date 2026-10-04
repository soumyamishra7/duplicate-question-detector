"""
Lightweight language detection for question text, using `langdetect`
(a pure-Python port of Google's language-detection library; no network
access needed — its n-gram profiles are bundled with the package).

The model and all its features (token overlap, TF-IDF, Word2Vec) were
trained on English-only Quora questions, so predictions on non-English
text are unreliable. This module just flags that risk; it doesn't block
or translate anything.
"""
from langdetect import detect_langs, LangDetectException

CONFIDENCE_THRESHOLD = 0.85
MIN_CHARS_FOR_RELIABLE_DETECTION = 15


def detect_language(text):
    """
    Return {"lang": "en"|<code>|"unknown", "confidence": float, "is_english": bool}.

    Short strings produce unreliable guesses, so anything under
    MIN_CHARS_FOR_RELIABLE_DETECTION characters is reported as "unknown"
    with is_english=True (benefit of the doubt — not enough signal to flag).
    """
    text = (text or "").strip()
    if len(text) < MIN_CHARS_FOR_RELIABLE_DETECTION:
        return {"lang": "unknown", "confidence": 0.0, "is_english": True}

    try:
        guesses = detect_langs(text)
    except LangDetectException:
        return {"lang": "unknown", "confidence": 0.0, "is_english": True}

    if not guesses:
        return {"lang": "unknown", "confidence": 0.0, "is_english": True}

    top = guesses[0]
    is_english = (top.lang == "en") or (top.prob < CONFIDENCE_THRESHOLD)
    return {"lang": top.lang, "confidence": round(top.prob, 3), "is_english": is_english}

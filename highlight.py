"""
Inline word-overlap highlighting between two questions.

Tokenizes each question (preserving original casing/punctuation for
display) and flags which word-tokens also appear — as meaningful
(non-stopword) words — in the other question. The frontend wraps flagged
tokens in a highlight span.
"""
import re

from features import STOP_WORDS

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[^\sA-Za-z0-9']+|\s+")


def _tokenize_for_display(text):
    """Split into tokens that, when joined, reconstruct the original text
    exactly (words, punctuation, and whitespace runs all kept as tokens)."""
    return _TOKEN_RE.findall(text or "")


def highlight_pairs(q1_raw, q2_raw):
    """
    Return (q1_tokens, q2_tokens), where each token is
    {"text": str, "shared": bool}. "shared" is True for word tokens
    (ignoring case) that are non-stopwords appearing in both questions.
    Punctuation/whitespace tokens are never marked shared.
    """
    def word_set(tokens):
        words = set()
        for t in tokens:
            low = t.lower()
            if low.isalnum() and low not in STOP_WORDS and len(low) > 1:
                words.add(low)
        return words

    t1 = _tokenize_for_display(q1_raw)
    t2 = _tokenize_for_display(q2_raw)

    w1 = word_set(t1)
    w2 = word_set(t2)
    shared = w1 & w2

    def mark(tokens):
        out = []
        for t in tokens:
            low = t.lower()
            is_shared = low.isalnum() and low in shared
            out.append({"text": t, "shared": is_shared})
        return out

    return mark(t1), mark(t2)

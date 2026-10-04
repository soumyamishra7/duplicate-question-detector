"""Unit tests for inline word-overlap highlighting (highlight.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from highlight import highlight_pairs


def _shared_words(tokens):
    return [t["text"].lower() for t in tokens if t["shared"]]


def test_highlight_pairs_marks_shared_content_words():
    t1, t2 = highlight_pairs("How can I lose weight fast?", "What is the quickest way to lose weight?")
    assert "lose" in _shared_words(t1)
    assert "weight" in _shared_words(t1)
    assert "lose" in _shared_words(t2)
    assert "weight" in _shared_words(t2)


def test_highlight_pairs_does_not_mark_stopwords_as_shared():
    t1, t2 = highlight_pairs("What is the best car?", "What is the best bike?")
    shared1 = _shared_words(t1)
    # "what", "is", "the" are stopwords and shouldn't be marked even though
    # they appear in both questions
    assert "what" not in shared1
    assert "is" not in shared1
    assert "the" not in shared1
    assert "best" in shared1


def test_highlight_pairs_no_overlap_for_unrelated_questions():
    t1, t2 = highlight_pairs("How do I bake a cake?", "What is the best programming language?")
    assert _shared_words(t1) == []
    assert _shared_words(t2) == []


def test_highlight_pairs_reconstructs_original_text():
    q1 = "How can I lose weight fast?"
    q2 = "What is the quickest way to lose weight?"
    t1, t2 = highlight_pairs(q1, q2)
    assert "".join(t["text"] for t in t1) == q1
    assert "".join(t["text"] for t in t2) == q2


def test_highlight_pairs_handles_empty_strings():
    t1, t2 = highlight_pairs("", "")
    assert t1 == []
    assert t2 == []


def test_highlight_pairs_case_insensitive_matching():
    t1, t2 = highlight_pairs("I love Python programming", "Python is a great language")
    assert "python" in _shared_words(t1)
    assert "python" in _shared_words(t2)

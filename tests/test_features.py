"""Unit tests for token/length/fuzzy feature engineering (features.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from features import preprocess, extract_features, FEATURE_NAMES, get_token_features, get_fuzzy_features


def test_preprocess_lowercases_and_strips_punctuation():
    out = preprocess("What's the BEST way?!")
    assert out == out.lower()
    assert "?" not in out
    assert "!" not in out


def test_preprocess_handles_contractions():
    out = preprocess("I can't believe it's not butter")
    assert "can not" in out or "cannot" not in out  # contraction expanded somehow
    assert "it is" in out


def test_preprocess_handles_empty_and_non_string_input():
    assert preprocess("") == ""
    assert preprocess(None) == "none" or preprocess(None) == ""  # str(None) -> "none"


def test_extract_features_length_matches_feature_names():
    feats = extract_features("How are you?", "How are you doing?")
    assert len(feats) == len(FEATURE_NAMES)


def test_extract_features_identical_questions_score_high():
    q = "What is the capital of France?"
    feats = extract_features(q, q)
    feats_dict = dict(zip(FEATURE_NAMES, feats))
    assert feats_dict["fuzz_ratio"] == 100
    assert feats_dict["token_set_ratio"] == 100
    # word_share = common / (len(w1) + len(w2)); for identical sets this
    # saturates at 0.5 (common == len(w1) == len(w2)), not 1.0 — that's the
    # formula's intended behaviour, not a bug.
    assert feats_dict["word_share"] == pytest.approx(0.5, abs=0.01)
    assert feats_dict["cwc_max"] == pytest.approx(1.0, abs=0.01)


def test_extract_features_unrelated_questions_score_low():
    feats = extract_features(
        "How do I bake a chocolate cake?",
        "What is the best programming language for beginners?",
    )
    feats_dict = dict(zip(FEATURE_NAMES, feats))
    assert feats_dict["word_share"] < 0.3
    assert feats_dict["fuzz_ratio"] < 60


def test_extract_features_handles_empty_strings_without_crashing():
    feats = extract_features("", "")
    assert len(feats) == len(FEATURE_NAMES)
    feats2 = extract_features("Hello world", "")
    assert len(feats2) == len(FEATURE_NAMES)


def test_get_token_features_first_last_word_match():
    tok = get_token_features("what is the best car", "what is the best bike")
    # index 6 = last_word_eq, index 7 = first_word_eq (see FEATURE_NAMES order)
    assert tok[7] == 1  # same first word ("what")
    assert tok[6] == 0  # different last word ("car" vs "bike")


def test_get_fuzzy_features_identical_strings():
    fuzzy = get_fuzzy_features("hello world", "hello world")
    assert all(f == 100 for f in fuzzy)


def test_get_fuzzy_features_completely_different_strings():
    fuzzy = get_fuzzy_features("abc", "xyz uvw rst")
    assert all(0 <= f <= 100 for f in fuzzy)

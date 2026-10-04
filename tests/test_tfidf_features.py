"""Unit tests for TF-IDF cosine-similarity features (tfidf_features.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tfidf_features import build_vectorizers, compute_tfidf_features, TFIDF_FEATURE_NAMES

CORPUS = [
    "What is the capital of France?",
    "What is the capital city of France?",
    "How do I bake a chocolate cake?",
    "What is the best programming language for beginners?",
    "How can I improve my English speaking skills?",
    "What are some tips to improve spoken English?",
]


def _vectorizers():
    return build_vectorizers(CORPUS)


def test_build_vectorizers_returns_fitted_vectorizers():
    word_vec, char_vec = _vectorizers()
    assert len(word_vec.vocabulary_) > 0
    assert len(char_vec.vocabulary_) > 0


def test_compute_tfidf_features_shape():
    word_vec, char_vec = _vectorizers()
    feats = compute_tfidf_features(
        "What is the capital of France?",
        "What is the capital city of France?",
        word_vec, char_vec,
    )
    assert len(feats) == len(TFIDF_FEATURE_NAMES) == 2


def test_compute_tfidf_features_identical_text_gives_cosine_one():
    word_vec, char_vec = _vectorizers()
    q = "What is the capital of France?"
    word_cos, char_cos = compute_tfidf_features(q, q, word_vec, char_vec)
    assert word_cos == 1.0 or abs(word_cos - 1.0) < 1e-6
    assert char_cos == 1.0 or abs(char_cos - 1.0) < 1e-6


def test_compute_tfidf_features_similar_pair_scores_higher_than_dissimilar():
    word_vec, char_vec = _vectorizers()
    similar_word_cos, _ = compute_tfidf_features(
        "What is the capital of France?",
        "What is the capital city of France?",
        word_vec, char_vec,
    )
    dissimilar_word_cos, _ = compute_tfidf_features(
        "What is the capital of France?",
        "How do I bake a chocolate cake?",
        word_vec, char_vec,
    )
    assert similar_word_cos > dissimilar_word_cos


def test_compute_tfidf_features_cosine_in_valid_range():
    word_vec, char_vec = _vectorizers()
    word_cos, char_cos = compute_tfidf_features(
        "Random unrelated text here",
        "Completely different sentence too",
        word_vec, char_vec,
    )
    assert 0.0 <= word_cos <= 1.0
    assert 0.0 <= char_cos <= 1.0

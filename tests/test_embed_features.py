"""Unit tests for Word2Vec embedding-similarity features (embed_features.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embed_features import build_word2vec, compute_embed_features, EMBED_FEATURE_NAMES

# A small but repetitive corpus so Word2Vec has enough co-occurrence signal
# to learn something meaningful for this unit test (a real model is trained
# on the full ~800k-question corpus; see training/train_model.py).
CORPUS = [
    "How can I lose weight fast",
    "What is the best way to lose weight",
    "How do I lose weight quickly",
    "Tips to lose weight and get fit",
    "How can I gain weight fast",
    "What is the best diet to lose weight",
    "How do I bake a chocolate cake",
    "What is the best recipe for chocolate cake",
    "How do I learn to code",
    "What is the best way to learn programming",
] * 20  # repeat for more co-occurrence signal in a tiny unit-test corpus


def _model():
    return build_word2vec(CORPUS, vector_size=20, window=3, min_count=1, epochs=10)


def test_build_word2vec_returns_model_with_vocab():
    model = _model()
    assert len(model.wv) > 0


def test_compute_embed_features_shape():
    model = _model()
    feats = compute_embed_features("How can I lose weight fast", "What is the best way to lose weight", model.wv)
    assert len(feats) == len(EMBED_FEATURE_NAMES) == 1


def test_compute_embed_features_in_valid_range():
    model = _model()
    feats = compute_embed_features("How do I bake a cake", "What is the best way to code", model.wv)
    assert 0.0 <= feats[0] <= 1.0


def test_compute_embed_features_identical_text_scores_near_one():
    model = _model()
    q = "How can I lose weight fast"
    feats = compute_embed_features(q, q, model.wv)
    assert feats[0] > 0.95


def test_compute_embed_features_handles_out_of_vocabulary_words():
    model = _model()
    # Words that never appeared in the tiny training corpus; should not crash,
    # and should fall back gracefully (zero vector -> midpoint similarity).
    feats = compute_embed_features("Zzyx qwerty asdf", "Completely unseen gibberish", model.wv)
    assert len(feats) == 1
    assert 0.0 <= feats[0] <= 1.0

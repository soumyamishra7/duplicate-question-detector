"""
Embedding-based similarity features for the Quora Question Pairs problem.

A Word2Vec model is trained on the Quora question corpus itself (no
internet / pretrained-model download required). Each question is
represented as the average of its words' embeddings, and the cosine
similarity between the two averaged vectors is used as a feature.

Unlike the TF-IDF features, this can catch paraphrases that share almost
no vocabulary (e.g. "lose weight fast" vs "shed pounds quickly") because
Word2Vec places semantically related words near each other in the vector
space, even if the exact words never co-occur in a question pair.
"""
import numpy as np
from gensim.models import Word2Vec

from features import preprocess

EMBED_FEATURE_NAMES = ["w2v_cosine"]

VECTOR_SIZE = 100


def build_word2vec(questions, vector_size=VECTOR_SIZE, window=5, min_count=2, epochs=8, seed=42):
    """Train a Word2Vec model on a corpus of (raw) questions."""
    sentences = [preprocess(q).split() for q in questions]
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        sg=1,          # skip-gram: better for smaller/medium corpora & rare words
        epochs=epochs,
        seed=seed,
    )
    return model


def _avg_vector(tokens, wv, dim):
    vecs = [wv[t] for t in tokens if t in wv]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0)


def _cosine(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_embed_features(q1_raw, q2_raw, wv):
    dim = wv.vector_size
    t1 = preprocess(q1_raw).split()
    t2 = preprocess(q2_raw).split()
    v1 = _avg_vector(t1, wv, dim)
    v2 = _avg_vector(t2, wv, dim)
    # Map cosine similarity (-1..1) to a 0..1 range for consistency with
    # the other similarity-style features (TF-IDF cosines, fuzzy ratios).
    cos = _cosine(v1, v2)
    return [(cos + 1.0) / 2.0]


def compute_embed_features_batch(df, wv, q1_col="question1", q2_col="question2"):
    dim = wv.vector_size
    rows = []
    for q1, q2 in zip(df[q1_col].fillna(""), df[q2_col].fillna("")):
        rows.append(compute_embed_features(q1, q2, wv))
    return np.array(rows, dtype=np.float32)

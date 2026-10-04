"""
TF-IDF based similarity features for the Quora Question Pairs problem.

Two vectorizers are fit on the training corpus:
  - a word-level TF-IDF (1-2 grams) that captures shared vocabulary/phrasing
  - a character-level TF-IDF (3-5 char n-grams, word boundary aware) that
    catches near-duplicates even when words are misspelled or reordered

Both vectorizers use sklearn's default L2 row-normalization, so the
row-wise dot product between a question-1 vector and a question-2 vector
*is* their cosine similarity — no extra normalization needed at inference.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from features import preprocess

TFIDF_FEATURE_NAMES = ["tfidf_word_cosine", "tfidf_char_cosine"]


def build_vectorizers(questions):
    """Fit word- and char-level TF-IDF vectorizers on a corpus of questions."""
    corpus = [preprocess(q) for q in questions]

    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_features=50000,
        sublinear_tf=True,
    )
    word_vec.fit(corpus)

    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_features=50000,
        sublinear_tf=True,
    )
    char_vec.fit(corpus)

    return word_vec, char_vec


def _row_cosine(vec, q1, q2):
    """Cosine similarity between TF-IDF(q1) and TF-IDF(q2) for a single pair."""
    a = vec.transform([q1])
    b = vec.transform([q2])
    # Both rows are L2-normalized by TfidfVectorizer, so dot product == cosine sim.
    sim = a.multiply(b).sum()
    return float(sim)


def compute_tfidf_features(q1_raw, q2_raw, word_vec, char_vec):
    q1 = preprocess(q1_raw)
    q2 = preprocess(q2_raw)
    word_cos = _row_cosine(word_vec, q1, q2)
    char_cos = _row_cosine(char_vec, q1, q2)
    return [word_cos, char_cos]


def compute_tfidf_features_batch(df, word_vec, char_vec, q1_col="question1", q2_col="question2"):
    q1_texts = [preprocess(q) for q in df[q1_col].fillna("")]
    q2_texts = [preprocess(q) for q in df[q2_col].fillna("")]

    A1 = word_vec.transform(q1_texts)
    A2 = word_vec.transform(q2_texts)
    word_cos = np.asarray(A1.multiply(A2).sum(axis=1)).ravel()

    B1 = char_vec.transform(q1_texts)
    B2 = char_vec.transform(q2_texts)
    char_cos = np.asarray(B1.multiply(B2).sum(axis=1)).ravel()

    return np.column_stack([word_cos, char_cos])

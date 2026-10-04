"""
EchoMatch — Quora Question Pairs duplicate-intent detector.

A small Flask web app that takes two questions and predicts the
probability that they carry the same intent, using token-based,
length-based, fuzzy (fuzzywuzzy), TF-IDF and Word2Vec features feeding a
tuned, calibrated LightGBM classifier.
"""
import os
import io
import csv
import logging
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, Response
from gensim.models import KeyedVectors
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from features import extract_features, FEATURE_NAMES
from tfidf_features import compute_tfidf_features, TFIDF_FEATURE_NAMES
from embed_features import compute_embed_features, EMBED_FEATURE_NAMES
from spelling import correct_text
from langid import detect_language
from highlight import highlight_pairs

ALL_FEATURE_NAMES = FEATURE_NAMES + TFIDF_FEATURE_NAMES + EMBED_FEATURE_NAMES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("echomatch")

app = Flask(__name__)

# Rate limiting: protects the (relatively expensive) /predict endpoint from
# accidental hammering or abuse. Uses Flask-Limiter's default in-memory
# backend, which is fine for a single-process deployment; point
# `storage_uri` at Redis (e.g. "redis://localhost:6379") for multi-worker
# or multi-host deployments, where limits must be shared across processes.
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

logger.info("Loading model artifacts...")
model = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
calibrator = joblib.load(os.path.join(MODEL_DIR, "calibrator.joblib"))
word_vec = joblib.load(os.path.join(MODEL_DIR, "tfidf_word.joblib"))
char_vec = joblib.load(os.path.join(MODEL_DIR, "tfidf_char.joblib"))
w2v = KeyedVectors.load(os.path.join(MODEL_DIR, "w2v_vectors.kv"))
logger.info("Model artifacts loaded.")

METRICS = {}
metrics_path = os.path.join(MODEL_DIR, "metrics.txt")
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        for line in f:
            if ":" in line:
                k, v = line.strip().split(":", 1)
                METRICS[k.strip()] = v.strip()

# Human-friendly groupings of the raw features, shown in the UI as the
# "linguistic breakdown" of a question pair.
DISPLAY_GROUPS = [
    {
        "label": "Length-based",
        "keys": ["q1_n_words", "q2_n_words", "abs_len_diff", "mean_len"],
    },
    {
        "label": "Token-based",
        "keys": ["word_share", "cwc_max", "ctc_max", "last_word_eq", "first_word_eq"],
    },
    {
        "label": "Fuzzy (fuzzywuzzy)",
        "keys": ["fuzz_ratio", "fuzz_partial_ratio", "token_sort_ratio", "token_set_ratio"],
    },
    {
        "label": "Semantic similarity (TF-IDF)",
        "keys": ["tfidf_word_cosine", "tfidf_char_cosine"],
    },
    {
        "label": "Meaning similarity (Word2Vec)",
        "keys": ["w2v_cosine"],
    },
]

PRETTY_NAMES = {
    "q1_n_words": "Words in Q1",
    "q2_n_words": "Words in Q2",
    "abs_len_diff": "Word-count difference",
    "mean_len": "Average word count",
    "word_share": "Shared-word ratio",
    "cwc_max": "Common content words",
    "ctc_max": "Common tokens (overall)",
    "last_word_eq": "Same last word",
    "first_word_eq": "Same first word",
    "fuzz_ratio": "Fuzzy ratio",
    "fuzz_partial_ratio": "Partial ratio",
    "token_sort_ratio": "Token sort ratio",
    "token_set_ratio": "Token set ratio",
    "tfidf_word_cosine": "Word-meaning overlap",
    "tfidf_char_cosine": "Spelling-level overlap",
    "w2v_cosine": "Learned meaning similarity",
}

BOOL_KEYS = {"last_word_eq", "first_word_eq"}
PERCENT_KEYS = {
    "word_share", "cwc_max", "cwc_min", "ctc_max", "ctc_min", "csc_max", "csc_min",
    "tfidf_word_cosine", "tfidf_char_cosine", "w2v_cosine",
}
SCORE_KEYS = {"fuzz_ratio", "fuzz_partial_ratio", "token_sort_ratio", "token_set_ratio"}


def _format_breakdown(feats_dict):
    breakdown = []
    for group in DISPLAY_GROUPS:
        items = []
        for key in group["keys"]:
            val = feats_dict.get(key)
            if val is None:
                continue
            if key in BOOL_KEYS:
                display = "Yes" if val else "No"
            elif key in PERCENT_KEYS:
                display = f"{val * 100:.0f}%"
            elif key in SCORE_KEYS:
                display = f"{val:.0f}/100"
            else:
                display = f"{val:.1f}" if isinstance(val, float) else str(val)
            items.append({"name": PRETTY_NAMES.get(key, key), "value": display})
        breakdown.append({"label": group["label"], "items": items})
    return breakdown


@lru_cache(maxsize=4096)
def run_prediction(q1, q2):
    """
    Pure, cacheable prediction function: (question1, question2) -> result dict.

    Cached in-process with an LRU cache (process memory only — fine for a
    single-worker deployment; a multi-worker deployment would see one cache
    per worker, which is still a net win since identical pairs are common
    in real traffic, but won't be shared across workers without an external
    cache like Redis).
    """
    feats = extract_features(q1, q2)
    tfidf_feats = compute_tfidf_features(q1, q2, word_vec, char_vec)
    embed_feats = compute_embed_features(q1, q2, w2v)
    all_feats = feats + tfidf_feats + embed_feats
    feats_dict = dict(zip(ALL_FEATURE_NAMES, all_feats))

    X = scaler.transform(np.array(all_feats).reshape(1, -1))
    raw_proba = float(model.predict_proba(X)[0, 1])
    proba = float(calibrator.predict([raw_proba])[0])
    label = "duplicate" if proba >= 0.5 else "not_duplicate"

    return {
        "probability": round(proba, 4),
        "label": label,
        "breakdown": _format_breakdown(feats_dict),
    }


@app.route("/")
def index():
    return render_template("index.html", metrics=METRICS)


@app.route("/healthz")
def healthz():
    """Liveness/readiness check for load balancers, Docker, etc."""
    return jsonify({"status": "ok"})


MAX_QUESTION_LEN = 2000


@app.route("/predict", methods=["POST"])
@limiter.limit("30 per minute")
def predict():
    data = request.get_json(force=True) or {}
    q1_raw = (data.get("question1") or "").strip()
    q2_raw = (data.get("question2") or "").strip()
    fix_typos = bool(data.get("fix_typos"))

    if not q1_raw or not q2_raw:
        return jsonify({"error": "Please enter both questions."}), 400

    if len(q1_raw) > MAX_QUESTION_LEN or len(q2_raw) > MAX_QUESTION_LEN:
        return jsonify({"error": f"Questions must be under {MAX_QUESTION_LEN} characters."}), 400

    # Language detection runs on exactly what the user typed, before any
    # typo correction, since that's the actual signal about what language
    # they're writing in.
    lang1 = detect_language(q1_raw)
    lang2 = detect_language(q2_raw)

    spelling_info = None
    q1, q2 = q1_raw, q2_raw
    if fix_typos:
        q1_corrected, q1_fixes = correct_text(q1_raw)
        q2_corrected, q2_fixes = correct_text(q2_raw)
        if q1_fixes or q2_fixes:
            spelling_info = {
                "question1": {"text": q1_corrected, "corrections": q1_fixes},
                "question2": {"text": q2_corrected, "corrections": q2_fixes},
            }
        q1, q2 = q1_corrected, q2_corrected

    try:
        result = run_prediction(q1, q2)
    except Exception:
        logger.exception("Prediction failed for a question pair")
        return jsonify({"error": "Something went wrong while comparing these questions."}), 500

    h1, h2 = highlight_pairs(q1, q2)
    result["highlight"] = {"question1": h1, "question2": h2}
    result["language"] = {"question1": lang1, "question2": lang2}
    if spelling_info:
        result["spelling"] = spelling_info

    return jsonify(result)


MAX_BATCH_ROWS = 500
MAX_BATCH_FILE_BYTES = 2 * 1024 * 1024  # 2 MB


@app.route("/predict_batch", methods=["POST"])
@limiter.limit("5 per minute")
def predict_batch():
    """
    Batch comparison from an uploaded CSV with `question1`/`question2`
    columns (matching the Kaggle train/test.csv format). Returns a CSV
    with `probability` and `label` columns appended.
    """
    f = request.files.get("file")
    if f is None or f.filename == "":
        return jsonify({"error": "Please choose a CSV file to upload."}), 400

    raw = f.read(MAX_BATCH_FILE_BYTES + 1)
    if len(raw) > MAX_BATCH_FILE_BYTES:
        return jsonify({"error": "File is too large (2 MB limit)."}), 400

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        return jsonify({"error": "Could not parse that file as CSV."}), 400

    df.columns = [c.strip().lower() for c in df.columns]
    if "question1" not in df.columns or "question2" not in df.columns:
        return jsonify({"error": "CSV must have 'question1' and 'question2' columns."}), 400

    if len(df) == 0:
        return jsonify({"error": "The uploaded CSV has no rows."}), 400
    if len(df) > MAX_BATCH_ROWS:
        return jsonify({"error": f"Please limit uploads to {MAX_BATCH_ROWS} rows (got {len(df)})."}), 400

    probabilities = []
    labels = []
    for q1, q2 in zip(df["question1"].fillna(""), df["question2"].fillna("")):
        q1, q2 = str(q1).strip()[:MAX_QUESTION_LEN], str(q2).strip()[:MAX_QUESTION_LEN]
        if not q1 or not q2:
            probabilities.append("")
            labels.append("error: empty question")
            continue
        try:
            result = run_prediction(q1, q2)
            probabilities.append(result["probability"])
            labels.append(result["label"])
        except Exception:
            logger.exception("Batch prediction failed for one row")
            probabilities.append("")
            labels.append("error")

    df["probability"] = probabilities
    df["label"] = labels

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    csv_bytes = buf.getvalue().encode("utf-8")

    return Response(
        csv_bytes,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=echomatch_results.csv"},
    )


@app.errorhandler(429)
def rate_limit_exceeded(e):
    return jsonify({"error": "Too many requests. Please slow down and try again shortly."}), 429


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)

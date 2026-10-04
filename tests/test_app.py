"""
Integration tests for the Flask app (routes, validation, caching, rate limiting).

These load the real trained model artifacts from model/, so they're slower
than the pure unit tests in test_features.py / test_tfidf_features.py /
test_embed_features.py, but they exercise the full request path.
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


@pytest.fixture(scope="module")
def client():
    from app import app, run_prediction
    app.config["TESTING"] = True
    run_prediction.cache_clear()
    with app.test_client() as c:
        yield c


def test_index_page_loads(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"EchoMatch" in r.data


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.get_json() == {"status": "ok"}


def test_predict_duplicate_pair(client):
    r = client.post(
        "/predict",
        json={
            "question1": "What is the capital of France?",
            "question2": "What is the capital city of France?",
        },
    )
    assert r.status_code == 200
    data = r.get_json()
    assert data["label"] == "duplicate"
    assert 0.0 <= data["probability"] <= 1.0
    assert len(data["breakdown"]) == 5  # length, token, fuzzy, tfidf, w2v groups


def test_predict_non_duplicate_pair(client):
    r = client.post(
        "/predict",
        json={
            "question1": "How do I bake a chocolate cake?",
            "question2": "What is the best programming language for beginners?",
        },
    )
    assert r.status_code == 200
    data = r.get_json()
    assert data["label"] == "not_duplicate"


def test_predict_missing_question_returns_400(client):
    r = client.post("/predict", json={"question1": "", "question2": "something"})
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_predict_missing_both_fields_returns_400(client):
    r = client.post("/predict", json={})
    assert r.status_code == 400


def test_predict_overlong_question_returns_400(client):
    r = client.post("/predict", json={"question1": "x" * 5000, "question2": "short"})
    assert r.status_code == 400


def test_predict_response_is_cached(client):
    from app import run_prediction
    run_prediction.cache_clear()
    payload = {"question1": "Why is the sky blue?", "question2": "What makes the sky appear blue?"}
    r1 = client.post("/predict", json=payload)
    info_after_first = run_prediction.cache_info()
    r2 = client.post("/predict", json=payload)
    info_after_second = run_prediction.cache_info()

    assert r1.get_json() == r2.get_json()
    assert info_after_second.hits == info_after_first.hits + 1


def test_predict_breakdown_contains_expected_groups(client):
    r = client.post(
        "/predict",
        json={"question1": "How can I learn Python?", "question2": "What is the best way to learn Python?"},
    )
    labels = [g["label"] for g in r.get_json()["breakdown"]]
    assert labels == [
        "Length-based",
        "Token-based",
        "Fuzzy (fuzzywuzzy)",
        "Semantic similarity (TF-IDF)",
        "Meaning similarity (Word2Vec)",
    ]


def test_predict_includes_highlight_and_language(client):
    r = client.post(
        "/predict",
        json={"question1": "How can I lose weight fast?", "question2": "What is the quickest way to lose weight?"},
    )
    data = r.get_json()
    assert "highlight" in data
    assert "question1" in data["highlight"] and "question2" in data["highlight"]
    assert "language" in data
    assert data["language"]["question1"]["lang"] == "en"


def test_predict_flags_non_english_question(client):
    r = client.post(
        "/predict",
        json={
            "question1": "Quelle est la capitale de la France et pourquoi est-elle importante?",
            "question2": "What is the capital of France?",
        },
    )
    data = r.get_json()
    assert data["language"]["question1"]["is_english"] is False


def test_predict_without_fix_typos_leaves_typos_uncorrected(client):
    r = client.post(
        "/predict",
        json={"question1": "How can I improve my Engish speeking?", "question2": "How do I get better at English?"},
    )
    data = r.get_json()
    assert "spelling" not in data


def test_predict_with_fix_typos_returns_corrections(client):
    r = client.post(
        "/predict",
        json={
            "question1": "How can I improve my Engish speeking skils?",
            "question2": "What should I do to get better at English?",
            "fix_typos": True,
        },
    )
    data = r.get_json()
    assert "spelling" in data
    assert len(data["spelling"]["question1"]["corrections"]) > 0


def test_predict_batch_returns_csv_with_predictions(client):
    csv_content = (
        b"question1,question2\n"
        b"What is the capital of France?,What is the capital city of France?\n"
        b"How do I bake a cake?,What is the best programming language?\n"
    )
    data = {"file": (io.BytesIO(csv_content), "test.csv")}
    r = client.post("/predict_batch", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    assert r.mimetype == "text/csv"
    body = r.data.decode()
    assert "probability" in body
    assert "label" in body
    assert body.count("\n") >= 3  # header + 2 data rows (+ trailing newline)


def test_predict_batch_missing_columns_returns_400(client):
    data = {"file": (io.BytesIO(b"a,b\n1,2\n"), "bad.csv")}
    r = client.post("/predict_batch", data=data, content_type="multipart/form-data")
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_predict_batch_no_file_returns_400(client):
    r = client.post("/predict_batch", data={}, content_type="multipart/form-data")
    assert r.status_code == 400


def test_predict_batch_empty_csv_returns_400(client):
    data = {"file": (io.BytesIO(b"question1,question2\n"), "empty.csv")}
    r = client.post("/predict_batch", data=data, content_type="multipart/form-data")
    assert r.status_code == 400


def test_predict_batch_too_many_rows_returns_400(client):
    big = b"question1,question2\n" + (b"a,b\n" * 600)
    data = {"file": (io.BytesIO(big), "big.csv")}
    r = client.post("/predict_batch", data=data, content_type="multipart/form-data")
    assert r.status_code == 400
    assert "500" in r.get_json()["error"]

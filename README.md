# EchoMatch — Quora Question Pairs Duplicate Detector

A small, fully working Flask web app that predicts whether two questions
have the same intent (the Kaggle "Quora Question Pairs" problem), built
around **token-based**, **length-based**, **fuzzy (fuzzywuzzy)**,
**TF-IDF similarity**, and **Word2Vec meaning similarity** features
feeding a tuned, calibrated LightGBM classifier — with gunicorn/Docker
deployment, rate limiting, caching, and an automated test suite underneath,
plus batch CSV comparison, optional typo correction, non-English
detection, shared-word highlighting, and client-side comparison history
on top.

## What's inside

```
echomatch/
├── app.py                  Flask application (routes below)
├── features.py             Token/length/fuzzy feature engineering
├── tfidf_features.py       TF-IDF cosine-similarity feature engineering
├── embed_features.py       Word2Vec cosine-similarity feature engineering
├── spelling.py             Offline typo correction (pyspellchecker)
├── langid.py                Offline language detection (langdetect)
├── highlight.py            Shared-word highlighting between two questions
├── requirements.txt        Runtime dependencies
├── requirements-dev.txt    Runtime + test dependencies (pytest)
├── pytest.ini
├── Dockerfile              Production container image (gunicorn inside)
├── .dockerignore
├── Procfile                For Heroku/Render-style platforms
├── run_prod.sh             Convenience script: run locally with gunicorn
├── model/
│   ├── model.joblib        Trained, tuned LightGBM classifier
│   ├── calibrator.joblib   Isotonic-regression probability calibrator
│   ├── scaler.joblib       StandardScaler fitted on the training features
│   ├── tfidf_word.joblib   Word-level (1-2gram) TF-IDF vectorizer
│   ├── tfidf_char.joblib   Char-level (3-5gram) TF-IDF vectorizer
│   ├── w2v_vectors.kv      Word2Vec embeddings trained on the question corpus
│   └── metrics.txt         Held-out test metrics
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── script.js
├── tests/
│   ├── test_features.py        Unit tests: token/length/fuzzy features
│   ├── test_tfidf_features.py  Unit tests: TF-IDF similarity features
│   ├── test_embed_features.py  Unit tests: Word2Vec similarity features
│   ├── test_spelling.py        Unit tests: typo correction
│   ├── test_langid.py          Unit tests: language detection
│   ├── test_highlight.py       Unit tests: shared-word highlighting
│   └── test_app.py             Integration tests: all routes, validation, caching
└── training/
    ├── train_model.py     Reference end-to-end training script
    ├── features.py        Same feature module, kept alongside the training script
    ├── tfidf_features.py  Same TF-IDF module, kept alongside the training script
    └── embed_features.py  Same Word2Vec module, kept alongside the training script
```

## Model performance (held-out test split)

See `model/metrics.txt`. The model was trained on the **full** `train.csv`
(323,429 train / 80,858 held-out test rows), tuned via a 12-trial
randomized hyperparameter search over LightGBM, then calibrated, reaching:

- **Accuracy ≈ 78.7%**
- **ROC AUC ≈ 0.875**
- **Log loss ≈ 0.412**
- **Brier score ≈ 0.139** (lower is better; measures how trustworthy the
  predicted probability itself is, not just the yes/no call)

This comfortably clears the 70% accuracy target.

**On calibration:** `model/calibrator.joblib` is a small isotonic-regression
curve fit on out-of-fold predictions (5-fold `cross_val_predict`), applied
on top of the LightGBM model's raw output. It turned out the model was
already close to well-calibrated out of the box — gradient boosting with
a log-loss objective tends to be — so this mainly sharpens probabilities
at the extremes rather than changing the overall accuracy. It was fit this
way (rather than wrapping the model in `CalibratedClassifierCV`'s ensemble
mode) specifically to avoid training and shipping 5 separate full-sized
models for a sub-0.1-point accuracy difference — that approach was tried
and discarded: it produced a 100MB+ model file for a ~0.05pp accuracy gain.

**On the Word2Vec feature:** `sentence-transformers`-style pretrained
embedding models are hosted on huggingface.co, which isn't reachable from
this build environment, so instead a Word2Vec model was trained from
scratch on the Quora question corpus itself (no internet required). It
successfully learns relations like *weight ↔ lose/fat/gain* and
*pounds ↔ kgs/kilos/lbs*, and became the 3rd most important feature out
of 25. It helps most when a pair already has *some* shared vocabulary; a
pair with **zero** shared words (e.g. "How can I lose weight fast?" vs
"What is the quickest way to shed pounds?") can still be missed, since
one embedding feature isn't always enough to outweigh 24 lexical features
that all read as "no overlap." A proper pretrained sentence-transformer
(if network access allows it in your environment) or averaging across more
epochs/a larger corpus would close this gap further.

## Features used

**Length-based**
- Character length and word count of each question
- Absolute difference and average of word counts

**Token-based**
- Overlap of non-stopword ("content") tokens, normalized by the smaller/larger question
- Overlap of stopword tokens
- Overlap of raw tokens
- Whether the first/last word matches

**Fuzzy (fuzzywuzzy)**
- `QRatio`, `partial_ratio`, `token_sort_ratio`, `token_set_ratio`

**TF-IDF similarity**
- Cosine similarity between word-level (1-2gram) TF-IDF vectors
- Cosine similarity between character-level (3-5gram) TF-IDF vectors —
  robust to misspellings and word reordering

**Word2Vec meaning similarity**
- Cosine similarity between each question's averaged word-embedding
  vector, from a Word2Vec model trained on the question corpus

All 25 features are standardized with `StandardScaler` before being passed
to the classifier.

## App features beyond the core prediction

**Batch CSV comparison.** Upload a CSV with `question1`/`question2`
columns (the Kaggle train/test.csv format works as-is) via the "Batch
compare" card, and get a CSV back with `probability` and `label` columns
appended. Capped at 500 rows / 2MB per upload, rate-limited to 5
uploads/minute per IP.

**Optional typo correction.** The "Fix obvious typos before comparing"
checkbox runs each question through `pyspellchecker` (a fully offline,
dictionary-based corrector — no network calls) before feature extraction,
and shows exactly what was changed. It's **off by default and
opt-in**, because a spell-checker without a real-world vocabulary can
occasionally "correct" a proper noun or piece of jargon it doesn't
recognize into something else (e.g. "Mathematica" → "mathematical") —
better to show the person what changed than silently alter their input.

**Non-English detection.** `langdetect` (also fully offline) flags when a
question looks like it's not English, since the model was trained
exclusively on English Quora questions and its predictions on other
languages aren't meaningful. This only warns — it doesn't block the
comparison, since language detection on short strings is itself
imperfect. Text under 15 characters is treated as "unknown" rather than
guessed at, since there's not enough signal to call it reliably.

**Shared-word highlighting.** Each result shows both questions with their
overlapping *meaningful* words (stopwords like "what"/"is"/"the" are
excluded) highlighted inline, so you can see at a glance what the
token-based features are actually keying off.

**Comparison history.** Every comparison is saved to the browser's
`localStorage` (last 20) and listed under "Recent comparisons" — nothing
is sent to or stored on the server. Clicking a past entry reloads it and
reruns the comparison.

## Running locally (development)

```bash
cd echomatch
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000** in your browser.

> Note: `fuzzywuzzy` works noticeably faster with `python-Levenshtein`
> installed (included in requirements.txt); without it, it still works
> but with a one-time warning.

This runs Flask's built-in development server, which is fine for trying
the app out but **isn't meant for production traffic** (Flask prints a
warning about this on startup). For anything beyond local testing, use one
of the options below.

## Running in production

**Option A — gunicorn directly:**

```bash
pip install -r requirements.txt
./run_prod.sh              # binds 0.0.0.0:8000
# or: PORT=9000 ./run_prod.sh
```

**Option B — Docker:**

```bash
docker build -t echomatch .
docker run -p 8000:8000 echomatch
```

Then open **http://localhost:8000**.

**Option C — Heroku/Render/etc.:** the included `Procfile` is ready to go;
these platforms detect it automatically.

Production notes baked in:
- **gunicorn** (2 workers × 2 threads by default — tune `--workers` to
  your CPU count in `run_prod.sh` / `Dockerfile` / `Procfile`) replaces
  Flask's dev server.
- **`/healthz`** returns `{"status": "ok"}` for load balancers, Docker
  `HEALTHCHECK`, or uptime monitors.
- **Rate limiting** (via Flask-Limiter): `/predict` is capped at 30
  requests/minute per IP, with a global cap of 200/day and 50/hour.
  Exceeding it returns `429` with a JSON error body. The default in-memory
  limiter is per-process; set `storage_uri` in `app.py` to a Redis URL
  (e.g. `redis://localhost:6379`) if you run multiple workers/hosts and
  need limits shared across them.
- **In-process caching**: identical `(question1, question2)` pairs are
  served from an LRU cache (`run_prediction`, 4096 entries) instead of
  re-computing features and re-running the model. Also per-process; a
  shared cache across workers would need something like Redis.
- **Structured error handling**: malformed input returns `400` with a
  JSON `error` message rather than a stack trace; unexpected failures
  during prediction are logged server-side and return a generic `500`
  rather than leaking internals to the client.

> **Note:** there's no Docker daemon available in the environment this
> project was built in, so the Dockerfile's structure was verified but
> the image itself has **not** been build-tested. It follows a standard
> `python:3.11-slim` + gunicorn + non-root-user pattern, but please do a
> trial `docker build` before relying on it.

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest
```

53 tests across seven files: pure unit tests for each feature/utility
module (`features.py`, `tfidf_features.py`, `embed_features.py`,
`spelling.py`, `langid.py`, `highlight.py` — fast, train tiny throwaway
models where needed) and integration tests for every Flask route
(`test_app.py` — loads the real model, so slightly slower), including the
batch CSV endpoint and its validation/limits.

## Re-training the model

The original Kaggle `train.csv` is **not** bundled in this zip (to keep
it small). To retrain:

1. Download `train.csv` from the Kaggle "Quora Question Pairs" competition
   and place it inside `training/`.
2. From the `training/` folder, run:
   ```bash
   pip install pandas scikit-learn lightgbm gensim joblib fuzzywuzzy python-Levenshtein distance
   python train_model.py
   ```
3. Copy the resulting `model.joblib`, `calibrator.joblib`, `scaler.joblib`,
   `tfidf_word.joblib`, `tfidf_char.joblib` and `w2v_vectors.kv` into `model/`.

## API

**`GET /healthz`** — liveness check.

```json
{ "status": "ok" }
```

**`POST /predict`** — rate-limited to 30 requests/minute per IP.

```json
{ "question1": "How can I be a good geologist?",
  "question2": "What should I do to be a great geologist?",
  "fix_typos": false }
```

`fix_typos` is optional (default `false`).

Success (`200`):

```json
{
  "probability": 0.7862,
  "label": "duplicate",
  "breakdown": [ { "label": "Length-based", "items": [...] }, ... ],
  "highlight": {
    "question1": [ { "text": "How", "shared": false }, { "text": " ", "shared": false }, ... ],
    "question2": [ ... ]
  },
  "language": {
    "question1": { "lang": "en", "confidence": 1.0, "is_english": true },
    "question2": { "lang": "en", "confidence": 1.0, "is_english": true }
  },
  "spelling": {
    "question1": { "text": "...corrected text...", "corrections": [ { "original": "Engish", "corrected": "english" } ] },
    "question2": { "text": "...", "corrections": [] }
  }
}
```

`spelling` is only present when `fix_typos: true` **and** at least one
correction was made.

Validation error (`400`) — empty question, or either question over 2000 characters:

```json
{ "error": "Please enter both questions." }
```

Rate limit exceeded (`429`):

```json
{ "error": "Too many requests. Please slow down and try again shortly." }
```

**`POST /predict_batch`** — multipart form upload, field name `file`;
rate-limited to 5 requests/minute per IP. Max 500 rows / 2MB.

Request: a CSV with `question1` and `question2` columns (column names
case-insensitive).

Success (`200`): the same CSV with `probability` and `label` columns
appended, returned as a file attachment (`Content-Type: text/csv`).

Validation errors (`400`): missing file, missing required columns, empty
CSV, or too many rows — each with a JSON `{"error": "..."}` body.

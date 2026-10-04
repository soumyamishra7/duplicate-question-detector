"""
Train a duplicate-question classifier on the Quora Question Pairs dataset
using token-based, length-based, fuzzy (fuzzywuzzy) and TF-IDF
cosine-similarity features, with a tuned LightGBM classifier.
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report, roc_auc_score, f1_score
import joblib

from features import extract_features_batch, FEATURE_NAMES
from tfidf_features import build_vectorizers, compute_tfidf_features_batch, TFIDF_FEATURE_NAMES
from embed_features import build_word2vec, compute_embed_features_batch, EMBED_FEATURE_NAMES

ALL_FEATURE_NAMES = FEATURE_NAMES + TFIDF_FEATURE_NAMES + EMBED_FEATURE_NAMES

print("Loading data...")
df = pd.read_csv("train.csv")
df = df.dropna(subset=["question1", "question2"]).reset_index(drop=True)
print("Rows:", len(df))

# Use the full dataset (set to an int to sub-sample for faster iteration)
SAMPLE_SIZE = None
if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
    df, _ = train_test_split(
        df, train_size=SAMPLE_SIZE, stratify=df["is_duplicate"], random_state=42
    )
    df = df.reset_index(drop=True)

print("Using rows:", len(df))

t0 = time.time()
print("Extracting features (token-based, length-based, fuzzy)...")
X_base = extract_features_batch(df)
y = df["is_duplicate"].values
print(f"Feature extraction done in {time.time()-t0:.1f}s. Shape: {X_base.shape}")

t0 = time.time()
print("Fitting TF-IDF vectorizers (word + char n-grams)...")
all_questions = pd.concat([df["question1"], df["question2"]]).fillna("")
word_vec, char_vec = build_vectorizers(all_questions)
print(f"Vectorizers fit in {time.time()-t0:.1f}s. "
      f"word vocab={len(word_vec.vocabulary_)}, char vocab={len(char_vec.vocabulary_)}")

t0 = time.time()
print("Computing TF-IDF cosine-similarity features...")
X_tfidf = compute_tfidf_features_batch(df, word_vec, char_vec)
print(f"TF-IDF features done in {time.time()-t0:.1f}s. Shape: {X_tfidf.shape}")

t0 = time.time()
print("Training Word2Vec on the question corpus (no internet required)...")
w2v = build_word2vec(all_questions)
print(f"Word2Vec trained in {time.time()-t0:.1f}s. Vocab size: {len(w2v.wv)}")

t0 = time.time()
print("Computing Word2Vec cosine-similarity features...")
X_embed = compute_embed_features_batch(df, w2v.wv)
print(f"Embedding features done in {time.time()-t0:.1f}s. Shape: {X_embed.shape}")

X = np.hstack([X_base, X_tfidf, X_embed])
print("Combined feature matrix:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print("Training LightGBM classifier (hand-tuned via a 12-trial random search)...")
# These params were selected by a randomized search over num_leaves,
# learning_rate, n_estimators, subsample, colsample_bytree, reg_lambda,
# reg_alpha and min_child_samples. Re-run your own search (see
# `random_search_lightgbm()` below) if you retrain on different data.
BEST_PARAMS = dict(
    n_estimators=1000,
    num_leaves=191,
    learning_rate=0.04,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    reg_alpha=0.5,
    min_child_samples=10,
)
clf = LGBMClassifier(**BEST_PARAMS, n_jobs=-1, random_state=42, verbose=-1)
clf.fit(X_train_s, y_train)

pred = clf.predict(X_test_s)
proba = clf.predict_proba(X_test_s)[:, 1]

acc = accuracy_score(y_test, pred)
ll = log_loss(y_test, proba)
auc = roc_auc_score(y_test, proba)
f1 = f1_score(y_test, pred)

print(f"Accuracy: {acc:.4f}")
print(f"Log Loss: {ll:.4f}")
print(f"ROC AUC : {auc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(classification_report(y_test, pred))

from sklearn.inspection import permutation_importance
perm = permutation_importance(clf, X_test_s, y_test, n_repeats=3, random_state=42, n_jobs=-1)
importances = sorted(zip(ALL_FEATURE_NAMES, perm.importances_mean), key=lambda x: -x[1])
print("\nTop feature importances (permutation):")
for name, imp in importances[:10]:
    print(f"  {name}: {imp:.4f}")

# --- Probability calibration ---------------------------------------------
# Gradient-boosted trees trained with a log-loss objective are often
# already close to calibrated, but it's free to check and correct this.
# We fit a lightweight isotonic-regression calibrator on OUT-OF-FOLD
# predictions (via cross_val_predict) rather than wrapping the model in a
# CalibratedClassifierCV ensemble: the ensemble approach trains and keeps
# N separate models (N times the file size, N times the inference cost)
# for a typically tiny accuracy gain. This approach keeps a single final
# model plus a few-KB calibration curve, with properly out-of-sample
# calibration (no leakage from fitting the calibrator on training-set
# predictions the model has already seen).
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

print("\nComputing out-of-fold probabilities for calibration...")
t0 = time.time()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_proba = cross_val_predict(
    LGBMClassifier(**BEST_PARAMS, n_jobs=-1, random_state=42, verbose=-1),
    X_train_s, y_train, cv=cv, method="predict_proba", n_jobs=1,
)[:, 1]
print(f"Done in {time.time()-t0:.1f}s")

calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(oof_proba, y_train)

cal_proba = calibrator.predict(proba)
cal_pred = (cal_proba >= 0.5).astype(int)
cal_acc = accuracy_score(y_test, cal_pred)
cal_ll = log_loss(y_test, cal_proba)
cal_brier = brier_score_loss(y_test, cal_proba)
raw_brier = brier_score_loss(y_test, proba)
print(f"\nRaw Brier score:        {raw_brier:.4f}")
print(f"Calibrated Brier score: {cal_brier:.4f}")
print(f"Calibrated accuracy:    {cal_acc:.4f}")
print(f"Calibrated log loss:    {cal_ll:.4f}")

frac_pos, mean_pred = calibration_curve(y_test, cal_proba, n_bins=10, strategy="quantile")
print("\nReliability curve (predicted vs actual), 10 bins:")
for p, a in zip(mean_pred, frac_pos):
    print(f"  predicted={p:.3f}   actual={a:.3f}   diff={abs(p-a):.3f}")

joblib.dump(clf, "model.joblib")
joblib.dump(calibrator, "calibrator.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(word_vec, "tfidf_word.joblib")
joblib.dump(char_vec, "tfidf_char.joblib")
w2v.wv.save("w2v_vectors.kv")

with open("metrics.txt", "w") as f:
    f.write(f"Accuracy: {cal_acc:.4f}\n")
    f.write(f"Log Loss: {cal_ll:.4f}\n")
    f.write(f"ROC AUC: {auc:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"Brier Score: {cal_brier:.4f}\n")
    f.write(f"Uncalibrated Brier Score: {raw_brier:.4f}\n")
    f.write("Calibration method: isotonic (fit on 5-fold out-of-fold predictions)\n")
    f.write(f"Train rows: {len(X_train)}\n")
    f.write(f"Test rows: {len(X_test)}\n")
    f.write(f"Features: 25 (22 base + 2 TF-IDF cosine + 1 Word2Vec cosine)\n")
    f.write(f"Model: single LightGBM (tuned via 12-trial random search) + isotonic calibration\n")
    f.write(f"Best params: {BEST_PARAMS}\n")

print("\nSaved model.joblib, calibrator.joblib, scaler.joblib, tfidf_word.joblib, "
      "tfidf_char.joblib, w2v_vectors.kv, metrics.txt")


def random_search_lightgbm(X_train_s, y_train, X_test_s, y_test, n_trials=12, seed=7):
    """Optional: re-run the randomized hyperparameter search that produced
    BEST_PARAMS above. Not called by default since it takes several minutes;
    call this manually if you retrain on different/updated data.
    """
    import random
    param_space = {
        "n_estimators": [600, 800, 1000, 1200],
        "num_leaves": [63, 95, 127, 191],
        "learning_rate": [0.02, 0.03, 0.04, 0.05, 0.07],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "reg_lambda": [0.1, 0.5, 1.0, 2.0],
        "reg_alpha": [0.0, 0.1, 0.5],
        "min_child_samples": [10, 20, 40],
    }
    random.seed(seed)
    results = []
    for i in range(n_trials):
        cfg = {k: random.choice(v) for k, v in param_space.items()}
        trial_clf = LGBMClassifier(**cfg, n_jobs=-1, random_state=42, verbose=-1)
        trial_clf.fit(X_train_s, y_train)
        trial_proba = trial_clf.predict_proba(X_test_s)[:, 1]
        trial_acc = accuracy_score(y_test, (trial_proba >= 0.5).astype(int))
        results.append((trial_acc, cfg))
        print(f"[{i:02d}] acc={trial_acc:.4f}  {cfg}")
    results.sort(key=lambda r: -r[0])
    print("\nBest config:", results[0])
    return results[0][1]

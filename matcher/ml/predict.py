"""
Loads the trained artifacts (TF-IDF vectorizer, scaler, logistic regression
model, PyTorch net, and metrics/threshold) exactly once per process and
exposes a single predict() function the views call.

Note: this pipeline deliberately does NOT use sentence-transformers for
semantic embeddings, even though that's a natural fit for a resume/JD
similarity feature. It was tried and removed — see the "About the ML stack"
note in the README — because sentence-transformers pulls in torch,
transformers, tokenizers, and huggingface-hub, which together exceeded
Render's free-tier 512MB RAM limit and crashed the whole app, both at
startup and mid-request. TF-IDF similarity plus the skill/experience
features carry the matching signal instead.
"""

import json

import joblib
import torch

from django.conf import settings

from .features import build_feature_dict, feature_dict_to_vector
from .nn_model import FitNet

_vectorizer = None
_scaler = None
_logistic_model = None
_nn_model = None
_metrics = None


def _artifacts_dir():
    return settings.ML_ARTIFACTS_DIR


def _load_artifacts():
    """Loads every saved artifact into module-level globals, once."""
    global _vectorizer, _scaler, _logistic_model, _nn_model, _metrics

    if _metrics is not None:
        return  # already loaded

    artifacts_dir = _artifacts_dir()

    _vectorizer = joblib.load(artifacts_dir / "tfidf_vectorizer.joblib")
    _scaler = joblib.load(artifacts_dir / "scaler.joblib")
    _logistic_model = joblib.load(artifacts_dir / "logistic_regression.joblib")

    with open(artifacts_dir / "metrics.json") as f:
        _metrics = json.load(f)

    input_size = len(_metrics["feature_names"])
    _nn_model = FitNet(input_size=input_size)
    _nn_model.load_state_dict(torch.load(artifacts_dir / "pytorch_nn.pt", map_location="cpu"))
    _nn_model.eval()


def warm():
    """
    Forces the artifacts to load right now instead of lazily on the first
    predict() call. Called once at process startup (see matcher/apps.py).
    Now that sentence-transformers is gone, this is cheap (small joblib
    files + a tiny PyTorch state dict) — kept mainly so the first real
    request isn't the one paying even that small cost.
    """
    _load_artifacts()


def predict(resume_text, jd_text):
    """
    Runs the full pipeline on one resume/JD pair and returns a dict with
    both models' scores/verdicts plus the human-readable skill/experience
    breakdown used by the result page.
    """
    _load_artifacts()

    features = build_feature_dict(resume_text, jd_text, _vectorizer)
    vector = feature_dict_to_vector(features).reshape(1, -1)
    scaled_vector = _scaler.transform(vector)

    logistic_score = float(_logistic_model.predict_proba(scaled_vector)[0][1])

    with torch.no_grad():
        logits = _nn_model(torch.tensor(scaled_vector, dtype=torch.float32))
        nn_score = float(torch.sigmoid(logits).item())

    thresholds = _metrics["thresholds"]
    primary_model_name = _metrics["primary_model"]

    scores_by_model = {"logistic_regression": logistic_score, "pytorch_nn": nn_score}
    primary_score = scores_by_model[primary_model_name]
    other_model_name = "pytorch_nn" if primary_model_name == "logistic_regression" else "logistic_regression"
    other_score = scores_by_model[other_model_name]

    def verdict_for(model_name, score):
        return "Good Fit" if score >= thresholds[model_name] else "Not a Fit"

    return {
        "score": primary_score,
        "verdict": verdict_for(primary_model_name, primary_score),
        "primary_model": primary_model_name,
        "other_model_name": other_model_name,
        "other_model_score": other_score,
        "other_model_verdict": verdict_for(other_model_name, other_score),
        "matched_skills": features["matched_skills"],
        "missing_skills": features["missing_skills"],
        "resume_years": features["resume_years"],
        "jd_years_required": features["jd_years_required"],
    }

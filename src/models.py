"""
models.py — Design component 2: ML Models

Loads the trained models saved by the notebooks and exposes two functions:
failure probability (classification) and remaining useful life (regression).
Contains no alert logic; that belongs to the prediction engine.
"""

from pathlib import Path
import joblib

# Build the path from this file's location, so the app works from any folder
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"


def load_classifier():
    """Load the AI4I failure model, its feature order and its thresholds."""
    model = joblib.load(MODEL_DIR / "rf_failure_model.pkl")
    features = joblib.load(MODEL_DIR / "feature_list.pkl")
    thresholds = joblib.load(MODEL_DIR / "thresholds.pkl")
    return model, features, thresholds


def load_rul_model():
    """Load the C-MAPSS RUL model, its feature order and its config."""
    model = joblib.load(MODEL_DIR / "rul_model.pkl")
    features = joblib.load(MODEL_DIR / "rul_feature_list.pkl")
    config = joblib.load(MODEL_DIR / "rul_config.pkl")
    return model, features, config


def predict_failure_probability(model, X):
    """Return the probability (0 to 1) that each machine will fail."""
    return model.predict_proba(X)[:, 1]


def predict_rul(model, X):
    """Return the predicted remaining useful life (in cycles) per engine row."""
    return model.predict(X)
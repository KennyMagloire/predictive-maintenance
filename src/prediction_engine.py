"""
prediction_engine.py — Design component 3: Prediction Engine

Turns the model's failure probability into what a maintenance team can act on:
a health score, a status (Critical / Warning / Normal) and a recommended action.
"""

import pandas as pd

from src.models import predict_failure_probability

# Advisory only: the system recommends, a person decides. It never acts on equipment.
ACTIONS = {
    "Critical": "Inspect promptly and prioritise for maintenance review.",
    "Warning": "Schedule an inspection and monitor the machine.",
    "Normal": "Continue routine monitoring.",
}


def classify_status(probability, thresholds):
    """Map a failure probability to an alert level using the tuned thresholds."""
    if probability >= thresholds["critical"]:
        return "Critical"
    if probability >= thresholds["warning"]:
        return "Warning"
    return "Normal"


def health_score(probability):
    """Convert failure probability to a 0-100 health score (100 = healthy)."""
    return round((1 - probability) * 100, 1)


def assess(model, X, thresholds):
    """
    Score every machine in X.

    X is the model-ready feature table from data_pipeline.prepare().
    Returns a table with the same index as X, so each result can be matched
    back to its original machine row.
    """
    probabilities = predict_failure_probability(model, X)

    result = pd.DataFrame(index=X.index)
    result["Failure probability"] = probabilities.round(3)
    result["Health score"] = [health_score(p) for p in probabilities]
    result["Status"] = [classify_status(p, thresholds) for p in probabilities]
    result["Action"] = result["Status"].map(ACTIONS)
    return result
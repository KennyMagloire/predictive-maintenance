"""
Unit tests for Design components 2 and 3: the models and the prediction engine.
"""

import numpy as np
import pandas as pd
import pytest

from src.models import load_classifier, load_rul_model, predict_failure_probability, predict_rul
from src.prediction_engine import ACTIONS, assess, classify_status, health_score

THRESHOLDS = {"critical": 0.40, "warning": 0.20}


class FakeModel:
    """A stand-in model that returns fixed probabilities, so the engine can be
    tested on its own without depending on the real trained model."""

    def __init__(self, probabilities):
        self.p = np.array(probabilities)

    def predict_proba(self, X):
        return np.column_stack([1 - self.p, self.p])


# ---------- prediction engine ----------

@pytest.mark.parametrize("probability, expected", [
    (0.00, "Normal"), (0.19, "Normal"),
    (0.20, "Warning"), (0.39, "Warning"),     # 0.20 is the Warning boundary
    (0.40, "Critical"), (1.00, "Critical"),   # 0.40 is the Critical boundary
])
def test_u17_status_at_each_threshold(probability, expected):
    assert classify_status(probability, THRESHOLDS) == expected


@pytest.mark.parametrize("probability, score", [
    (0.0, 100.0), (1.0, 0.0), (0.25, 75.0), (0.4, 60.0),
])
def test_u18_health_score(probability, score):
    assert health_score(probability) == pytest.approx(score)


def test_u19_every_status_has_a_recommended_action():
    assert set(ACTIONS) == {"Critical", "Warning", "Normal"}
    assert all(isinstance(text, str) and text for text in ACTIONS.values())


def test_u20_assess_scores_and_keeps_the_original_row_labels():
    X = pd.DataFrame({"a": [1, 2, 3]}, index=[10, 20, 30])   # labels must survive
    result = assess(FakeModel([0.10, 0.25, 0.90]), X, THRESHOLDS)
    assert list(result.index) == [10, 20, 30]
    assert list(result["Status"]) == ["Normal", "Warning", "Critical"]
    assert list(result["Action"]) == [ACTIONS["Normal"], ACTIONS["Warning"], ACTIONS["Critical"]]
    assert list(result["Health score"]) == pytest.approx([90.0, 75.0, 10.0])


# ---------- saved models ----------

def test_u21_classifier_artefacts_load_and_are_consistent():
    model, features, thresholds = load_classifier()
    assert len(features) == 9
    assert thresholds["critical"] == pytest.approx(0.40)
    assert thresholds["critical"] > thresholds["warning"] > 0


def test_u22_rul_artefacts_load_and_are_consistent():
    model, features, config = load_rul_model()
    assert len(features) == 17
    assert config["rul_cap"] == 125


def test_u23_failure_probability_is_a_valid_probability():
    model, features, _ = load_classifier()
    healthy = {"Air temperature": 298.1, "Process temperature": 308.6,
               "Rotational speed": 1551.0, "Torque": 42.8, "Tool wear": 0.0,
               "Temp difference": 10.5, "Power": 42.8 * 1551 * 2 * np.pi / 60,
               "Overstrain": 0.0, "Type_encoded": 1}
    X = pd.DataFrame([healthy])[features]
    p = predict_failure_probability(model, X)
    assert p.shape == (1,) and 0.0 <= p[0] <= 1.0


def test_u24_rul_prediction_stays_inside_the_trained_range():
    model, features, config = load_rul_model()
    X = pd.DataFrame(np.zeros((3, len(features))), columns=features)
    predictions = predict_rul(model, X)
    assert ((predictions >= 0) & (predictions <= config["rul_cap"])).all()
"""
Integration tests (components working together) and ML-specific tests
(does the MODEL behave, not just the code).

Uses the real dataset and the real saved models.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.data_pipeline import SENSORS, load_dataset, prepare
from src.models import load_classifier, predict_failure_probability
from src.prediction_engine import assess

DATA = Path(__file__).resolve().parent.parent / "data"

HEALTHY = {"Type": "M", "Air temperature": 298.1, "Process temperature": 308.6,
           "Rotational speed": 1551.0, "Torque": 42.8, "Tool wear": 0.0}
OVERSTRAINED = {"Type": "L", "Air temperature": 298.9, "Process temperature": 309.1,
                "Rotational speed": 1410.0, "Torque": 65.7, "Tool wear": 191.0}


# ---------- shared setup (built once per file, then reused) ----------

@pytest.fixture(scope="module")
def classifier():
    return load_classifier()


@pytest.fixture(scope="module")
def ai4i():
    return load_dataset(DATA / "ai4i2020.csv")


@pytest.fixture(scope="module")
def scored(ai4i, classifier):
    model, features, thresholds = classifier
    X, rejected, problems = prepare(ai4i, features)
    return X, rejected, assess(model, X, thresholds)


@pytest.fixture(scope="module")
def holdout(ai4i, classifier):
    """The same stratified 80/20 split used in the notebook (random_state=42)."""
    model, features, thresholds = classifier
    X, _, _ = prepare(ai4i, features)
    y = ai4i.loc[X.index, "Machine failure"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    return X_test, y_test


def run_pipeline(path, classifier):
    model, features, thresholds = classifier
    df = load_dataset(path)
    X, rejected, problems = prepare(df, features)
    return df, X, rejected, problems, assess(model, X, thresholds)


def probabilities_for(classifier, *rows):
    model, features, _ = classifier
    X, _, _ = prepare(pd.DataFrame(list(rows)), features)
    return predict_failure_probability(model, X)


# ---------- INTEGRATION TESTS ----------

def test_i01_full_dataset_end_to_end_golden_numbers(scored):
    X, rejected, result = scored
    counts = result["Status"].value_counts()
    assert len(rejected) == 0
    assert counts["Normal"] == 9651
    assert counts["Critical"] == 329
    assert counts["Warning"] == 20


def test_i02_broken_data_file(classifier):
    df, X, rejected, problems, result = run_pipeline(DATA / "test_broken.csv", classifier)
    assert len(rejected) == 3 and len(X) == 3
    status = dict(zip(df.loc[result.index, "Product ID"], result["Status"]))
    assert status == {"T001": "Normal", "T002": "Critical", "T006": "Critical"}


def test_i03_messy_data_file(classifier):
    df, X, rejected, problems, result = run_pipeline(DATA / "test_messy.csv", classifier)
    assert len(rejected) == 2 and len(X) == 2
    assert any("duplicate" in p for p in problems)
    assert list(df.loc[result.index, "Product ID"]) == ["M001", "M004"]


# ---------- ML-SPECIFIC TESTS ----------

def test_m01_performance_gate_on_held_out_data(holdout, classifier):
    """If the model ever degrades, the build fails."""
    X_test, y_test = holdout
    model, features, thresholds = classifier
    predicted = predict_failure_probability(model, X_test) >= thresholds["critical"]
    precision = precision_score(y_test, predicted)
    recall = recall_score(y_test, predicted)
    f1 = f1_score(y_test, predicted)
    print(f"\nHeld-out: precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
    assert recall >= 0.80
    assert precision >= 0.90


def test_m02_behaves_sensibly_on_clear_cases(classifier):
    """Behavioural test: a healthy machine is Normal; a worn, overloaded one is Critical."""
    _, _, thresholds = classifier
    p_healthy, p_overstrained = probabilities_for(classifier, HEALTHY, OVERSTRAINED)
    assert p_healthy < thresholds["warning"]
    assert p_overstrained >= thresholds["critical"]


def test_m03_same_input_gives_same_output(scored, classifier):
    """Equal to within floating-point rounding, and identical alert statuses.

    Exact equality is the wrong test: the forest predicts in parallel
    (n_jobs=-1), so the order of floating-point additions can differ by ~1e-16.
    """
    model, _, thresholds = classifier
    X, _, _ = scored
    first = predict_failure_probability(model, X.head(200))
    second = predict_failure_probability(model, X.head(200))
    assert np.allclose(first, second, atol=1e-9)
    status_first = first >= thresholds["critical"]
    status_second = second >= thresholds["critical"]
    assert np.array_equal(status_first, status_second)


def test_m04_row_order_does_not_change_a_machines_score(scored, classifier):
    model, _, _ = classifier
    X, _, _ = scored
    sample = X.head(300)
    shuffled = sample.sample(frac=1, random_state=1)
    original = pd.Series(predict_failure_probability(model, sample), index=sample.index)
    reordered = pd.Series(predict_failure_probability(model, shuffled), index=shuffled.index)
    assert np.allclose(original.sort_index(), reordered.sort_index())


def test_m05_small_sensor_noise_rarely_changes_the_alert(ai4i, classifier, scored):
    """Robustness: +/-1% jitter on every sensor should flip very few statuses."""
    model, features, thresholds = classifier
    _, _, baseline = scored
    rng = np.random.default_rng(0)
    noisy = ai4i.copy()
    for column in SENSORS:
        noisy[column] = noisy[column] * (1 + rng.uniform(-0.01, 0.01, len(noisy)))
    X_noisy, _, _ = prepare(noisy, features)
    after = assess(model, X_noisy, thresholds)
    common = baseline.index.intersection(after.index)
    changed = (baseline.loc[common, "Status"] != after.loc[common, "Status"]).mean()
    print(f"\nStatuses changed by 1% noise: {changed:.2%}")
    assert changed < 0.10


def test_m06_failure_mode_columns_are_not_model_inputs(classifier):
    """Target-leakage guard."""
    _, features, _ = classifier
    forbidden = {"TWF", "HDF", "PWF", "OSF", "RNF", "Machine failure"}
    assert forbidden.isdisjoint(features)


def test_m07_saved_models_match_the_training_library_version():
    assert sklearn.__version__ == "1.6.1"


def test_m08_recall_by_machine_type(holdout, classifier, ai4i):
    """Slice test: documents defect DEF-03 (weak recall on Type H)."""
    X_test, y_test = holdout
    model, _, thresholds = classifier
    predicted = predict_failure_probability(model, X_test) >= thresholds["critical"]
    types = ai4i.loc[X_test.index, "Type"].to_numpy()
    for machine_type in ["L", "M", "H"]:
        mask = types == machine_type
        failures = int(y_test.to_numpy()[mask].sum())
        if failures:
            r = recall_score(y_test.to_numpy()[mask], predicted[mask])
            print(f"\nType {machine_type}: recall={r:.2f} ({failures} failures)")
            if machine_type in ("L", "M"):
                assert r >= 0.75
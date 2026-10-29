"""
Unit tests for Design component 1: the data pipeline.

Each test builds a tiny table in memory, so it runs in milliseconds and
needs no files. Test IDs (TC-U..) are used in the report's test case log.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import (clean, validate, engineer_features, prepare)


def make_row(**overrides):
    """One valid machine record. Pass a dict with {'Torque': -5} to break a field."""
    row = {"Product ID": "T001", "Type": "M", "Air temperature": 298.1,
           "Process temperature": 308.6, "Rotational speed": 1551.0,
           "Torque": 42.8, "Tool wear": 0.0}
    row.update(overrides)
    return row


def frame(*rows):
    return pd.DataFrame(list(rows))


# ---------- validate() ----------

def test_u01_valid_record_is_accepted():
    usable, rejected, problems = validate(frame(make_row()))
    assert len(usable) == 1 and len(rejected) == 0 and problems == []


def test_u02_missing_sensor_value_is_rejected():
    usable, rejected, problems = validate(frame(make_row(**{"Torque": np.nan})))
    assert len(usable) == 0 and len(rejected) == 1
    assert any("missing" in p for p in problems)


@pytest.mark.parametrize("field, bad_value", [
    ("Torque", -5.0),                 # negative torque is impossible
    ("Rotational speed", 0.0),        # a stopped machine is not a reading
    ("Air temperature", 500.0),       # beyond any plausible environment
    ("Tool wear", -1.0),
])
def test_u03_impossible_values_are_rejected(field, bad_value):
    usable, rejected, problems = validate(frame(make_row(**{field: bad_value})))
    assert len(rejected) == 1
    assert any("out-of-range" in p for p in problems)


@pytest.mark.parametrize("torque, accepted", [
    (0.1, True), (100.0, True),       # exactly on the limits: accepted
    (0.09, False), (100.1, False),    # just outside: rejected
])
def test_u04_range_boundaries(torque, accepted):
    usable, rejected, _ = validate(frame(make_row(**{"Torque": torque})))
    assert (len(usable) == 1) is accepted


def test_u05_unrecognised_machine_type_is_rejected():
    usable, rejected, problems = validate(frame(make_row(Type="X")))
    assert len(rejected) == 1
    assert any("machine type" in p for p in problems)


def test_u06_one_bad_row_does_not_reject_good_rows():
    good = make_row(**{"Product ID": "OK"})
    bad = make_row(**{"Product ID": "BAD", "Torque": -5.0})
    usable, rejected, _ = validate(frame(good, bad))
    assert list(usable["Product ID"]) == ["OK"]
    assert list(rejected["Product ID"]) == ["BAD"]


# ---------- clean() ----------

def test_u07_clean_fixes_machine_type_spacing_and_case():
    cleaned, _ = clean(frame(make_row(Type=" m ")))
    assert cleaned["Type"].iloc[0] == "M"


def test_u08_clean_removes_exact_duplicates_and_reports_it():
    cleaned, notes = clean(frame(make_row(), make_row()))
    assert len(cleaned) == 1
    assert any("duplicate" in n for n in notes)


def test_u09_clean_turns_text_in_a_sensor_column_into_missing():
    cleaned, notes = clean(frame(make_row(Torque="abc")))
    assert pd.isnull(cleaned["Torque"].iloc[0])
    assert any("non-numeric" in n for n in notes)


def test_u10_clean_never_fills_in_missing_values():
    cleaned, _ = clean(frame(make_row(**{"Torque": np.nan})))
    assert pd.isnull(cleaned["Torque"].iloc[0])      # still missing: no imputation


# ---------- engineer_features() ----------

def test_u11_temperature_difference():
    out = engineer_features(frame(make_row(**{"Air temperature": 300.0,
                                              "Process temperature": 310.0})))
    assert out["Temp difference"].iloc[0] == pytest.approx(10.0)


def test_u12_power_formula_on_a_hand_checkable_case():
    # 10 Nm at 60 rpm: power = torque * speed * 2*pi/60 = 20*pi watts
    out = engineer_features(frame(make_row(**{"Torque": 10.0, "Rotational speed": 60.0})))
    assert out["Power"].iloc[0] == pytest.approx(20 * math.pi)


def test_u13_overstrain_is_tool_wear_times_torque():
    out = engineer_features(frame(make_row(**{"Tool wear": 100.0, "Torque": 10.0})))
    assert out["Overstrain"].iloc[0] == pytest.approx(1000.0)


@pytest.mark.parametrize("machine_type, code", [("L", 0), ("M", 1), ("H", 2)])
def test_u14_type_encoding(machine_type, code):
    out = engineer_features(frame(make_row(Type=machine_type)))
    assert out["Type_encoded"].iloc[0] == code


# ---------- prepare() ----------

def test_u15_prepare_returns_columns_in_the_requested_order():
    wanted = ["Torque", "Power", "Air temperature"]
    X, rejected, problems = prepare(frame(make_row()), wanted)
    assert list(X.columns) == wanted


def test_u16_prepare_returns_none_when_every_row_is_invalid():
    X, rejected, problems = prepare(frame(make_row(**{"Torque": -5.0})), ["Torque"])
    assert X is None and len(rejected) == 1
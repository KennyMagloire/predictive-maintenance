"""
data_pipeline.py — Design component 1: Data Pipeline

Loads equipment data, cleans what can be fixed safely, validates the rest,
and engineers the features the models require.
Implements REQ-F01 to REQ-F04.
"""

import numpy as np
import pandas as pd

SENSORS = ["Air temperature", "Process temperature",
           "Rotational speed", "Torque", "Tool wear"]

ENGINEERED = ["Temp difference", "Power", "Overstrain"]

VALID_TYPES = ["L", "M", "H"]

# Physically valid operating ranges, used to detect sensor faults (resolves DEF-02)
VALID_RANGES = {
    "Air temperature":     (290.0, 310.0),
    "Process temperature": (300.0, 320.0),
    "Rotational speed":    (1000.0, 3000.0),
    "Torque":              (0.1, 100.0),
    "Tool wear":           (0.0, 300.0),
}


def load_dataset(path):
    """
    Load the equipment dataset from a CSV file.

    The raw AI4I file carries units in its column names, for example
    'Air temperature [K]'. They are stripped so that the names match the
    ones the models were trained on.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.replace(r"\s*\[.*?\]", "", regex=True)
    return df


def clean(df):
    """
    Fix only what can be fixed safely. Returns (cleaned, notes).

    - Machine type: stray spaces and lower case are corrected.
    - Sensor columns: text is converted to numbers; anything that is not a
      number becomes 'missing' (it is rejected later, never guessed).
    - Exact duplicate rows are removed.
    Missing values are deliberately NOT filled in: a missing or impossible
    reading is often the sign of a failed sensor, and inventing a value would
    hide it.
    """
    notes = []
    df = df.copy()

    if "Type" in df.columns:
        df["Type"] = df["Type"].astype("string").str.strip().str.upper()

    for col in SENSORS:
        if col in df.columns:
            before = int(df[col].isnull().sum())
            df[col] = pd.to_numeric(df[col], errors="coerce")
            added = int(df[col].isnull().sum()) - before
            if added > 0:
                notes.append(f"{added} non-numeric value(s) in {col} treated as missing")

    duplicates = int(df.duplicated().sum())
    if duplicates > 0:
        df = df.drop_duplicates()
        notes.append(f"{duplicates} duplicate record(s) removed")

    return df, notes


def validate(df):
    """
    Check incoming data for problems before prediction.

    Returns (usable, rejected, problems): the rows that passed, the rows that
    failed, and a list of messages describing what was wrong.
    Resolves DEF-01 (missing values accepted silently) and
    DEF-02 (physically impossible values treated as machine failure).
    """
    problems = []

    missing = df[SENSORS].isnull().any(axis=1)
    if missing.any():
        problems.append(f"{int(missing.sum())} record(s) have missing sensor readings")

    out_of_range = pd.Series(False, index=df.index)
    for col, (low, high) in VALID_RANGES.items():
        if col in df.columns:
            bad = (df[col] < low) | (df[col] > high)
            if bad.any():
                problems.append(f"{int(bad.sum())} record(s) have out-of-range {col}")
            out_of_range |= bad

    bad_type = ~df["Type"].isin(VALID_TYPES)
    if bad_type.any():
        problems.append(f"{int(bad_type.sum())} record(s) have an unrecognised machine type")

    reject = missing | out_of_range | bad_type
    usable = df[~reject].copy()
    rejected = df[reject].copy()
    return usable, rejected, problems


def engineer_features(df):
    """
    Add the three physics-based features derived from the failure mechanisms.
    Implements REQ-F03.
    """
    df = df.copy()
    df["Temp difference"] = df["Process temperature"] - df["Air temperature"]
    df["Power"] = df["Torque"] * df["Rotational speed"] * (2 * np.pi / 60)
    df["Overstrain"] = df["Tool wear"] * df["Torque"]
    df["Type_encoded"] = df["Type"].map({"L": 0, "M": 1, "H": 2})
    return df


def prepare(df, feature_list):
    """
    Full pipeline: clean, validate, engineer features, return model-ready data.

    Returns (features, rejected, problems). 'problems' lists both what was
    cleaned and what was rejected, so nothing happens silently.
    """
    cleaned, notes = clean(df)
    usable, rejected, problems = validate(cleaned)
    problems = notes + problems
    if len(usable) == 0:
        return None, rejected, problems
    prepared = engineer_features(usable)
    return prepared[feature_list], rejected, problems
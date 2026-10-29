"""
storage.py — History store (SQLite)

Saves every scored batch so that readings accumulate over time. This stored
history is the seed of a future retraining dataset: today's scored readings
become tomorrow's historical data.
"""

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data_pipeline import SENSORS

COLUMNS = (["Product ID", "Type"] + SENSORS
           + ["Failure probability", "Health score", "Status"])

TEXT_COLUMNS = {"Product ID", "Type", "Status"}


def _connect(path):
    """Open the database; 'closing' guarantees the file is released afterwards."""
    return closing(sqlite3.connect(path))


def init_db(path):
    """Create the readings table if it does not exist (safe to call repeatedly)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    definitions = ", ".join(
        f'"{name}" {"TEXT" if name in TEXT_COLUMNS else "REAL"}' for name in COLUMNS)
    with _connect(path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS readings ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            f"saved_at TEXT NOT NULL, {definitions})")
        conn.commit()


def save_results(path, results):
    """Append a scored table to the history. Returns the number of rows saved."""
    if len(results) == 0:
        return 0
    saved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    names = ", ".join(f'"{name}"' for name in COLUMNS)
    placeholders = ", ".join("?" for _ in range(len(COLUMNS) + 1))
    rows = [(saved_at,) + row
            for row in results[COLUMNS].itertuples(index=False, name=None)]
    with _connect(path) as conn:
        conn.executemany(
            f"INSERT INTO readings (saved_at, {names}) VALUES ({placeholders})", rows)
        conn.commit()
    return len(rows)


def load_history(path, limit=None):
    """Return the stored readings, newest first."""
    query = "SELECT * FROM readings ORDER BY id DESC"
    params = ()
    if limit is not None:
        query += " LIMIT ?"
        params = (int(limit),)
    with _connect(path) as conn:
        return pd.read_sql_query(query, conn, params=params)


def count_records(path):
    """Return how many readings are stored."""
    with _connect(path) as conn:
        return conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
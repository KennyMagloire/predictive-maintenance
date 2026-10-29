"""
Tests for the history store (SQLite). Written BEFORE the code exists.
Each test uses a temporary database file, so nothing touches real data.
"""

import pandas as pd

from src.data_pipeline import SENSORS
from src.storage import count_records, init_db, load_history, save_results


def make_results(*product_ids):
    """A small scored table shaped like the dashboard's results."""
    rows = []
    for pid in product_ids:
        row = {"Product ID": pid, "Type": "M"}
        row.update({name: 1.0 for name in SENSORS})
        row.update({"Failure probability": 0.5, "Health score": 50.0, "Status": "Critical"})
        rows.append(row)
    return pd.DataFrame(rows)


def test_t01_new_database_is_empty(tmp_path):
    db = tmp_path / "history.db"
    init_db(db)
    assert count_records(db) == 0


def test_t02_init_twice_does_not_lose_data(tmp_path):
    db = tmp_path / "history.db"
    init_db(db)
    save_results(db, make_results("A"))
    init_db(db)
    assert count_records(db) == 1


def test_t03_saving_returns_the_number_of_rows_saved(tmp_path):
    db = tmp_path / "history.db"
    init_db(db)
    assert save_results(db, make_results("A", "B", "C")) == 3
    assert count_records(db) == 3


def test_t04_what_is_saved_can_be_read_back_unchanged(tmp_path):
    db = tmp_path / "history.db"
    init_db(db)
    save_results(db, make_results("A"))
    history = load_history(db)
    assert history["Product ID"].iloc[0] == "A"
    assert history["Status"].iloc[0] == "Critical"
    assert history["Failure probability"].iloc[0] == 0.5


def test_t05_history_accumulates_across_batches(tmp_path):
    db = tmp_path / "history.db"
    init_db(db)
    save_results(db, make_results("A"))
    save_results(db, make_results("B", "C"))
    assert count_records(db) == 3


def test_t06_newest_batch_comes_first(tmp_path):
    db = tmp_path / "history.db"
    init_db(db)
    save_results(db, make_results("OLD"))
    save_results(db, make_results("NEW"))
    assert load_history(db)["Product ID"].iloc[0] == "NEW"


def test_t07_saving_an_empty_table_is_harmless(tmp_path):
    db = tmp_path / "history.db"
    init_db(db)
    assert save_results(db, make_results()) == 0
    assert count_records(db) == 0
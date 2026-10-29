"""Shared test setup: keep the history database out of the real data folder."""

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def isolated_history_db(tmp_path_factory):
    path = tmp_path_factory.mktemp("history") / "history.db"
    os.environ["HISTORY_DB"] = str(path)
    yield
    os.environ.pop("HISTORY_DB", None)
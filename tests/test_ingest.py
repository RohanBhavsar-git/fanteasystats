"""
Tests for src/ingest.py: _retry_transient (Phase 8's fix for a real GitHub
Actions failure: "Connection reset by peer" downloading from
nflverse-data's release CDN on a cold, cache-less CI checkout), and
get_injuries' junk-value filter.
"""

import sys
import time
from pathlib import Path

import pandas as pd
import polars as pl
import pytest
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import src.ingest as ingest_module  # noqa: E402
from src.ingest import _retry_transient  # noqa: E402


def _connection_error_with_404():
    response = requests.Response()
    response.status_code = 404
    http_404 = requests.exceptions.HTTPError(response=response)
    err = ConnectionError("Failed to download ...")
    err.__cause__ = http_404
    return err


def _connection_error_transient():
    return ConnectionError("('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer'))")


def test_retry_transient_succeeds_after_one_transient_failure(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)  # don't actually wait in tests

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise _connection_error_transient()
        return "ok"

    result = _retry_transient(flaky, max_attempts=3, backoff_seconds=0.01)
    assert result == "ok"
    assert calls["n"] == 2


def test_retry_transient_reraises_after_exhausting_attempts(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)

    calls = {"n": 0}

    def always_flaky():
        calls["n"] += 1
        raise _connection_error_transient()

    with pytest.raises(ConnectionError):
        _retry_transient(always_flaky, max_attempts=3, backoff_seconds=0.01)
    assert calls["n"] == 3  # every attempt used, no more


def test_retry_transient_does_not_retry_a_404():
    """
    An HTTP 404 means the resource genuinely doesn't exist (e.g. an
    unpublished season) -- retrying wastes time before failing anyway, and
    src/pipeline.py's _is_unpublished_season_error relies on the 404
    surfacing immediately, not after 3 rounds of backoff.
    """
    calls = {"n": 0}

    def not_found():
        calls["n"] += 1
        raise _connection_error_with_404()

    with pytest.raises(ConnectionError):
        _retry_transient(not_found, max_attempts=3, backoff_seconds=0.01)
    assert calls["n"] == 1  # no retries for a 404


def test_retry_transient_passes_args_and_kwargs_through():
    def add(a, b, c=0):
        return a + b + c

    assert _retry_transient(add, 1, 2, c=3) == 6


def test_get_injuries_strips_whitespace_padded_junk_before_matching(monkeypatch):
    """
    Regression test for a real bug found auditing the Practice Report
    feature for predictiveness (2026): the junk-value filter matched a
    bare "\\n" exactly, but the actual junk value present in nflverse's
    real data is "\\n" plus trailing spaces, which slid through
    unfiltered into 212 real practice_status rows -- a column the
    Practice Report tab displays directly. Fixed by stripping before
    comparing, which also catches any other whitespace-only variant, not
    just this one exact string. Pinned with both the padded and bare
    forms so a future refactor can't quietly go back to an exact-match
    comparison.
    """
    fake_raw = pl.DataFrame({
        "gsis_id": ["00-0001", "00-0002", "00-0003", "00-0004"],
        "season": [2099, 2099, 2099, 2099],
        "week": [1, 1, 1, 1],
        "report_status": ["Questionable", "\n    ", "Out", None],
        "practice_status": [
            "Full Participation in Practice", "\n    ", "\n", "Did Not Participate In Practice",
        ],
    })

    monkeypatch.setattr(ingest_module.nfl, "load_injuries", lambda seasons: fake_raw)
    monkeypatch.setattr(ingest_module, "_read_cache_parquet", lambda name: None)
    monkeypatch.setattr(ingest_module, "_write_cache_parquet", lambda df, name: None)

    out = ingest_module.get_injuries([2099], refresh=True)

    assert pd.isna(out.loc[1, "report_status"])   # "\n    " (padded) -- the variant that slipped through before
    assert pd.isna(out.loc[1, "practice_status"])  # same row, same padded variant
    assert pd.isna(out.loc[2, "practice_status"])  # bare "\n" -- already worked before this fix
    assert out.loc[0, "report_status"] == "Questionable"  # a real value survives unchanged
    assert out.loc[3, "practice_status"] == "Did Not Participate In Practice"

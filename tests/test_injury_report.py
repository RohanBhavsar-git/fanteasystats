"""
Tests for src/injury_report.py -- synthetic data only, no network calls.
Covers the scoping rules (season/game_type/position), the crosswalk join
for sleeper_id, and the JSON-round-trip-safety fix (pandas silently
reverting None back to NaN on an all-null float64 column, which produced
a literal non-JSON NaN token in a real committed export before this was
caught -- see PROJECT_CONTEXT.md's Practice Report findings).
"""

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.injury_report import PRACTICE_REPORT_POSITIONS, build_practice_report_export, build_weekly_practice_report  # noqa: E402


def _injuries_frame():
    return pd.DataFrame([
        # A real, well-formed skill-position REG-week row.
        {
            "season": 2025, "week": 1, "game_type": "REG", "gsis_id": "00-0001",
            "position": "QB", "team": "KC", "full_name": "Player One",
            "report_status": "Questionable", "report_primary_injury": "Knee", "report_secondary_injury": pd.NA,
            "practice_status": "Limited Participation in Practice", "practice_primary_injury": "Knee",
            "practice_secondary_injury": pd.NA,
        },
        # Same week, no gsis_id -> crosswalk match at all (a real, disclosed gap).
        {
            "season": 2025, "week": 1, "game_type": "REG", "gsis_id": "00-0002",
            "position": "RB", "team": "SF", "full_name": "Player Two",
            "report_status": pd.NA, "report_primary_injury": pd.NA, "report_secondary_injury": pd.NA,
            "practice_status": "Full Participation in Practice", "practice_primary_injury": "Ankle",
            "practice_secondary_injury": pd.NA,
        },
        # Out of position scope (IDP) -- must be dropped.
        {
            "season": 2025, "week": 1, "game_type": "REG", "gsis_id": "00-0003",
            "position": "LB", "team": "SF", "full_name": "Player Three",
            "report_status": "Out", "report_primary_injury": "Ankle", "report_secondary_injury": pd.NA,
            "practice_status": "Did Not Participate In Practice", "practice_primary_injury": "Ankle",
            "practice_secondary_injury": pd.NA,
        },
        # Playoff game_type -- a fantasy week never covers this; must be dropped.
        {
            "season": 2025, "week": 19, "game_type": "WC", "gsis_id": "00-0004",
            "position": "WR", "team": "BUF", "full_name": "Player Four",
            "report_status": "Doubtful", "report_primary_injury": "Hamstring", "report_secondary_injury": pd.NA,
            "practice_status": "Limited Participation in Practice", "practice_primary_injury": "Hamstring",
            "practice_secondary_injury": pd.NA,
        },
        # Different season -- must be dropped when scoping to 2025.
        {
            "season": 2024, "week": 1, "game_type": "REG", "gsis_id": "00-0001",
            "position": "QB", "team": "KC", "full_name": "Player One",
            "report_status": pd.NA, "report_primary_injury": pd.NA, "report_secondary_injury": pd.NA,
            "practice_status": "Full Participation in Practice", "practice_primary_injury": pd.NA,
            "practice_secondary_injury": pd.NA,
        },
    ])


def _crosswalk_frame():
    return pd.DataFrame([
        {"gsis_id": "00-0001", "sleeper_id": "1111"},
        # 00-0002 deliberately absent -- the real "no crosswalk match" case.
    ])


def test_scopes_to_season_game_type_and_position():
    by_week = build_weekly_practice_report(_injuries_frame(), _crosswalk_frame(), season=2025)
    assert list(by_week.keys()) == ["1"]
    names = {r["name"] for r in by_week["1"]}
    assert names == {"Player One", "Player Two"}  # LB and WC-week rows dropped, 2024 row dropped


def test_crosswalk_join_and_null_gap_disclosed():
    by_week = build_weekly_practice_report(_injuries_frame(), _crosswalk_frame(), season=2025)
    by_name = {r["name"]: r for r in by_week["1"]}
    assert by_name["Player One"]["sleeper_id"] == "1111"
    assert by_name["Player Two"]["sleeper_id"] is None  # no crosswalk match -- null, not dropped


def test_empty_season_returns_empty_dict():
    assert build_weekly_practice_report(_injuries_frame(), _crosswalk_frame(), season=2099) == {}


def test_export_is_json_round_trip_safe_with_no_literal_nan():
    """
    Regression test for the real bug this feature shipped with once:
    pandas silently reverts a float64 column's None back to NaN when every
    value in that column is null within a given week's group (e.g.
    report_secondary_injury, rarely populated at all) -- json.dumps then
    writes a literal `NaN` token, and re-parsing it produces a float NaN
    that fails `NaN == NaN`, breaking the write-then-verify round trip
    scripts/weekly_update.py and scripts/archive_season.py both rely on.
    """
    export = build_practice_report_export(_injuries_frame(), _crosswalk_frame(), season=2025, source_updated_at="2026-01-01T00:00:00Z")
    serialized = json.dumps(export)
    assert "NaN" not in serialized
    reparsed = json.loads(serialized)
    assert reparsed == export


def test_position_scope_excludes_team_defense_and_idp():
    assert "DEF" not in PRACTICE_REPORT_POSITIONS
    assert "LB" not in PRACTICE_REPORT_POSITIONS

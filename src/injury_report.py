"""
FanTeasy Stats -- Practice Report.

Wraps nflverse's official weekly injury/practice report (src/ingest.py's
get_injuries) into a per-week export block for the dashboard's Practice
Report tab. Deliberately independent of the projection pipeline
(assemble_player_advanced_stats) -- this data needs no predictions/usage/
xfp merge, it's just a real report, so it's assigned onto the export
payload directly as its own top-level key by the two build scripts,
the same way they already bolt on `simulation`/`kicker_stats`.

Two things worth stating plainly, both checked directly against the real
data before writing this module (see notebooks/08_practice_report.ipynb):

1. This is NOT the same information as the Injury tab / Lineup Risks
   panel, which read Sleeper's own `injury_status` (a roster-level flag --
   Out/Doubtful/Questionable, but also IR/PUP/Suspended, none of which are
   weekly-report concepts at all). Cross-referencing real rostered players
   in this league found real, frequent disagreement in BOTH directions:
   players nflverse's official report already lists Questionable/Doubtful/
   Out while Sleeper's injury_status still reads healthy (Sleeper hasn't
   caught up to this week's report yet), and players Sleeper flags
   Questionable with no current nflverse report_status at all (Sleeper
   flagging something the official report doesn't currently carry). PUP/IR
   is its own case: a player Sleeper marks PUP can still show up on a real
   weekly practice report (e.g. "Limited Participation") as they work back
   -- information the static PUP tag alone doesn't carry. These are two
   real, complementary signals, not the same thing under two names, so
   this tab exists alongside the Injury tab rather than replacing it.

2. Practice participation is measurably predictive, not just informational
   -- see notebooks/08_practice_report.ipynb for the full walk-forward-style
   check across 2018-2025. Two headline numbers: players who Did Not
   Participate in practice that week went on to play in only ~16% of those
   weeks (vs. ~84% for Full Participation); and CONDITIONAL on playing
   anyway, a Limited-participation week produced a real fantasy-points
   discount (~94-95% of that player's own healthy-week median, confirmed
   independently via snap-share) while a DNP-but-played week's discount was
   roughly triple that (~83-95%). The OFFICIAL report_status designation is
   the sharper "will they play at all" signal (Out/Doubtful essentially
   never play; Questionable is a real coin flip); practice_status adds the
   complementary "how are they trending" read. This finding is reported,
   not wired into the projection model -- CLAUDE.md's model scope stays
   tabular regression on real usage/efficiency features, not a practice-
   report input.
"""

from __future__ import annotations

import pandas as pd

# Individual defensive/offensive-line players show up on the same weekly
# report but aren't fantasy-relevant in this standard-format league (no
# IDP scoring) -- same skill-position framing the Players tab's own
# position filter uses (K included, team DEF excluded: a report entry
# describes an individual player, and a team defense unit never appears
# on an individual practice report).
PRACTICE_REPORT_POSITIONS = ["QB", "RB", "WR", "TE", "K"]


def build_weekly_practice_report(
    injuries: pd.DataFrame,
    crosswalk: pd.DataFrame,
    season: int,
) -> dict:
    """
    One entry per real REG week that has at least one report -- {week_str:
    [record, ...]}. Scoped to `season` and PRACTICE_REPORT_POSITIONS, and to
    game_type == 'REG' (a fantasy league's schedule -- including its own
    playoff weeks -- always runs on the NFL's REGULAR season weeks; the NFL's
    own postseason, game_type WC/DIV/CON/SB, is never a fantasy week here).

    Each record carries BOTH report_status/report_primary_injury (the
    official game designation, when the player has one) and practice_status/
    practice_primary_injury (that week's practice participation) rather than
    collapsing to a single "status" -- see this module's docstring for why
    they occasionally diverge in a way worth seeing directly (e.g. an
    illness-driven Questionable layered on a separately-tracked physical
    injury), and this project's non-negotiable against presenting derived
    values with more confidence than the source data supports.

    `sleeper_id` is attached via the crosswalk where a match exists and left
    null otherwise (a real, disclosed gap -- not every player on a real NFL
    practice report has a DynastyProcess crosswalk entry, most often a
    practice-squad body with no fantasy footprint at all) -- index.html
    still shows the row with its real gsis_id-sourced name/position/team,
    just without rostered-league matching or a Player Detail link.

    Returns {} if `injuries` has zero rows for this season (e.g. a brand
    new season before its first weekly report has posted) -- an honestly
    empty report, not a missing key, so the dashboard's own "no report yet"
    empty state has something real to check against.
    """
    season_rows = injuries[
        (injuries["season"] == season)
        & (injuries["game_type"] == "REG")
        & (injuries["position"].isin(PRACTICE_REPORT_POSITIONS))
    ].copy()
    if season_rows.empty:
        return {}

    cw = crosswalk.dropna(subset=["gsis_id", "sleeper_id"]).drop_duplicates(subset=["gsis_id"])
    season_rows = season_rows.merge(cw[["gsis_id", "sleeper_id"]], on="gsis_id", how="left")

    by_week: dict[str, list[dict]] = {}
    record_cols = [
        "sleeper_id", "gsis_id", "full_name", "position", "team",
        "report_status", "report_primary_injury", "report_secondary_injury",
        "practice_status", "practice_primary_injury", "practice_secondary_injury",
    ]
    for week, group in season_rows.groupby("week"):
        records = group[record_cols].rename(columns={"full_name": "name"})
        # .astype(object) FIRST -- a column that's entirely null within this
        # one week's group (report_secondary_injury most often) stays
        # float64, and pandas silently reverts a float64 column's NaN
        # back from None to NaN on assignment; object dtype holds None
        # natively, so the JSON export gets real `null`, not a literal
        # non-JSON `NaN` token that also breaks Python's own re-parse
        # equality (NaN != NaN) on the write-verify round trip below.
        records = records.astype(object).where(pd.notna(records), None)
        by_week[str(int(week))] = records.to_dict(orient="records")

    return by_week


def build_practice_report_export(
    injuries: pd.DataFrame,
    crosswalk: pd.DataFrame,
    season: int,
    source_updated_at: str | None,
) -> dict:
    """
    Full `practice_report` top-level export block: the per-week data from
    build_weekly_practice_report plus the honest freshness/scope metadata
    the dashboard needs to present this as weekly (not daily) data that can
    lag the live report -- see get_injuries_source_updated_at's docstring
    for why `source_updated_at` is file-level, not per-row, and why that's
    the honest ceiling here rather than a gap to paper over.
    """
    return {
        "position_scope": PRACTICE_REPORT_POSITIONS,
        "source_updated_at": source_updated_at,
        "weeks": build_weekly_practice_report(injuries, crosswalk, season),
    }

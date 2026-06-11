"""
team_import.py — the credential-free "paste sink".

A user fetches their own FIFA team in their own browser (the bookmarklet, or by
opening /api/en/fantasy/team while logged in) and hands us only the resulting
JSON. We parse it into a canonical squad and run the rolling-captaincy analysis.
No session cookie ever reaches the server — we only ever receive player ids.

Standalone and not wired to Flask yet. The eventual web route will call
parse_team() on the posted blob, then captain.analyze_squad_ids().

CLI test harness:
    pbpaste | python team_import.py          # macOS, with a team JSON on the clipboard
    python team_import.py < team.json

Dependencies: pandas (only in the analysis glue); the parser itself is stdlib.
"""

import json
import sys

POSITIONS = ("GK", "DEF", "MID", "FWD")

PROJECTIONS_CSV = "data/projections.csv"
FIXTURES_CSV = "data/processed/fixtures.csv"


class TeamParseError(ValueError):
    """Raised when a pasted blob isn't a recognisable FIFA team payload.
    Carries a plain, user-facing message (safe to show in the UI later)."""


# ─────────────────────────────────────────────────────────────────────────────
# Parsing  (pure stdlib — trivially unit-testable)
# ─────────────────────────────────────────────────────────────────────────────

def parse_team(blob):
    """Parse a pasted team blob into a canonical squad dict:

        {team_id, captain, vice,
         lineup:   {GK: [...], DEF: [...], MID: [...], FWD: [...]},
         bench:    [ids, in bench order],
         starters: [ids, the XI],
         player_ids: [starters + bench]}

    Accepts a dict or a JSON string, and either the raw team object
    ({"id":..,"lineup":..}) or the API envelope ({"success": {...}, "errors": [...]}).
    Lenient about whitespace. Raises TeamParseError with a plain message on
    anything unrecognisable.

    Note: all 15 players are captain candidates, so callers running the analysis
    should use `player_ids`. A "bench" player isn't out of contention — the same
    mid-round flexibility that powers rolling captaincy lets you sub a bench
    player into the XI before their match and armband them (swap them in for a
    compatible starter; the formation stays valid). `starters`/`bench` are kept
    only for display/formation purposes.
    """
    data = _coerce_to_dict(blob)

    # Unwrap the API envelope if present.
    if isinstance(data.get("success"), dict):
        errors = data.get("errors") or []
        if errors and not data["success"]:
            raise TeamParseError(f"Team payload reported an error: {_first_error(errors)}")
        data = data["success"]

    lineup_raw = data.get("lineup")
    if not isinstance(lineup_raw, dict):
        raise TeamParseError("No 'lineup' found — is this a FIFA team payload?")

    lineup, starters = {}, []
    for pos in POSITIONS:
        ids = [_as_id(x) for x in (lineup_raw.get(pos) or [])]
        lineup[pos] = ids
        starters.extend(ids)

    # Bench: prefer the ordered benchOrder; fall back to the bench-by-position dict.
    bench = [_as_id(x) for x in (data.get("benchOrder") or [])]
    if not bench and isinstance(data.get("bench"), dict):
        for pos in POSITIONS:
            bench.extend(_as_id(x) for x in (data["bench"].get(pos) or []))

    if not starters:
        raise TeamParseError("Parsed a lineup but found no starting players.")

    return {
        "team_id": data.get("id"),
        "captain": _as_id_or_none(data.get("captain")),
        "vice": _as_id_or_none(data.get("vice")),
        "lineup": lineup,
        "bench": bench,
        "starters": starters,
        "player_ids": starters + bench,
    }


def _coerce_to_dict(blob):
    if isinstance(blob, dict):
        return blob
    if isinstance(blob, (bytes, bytearray)):
        blob = blob.decode("utf-8", "replace")
    if not isinstance(blob, str):
        raise TeamParseError(f"Unsupported input type: {type(blob).__name__}")
    text = blob.strip()
    if not text:
        raise TeamParseError("Empty input — paste your FIFA team JSON.")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise TeamParseError(f"That isn't valid JSON ({e.msg} at position {e.pos}).")


def _as_id(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        raise TeamParseError(f"Player id is not a whole number: {x!r}")


def _as_id_or_none(x):
    return None if x is None else _as_id(x)


def _first_error(errors):
    e = errors[0]
    return e.get("message", "unknown error") if isinstance(e, dict) else str(e)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis glue  (pulls in pandas + captain)
# ─────────────────────────────────────────────────────────────────────────────

def captaincy_from_blob(blob, projections_csv=PROJECTIONS_CSV, fixtures_csv=FIXTURES_CSV):
    """End-to-end: pasted blob -> (parsed_team, [per-round analysis]). Loads the
    projections/fixtures from disk (standalone use). The web route will instead
    pass its in-memory projections to captain.analyze_squad_ids directly."""
    import pandas as pd
    import captain

    team = parse_team(blob)
    proj = pd.read_csv(projections_csv)
    fixtures = pd.read_csv(fixtures_csv)
    # All 15 are captain candidates (bench players can be subbed into the XI
    # mid-round and armbanded) — so analyse player_ids, not just the XI.
    rounds = captain.analyze_squad_ids(team["player_ids"], proj, fixtures)
    return team, rounds


def main():
    import captain

    team, rounds = captaincy_from_blob(sys.stdin.read())
    print(f"Parsed team #{team['team_id']}: {len(team['starters'])} starters, "
          f"{len(team['bench'])} bench. captain={team['captain']} vice={team['vice']}\n")
    for rd in rounds:
        captain.print_round_table(rd)


if __name__ == "__main__":
    main()

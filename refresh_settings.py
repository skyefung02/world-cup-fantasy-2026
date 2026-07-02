"""
refresh_settings.py — sync wc_settings.json to your live FIFA fantasy team.

Run this after a round rolls over (or any time your team changes) so the solver
optimises FROM your actual squad instead of building a fresh one. It fetches your
team via the FIFA_SID cookie (.env), then rewrites the mutable fields of
wc_settings.json in place:

    initial_squad   ← your 15 current player ids (lineup + bench)
    ft              ← freeTransfers reported by FIFA
    itb             ← stage budget − current squad value (see note below)
    used_wildcard / used_twelfth_man / used_max_captain / used_qual_booster
                    ← derived from which boosters FIFA shows as already played

It does NOT touch next_round — you set that to whichever round you want to solve.

ITB note: FIFA doesn't expose your bank balance in the team payload, and it
tracks selling prices separately from current prices (which the solver can't
see). So itb is computed as (stage budget − squad value at CURRENT prices),
which is exact only if no player prices have moved. Check the value it prints
against the ITB shown in the FIFA app and pass --itb <value> to override.

Usage:
    python refresh_settings.py                 # refresh from live team
    python refresh_settings.py --itb 1.5       # force a specific bank value
    python refresh_settings.py --dry-run       # print changes, don't write
"""

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

import fifa_team
import team_import
from wc_solver import BUDGET, KNOCKOUT_BUDGET, SETTINGS_USER_PATH, load_settings

PROJECTIONS_FALLBACK = "data/projections.csv"


def derived_used_flags(team_payload: dict) -> dict:
    """Map FIFA's booster fields to the solver's used_* flags.

    FIFA records a played booster as a non-null round id (wildCard, qualification)
    or a populated dict (twelfthMan). maxCaptain stays None until played.

    Max Captain is a deliberate exception: it carries no scoring term in the
    solver objective (unlike WC/12th Man/Qual Booster), so letting use_mc float
    free just risks a meaningless MC tag in the output. We always force it to 1
    ("used" = banned) so the solver ignores it, regardless of the API state.
    """
    tm = team_payload.get("twelfthMan") or {}
    return {
        "used_wildcard":     int(team_payload.get("wildCard") is not None),
        "used_twelfth_man":  int(bool(tm.get("playerId"))),
        "used_max_captain":  1,  # unmodeled booster — always suppress (see docstring)
        "used_qual_booster": int(team_payload.get("qualification") is not None),
    }


def compute_itb(squad_ids: list[int], budget: float, projection_file: str) -> tuple[float, float]:
    """Return (itb, squad_value) using current prices from the projection CSV."""
    proj = pd.read_csv(projection_file).set_index("id")
    missing = [pid for pid in squad_ids if pid not in proj.index]
    if missing:
        raise SystemExit(
            f"Player id(s) {missing} not found in {projection_file}. "
            f"Refresh projections first, or check the team payload."
        )
    squad_value = float(proj.loc[squad_ids, "price"].sum())
    return round(budget - squad_value, 1), round(squad_value, 1)


def write_settings_inplace(path: Path, updates: dict) -> None:
    """Apply `updates` to the user settings JSON, preserving key order and keeping
    the initial_squad list on a single line for readability."""
    with open(path) as f:
        settings = json.load(f)          # dict preserves insertion order (py3.7+)

    for k, v in updates.items():
        settings[k] = v

    text = json.dumps(settings, indent=4, ensure_ascii=False)
    # Collapse the multi-line initial_squad array back onto one line.
    text = re.sub(
        r'("initial_squad":\s*)\[[^\]]*\]',
        lambda m: m.group(1) + "[" + ", ".join(str(i) for i in settings["initial_squad"]) + "]",
        text,
    )
    with open(path, "w") as f:
        f.write(text + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync wc_settings.json to your live FIFA team")
    parser.add_argument("--itb", type=float, default=None,
                        help="Override the computed bank value (£m)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the changes without writing the file")
    args = parser.parse_args()

    load_dotenv(os.path.join(os.getcwd(), ".env"))
    sid = os.environ.get("FIFA_SID")
    if not sid:
        raise SystemExit("FIFA_SID is not set in .env — add it and retry.")

    # Merged settings tell us the target round, stage rounds, and projection file.
    options = load_settings()
    next_round = int(options.get("next_round", 1))
    group_stage_rounds = set(options.get("group_stage_rounds", [1, 2, 3]))
    projection_file = options.get("projection_file", PROJECTIONS_FALLBACK)

    payload = fifa_team.fetch_team(sid)
    team = team_import.parse_team(payload)
    squad_ids = team["player_ids"]
    if len(squad_ids) != 15:
        raise SystemExit(f"Expected 15 players, got {len(squad_ids)} — aborting.")

    budget = BUDGET if next_round in group_stage_rounds else KNOCKOUT_BUDGET
    itb, squad_value = compute_itb(squad_ids, budget, projection_file)
    if args.itb is not None:
        itb = args.itb

    updates = {
        "initial_squad": squad_ids,
        "ft": int(payload.get("freeTransfers", 1)),
        "itb": itb,
        **derived_used_flags(payload),
    }

    print(f"FIFA team #{team['team_id']}  |  solving round {next_round}  "
          f"(budget £{budget:.0f}m, squad value £{squad_value:.1f}m)")
    print(json.dumps(updates, indent=4, ensure_ascii=False))
    if args.itb is None:
        print(f"\nitb computed from current prices (£{budget:.0f}m − £{squad_value:.1f}m). "
              f"Verify against the FIFA app; use --itb to override.")

    if args.dry_run:
        print("\n--dry-run: wc_settings.json not modified.")
        return

    write_settings_inplace(SETTINGS_USER_PATH, updates)
    print(f"\nUpdated {SETTINGS_USER_PATH}.")


if __name__ == "__main__":
    main()

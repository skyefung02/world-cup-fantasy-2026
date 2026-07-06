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

Eliminated holds: when a team is knocked out, build_projections drops its players
(FIFA removes them from the live feed), but you still HOLD any you own as dead
0-point assets. refresh backfills a zero-point placeholder row into projections.csv
for each such held player — priced from a persistent price cache (data/price_cache.csv,
which refresh keeps up to date) or, as a bootstrap, the newest solver result that
still listed them — so the solver holds the dead asset and plans to sell it for one
transfer. Use --dead-price ID=VALUE to override a recovered price.

ITB note: FIFA doesn't expose your bank balance in the team payload, and it
tracks selling prices separately from current prices (which the solver can't
see). So itb is computed as (stage budget − squad value at CURRENT prices),
which is exact only if no player prices have moved. Check the value it prints
against the ITB shown in the FIFA app and pass --itb <value> to override.

Usage:
    python refresh_settings.py                 # refresh from live team
    python refresh_settings.py --itb 1.5       # force a specific bank value
    python refresh_settings.py --dead-price 173=10.0   # price an eliminated hold
    python refresh_settings.py --dry-run       # print changes, don't write
"""

import argparse
import glob
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

# Persistent {id → price/meta} snapshot. build_projections drops players whose
# team is eliminated (FIFA strips them from the live feed), but you still HOLD
# them as dead assets. We remember their last-known price here so refresh can
# rebuild a zero-point placeholder row for the solver. See reconcile_projections.
PRICE_CACHE_PATH = Path("data/price_cache.csv")
RESULTS_DIR      = "data/results"
META_COLS        = ["id", "player", "position", "price", "team", "abbr"]


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


def _meta_from_row(pid: int, row) -> dict:
    """Extract the META_COLS record from a projections/results/cache row."""
    return {
        "id":       pid,
        "player":   row.get("player", row.get("name", f"id{pid}")),
        "position": row["position"],
        "price":    float(row["price"]),
        "team":     row.get("team", ""),
        "abbr":     row.get("abbr", ""),
    }


def load_price_cache() -> dict[int, dict]:
    """Read the persistent price snapshot into {id → meta}. Missing file → {}."""
    if not PRICE_CACHE_PATH.exists():
        return {}
    cache = pd.read_csv(PRICE_CACHE_PATH)
    return {int(r["id"]): _meta_from_row(int(r["id"]), r) for _, r in cache.iterrows()}


def save_price_cache(cache: dict[int, dict]) -> None:
    pd.DataFrame([cache[k] for k in sorted(cache)], columns=META_COLS).to_csv(
        PRICE_CACHE_PATH, index=False)


def newest_results_lookup(pid: int) -> dict | None:
    """Recover a player's last-known meta from the most recent solver result CSV
    that still contains them — the bootstrap source when the price cache predates
    the player's elimination (or doesn't exist yet)."""
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "wc_*_iter0.csv")), reverse=True):
        try:
            res = pd.read_csv(f)
        except Exception:
            continue
        hit = res[res["id"] == pid]
        if len(hit):
            return _meta_from_row(pid, hit.iloc[0])
    return None


def _zero_row(proj: pd.DataFrame, meta: dict) -> dict:
    """A projections row with every numeric column 0 and text columns blank, then
    the identity/price fields from `meta`. Zero xMins/Pts/PPlay everywhere makes the
    solver treat the player as a worthless held asset it will transfer out."""
    row = {c: (0 if pd.api.types.is_numeric_dtype(proj[c]) else "") for c in proj.columns}
    row.update({k: meta[k] for k in META_COLS if k in proj.columns})
    return row


def reconcile_projections(squad_ids: list[int], projection_file: str,
                          dead_prices: dict[int, float], dry_run: bool) -> pd.DataFrame:
    """Ensure every held player has a projections row, then return the (possibly
    patched) projections DataFrame.

    1. Upsert all currently-active players into the price cache, so a player's
       price is remembered for future rounds even after their team is eliminated.
    2. For any held player missing from projections (their team went out), rebuild
       a zero-point placeholder row using price/meta from the cache, or — as a
       bootstrap — the newest solver result that still listed them. A --dead-price
       entry overrides the recovered price.

    Writes projections.csv and the cache in place unless dry_run.
    """
    proj    = pd.read_csv(projection_file)
    present = set(proj["id"].astype(int))
    cache   = load_price_cache()

    for _, r in proj.iterrows():                       # (1) remember active prices
        cache[int(r["id"])] = _meta_from_row(int(r["id"]), r)

    injected, unresolved = [], []
    for pid in (p for p in squad_ids if p not in present):   # (2) held but dropped
        meta = cache.get(pid) or newest_results_lookup(pid)
        if meta is None:
            unresolved.append(pid)
            continue
        meta = dict(meta)
        if pid in dead_prices:
            meta["price"] = dead_prices[pid]
        proj = pd.concat([proj, pd.DataFrame([_zero_row(proj, meta)])], ignore_index=True)
        injected.append(meta)

    if unresolved:
        raise SystemExit(
            f"Held player id(s) {unresolved} are missing from {projection_file} with no "
            f"cached or historical price (never refreshed while they were active?). "
            f"Pass --dead-price ID=VALUE for each, e.g. --dead-price {unresolved[0]}=10.0"
        )

    for m in injected:
        print(f"⚠ {m['player']} ({m['team'] or '?'}) eliminated — held as "
              f"£{float(m['price']):.1f}m dead asset (0 pts); solver will plan to sell it.")

    if injected:
        proj = proj.sort_values("id").reset_index(drop=True)
    if not dry_run:
        proj.to_csv(projection_file, index=False)
        save_price_cache(cache)
    return proj


def compute_itb(proj: pd.DataFrame, squad_ids: list[int], budget: float) -> tuple[float, float]:
    """Return (itb, squad_value) using current prices from the projection frame.
    Assumes reconcile_projections has already backfilled any eliminated holds."""
    priced = proj.set_index("id")
    missing = [pid for pid in squad_ids if pid not in priced.index]
    if missing:  # reconcile_projections should have prevented this
        raise SystemExit(f"Player id(s) {missing} still unpriced after reconciliation.")
    squad_value = float(priced.loc[squad_ids, "price"].sum())
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
    parser.add_argument("--dead-price", action="append", default=[], metavar="ID=VALUE",
                        help="Override the price of a held-but-eliminated player, e.g. "
                             "--dead-price 173=10.0 (repeatable)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the changes without writing any files")
    args = parser.parse_args()

    dead_prices: dict[int, float] = {}
    for entry in args.dead_price:
        try:
            pid, val = entry.split("=")
            dead_prices[int(pid)] = float(val)
        except ValueError:
            raise SystemExit(f"Bad --dead-price '{entry}' — expected ID=VALUE, e.g. 173=10.0")

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
    proj = reconcile_projections(squad_ids, projection_file, dead_prices, args.dry_run)
    itb, squad_value = compute_itb(proj, squad_ids, budget)
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

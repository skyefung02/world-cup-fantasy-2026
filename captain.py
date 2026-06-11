"""
captain.py

Optimal *rolling captaincy* for the group stage, one round at a time.

The game lets you move the armband to a not-yet-kicked-off match after earlier
matches have finished. So within a round the kickoffs form a sequence of
"blocks" (sets of games starting at the same time), and at each block you can
either KEEP the armband on that block's best player (locking in their realised
score) or ROLL it forward to a later block. That is an optimal-stopping problem,
solved here by backward induction.

For each round independently:
  * Group your squad into kickoff blocks (games sharing a kickoff time), ordered
    chronologically.
  * Each block's candidate captain = your highest-projected player in that block.
  * Work backwards. The last block's value U is just its candidate's projection
    (you must keep). For an earlier block k:
        U[k] = E[max(X_k, U[k+1])]
    where X_k is the captain's realised score and U[k+1] is the value of rolling.
  * The keep/roll rule at block k is: keep if realised score >= U[k+1], else roll.

FIRST PASS modelling assumption: X_k ~ Poisson(mu), mu = candidate's projected
points. This is a rough approximation — fantasy points have a floor (a starter
rarely scores ~0), a heavier upside tail, and variance != mean. The expectation
is isolated in captain_score_expectation() so it can later be swapped for a
component Monte Carlo (sampling goals/assists/clean-sheet/etc. from the per-round
columns already in projections.csv) without touching the backward induction.

Dependencies: pandas only (the Poisson survival function is computed in stdlib,
so this stays light and matches whatever env serves the Flask app).
"""

import math

import pandas as pd

# ── Your 15-player squad: list of FIFA player IDs (see /my-team for yours) ──
MY_SQUAD = [757, 742, 1086, 542, 1229, 1366, 543, 529, 855, 38, 1236, 423, 1709, 270, 256]

PROJECTIONS_CSV = "data/projections.csv"
FIXTURES_CSV = "data/processed/fixtures.csv"
ROUNDS = [1, 2, 3]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def squad_records_from_df(df, squad_ids, warn=True):
    """Filter a projections DataFrame to the squad and return rows as dicts.
    Used by both the CLI (df from projections.csv) and the website (df from the
    app's in-memory projections). Warns on ids not found — projections only
    contain 'playing' players, so an injured/dropped pick can silently vanish."""
    found = df[df["id"].isin(squad_ids)].copy()
    if warn:
        missing = [pid for pid in squad_ids if pid not in set(found["id"])]
        if missing:
            print(f"⚠ {len(missing)} squad id(s) not found in projections "
                  f"(injured/dropped?): {missing}\n")
    return found.to_dict("records")


def load_squad(squad_ids):
    """CLI convenience: load projections.csv and filter to the squad."""
    return squad_records_from_df(pd.read_csv(PROJECTIONS_CSV), squad_ids)


def team_kickoffs(fixtures, rnd):
    """Map team abbr -> kickoff datetime string for a given round."""
    sub = fixtures[fixtures["round_id"] == rnd]
    ko = {}
    for _, row in sub.iterrows():
        ko[row["home_abbr"]] = row["date"]
        ko[row["away_abbr"]] = row["date"]
    return ko


def fmt_kickoff(date_str):
    """ISO timestamp -> compact readable label, e.g. 'Thu 11 Jun 20:00'."""
    return pd.to_datetime(date_str).strftime("%a %d %b %H:%M")


# ─────────────────────────────────────────────────────────────────────────────
# Block construction
# ─────────────────────────────────────────────────────────────────────────────

def build_blocks(squad, kickoffs, rnd):
    """Group squad players by their team's kickoff time for this round, return a
    list of blocks ordered chronologically. Each block is a dict:
        {"kickoff": <iso str>, "players": [player_dict, ...]}
    Players whose team isn't playing this round are skipped with a warning."""
    by_kickoff = {}
    for p in squad:
        ko = kickoffs.get(p["abbr"])
        if ko is None:
            print(f"⚠ {p['player']} ({p['abbr']}) has no fixture in round {rnd} — skipped.")
            continue
        by_kickoff.setdefault(ko, []).append(p)

    blocks = [{"kickoff": ko, "players": players} for ko, players in by_kickoff.items()]
    blocks.sort(key=lambda b: pd.to_datetime(b["kickoff"]))
    return blocks


def block_candidate(block, rnd):
    """The block's captain candidate: highest projected player for this round."""
    return max(block["players"], key=lambda p: p[f"{rnd}_Pts"])


# ─────────────────────────────────────────────────────────────────────────────
# Expectation kernel  ── swap this out for component Monte Carlo later ──
# ─────────────────────────────────────────────────────────────────────────────

def _poisson_sf(n, mu):
    """Survival function P(X > n) for X ~ Poisson(mu), n an integer. Pure stdlib —
    computed by summing the CDF iteratively, which is exact and fast for the small
    means (~5-10) we deal with here. Matches scipy.stats.poisson.sf, including
    sf(n<0) = 1.0."""
    if n < 0:
        return 1.0
    term = math.exp(-mu)  # i = 0 term of the pmf
    cdf = term
    for i in range(1, n + 1):
        term *= mu / i
        cdf += term
    return max(0.0, 1.0 - cdf)


def captain_score_expectation(candidate, rnd, threshold):
    """Expected value of KEEPING the armband on this candidate, given that rolling
    is worth `threshold`. Returns E[max(X, threshold)] for the captain's realised
    score X.

    FIRST PASS: X ~ Poisson(mu), mu = candidate's projected points. Closed form
    (verified to equal E[max(X, threshold)] exactly, integer thresholds included):
        m = int(threshold) + 1
        E = threshold + mu*sf(m-2, mu) - threshold*sf(m-1, mu)

    TODO: replace the body with a component Monte Carlo over the per-round
    projections.csv columns (xG, xA, PCleanSheet, AppPts, ...) and return
    mean(max(sim_total, threshold)). The backward induction is agnostic to how
    this expectation is produced.
    """
    mu = candidate[f"{rnd}_Pts"]
    m = int(threshold) + 1
    return threshold + mu * _poisson_sf(m - 2, mu) - threshold * _poisson_sf(m - 1, mu)


def keep_probability(mu, threshold):
    """P(realised score >= threshold) for X ~ Poisson(mu) — i.e. the chance this
    block ends the roll. Inclusive at integer thresholds (uses ceil)."""
    return _poisson_sf(math.ceil(threshold) - 1, mu)


def backward_induction(blocks, rnd):
    """Returns (candidates, U) where candidates[k] is block k's captain candidate
    and U[k] is the value of being at block k and playing optimally onward."""
    candidates = [block_candidate(b, rnd) for b in blocks]
    n = len(candidates)
    U = [0.0] * n

    U[n - 1] = candidates[n - 1][f"{rnd}_Pts"]  # last block: must keep
    for k in range(n - 2, -1, -1):
        U[k] = captain_score_expectation(candidates[k], rnd, threshold=U[k + 1])
    return candidates, U


def analyze_round(squad, fixtures, rnd):
    """Run the full per-round analysis and return a JSON-friendly dict for both
    the CLI printer and the web template. Shape:

        {round, blocks: [...], rolling_ev, static: {player, value}|None,
         uplift, uplift_pct, single_block}

    Each block: {index, kickoff_label, player, team, abbr, position, proj,
                 threshold|None, keep_prob|None, roll_to|None, is_start, is_last}
    """
    ptcol = f"{rnd}_Pts"
    blocks = build_blocks(squad, team_kickoffs(fixtures, rnd), rnd)
    out = {"round": rnd, "blocks": [], "rolling_ev": None,
           "static": None, "uplift": 0.0, "uplift_pct": 0.0, "single_block": False}
    if not blocks:
        return out

    candidates, U = backward_induction(blocks, rnd)
    n = len(blocks)
    for k, (block, cand) in enumerate(zip(blocks, candidates)):
        is_last = (k == n - 1)
        threshold = None if is_last else U[k + 1]
        out["blocks"].append({
            "index": k + 1,
            "kickoff": block["kickoff"],            # raw ISO — used to match fixture finality
            "kickoff_label": fmt_kickoff(block["kickoff"]),
            "id": cand["id"],                       # candidate player id — for realized lookup
            "player": cand["player"],
            "team": cand["team"],
            "abbr": cand["abbr"],
            "position": cand["position"],
            "proj": round(float(cand[ptcol]), 2),
            "value": round(float(U[k]), 2),         # U[k] = value of playing from this block on
            "threshold": None if is_last else round(float(threshold), 2),
            "keep_prob": None if is_last else round(keep_probability(cand[ptcol], threshold) * 100, 1),
            "roll_to": None if is_last else k + 2,
            "is_start": k == 0,
            "is_last": is_last,
        })

    out["rolling_ev"] = round(float(U[0]), 2)
    if n == 1:
        out["single_block"] = True
        return out

    static_idx = max(range(n), key=lambda i: candidates[i][ptcol])
    sc = candidates[static_idx]
    sv = float(sc[ptcol])
    out["static"] = {"player": sc["player"], "value": round(sv, 2)}
    out["uplift"] = round(float(U[0]) - sv, 2)
    out["uplift_pct"] = round((float(U[0]) - sv) / sv * 100, 1) if sv else 0.0
    return out


def analyze_squad_ids(squad_ids, projections_df, fixtures, rounds=ROUNDS):
    """Run analyze_round for each round given a list of player ids plus the
    projections/fixtures frames. Shared core for any caller — the CLI, the paste
    sink (team_import), and the eventual web route all funnel through here so the
    captaincy logic lives in exactly one place."""
    squad = squad_records_from_df(projections_df, squad_ids, warn=False)
    return [analyze_round(squad, fixtures, rnd) for rnd in rounds]


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def print_round_table(rd):
    """Print the decision table for one analyze_round() result."""
    print(f"=== Round {rd['round']} Captain Decision Table ===")
    if not rd["blocks"]:
        print("No squad players have a fixture this round.\n")
        return

    first = rd["blocks"][0]
    print(f"Start armband on: {first['player']} ({first['team']}) "
          f"— projected {first['proj']:.2f} pts\n")

    for b in rd["blocks"]:
        print(f"Block {b['index']} — {b['kickoff_label']}")
        print(f"  Captain: {b['player']} ({b['team']}) — {b['proj']:.2f} pts")
        if b["is_last"]:
            label = "only kickoff block this round" if rd["single_block"] else "last block"
            print(f"  Keep always ({label})")
        else:
            print(f"  Keep if score >= {b['threshold']:.2f} | else roll to Block "
                  f"{b['roll_to']}  (keep {b['keep_prob']:.0f}%)")
        print()

    if rd["single_block"]:
        print("Only one kickoff block this round — rolling captaincy has no value.")
        print(f"Expected captain points: {rd['rolling_ev']:.2f}\n")
        return

    print(f"Expected captain points (optimal rolling): {rd['rolling_ev']:.2f}")
    print(f"vs. static best captain ({rd['static']['player']}): {rd['static']['value']:.2f}")
    print(f"Uplift from rolling: {rd['uplift']:+.2f} pts ({rd['uplift_pct']:+.1f}%)\n")


def main():
    squad = load_squad(MY_SQUAD)
    fixtures = pd.read_csv(FIXTURES_CSV)
    for rnd in ROUNDS:
        print_round_table(analyze_round(squad, fixtures, rnd))


if __name__ == "__main__":
    main()

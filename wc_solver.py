"""
wc_solver.py — World Cup Fantasy 2026 ILP squad optimiser
Solver backend: sasoptpy (model builder) + highspy (HiGHS MIP solver)

Adapted from open-fpl-solver (github.com/sertalpbilal/FPL-Optimized)
Key differences from the FPL solver:
  - No FPL API calls — data loaded from local projection CSV
  - Max 2 players per country (vs. 3 per club in FPL)
  - Simplified transfer system (configurable FTs + hit cost)
  - WC-specific chips: Power Play, No Limits, Limitless
  - No price-change / 50% profit rule
  - No season-specific special cases (AFCON etc.)

⚠️  All WC-specific constants are marked with [WC-RULE].
    Verify against official 2026 rules before running for real.
"""

import csv
import datetime
import json
import os
from pathlib import Path

import highspy
import numpy as np
import pandas as pd
import sasoptpy as so

# ── WC Fantasy 2026 constants ─────────────────────────────────────────────────
# [WC-RULE] Verify all of these before tournament launch

SQUAD_SIZE       = 15
LINEUP_SIZE      = 11
BUDGET           = 100.0    # £100m starting budget
MAX_PER_COUNTRY  = 2        # max players from the same national team  [WC-RULE]

# Squad composition by position  [WC-RULE]
SQUAD_QUOTA = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}

# Minimum / maximum starters per position (valid formations)
LINEUP_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
LINEUP_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}

# Chips  [WC-RULE] — names and behaviour need confirming for 2026
#   power_play : captain's points count 3× instead of 2× (like Triple Captain)
#   no_limits  : all 15 players score (like Bench Boost)
#   limitless  : pick any squad for one round, no budget/country limit (like Free Hit)
CHIP_NAMES = ["power_play", "no_limits", "limitless"]

POSITIONS        = list(SQUAD_QUOTA.keys())   # ["GK", "DEF", "MID", "FWD"]
BENCH_ORDER      = [0, 1, 2, 3]               # slot 0 = GK bench slot
BINARY_THRESHOLD = 0.5


# ── Data loading ──────────────────────────────────────────────────────────────

def load_projection_data(filepath: str | Path) -> pd.DataFrame:
    """
    Load the per-round projection CSV.

    Expected columns:
        id, name, position, price, team, abbr,
        1_Pts, 1_xMins, 2_Pts, 2_xMins, 3_Pts, 3_xMins, ...

    Returns a DataFrame indexed by integer player id.
    """
    df = pd.read_csv(filepath)

    required = {"id", "name", "position", "price", "team"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Projection CSV is missing columns: {missing}")

    df["id"] = df["id"].astype(int)
    df = df.set_index("id")
    return df


def prep_wc_data(options: dict) -> dict:
    """
    Load and validate projection data; build the data dict consumed by solve_wc().
    """
    projection_path = options.get("projection_file", "data/wc_projections.csv")
    df = load_projection_data(projection_path)

    # Solve window
    next_round = int(options.get("next_round", 1))
    horizon    = int(options.get("horizon", 3))
    rounds     = list(range(next_round, next_round + horizon))

    # Validate columns
    for r in rounds:
        for suffix in ["_Pts", "_xMins"]:
            col = f"{r}{suffix}"
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' missing from projection file. "
                    f"Check your horizon ({horizon}) and next_round ({next_round})."
                )

    # Derived totals used for player-pool filtering
    df["total_ev"]   = df[[f"{r}_Pts"   for r in rounds]].sum(axis=1)
    df["total_mins"] = df[[f"{r}_xMins" for r in rounds]].sum(axis=1)

    # Filter: unavailable players
    n_before = len(df)
    if "status" in df.columns:
        df = df[df["status"].str.lower().isin(["playing", "available", ""])]

    # Filter: minimum expected minutes across the horizon
    xmin_lb = options.get("xmin_lb", 0)
    if xmin_lb > 0:
        # Always keep players the user has explicitly locked or already owns
        safe_ids = set(options.get("initial_squad", []) + options.get("locked", []))
        df = df[(df["total_mins"] >= xmin_lb) | df.index.isin(safe_ids)]

    print(f"Player pool: {n_before} → {len(df)} players after filtering")

    buy_price     = df["price"].to_dict()          # raw float, e.g. 10.5 = £10.5m
    initial_squad = [int(i) for i in options.get("initial_squad", [])]
    itb           = float(options.get("itb", BUDGET))
    initial_ft    = int(options.get("ft", 1))       # free transfers available now  [WC-RULE]

    return {
        "df":             df,
        "rounds":         rounds,
        "next_round":     next_round,
        "buy_price":      buy_price,
        "initial_squad":  initial_squad,
        "itb":            itb,
        "ft":             initial_ft,
    }


# ── Core ILP model ────────────────────────────────────────────────────────────

def solve_wc(data: dict, options: dict) -> list[dict]:
    """
    Build and solve the WC Fantasy ILP for the configured round window.

    Returns a list of solution dicts, each with:
        picks       — DataFrame (one row per player × round in the solution)
        score       — objective value (use for ranking when num_iterations > 1)
        total_xp    — raw lineup xP sum
        summary     — human-readable action string
        statistics  — per-round stats dict
        buy / sell  — transfer names for next_round
        chip        — chip(s) used across the horizon
        iter        — iteration index
    """

    # ── Options ───────────────────────────────────────────────────────────────
    horizon        = int(options.get("horizon", 3))
    objective      = options.get("objective", "decay")        # "decay" | "regular"
    decay_base     = float(options.get("decay_base", 0.85))
    bench_weights  = {int(k): v for k, v in options.get(
                        "bench_weights", {0: 0.03, 1: 0.21, 2: 0.06, 3: 0.002}).items()}
    hit_cost       = float(options.get("hit_cost", 4))        # pts per extra transfer  [WC-RULE]
    ft_value       = float(options.get("ft_value", 1.5))      # value of saving a FT
    vcap_weight    = float(options.get("vcap_weight", 0.1))
    itb_value      = float(options.get("itb_value", 0.0))
    num_iterations = int(options.get("num_iterations", 1))
    chip_limits    = options.get("chip_limits", {
        "power_play": 1, "no_limits": 1, "limitless": 1,
    })

    # ── Data ──────────────────────────────────────────────────────────────────
    df            = data["df"]
    rounds        = data["rounds"]
    next_round    = data["next_round"]
    buy_price     = data["buy_price"]
    initial_squad = data["initial_squad"]
    itb           = data["itb"]
    initial_ft    = data["ft"]

    players    = df.index.tolist()
    teams      = df["team"].unique().tolist()
    all_rounds = [next_round - 1, *rounds]   # index t-1 used for initial state

    player_pos  = df["position"].to_dict()
    player_team = df["team"].to_dict()
    points_pw   = {(p, r): float(df.loc[p, f"{r}_Pts"])   for p in players for r in rounds}
    mins_pw     = {(p, r): float(df.loc[p, f"{r}_xMins"]) for p in players for r in rounds}

    # ── Model ─────────────────────────────────────────────────────────────────
    model = so.Model(name=f"wc_h{horizon}_{objective[0]}{decay_base}")

    # ── Decision variables ─────────────────────────────────────────────────────

    # Core squad / selection
    squad    = model.add_variables(players, all_rounds, name="squad",   vartype=so.binary)
    squad_ll = model.add_variables(players, rounds,     name="sq_ll",   vartype=so.binary)
    lineup   = model.add_variables(players, rounds,     name="lineup",  vartype=so.binary)
    captain  = model.add_variables(players, rounds,     name="captain", vartype=so.binary)
    vicecap  = model.add_variables(players, rounds,     name="vicecap", vartype=so.binary)
    bench    = model.add_variables(players, rounds, BENCH_ORDER, name="bench", vartype=so.binary)

    # Transfers + budget
    transfer_in         = model.add_variables(players, rounds, name="tr_in",  vartype=so.binary)
    transfer_out        = model.add_variables(players, rounds, name="tr_out", vartype=so.binary)
    in_the_bank         = model.add_variables(all_rounds, name="itb",  vartype=so.continuous, lb=0)
    penalized_transfers = model.add_variables(rounds,     name="pt",   vartype=so.integer, lb=0)
    ft_var              = model.add_variables(all_rounds, name="ft",   vartype=so.integer, lb=0, ub=5)

    # Chips
    # use_pp is per-player (only the captain carries it), others are per-round
    use_pp = model.add_variables(players, rounds, name="use_pp", vartype=so.binary)
    use_nl = model.add_variables(rounds,          name="use_nl", vartype=so.binary)
    use_ll = model.add_variables(rounds,          name="use_ll", vartype=so.binary)

    # Derived expressions
    use_pp_gw      = {r: so.expr_sum(use_pp[p, r] for p in players) for r in rounds}
    squad_count    = {r: so.expr_sum(squad[p, r]    for p in players) for r in rounds}
    squad_ll_count = {r: so.expr_sum(squad_ll[p, r] for p in players) for r in rounds}
    num_transfers  = {r: so.expr_sum(transfer_out[p, r] for p in players) for r in rounds}
    bought_amount  = {r: so.expr_sum(buy_price[p] * transfer_in[p, r]  for p in players) for r in rounds}
    sold_amount    = {r: so.expr_sum(buy_price[p] * transfer_out[p, r] for p in players) for r in rounds}

    lineup_pos = {
        (pos, r): so.expr_sum(lineup[p, r] for p in players if player_pos[p] == pos)
        for pos in POSITIONS for r in rounds
    }
    squad_pos = {
        (pos, r): so.expr_sum(squad[p, r] for p in players if player_pos[p] == pos)
        for pos in POSITIONS for r in rounds
    }
    squad_ll_pos = {
        (pos, r): so.expr_sum(squad_ll[p, r] for p in players if player_pos[p] == pos)
        for pos in POSITIONS for r in rounds
    }

    # ── Initial conditions ─────────────────────────────────────────────────────
    if initial_squad:
        model.add_constraints(
            (squad[p, next_round - 1] == (1 if p in initial_squad else 0) for p in players),
            name="initial_squad_state",
        )
    else:
        # Preseason: no existing squad — all previous-state vars are zero
        model.add_constraints(
            (squad[p, next_round - 1] == 0 for p in players),
            name="preseason_empty_prev",
        )

    model.add_constraint(in_the_bank[next_round - 1] == itb, name="initial_itb")
    model.add_constraint(ft_var[next_round] == initial_ft,   name="initial_ft")

    # ── Squad size constraints ─────────────────────────────────────────────────
    model.add_constraints(
        (squad_count[r] == SQUAD_SIZE for r in rounds),
        name="squad_size",
    )
    # Limitless squad must also be 15 if the chip is active, else 0
    model.add_constraints(
        (squad_ll_count[r] == SQUAD_SIZE * use_ll[r] for r in rounds),
        name="squad_ll_size",
    )

    # ── Lineup size (all 15 score when No Limits active) ──────────────────────
    model.add_constraints(
        (
            so.expr_sum(lineup[p, r] for p in players)
            == LINEUP_SIZE + (SQUAD_SIZE - LINEUP_SIZE) * use_nl[r]
            for r in rounds
        ),
        name="lineup_size",
    )

    # ── Bench slots (empty when No Limits active) ──────────────────────────────
    model.add_constraints(
        (
            so.expr_sum(bench[p, r, 0] for p in players if player_pos[p] == "GK")
            == 1 - use_nl[r]
            for r in rounds
        ),
        name="bench_gk",
    )
    model.add_constraints(
        (
            so.expr_sum(bench[p, r, o] for p in players) == 1 - use_nl[r]
            for r in rounds for o in [1, 2, 3]
        ),
        name="bench_outfield",
    )

    # ── Captain / vicecap ──────────────────────────────────────────────────────
    model.add_constraints(
        (so.expr_sum(captain[p, r] for p in players) == 1 for r in rounds),
        name="one_captain",
    )
    model.add_constraints(
        (so.expr_sum(vicecap[p, r] for p in players) == 1 for r in rounds),
        name="one_vicecap",
    )
    model.add_constraints(
        (captain[p, r] <= lineup[p, r] for p in players for r in rounds),
        name="captain_in_lineup",
    )
    model.add_constraints(
        (vicecap[p, r] <= lineup[p, r] for p in players for r in rounds),
        name="vicecap_in_lineup",
    )
    model.add_constraints(
        (captain[p, r] + vicecap[p, r] <= 1 for p in players for r in rounds),
        name="cap_vc_distinct",
    )

    # Power Play: only the captain can carry it (so we get 3× instead of 2×)
    model.add_constraints(
        (use_pp[p, r] <= captain[p, r] for p in players for r in rounds),
        name="pp_captain_only",
    )

    # ── Lineup ↔ squad relationships ───────────────────────────────────────────
    # On normal rounds: must be in regular squad to start or bench
    model.add_constraints(
        (lineup[p, r] <= squad[p, r] + use_ll[r] for p in players for r in rounds),
        name="lineup_from_squad",
    )
    model.add_constraints(
        (
            bench[p, r, o] <= squad[p, r] + use_ll[r]
            for p in players for r in rounds for o in BENCH_ORDER
        ),
        name="bench_from_squad",
    )
    # On Limitless rounds: must be in the limitless squad instead
    model.add_constraints(
        (lineup[p, r] <= squad_ll[p, r] + 1 - use_ll[r] for p in players for r in rounds),
        name="lineup_from_sq_ll",
    )
    model.add_constraints(
        (
            bench[p, r, o] <= squad_ll[p, r] + 1 - use_ll[r]
            for p in players for r in rounds for o in BENCH_ORDER
        ),
        name="bench_from_sq_ll",
    )
    # Can't be in lineup AND on bench simultaneously
    model.add_constraints(
        (
            lineup[p, r] + so.expr_sum(bench[p, r, o] for o in BENCH_ORDER) <= 1
            for p in players for r in rounds
        ),
        name="lineup_bench_exclusive",
    )

    # ── Formation constraints ──────────────────────────────────────────────────
    for pos in POSITIONS:
        model.add_constraints(
            (lineup_pos[pos, r] >= LINEUP_MIN[pos] for r in rounds),
            name=f"formation_lb_{pos}",
        )
        # No Limits unlocks the upper bound (all 15 play, formation matters less)
        model.add_constraints(
            (lineup_pos[pos, r] <= LINEUP_MAX[pos] + use_nl[r] for r in rounds),
            name=f"formation_ub_{pos}",
        )
        model.add_constraints(
            (squad_pos[pos, r] == SQUAD_QUOTA[pos] for r in rounds),
            name=f"squad_quota_{pos}",
        )
        model.add_constraints(
            (squad_ll_pos[pos, r] == SQUAD_QUOTA[pos] * use_ll[r] for r in rounds),
            name=f"squad_ll_quota_{pos}",
        )

    # ── Country limits ─────────────────────────────────────────────────────────
    model.add_constraints(
        (
            so.expr_sum(squad[p, r] for p in players if player_team[p] == t)
            <= MAX_PER_COUNTRY
            for t in teams for r in all_rounds
        ),
        name="country_limit",
    )
    model.add_constraints(
        (
            so.expr_sum(squad_ll[p, r] for p in players if player_team[p] == t)
            <= MAX_PER_COUNTRY * use_ll[r]
            for t in teams for r in rounds
        ),
        name="country_limit_ll",
    )

    # ── Transfer continuity ────────────────────────────────────────────────────
    model.add_constraints(
        (
            squad[p, r] == squad[p, r - 1] + transfer_in[p, r] - transfer_out[p, r]
            for p in players for r in rounds
        ),
        name="squad_continuity",
    )
    model.add_constraints(
        (transfer_in[p, r] + transfer_out[p, r] <= 1 for p in players for r in rounds),
        name="no_in_and_out_same_round",
    )
    # No transfers during Limitless (regular squad carries over unchanged)
    model.add_constraints(
        (transfer_in[p, r]  <= 1 - use_ll[r] for p in players for r in rounds),
        name="no_tr_in_ll",
    )
    model.add_constraints(
        (transfer_out[p, r] <= 1 - use_ll[r] for p in players for r in rounds),
        name="no_tr_out_ll",
    )

    # ── Budget continuity ──────────────────────────────────────────────────────
    model.add_constraints(
        (
            in_the_bank[r] == in_the_bank[r - 1] + sold_amount[r] - bought_amount[r]
            for r in rounds
        ),
        name="budget_continuity",
    )
    # Limitless budget: must afford the limitless squad from current bank + squad value
    model.add_constraints(
        (
            so.expr_sum(buy_price[p] * squad[p, r - 1] for p in players)
            + in_the_bank[r - 1]
            >= so.expr_sum(buy_price[p] * squad_ll[p, r] for p in players)
            for r in rounds
        ),
        name="limitless_budget",
    )

    # ── Free transfer tracking ─────────────────────────────────────────────────
    # [WC-RULE] The exact FT rules for WC 2026 need confirming.
    # Current model: 1 FT per round that doesn't roll over. Extra transfers cost
    # hit_cost points each. Set hit_cost=0 for unlimited free transfers.
    #
    # For a fuller rolling-FT model (like FPL's 1→5 state machine), see the
    # open-fpl-solver — adapt if the official rules support rollover.
    m = 20  # big-M for FT floor/cap constraints

    # ft_above[r] = 1  iff  raw_ft_next[r] > 5
    ft_above = model.add_variables(rounds, name="ft_above", vartype=so.binary)
    # ft_below[r] = 1  iff  raw_ft_next[r] <= 0
    ft_below = model.add_variables(rounds, name="ft_below", vartype=so.binary)

    raw_ft_next = {r: ft_var[r] - num_transfers[r] + 1 for r in rounds}

    for r in rounds:
        # Penalize transfers beyond the FT allowance
        model.add_constraint(
            penalized_transfers[r] >= num_transfers[r] - ft_var[r],
            name=f"hit_calc_{r}",
        )

        if r + 1 in rounds:
            # Clamp raw_ft_next into [1, 5] for the next round
            model.add_constraint(raw_ft_next[r] >= 6 - m * (1 - ft_above[r]), name=f"fta_lb_{r}")
            model.add_constraint(raw_ft_next[r] <= 5 + m * ft_above[r],       name=f"fta_ub_{r}")
            model.add_constraint(raw_ft_next[r] <= 0 + m * (1 - ft_below[r]), name=f"ftb_ub_{r}")
            model.add_constraint(raw_ft_next[r] >= 1 - m * ft_below[r],       name=f"ftb_lb_{r}")

            # above cap → next ft = 5
            model.add_constraint(ft_var[r + 1] <= 5 + m * (1 - ft_above[r]), name=f"ft_cap_ub_{r}")
            model.add_constraint(ft_var[r + 1] >= 5 - m * (1 - ft_above[r]), name=f"ft_cap_lb_{r}")
            # below floor → next ft = 1
            model.add_constraint(ft_var[r + 1] <= 1 + m * (1 - ft_below[r]), name=f"ft_flr_ub_{r}")
            model.add_constraint(ft_var[r + 1] >= 1 - m * (1 - ft_below[r]), name=f"ft_flr_lb_{r}")
            # in range → next ft = raw value
            model.add_constraint(
                ft_var[r + 1] - raw_ft_next[r] <= m * (ft_above[r] + ft_below[r]),
                name=f"ft_inrange_ub_{r}",
            )
            model.add_constraint(
                raw_ft_next[r] - ft_var[r + 1] <= m * (ft_above[r] + ft_below[r]),
                name=f"ft_inrange_lb_{r}",
            )

    # ── Chip limits + mutual exclusion ─────────────────────────────────────────
    model.add_constraints(
        (use_pp_gw[r] + use_nl[r] + use_ll[r] <= 1 for r in rounds),
        name="one_chip_per_round",
    )
    model.add_constraint(
        so.expr_sum(use_pp_gw[r] for r in rounds) <= chip_limits.get("power_play", 1),
        name="pp_limit",
    )
    model.add_constraint(
        so.expr_sum(use_nl[r] for r in rounds) <= chip_limits.get("no_limits", 1),
        name="nl_limit",
    )
    model.add_constraint(
        so.expr_sum(use_ll[r] for r in rounds) <= chip_limits.get("limitless", 1),
        name="ll_limit",
    )
    # squad_ll vars must be zero when Limitless isn't active
    model.add_constraints(
        (squad_ll[p, r] <= use_ll[r] for p in players for r in rounds),
        name="ll_squad_scope",
    )

    # ── Optional user constraints ──────────────────────────────────────────────
    if options.get("banned"):
        banned = [p for p in options["banned"] if p in players]
        model.add_constraints(
            (so.expr_sum(squad[p, r] for r in rounds) == 0 for p in banned),
            name="banned_players",
        )

    if options.get("locked"):
        locked = [p for p in options["locked"] if p in players]
        model.add_constraints(
            (squad[p, r] + squad_ll[p, r] == 1 for p in locked for r in rounds),
            name="locked_players",
        )

    if options.get("num_transfers") is not None:
        model.add_constraint(
            so.expr_sum(transfer_in[p, next_round] for p in players)
            == int(options["num_transfers"]),
            name="forced_transfer_count",
        )

    if options.get("force_chip"):
        # e.g. {"limitless": 2, "power_play": 1}  → force chip in that round
        for chip, rnd in options["force_chip"].items():
            if chip == "power_play":
                model.add_constraint(use_pp_gw[rnd] == 1, name=f"force_pp_{rnd}")
            elif chip == "no_limits":
                model.add_constraint(use_nl[rnd] == 1,    name=f"force_nl_{rnd}")
            elif chip == "limitless":
                model.add_constraint(use_ll[rnd] == 1,    name=f"force_ll_{rnd}")

    # ── Objective ──────────────────────────────────────────────────────────────
    # Scoring:
    #   lineup player  → 1× points
    #   captain        → +1× (total 2×)
    #   power_play     → +1× on top of captain (total 3×)
    #   vicecap        → vcap_weight × points (e.g. 0.1 for VC coverage)
    #   bench slot o   → bench_weights[o] × points (expected sub probability)
    gw_xp = {
        r: so.expr_sum(
            points_pw[p, r] * (
                lineup[p, r]
                + captain[p, r]
                + use_pp[p, r]
                + vcap_weight * vicecap[p, r]
                + so.expr_sum(bench_weights[o] * bench[p, r, o] for o in BENCH_ORDER)
            )
            for p in players
        )
        for r in rounds
    }

    # FT value: reward for having more free transfers available (saved option value)
    gw_ft_gain = {
        r: ft_value * (ft_var[r] - ft_var.get(r - 1, ft_var[next_round]))
        for r in rounds
    }

    gw_total = {
        r: gw_xp[r]
           - hit_cost * penalized_transfers[r]
           + itb_value * in_the_bank[r]
        for r in rounds
    }

    if objective == "regular":
        obj = so.expr_sum(gw_total[r] for r in rounds)
    else:  # "decay": weight nearer rounds more heavily
        obj = so.expr_sum(
            gw_total[r] * pow(decay_base, r - next_round) for r in rounds
        )

    model.set_objective(-obj, sense="N", name="wc_obj")

    # ── Solve loop (supports num_iterations > 1 for diverse solutions) ─────────
    solutions = []

    for iteration in range(num_iterations):
        tmp_dir = Path("tmp")
        tmp_dir.mkdir(exist_ok=True)
        mps_path = tmp_dir / f"wc_h{horizon}_i{iteration}.mps"
        model.export_mps(str(mps_path))

        secs    = int(options.get("secs", 300))
        gap     = float(options.get("gap", 0.0))
        verbose = bool(options.get("verbose", False))

        h = highspy.Highs()
        h.readModel(str(mps_path))
        h.setOptionValue("parallel",       "on")
        h.setOptionValue("time_limit",     secs)
        h.setOptionValue("mip_rel_gap",    gap)
        h.setOptionValue("log_to_console", verbose)
        h.run()

        sol = h.getSolution()
        for idx, var in enumerate(model.get_variables()):
            var.set_value(sol.col_value[idx])

        if options.get("delete_tmp", True):
            try:
                mps_path.unlink()
            except Exception:
                pass

        # ── Extract result picks ───────────────────────────────────────────────
        picks = []
        for r in rounds:
            for p in players:
                in_sq   = squad[p, r].get_value() + squad_ll[p, r].get_value()
                is_out  = transfer_out[p, r].get_value()
                if in_sq + is_out < BINARY_THRESHOLD:
                    continue

                row        = df.loc[p]
                is_lineup  = int(lineup[p, r].get_value()  > BINARY_THRESHOLD)
                is_cap     = int(captain[p, r].get_value() > BINARY_THRESHOLD)
                is_vc      = int(vicecap[p, r].get_value() > BINARY_THRESHOLD)
                is_pp      = int(use_pp[p, r].get_value()  > BINARY_THRESHOLD)
                is_tr_in   = int(transfer_in[p, r].get_value()  > BINARY_THRESHOLD)
                is_tr_out  = int(is_out > BINARY_THRESHOLD)
                is_ll_rnd  = int(use_ll[r].get_value() > BINARY_THRESHOLD)
                is_nl_rnd  = int(use_nl[r].get_value() > BINARY_THRESHOLD)

                bench_slot = -1
                for o in BENCH_ORDER:
                    if bench[p, r, o].get_value() > BINARY_THRESHOLD:
                        bench_slot = o

                chip_text  = ("PP" if is_pp else "NL" if is_nl_rnd else "LL" if is_ll_rnd else "")
                multiplier = is_lineup + is_cap + is_pp   # 1, 2, or 3

                picks.append({
                    "id":           p,
                    "round":        r,
                    "name":         row["name"],
                    "position":     row["position"],
                    "team":         row["team"],
                    "abbr":         row.get("abbr", ""),
                    "price":        buy_price[p],
                    "xP":           round(points_pw[p, r], 2),
                    "xMins":        mins_pw[p, r],
                    "squad":        int(in_sq > BINARY_THRESHOLD),
                    "lineup":       is_lineup,
                    "bench":        bench_slot,
                    "captain":      is_cap,
                    "vicecaptain":  is_vc,
                    "power_play":   is_pp,
                    "transfer_in":  is_tr_in,
                    "transfer_out": is_tr_out,
                    "multiplier":   multiplier,
                    "xp_contrib":   round(points_pw[p, r] * multiplier, 2),
                    "chip":         chip_text,
                    "ft":           round(ft_var[r].get_value()),
                    "iter":         iteration,
                })

        picks_df = pd.DataFrame(picks).sort_values(
            by=["round", "squad", "lineup", "bench", "position"],
            ascending=[True, False, False, True, True],
        )

        # ── Summary string ─────────────────────────────────────────────────────
        summary, move_summary = _build_summary(
            picks_df, rounds, next_round,
            in_the_bank, ft_var, penalized_transfers, num_transfers,
            use_ll, use_nl, use_pp_gw,
        )

        total_xp  = picks_df["xp_contrib"].sum()
        obj_value = -model.get_objective_value()

        # Per-round statistics
        statistics = {}
        for r in rounds:
            lineup_rows = picks_df[(picks_df["round"] == r) & (picks_df["lineup"] == 1)]
            chip_rows   = picks_df[(picks_df["round"] == r) & (picks_df["chip"] != "")]
            chip_str    = chip_rows["chip"].iloc[0] if not chip_rows.empty else ""
            statistics[r] = {
                "itb":  round(in_the_bank[r].get_value(), 2),
                "ft":   round(ft_var[r].get_value()),
                "pt":   round(penalized_transfers[r].get_value()),
                "nt":   round(num_transfers[r].get_value()),
                "xP":   round(lineup_rows["xp_contrib"].sum(), 2),
                "chip": chip_str or None,
            }

        buy_names  = picks_df[(picks_df["round"] == next_round) & (picks_df["transfer_in"] == 1)]["name"].tolist()
        sell_names = picks_df[(picks_df["round"] == next_round) & (picks_df["transfer_out"] == 1)]["name"].tolist()

        solutions.append({
            "iter":       iteration,
            "picks":      picks_df,
            "total_xp":   round(total_xp, 2),
            "score":      round(obj_value, 3),
            "summary":    summary,
            "statistics": statistics,
            "buy":        ", ".join(buy_names) or "-",
            "sell":       ", ".join(sell_names) or "-",
            "chip":       _chip_summary(rounds, use_ll, use_nl, use_pp_gw),
        })

        if num_iterations == 1:
            return solutions

        # Force next iteration to differ on transfer-ins for next_round
        transferred_in    = [p for p in players if transfer_in[p, next_round].get_value() > BINARY_THRESHOLD]
        not_transferred   = [p for p in players if transfer_in[p, next_round].get_value() < BINARY_THRESHOLD]
        iter_cut = (
            so.expr_sum(1 - transfer_in[p, next_round] for p in transferred_in)
            + so.expr_sum(transfer_in[p, next_round]   for p in not_transferred)
        )
        model.add_constraint(iter_cut >= 1, name=f"iter_cut_{iteration}")

    return solutions


# ── Output helpers ─────────────────────────────────────────────────────────────

def _chip_summary(rounds, use_ll, use_nl, use_pp_gw) -> str:
    parts = []
    for r in rounds:
        if use_ll[r].get_value()     > BINARY_THRESHOLD: parts.append(f"LL(R{r})")
        if use_nl[r].get_value()     > BINARY_THRESHOLD: parts.append(f"NL(R{r})")
        if use_pp_gw[r].get_value()  > BINARY_THRESHOLD: parts.append(f"PP(R{r})")
    return ", ".join(parts) or "-"


def _build_summary(
    picks_df, rounds, next_round,
    in_the_bank, ft_var, penalized_transfers, num_transfers,
    use_ll, use_nl, use_pp_gw,
) -> tuple[str, dict]:
    lines = []
    move_summary = {"buy": [], "sell": [], "chip": []}

    for r in rounds:
        lines.append(f"── Round {r} " + "─" * 40)

        chip_parts = []
        if use_ll[r].get_value()    > BINARY_THRESHOLD: chip_parts.append("LIMITLESS")
        if use_nl[r].get_value()    > BINARY_THRESHOLD: chip_parts.append("NO LIMITS")
        if use_pp_gw[r].get_value() > BINARY_THRESHOLD: chip_parts.append("POWER PLAY")
        if chip_parts:
            label = " + ".join(chip_parts)
            lines.append(f"  CHIP: {label}")
            move_summary["chip"].extend(chip_parts)

        itb_prev = in_the_bank[r - 1].get_value()
        itb_now  = in_the_bank[r].get_value()
        ft_now   = round(ft_var[r].get_value())
        pt_now   = round(penalized_transfers[r].get_value())
        nt_now   = round(num_transfers[r].get_value())
        lines.append(
            f"  ITB: £{itb_prev:.1f}m → £{itb_now:.1f}m  |  "
            f"FT: {ft_now}  |  Transfers: {nt_now}  |  Hits: {pt_now}"
        )

        buys  = picks_df[(picks_df["round"] == r) & (picks_df["transfer_in"] == 1)]["name"].tolist()
        sells = picks_df[(picks_df["round"] == r) & (picks_df["transfer_out"] == 1)]["name"].tolist()
        if sells: lines.append(f"  OUT: {', '.join(sells)}")
        if buys:  lines.append(f"  IN:  {', '.join(buys)}")
        if r == next_round:
            move_summary["buy"].extend(buys)
            move_summary["sell"].extend(sells)

        lineup_rows = picks_df[(picks_df["round"] == r) & (picks_df["lineup"] == 1)]
        for pos in ["GK", "DEF", "MID", "FWD"]:
            pos_players = lineup_rows[lineup_rows["position"] == pos]
            entries = []
            for _, row in pos_players.iterrows():
                tag = ""
                if row["captain"]:     tag = " [C]" + (" [PP]" if row["power_play"] else "")
                elif row["vicecaptain"]: tag = " [V]"
                entries.append(f"{row['name']}{tag} ({row['xP']})")
            if entries:
                lines.append(f"  {pos:3}: {', '.join(entries)}")

        bench_rows = picks_df[(picks_df["round"] == r) & (picks_df["bench"] >= 0)].sort_values("bench")
        lines.append(f"  BEN: {', '.join(bench_rows['name'].tolist())}")
        xp_tot = picks_df[(picks_df["round"] == r) & (picks_df["lineup"] == 1)]["xp_contrib"].sum()
        lines.append(f"  Lineup xP: {xp_tot:.2f}")
        lines.append("")

    return "\n".join(lines), move_summary


# ── Entry point ────────────────────────────────────────────────────────────────

def solve_regular_wc(runtime_options: dict | None = None):
    """
    Load settings, run the solver, print and save results.

    Call directly or pass runtime_options to override any default setting.
    Settings are also read from data/wc_settings.json if it exists
    (runtime_options always take highest priority).
    """

    # Defaults — all overridable via wc_settings.json or runtime_options
    options: dict = {
        # Data
        "projection_file":  "data/wc_projections.csv",
        "next_round":       1,
        "horizon":          3,

        # Solver
        "objective":        "decay",    # "decay" | "regular"
        "decay_base":       0.85,
        "secs":             300,        # solver time limit (seconds)
        "gap":              0.0,        # MIP optimality gap (0 = optimal)
        "verbose":          False,
        "num_iterations":   1,
        "delete_tmp":       True,

        # Team state
        "initial_squad":    [],         # list of player IDs; empty = preseason pick
        "itb":              100.0,      # in-the-bank (£m)
        "ft":               1,          # free transfers available  [WC-RULE]

        # Scoring weights
        "hit_cost":         4,          # points per extra transfer  [WC-RULE]
        "ft_value":         1.5,        # value of a saved free transfer
        "bench_weights":    {0: 0.03, 1: 0.21, 2: 0.06, 3: 0.002},
        "vcap_weight":      0.1,
        "itb_value":        0.0,

        # Chips  [WC-RULE]
        "chip_limits": {"power_play": 1, "no_limits": 1, "limitless": 1},
        "force_chip":       {},         # e.g. {"limitless": 2}  → use LL in round 2

        # Player pool filtering
        "xmin_lb":          0,          # min total xMins across horizon (0 = no filter)

        # Optional constraints
        "banned":           [],         # player IDs to exclude
        "locked":           [],         # player IDs to force into every squad
        "num_transfers":    None,       # fix exact transfer count for next_round

        # Output
        "print_summary":    True,
        "save_results":     True,
    }

    # Layer 1: settings file
    settings_path = Path("data/wc_settings.json")
    if settings_path.exists():
        with open(settings_path) as f:
            options.update(json.load(f))

    # Layer 2: runtime_options (highest priority)
    if runtime_options:
        options.update(runtime_options)

    data     = prep_wc_data(options)
    response = solve_wc(data, options)

    # Save + print
    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for result in response:
        if options["print_summary"]:
            print(
                f"\n=== Solution {result['iter'] + 1}  "
                f"| Score: {result['score']:.2f}  "
                f"| xP: {result['total_xp']:.2f} ==="
            )
            print(result["summary"])

        if options["save_results"]:
            out_path = results_dir / f"wc_{stamp}_iter{result['iter']}.csv"
            result["picks"].to_csv(out_path, index=False)
            print(f"Saved → {out_path}")

    # Return a ranked summary table
    summary_rows = [
        {
            "iter":     r["iter"] + 1,
            "score":    r["score"],
            "total_xp": r["total_xp"],
            "buy":      r["buy"],
            "sell":     r["sell"],
            "chip":     r["chip"],
        }
        for r in response
    ]
    result_table = (
        pd.DataFrame(summary_rows)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

    if options["print_summary"]:
        print("\n" + result_table.to_string(index=False))

    return result_table, response


if __name__ == "__main__":
    solve_regular_wc()

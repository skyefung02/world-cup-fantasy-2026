"""
wc_solver.py — World Cup Fantasy 2026 ILP squad optimiser
Solver backend: sasoptpy (model builder) + highspy (HiGHS MIP solver)

Adapted from open-fpl-solver (github.com/solioanalytics/open-fpl-solver),
licensed under the Apache License 2.0. Changes made: data pipeline replaced
with WC Fantasy CSV inputs; FPL-specific chips/rules replaced with WC 2026
booster system; country limits made stage-dependent; transfer allocation
changed to match official WC 2026 rules.

Settings priority (lowest → highest):
    wc_settings_defaults.json  →  wc_settings.json  →  --config file  →  runtime_options dict
"""

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

import highspy
import pandas as pd
import sasoptpy as so

# ── Constants ─────────────────────────────────────────────────────────────────

SQUAD_SIZE       = 15
LINEUP_SIZE      = 11
BUDGET           = 100.0     # £m starting budget
KNOCKOUT_BUDGET  = 105.0     # £m from R32 onwards (+£5m per official rules)

SQUAD_QUOTA = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
LINEUP_MIN  = {"GK": 1, "DEF": 3, "MID": 3, "FWD": 1}   # MID min = 3 (smallest valid formation is X-3-X)
LINEUP_MAX  = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}

POSITIONS    = list(SQUAD_QUOTA.keys())
BENCH_ORDER  = [0, 1, 2, 3]   # slot 0 = GK bench slot
BINARY_THRESHOLD = 0.5

SETTINGS_DEFAULTS_PATH = Path("wc_settings_defaults.json")
SETTINGS_USER_PATH     = Path("wc_settings.json")


# ── Settings loading ──────────────────────────────────────────────────────────

def load_settings(config_path: str | None = None) -> dict:
    """
    Load settings in priority order:
      1. wc_settings_defaults.json  (all keys + defaults)
      2. wc_settings.json           (user personal overrides)
      3. --config file              (scenario-level overrides, if provided)
    """
    options = {}

    if SETTINGS_DEFAULTS_PATH.exists():
        with open(SETTINGS_DEFAULTS_PATH) as f:
            options.update(json.load(f))

    if SETTINGS_USER_PATH.exists():
        with open(SETTINGS_USER_PATH) as f:
            options.update(json.load(f))

    if config_path:
        p = Path(config_path)
        if p.exists():
            with open(p) as f:
                options.update(json.load(f))
        else:
            print(f"Warning: --config file '{config_path}' not found, skipping.")

    return options


# ── Data loading ──────────────────────────────────────────────────────────────

def load_projection_data(filepath: str | Path) -> pd.DataFrame:
    """
    Load the per-round projection CSV.

    Required columns : id, player (or name), position, price, team
    Optional columns : abbr, status, qual_prob
    Per-round columns: {round_id}_Pts, {round_id}_xMins  (one pair per round)

    Returns a DataFrame indexed by integer player id.
    The 'player' column is normalised to 'name' internally.
    """
    df = pd.read_csv(filepath)

    # Normalise player name column: build_projections.py outputs 'player'
    if "player" in df.columns and "name" not in df.columns:
        df = df.rename(columns={"player": "name"})

    required = {"id", "name", "position", "price", "team"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Projection CSV is missing required columns: {missing}")
    df["id"] = df["id"].astype(int)
    return df.set_index("id")


def prep_wc_data(options: dict) -> dict:
    """
    Load and validate projection data; build the data dict consumed by solve_wc().
    """
    projection_path = options.get("projection_file", "data/projections.csv")
    df = load_projection_data(projection_path)

    next_round = int(options.get("next_round", 1))
    horizon    = int(options.get("horizon", 3))
    rounds     = list(range(next_round, next_round + horizon))

    for r in rounds:
        for suffix in ("_Pts", "_xMins"):
            if f"{r}{suffix}" not in df.columns:
                raise ValueError(
                    f"Column '{r}{suffix}' missing from projection file "
                    f"(next_round={next_round}, horizon={horizon})."
                )

    df["total_ev"]   = df[[f"{r}_Pts"   for r in rounds]].sum(axis=1)
    df["total_mins"] = df[[f"{r}_xMins" for r in rounds]].sum(axis=1)

    # Filter unavailable players (keep locked/owned players regardless)
    n_before  = len(df)
    safe_ids  = set(options.get("initial_squad", []) + options.get("locked", []))

    if "status" in df.columns:
        df = df[df["status"].str.lower().isin(["playing", "available", ""]) | df.index.isin(safe_ids)]

    xmin_lb = options.get("xmin_lb", 0)
    if xmin_lb > 0:
        df = df[(df["total_mins"] >= xmin_lb) | df.index.isin(safe_ids)]

    print(f"Player pool: {n_before} → {len(df)} after filtering")

    # Determine starting budget (increases by £5m from R32 onwards)
    group_stage_rounds = set(options.get("group_stage_rounds", [1, 2, 3]))
    in_knockout        = all(r not in group_stage_rounds for r in rounds)
    default_budget     = KNOCKOUT_BUDGET if in_knockout else BUDGET
    itb                = float(options.get("itb", default_budget))

    return {
        "df":                 df,
        "rounds":             rounds,
        "next_round":         next_round,
        "group_stage_rounds": group_stage_rounds,
        "buy_price":          df["price"].to_dict(),
        "qual_prob":          df["qual_prob"].to_dict() if "qual_prob" in df.columns else {},
        "initial_squad":      [int(i) for i in options.get("initial_squad", [])],
        "itb":                itb,
        "ft":                 int(options.get("ft", 1)),
    }


# ── ILP model ─────────────────────────────────────────────────────────────────

def solve_wc(data: dict, options: dict) -> list[dict]:
    """
    Build and solve the WC Fantasy 2026 ILP.

    Returns a list of solution dicts (one per iteration), each with:
        picks       — DataFrame, one row per (player, round) in the solution
        score       — objective value (descending = better)
        total_xp    — raw lineup xP sum
        summary     — human-readable action string
        statistics  — per-round stats dict
        buy / sell  — transfer names for next_round
        boosters    — boosters used across the horizon
        iter        — iteration index
    """

    # ── Options ───────────────────────────────────────────────────────────────
    horizon        = int(options.get("horizon", 3))
    objective      = options.get("objective", "decay")
    decay_base     = float(options.get("decay_base", 0.85))
    bench_weights  = {int(k): v for k, v in options.get(
                         "bench_weights", {"0": 0.03, "1": 0.21, "2": 0.06, "3": 0.002}).items()}
    hit_cost       = float(options.get("hit_cost", 3))   # 3 pts per extra transfer (official rule)
    vcap_weight    = float(options.get("vcap_weight", 0.1))
    itb_value      = float(options.get("itb_value", 0.0))
    num_iterations = int(options.get("num_iterations", 1))

    booster_limits = options.get("booster_limits", {
        "wildcard":   1,
        "twelfth_man":  1,
        "max_captain":  1,
        "qual_booster": 1,
    })

    # ── Data ──────────────────────────────────────────────────────────────────
    df                 = data["df"]
    rounds             = data["rounds"]
    next_round         = data["next_round"]
    group_stage_rounds = data["group_stage_rounds"]
    buy_price          = data["buy_price"]
    qual_prob          = data["qual_prob"]
    initial_squad      = data["initial_squad"]
    itb                = data["itb"]
    initial_ft         = data["ft"]

    players    = df.index.tolist()
    teams      = df["team"].unique().tolist()
    all_rounds = [next_round - 1, *rounds]

    player_pos  = df["position"].to_dict()
    player_team = df["team"].to_dict()
    points_pw   = {(p, r): float(df.loc[p, f"{r}_Pts"])   for p in players for r in rounds}
    mins_pw     = {(p, r): float(df.loc[p, f"{r}_xMins"]) for p in players for r in rounds}

    # FT schedule: base allocation per round (99 = effectively unlimited)
    ft_schedule = options.get("ft_schedule", {})
    base_ft = {r: int(ft_schedule.get(str(r), ft_schedule.get(r, 1))) for r in rounds}

    # Country limit per round (varies by stage)
    country_limit_by_round = options.get("country_limit_by_round", {})
    def country_limit(r: int) -> int:
        return int(country_limit_by_round.get(str(r), country_limit_by_round.get(r, 3)))

    # Which group-stage round pairs allow FT carryover
    group_consecutive_pairs = [
        (r, r + 1) for r in rounds
        if r + 1 in rounds
        and r in group_stage_rounds
        and r + 1 in group_stage_rounds
    ]

    # ── Model ─────────────────────────────────────────────────────────────────
    model = so.Model(name=f"wc_h{horizon}_{objective}")

    # ── Variables ─────────────────────────────────────────────────────────────

    squad        = model.add_variables(players, all_rounds, name="squad",   vartype=so.binary)
    lineup       = model.add_variables(players, rounds,     name="lineup",  vartype=so.binary)
    captain      = model.add_variables(players, rounds,     name="captain", vartype=so.binary)
    vicecap      = model.add_variables(players, rounds,     name="vicecap", vartype=so.binary)
    bench        = model.add_variables(players, rounds, BENCH_ORDER, name="bench", vartype=so.binary)
    transfer_in  = model.add_variables(players, rounds, name="tr_in",  vartype=so.binary)
    transfer_out = model.add_variables(players, rounds, name="tr_out", vartype=so.binary)
    in_the_bank  = model.add_variables(all_rounds, name="itb", vartype=so.continuous, lb=0)
    penalized_transfers = model.add_variables(rounds, name="pt", vartype=so.integer, lb=0)
    ft_var       = model.add_variables(rounds, name="ft", vartype=so.integer, lb=0, ub=15)

    # Boosters
    use_wc  = model.add_variables(rounds, name="use_wc",  vartype=so.binary)  # Wildcard
    use_tm  = model.add_variables(rounds, name="use_tm",  vartype=so.binary)  # 12th Man
    use_mc  = model.add_variables(rounds, name="use_mc",  vartype=so.binary)  # Max Captain
    use_qb  = model.add_variables(rounds, name="use_qb",  vartype=so.binary)  # Qual Booster

    # 12th Man player: one extra player outside all squad/budget/country constraints
    tm_player = model.add_variables(players, rounds, name="tm_pl", vartype=so.binary)

    # FT carryover binary (group stage only)
    saved_ft = model.add_variables(rounds, name="saved_ft", vartype=so.binary)

    # ── Derived expressions ───────────────────────────────────────────────────
    num_transfers = {r: so.expr_sum(transfer_out[p, r] for p in players) for r in rounds}
    bought_amount = {r: so.expr_sum(buy_price[p] * transfer_in[p, r]  for p in players) for r in rounds}
    sold_amount   = {r: so.expr_sum(buy_price[p] * transfer_out[p, r] for p in players) for r in rounds}
    lineup_pos    = {
        (pos, r): so.expr_sum(lineup[p, r] for p in players if player_pos[p] == pos)
        for pos in POSITIONS for r in rounds
    }
    squad_pos = {
        (pos, r): so.expr_sum(squad[p, r] for p in players if player_pos[p] == pos)
        for pos in POSITIONS for r in rounds
    }

    # ── Initial conditions ────────────────────────────────────────────────────
    model.add_constraints(
        (squad[p, next_round - 1] == (1 if p in initial_squad else 0) for p in players),
        name="initial_squad",
    )
    model.add_constraint(in_the_bank[next_round - 1] == itb, name="initial_itb")
    model.add_constraint(ft_var[next_round] == initial_ft,   name="initial_ft")

    # ── Squad size ────────────────────────────────────────────────────────────
    model.add_constraints(
        (so.expr_sum(squad[p, r] for p in players) == SQUAD_SIZE for r in rounds),
        name="squad_size",
    )

    # ── Lineup size ───────────────────────────────────────────────────────────
    model.add_constraints(
        (so.expr_sum(lineup[p, r] for p in players) == LINEUP_SIZE for r in rounds),
        name="lineup_size",
    )

    # ── Bench ─────────────────────────────────────────────────────────────────
    model.add_constraints(
        (so.expr_sum(bench[p, r, 0] for p in players if player_pos[p] == "GK") == 1
         for r in rounds),
        name="bench_gk",
    )
    model.add_constraints(
        (so.expr_sum(bench[p, r, o] for p in players) == 1
         for r in rounds for o in [1, 2, 3]),
        name="bench_outfield",
    )

    # ── Captain / vicecap ─────────────────────────────────────────────────────
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

    # ── Lineup ↔ squad ────────────────────────────────────────────────────────
    model.add_constraints(
        (lineup[p, r] <= squad[p, r] for p in players for r in rounds),
        name="lineup_in_squad",
    )
    model.add_constraints(
        (bench[p, r, o] <= squad[p, r]
         for p in players for r in rounds for o in BENCH_ORDER),
        name="bench_in_squad",
    )
    model.add_constraints(
        (lineup[p, r] + so.expr_sum(bench[p, r, o] for o in BENCH_ORDER) <= 1
         for p in players for r in rounds),
        name="lineup_bench_exclusive",
    )

    # ── Formation constraints ─────────────────────────────────────────────────
    for pos in POSITIONS:
        model.add_constraints(
            (lineup_pos[pos, r] >= LINEUP_MIN[pos] for r in rounds),
            name=f"formation_lb_{pos}",
        )
        model.add_constraints(
            (lineup_pos[pos, r] <= LINEUP_MAX[pos] for r in rounds),
            name=f"formation_ub_{pos}",
        )
        model.add_constraints(
            (squad_pos[pos, r] == SQUAD_QUOTA[pos] for r in rounds),
            name=f"squad_quota_{pos}",
        )

    # ── Country limits (stage-dependent) ─────────────────────────────────────
    # Sanitise team names for MPS constraint names — spaces/accents break parsers
    for t in teams:
        t_key = re.sub(r"[^A-Za-z0-9_]", "_", t)
        for r in all_rounds:
            lim = country_limit(r)
            model.add_constraint(
                so.expr_sum(squad[p, r] for p in players if player_team[p] == t) <= lim,
                name=f"country_lim_{t_key}_{r}",
            )

    # ── Transfer continuity ───────────────────────────────────────────────────
    model.add_constraints(
        (squad[p, r] == squad[p, r - 1] + transfer_in[p, r] - transfer_out[p, r]
         for p in players for r in rounds),
        name="squad_continuity",
    )
    model.add_constraints(
        (transfer_in[p, r] + transfer_out[p, r] <= 1 for p in players for r in rounds),
        name="no_in_and_out",
    )
    # Wildcard: no transfer restrictions (but we still track them for output)
    # Non-wildcard: transfers must not exceed squad size (already bounded by binary vars)

    # ── Budget continuity ─────────────────────────────────────────────────────
    model.add_constraints(
        (in_the_bank[r] == in_the_bank[r - 1] + sold_amount[r] - bought_amount[r]
         for r in rounds),
        name="budget_continuity",
    )

    # ── Free transfer tracking ────────────────────────────────────────────────
    #
    # Group stage:
    #   - base FT allocation per round (from ft_schedule)
    #   - up to 1 unused FT carries over to the next group-stage round
    #   - Wildcard: no hits in that round, no carryover afterward
    #
    # Knockout: fixed allocation per round, no carryover
    #
    # Transfer hits: each transfer beyond ft_var costs hit_cost points
    m = 20  # big-M (nobody will ever have >20 FTs)

    for r in rounds:
        # Hits: penalized_transfers >= num_transfers - ft_var (Wildcard grants unlimited)
        model.add_constraint(
            penalized_transfers[r] >= num_transfers[r] - ft_var[r] - SQUAD_SIZE * use_wc[r],
            name=f"hit_calc_{r}",
        )
        # Wildcard suppresses hits entirely
        model.add_constraint(
            penalized_transfers[r] <= SQUAD_SIZE * (1 - use_wc[r]),
            name=f"wc_no_hits_{r}",
        )

        if r + 1 in rounds:
            if (r, r + 1) in group_consecutive_pairs:
                # ── Group-stage carryover ──────────────────────────────────────
                # saved_ft[r] = 1  iff  (ft_var[r] - num_transfers[r] >= 1)
                #                  and  Wildcard not active
                net = ft_var[r] - num_transfers[r]

                # saved_ft[r] = 0 when Wildcard active
                model.add_constraint(saved_ft[r] <= 1 - use_wc[r], name=f"sft_wc_{r}")
                # saved_ft[r] = 1  →  net >= 1
                model.add_constraint(net >= 1 - m * (1 - saved_ft[r]),  name=f"sft_lb_{r}")
                # saved_ft[r] = 0  →  net <= 0  (all FTs used or hits taken)
                model.add_constraint(net <= m * saved_ft[r],             name=f"sft_ub_{r}")

                # Next round FT = base allocation + carryover (max 1)
                model.add_constraint(
                    ft_var[r + 1] == base_ft[r + 1] + saved_ft[r],
                    name=f"ft_grp_transition_{r}",
                )
            else:
                # ── Knockout: fixed allocation, no carryover ───────────────────
                model.add_constraint(
                    ft_var[r + 1] == base_ft[r + 1],
                    name=f"ft_kno_transition_{r}",
                )

    # ── Booster constraints ───────────────────────────────────────────────────

    # At most one booster active per round
    model.add_constraints(
        (use_wc[r] + use_tm[r] + use_mc[r] + use_qb[r] <= 1 for r in rounds),
        name="one_booster_per_round",
    )

    # Each booster used at most once total
    model.add_constraint(so.expr_sum(use_wc[r] for r in rounds) <= booster_limits.get("wildcard",    1), name="wc_limit")
    model.add_constraint(so.expr_sum(use_tm[r] for r in rounds) <= booster_limits.get("twelfth_man", 1), name="tm_limit")
    model.add_constraint(so.expr_sum(use_mc[r] for r in rounds) <= booster_limits.get("max_captain", 1), name="mc_limit")
    model.add_constraint(so.expr_sum(use_qb[r] for r in rounds) <= booster_limits.get("qual_booster",1), name="qb_limit")

    # Wildcard cannot be used in round 1 (group stage MD1 = preseason unlimited already)
    wildcard_banned = set(options.get("wildcard_banned_rounds", [next_round]
                          if next_round in group_stage_rounds and next_round == min(group_stage_rounds)
                          else []))
    for r in rounds:
        if r in wildcard_banned:
            model.add_constraint(use_wc[r] == 0, name=f"wc_banned_{r}")

    # Qualification Booster only available from R32 onwards
    for r in rounds:
        if r in group_stage_rounds:
            model.add_constraint(use_qb[r] == 0, name=f"qb_group_banned_{r}")

    # 12th Man: exactly 1 player selected when active, 0 otherwise
    model.add_constraints(
        (so.expr_sum(tm_player[p, r] for p in players) == use_tm[r] for r in rounds),
        name="tm_count",
    )
    # 12th Man cannot be someone already in your squad
    model.add_constraints(
        (tm_player[p, r] + squad[p, r] <= 1 for p in players for r in rounds),
        name="tm_not_in_squad",
    )
    # 12th Man is not subject to captain/vicecap (enforced by tm_player being separate)
    # No budget or country constraint on tm_player (per official rules)

    # Force boosters (from options: e.g. "use_wildcard": [2])
    for r in options.get("use_wildcard",    []): model.add_constraint(use_wc[r] == 1, name=f"force_wc_{r}")
    for r in options.get("use_twelfth_man", []): model.add_constraint(use_tm[r] == 1, name=f"force_tm_{r}")
    for r in options.get("use_max_captain", []): model.add_constraint(use_mc[r] == 1, name=f"force_mc_{r}")
    for r in options.get("use_qual_booster",[]): model.add_constraint(use_qb[r] == 1, name=f"force_qb_{r}")

    # Disable boosters already used (set limit to 0)
    used_wc = int(options.get("used_wildcard",    0))
    used_tm = int(options.get("used_twelfth_man", 0))
    used_mc = int(options.get("used_max_captain", 0))
    used_qb = int(options.get("used_qual_booster",0))
    if used_wc: model.add_constraint(so.expr_sum(use_wc[r] for r in rounds) == 0, name="wc_exhausted")
    if used_tm: model.add_constraint(so.expr_sum(use_tm[r] for r in rounds) == 0, name="tm_exhausted")
    if used_mc: model.add_constraint(so.expr_sum(use_mc[r] for r in rounds) == 0, name="mc_exhausted")
    if used_qb: model.add_constraint(so.expr_sum(use_qb[r] for r in rounds) == 0, name="qb_exhausted")

    # ── Optional constraints ──────────────────────────────────────────────────
    if options.get("banned"):
        banned = [p for p in options["banned"] if p in players]
        model.add_constraints(
            (so.expr_sum(squad[p, r] for r in rounds) == 0 for p in banned),
            name="banned_players",
        )

    if options.get("locked"):
        locked = [p for p in options["locked"] if p in players]
        model.add_constraints(
            (squad[p, r] == 1 for p in locked for r in rounds),
            name="locked_players",
        )

    if options.get("num_transfers") is not None:
        model.add_constraint(
            so.expr_sum(transfer_in[p, next_round] for p in players) == int(options["num_transfers"]),
            name="forced_transfer_count",
        )

    # ── Qual Booster linearisation ────────────────────────────────────────────
    # use_qb[r] * lineup[p,r] is bilinear — MPS format doesn't support it.
    # Introduce qb_lineup[p,r] = use_qb[r] AND lineup[p,r] via standard
    # binary linearisation. Skipped entirely when qual_prob data isn't present
    # (i.e. all group-stage runs), keeping the model small.
    has_qual_prob = bool(qual_prob)
    if has_qual_prob:
        qb_lineup = model.add_variables(players, rounds, name="qb_lu", vartype=so.binary)
        model.add_constraints(
            (qb_lineup[p, r] <= use_qb[r]    for p in players for r in rounds),
            name="qbl_ub_wc",
        )
        model.add_constraints(
            (qb_lineup[p, r] <= lineup[p, r] for p in players for r in rounds),
            name="qbl_ub_lu",
        )
        model.add_constraints(
            (qb_lineup[p, r] >= use_qb[r] + lineup[p, r] - 1 for p in players for r in rounds),
            name="qbl_lb",
        )
        qb_xp = {
            r: 2 * so.expr_sum(qual_prob.get(p, 0) * qb_lineup[p, r] for p in players)
            for r in rounds
        }
    else:
        qb_xp = {r: 0 for r in rounds}

    # ── Objective ─────────────────────────────────────────────────────────────
    #
    # Scoring:
    #   lineup player  →  1× points
    #   captain        →  +1× (total 2×)
    #   vicecap        →  vcap_weight× (expected VC coverage value)
    #   bench slot o   →  bench_weights[o]× (sub probability)
    #   12th Man       →  1× (no captain multiplier, no bench weight)
    #   Qual Booster   →  +2 pts × qual_prob for each starting XI player (linearised)
    #
    gw_xp = {
        r: so.expr_sum(
            points_pw[p, r] * (
                lineup[p, r]
                + captain[p, r]
                + vcap_weight * vicecap[p, r]
                + so.expr_sum(bench_weights[o] * bench[p, r, o] for o in BENCH_ORDER)
            )
            for p in players
        )
        # 12th Man contribution
        + so.expr_sum(points_pw[p, r] * tm_player[p, r] for p in players)
        # Qualification Booster (linearised; 0 during group stage)
        + qb_xp[r]
        for r in rounds
    }

    gw_total = {
        r: gw_xp[r] - hit_cost * penalized_transfers[r] + itb_value * in_the_bank[r]
        for r in rounds
    }

    if objective == "regular":
        obj = so.expr_sum(gw_total[r] for r in rounds)
    else:
        obj = so.expr_sum(gw_total[r] * pow(decay_base, r - next_round) for r in rounds)

    model.set_objective(-obj, sense="N", name="wc_obj")

    # ── Solve loop ────────────────────────────────────────────────────────────
    solutions = []

    for iteration in range(num_iterations):
        tmp_dir  = Path("tmp")
        tmp_dir.mkdir(exist_ok=True)
        mps_path = tmp_dir / f"wc_h{horizon}_i{iteration}.mps"
        model.export_mps(str(mps_path))
        print(f"Exported model: wc_h{horizon}_i{iteration}")

        h = highspy.Highs()
        h.readModel(str(mps_path))
        h.setOptionValue("parallel",       "on")
        h.setOptionValue("time_limit",     int(options.get("secs", 300)))
        h.setOptionValue("mip_rel_gap",    float(options.get("gap", 0.0)))
        h.setOptionValue("log_to_console", bool(options.get("verbose", False)))
        h.run()

        sol = h.getSolution()
        for idx, var in enumerate(model.get_variables()):
            var.set_value(sol.col_value[idx])

        if options.get("delete_tmp", True):
            try: mps_path.unlink()
            except Exception: pass

        # ── Extract picks ──────────────────────────────────────────────────
        picks = []
        for r in rounds:
            for p in players:
                in_sq  = squad[p, r].get_value()
                is_out = transfer_out[p, r].get_value()
                if in_sq + is_out < BINARY_THRESHOLD:
                    continue

                is_lineup  = int(lineup[p, r].get_value()  > BINARY_THRESHOLD)
                is_cap     = int(captain[p, r].get_value() > BINARY_THRESHOLD)
                is_vc      = int(vicecap[p, r].get_value() > BINARY_THRESHOLD)
                is_tr_in   = int(transfer_in[p, r].get_value()  > BINARY_THRESHOLD)
                is_tr_out  = int(is_out > BINARY_THRESHOLD)
                is_wc      = int(use_wc[r].get_value() > BINARY_THRESHOLD)
                is_tm_rnd  = int(use_tm[r].get_value() > BINARY_THRESHOLD)
                is_mc      = int(use_mc[r].get_value() > BINARY_THRESHOLD)
                is_qb      = int(use_qb[r].get_value() > BINARY_THRESHOLD)

                bench_slot = -1
                for o in BENCH_ORDER:
                    if bench[p, r, o].get_value() > BINARY_THRESHOLD:
                        bench_slot = o

                # TM tag is reserved for the tm_player row added below;
                # regular squad players only carry WC/MC/QB round markers
                booster = ("WC" if is_wc else "MC" if is_mc else "QB" if is_qb else "")
                multiplier = is_lineup + is_cap
                row = df.loc[p]

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
                    "transfer_in":  is_tr_in,
                    "transfer_out": is_tr_out,
                    "multiplier":   multiplier,
                    "xp_contrib":   round(points_pw[p, r] * multiplier, 2),
                    "booster":      booster,
                    "ft":           round(ft_var[r].get_value()),
                    "iter":         iteration,
                })

            # 12th Man row (separate player outside regular squad)
            for p in players:
                if tm_player[p, r].get_value() > BINARY_THRESHOLD:
                    row = df.loc[p]
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
                        "squad":        0,
                        "lineup":       1,   # scores points
                        "bench":        -1,
                        "captain":      0,
                        "vicecaptain":  0,
                        "transfer_in":  0,
                        "transfer_out": 0,
                        "multiplier":   1,
                        "xp_contrib":   round(points_pw[p, r], 2),
                        "booster":      "TM",
                        "ft":           round(ft_var[r].get_value()),
                        "iter":         iteration,
                    })

        picks_df = pd.DataFrame(picks).sort_values(
            by=["round", "squad", "lineup", "bench", "position"],
            ascending=[True, False, False, True, True],
        )

        # ── Per-round statistics ───────────────────────────────────────────
        statistics = {}
        for r in rounds:
            lu  = picks_df[(picks_df["round"] == r) & (picks_df["lineup"] == 1)]
            bst = picks_df[(picks_df["round"] == r) & (picks_df["booster"] != "")]
            statistics[r] = {
                "itb":     round(in_the_bank[r].get_value(), 2),
                "ft":      round(ft_var[r].get_value()),
                "pt":      round(penalized_transfers[r].get_value()),
                "nt":      round(num_transfers[r].get_value()),
                "xP":      round(lu["xp_contrib"].sum(), 2),
                "booster": bst["booster"].iloc[0] if not bst.empty else None,
            }

        summary, move_summary = _build_summary(
            picks_df, rounds, next_round,
            in_the_bank, ft_var, penalized_transfers, num_transfers,
            use_wc, use_tm, use_mc, use_qb,
        )

        buy_names  = picks_df[(picks_df["round"] == next_round) & (picks_df["transfer_in"] == 1)]["name"].tolist()
        sell_names = picks_df[(picks_df["round"] == next_round) & (picks_df["transfer_out"] == 1)]["name"].tolist()

        solutions.append({
            "iter":       iteration,
            "picks":      picks_df,
            "total_xp":   round(picks_df["xp_contrib"].sum(), 2),
            "score":      round(-model.get_objective_value(), 3),
            "summary":    summary,
            "statistics": statistics,
            "buy":        ", ".join(buy_names) or "-",
            "sell":       ", ".join(sell_names) or "-",
            "boosters":   _booster_summary(rounds, use_wc, use_tm, use_mc, use_qb),
        })

        if num_iterations == 1:
            return solutions

        # Force next iteration to differ on which players are transferred in
        tr_in_now = [p for p in players if transfer_in[p, next_round].get_value() > BINARY_THRESHOLD]
        tr_in_not = [p for p in players if transfer_in[p, next_round].get_value() < BINARY_THRESHOLD]
        cut = (
            so.expr_sum(1 - transfer_in[p, next_round] for p in tr_in_now)
            + so.expr_sum(transfer_in[p, next_round]   for p in tr_in_not)
        )
        model.add_constraint(cut >= 1, name=f"iter_cut_{iteration}")

    return solutions


# ── Output helpers ─────────────────────────────────────────────────────────────

def _booster_summary(rounds, use_wc, use_tm, use_mc, use_qb) -> str:
    parts = []
    for r in rounds:
        if use_wc[r].get_value() > BINARY_THRESHOLD: parts.append(f"WC(R{r})")
        if use_tm[r].get_value() > BINARY_THRESHOLD: parts.append(f"TM(R{r})")
        if use_mc[r].get_value() > BINARY_THRESHOLD: parts.append(f"MC(R{r})")
        if use_qb[r].get_value() > BINARY_THRESHOLD: parts.append(f"QB(R{r})")
    return ", ".join(parts) or "-"


def _build_summary(picks_df, rounds, next_round,
                   in_the_bank, ft_var, penalized_transfers, num_transfers,
                   use_wc, use_tm, use_mc, use_qb):
    lines = []
    move_summary = {"buy": [], "sell": [], "booster": []}

    for r in rounds:
        lines.append(f"── Round {r} " + "─" * 42)

        booster_parts = []
        if use_wc[r].get_value() > BINARY_THRESHOLD: booster_parts.append("WILDCARD")
        if use_tm[r].get_value() > BINARY_THRESHOLD: booster_parts.append("12TH MAN")
        if use_mc[r].get_value() > BINARY_THRESHOLD: booster_parts.append("MAX CAPTAIN")
        if use_qb[r].get_value() > BINARY_THRESHOLD: booster_parts.append("QUAL BOOSTER")
        if booster_parts:
            lines.append(f"  BOOSTER: {' + '.join(booster_parts)}")
            move_summary["booster"].extend(booster_parts)

        itb_prev = in_the_bank[r - 1].get_value()
        itb_now  = in_the_bank[r].get_value()
        ft_now   = round(ft_var[r].get_value())
        pt_now   = round(penalized_transfers[r].get_value())
        nt_now   = round(num_transfers[r].get_value())
        lines.append(
            f"  ITB: £{itb_prev:.1f}m → £{itb_now:.1f}m  |  "
            f"FT: {ft_now}  |  Transfers: {nt_now}  |  Hits: {pt_now}"
        )

        sells = picks_df[(picks_df["round"] == r) & (picks_df["transfer_out"] == 1)]["name"].tolist()
        buys  = picks_df[(picks_df["round"] == r) & (picks_df["transfer_in"]  == 1)]["name"].tolist()
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
                tag = (" [C]" if row["captain"] else " [V]" if row["vicecaptain"] else "")
                suffix = " [TM]" if row["booster"] == "TM" else ""
                entries.append(f"{row['name']}{tag}{suffix} ({row['xP']})")
            if entries:
                lines.append(f"  {pos:3}: {', '.join(entries)}")

        bench_rows = picks_df[
            (picks_df["round"] == r) & (picks_df["bench"] >= 0)
        ].sort_values("bench")
        lines.append(f"  BEN: {', '.join(bench_rows['name'].tolist())}")
        lines.append(f"  Lineup xP: {lineup_rows['xp_contrib'].sum():.2f}")
        lines.append("")

    return "\n".join(lines), move_summary


# ── Entry point ────────────────────────────────────────────────────────────────

def solve_regular_wc(runtime_options: dict | None = None):
    """
    Load settings, run the solver, print and save results.

    Can be called directly with a runtime_options dict, or from the CLI.
    """

    # ── CLI arguments ─────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="WC Fantasy 2026 ILP solver",
        add_help=True,
    )
    parser.add_argument("--config",          type=str,  default=None,
                        help="Path to a JSON settings file (merged on top of wc_settings.json)")
    parser.add_argument("--next_round",      type=int,  default=None)
    parser.add_argument("--horizon",         type=int,  default=None)
    parser.add_argument("--num_iterations",  type=int,  default=None)
    parser.add_argument("--use_wildcard",    type=int,  nargs="+", default=None,
                        help="Round(s) to force Wildcard (e.g. --use_wildcard 2)")
    parser.add_argument("--use_twelfth_man", type=int,  nargs="+", default=None)
    parser.add_argument("--use_max_captain", type=int,  nargs="+", default=None)
    parser.add_argument("--use_qual_booster",type=int,  nargs="+", default=None)
    parser.add_argument("--secs",            type=int,  default=None)
    parser.add_argument("--verbose",         action="store_true", default=None)

    # Only parse args when invoked from CLI (not when called from a notebook)
    cli_args = vars(parser.parse_args()) if runtime_options is None else {}
    config_path = cli_args.pop("config", None)

    # ── Load settings (defaults → user → --config file) ───────────────────────
    options = load_settings(config_path)

    # ── Apply non-None CLI args on top ────────────────────────────────────────
    for k, v in cli_args.items():
        if v is not None:
            options[k] = v

    # ── Apply runtime_options with highest priority ────────────────────────────
    if runtime_options:
        options.update(runtime_options)

    # ── Solve ─────────────────────────────────────────────────────────────────
    data     = prep_wc_data(options)
    response = solve_wc(data, options)

    # ── Save + print ──────────────────────────────────────────────────────────
    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for result in response:
        if options.get("print_summary", True):
            print(
                f"\n=== Solution {result['iter'] + 1}  "
                f"| Score: {result['score']:.2f}  "
                f"| xP: {result['total_xp']:.2f}  "
                f"| Boosters: {result['boosters']} ==="
            )
            print(result["summary"])

        if options.get("save_results", True):
            out_path = results_dir / f"wc_{stamp}_iter{result['iter']}.csv"
            result["picks"].to_csv(out_path, index=False)
            print(f"Saved → {out_path}")

    summary_rows = [
        {"iter": r["iter"] + 1, "score": r["score"], "total_xp": r["total_xp"],
         "buy": r["buy"], "sell": r["sell"], "boosters": r["boosters"]}
        for r in response
    ]
    result_table = (
        pd.DataFrame(summary_rows)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

    if options.get("print_summary", True):
        print("\n" + result_table.to_string(index=False))

    return result_table, response


if __name__ == "__main__":
    solve_regular_wc()

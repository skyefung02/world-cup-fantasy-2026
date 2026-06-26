import os

import numpy as np
import pandas as pd

from scoring import (
    GOAL_PTS, CLEAN_SHEET_PTS, GOALS_CONCEDED_PTS, ASSIST_PTS,
    PEN_PROB, PEN_CONVERSION, SET_PIECE_ASSIST_PROB,
    SCOUTING_BONUS_PTS, SCOUTING_RAMP_LO, SCOUTING_RAMP_HI,
)

PROCESSED_DIR = "data/processed"
XMINS_PATH = "data/xmins.csv"
DEFAULT_XMINS_PATH = "data/default_xmins.csv"
ROUNDS = (1, 2, 3)  # group-stage rounds (hardcoded across app/templates)
XG_OVERRIDES_PATH = "data/xg_overrides.csv"
SCOUTING_OVERRIDES_PATH = "data/scouting_overrides.csv"
SET_PIECE_TAKERS_PATH = "data/set_piece_takers.csv"

# Knockout stage: team-level Monte Carlo aggregates (opponent-mix xG/xGA + P(play))
# from build_knockout_projections.build_team_rounds, plus confirmed-fixture cards.
# The base state crosses these with the per-player constants so the live engine can
# project rounds 4..8 with the same xMins/share/scouting edits as the group stage.
KO_TEAM_ROUNDS_PATH = "data/knockout_team_rounds.csv"
KO_FIXTURES_PATH = "data/knockout_fixtures.csv"
KO_ROUNDS = (4, 5, 6, 7, 8)  # R32, R16, QF, SF, Final/3rd-place

PEN_LOCK_FRACTION = PEN_PROB * PEN_CONVERSION

# Elo->goals calibration (fit against SilverBulletin 48-team group-stage projections,
# measured on the full pipeline incl. the tactical-tilt multiplier below).
# ELO_DIVISOR widens the strength spread; GOALS_SCALE corrects a uniform level offset
# on both xG scored and conceded. See win_expectancy / expected_goals.
ELO_DIVISOR = 348
GOALS_SCALE = 0.935


# --- Model functions ---

def win_expectancy(elo_team, elo_opp):
    """Standard Elo win expectancy formula. Neutral ground.

    Divisor calibrated to 348 (vs the textbook 400) so the strength->goals
    spread matches the SilverBulletin reference; the wider scale stops the
    model from compressing the gap between elite and minnow sides.
    """
    return 1 / (1 + 10 ** ((elo_opp - elo_team) / ELO_DIVISOR))


def expected_goals(we):
    """Quartic polynomial: win expectancy → xG scored. Two-regime model."""
    low = (
        3.90388 * we**4
        - 0.58486 * we**3
        - 2.98315 * we**2
        + 3.13160 * we
        + 0.33193
    )
    high = (
        308097.45501 * (we - 0.9)**4
        - 42803.04696 * (we - 0.9)**3
        + 2116.35304  * (we - 0.9)**2
        - 9.61869     * (we - 0.9)
        + 2.86899
    )
    return np.where(we < 0.9, low, high)


def clean_sheet_prob(xg_conceded):
    return np.exp(-xg_conceded)


def appearance_pts(xmins):
    if xmins == 0:
        return 0
    elif xmins < 60:
        return 1
    else:
        return 2


def _appearance_pts_vec(xmins_arr):
    return np.where(xmins_arr == 0, 0.0, np.where(xmins_arr < 60, 1.0, 2.0))


# ─────────────────────────────────────────────────────────────────────────────
# Cached base state
#
# Everything that does NOT depend on user xmins/override edits gets computed
# once at first access and cached. Restart the app to refresh (e.g. after
# fetch_data or weight_table edits).
# ─────────────────────────────────────────────────────────────────────────────

_base_state = None


def get_base_state(force=False):
    global _base_state
    if _base_state is None or force:
        _base_state = _precompute_base_state()
    return _base_state


def invalidate_base_state():
    global _base_state
    _base_state = None


def get_default_xmins_map():
    """Return {player_id: int(default_xmins)} from the cached base state."""
    df = get_base_state()
    unique = df.drop_duplicates("id")
    return {int(pid): int(round(x)) for pid, x in zip(unique["id"], unique["default_xmins"])}


# ─────────────────────────────────────────────────────────────────────────────
# xMins maps are per-round: canonical shape is nested {player_id: {round_id: mins}}.
# normalize_xmins() accepts either that or the legacy flat {player_id: mins} (which
# it expands to all rounds), so old CSVs / session blobs / API payloads still load.
# ─────────────────────────────────────────────────────────────────────────────

def normalize_xmins(raw):
    """Coerce a flat or nested xmins map into nested {int(id): {int(round): int(mins)}}."""
    out = {}
    for pid, val in (raw or {}).items():
        pid = int(pid)
        if isinstance(val, dict):
            rounds = {int(r): int(m) for r, m in val.items()}
        else:
            rounds = {r: int(val) for r in ROUNDS}
        if rounds:
            out[pid] = rounds
    return out


def _flatten_xmins(nested):
    """Nested {id: {round: mins}} → flat {(id, round): mins} for a MultiIndex lookup."""
    return {
        (int(pid), int(rnd)): int(mins)
        for pid, rounds in (nested or {}).items()
        for rnd, mins in rounds.items()
    }


def _precompute_base_state():
    """Build the immutable per-fixture × per-player table."""
    df = pd.read_csv(f"{PROCESSED_DIR}/player_fixtures.csv")

    # percentSelected may be missing on pre-scouting-bonus snapshots; default to 0
    # (treat as low-owned → conservative, lets bonus still compute)
    if "percentSelected" not in df.columns:
        df["percentSelected"] = 0.0
    df["percentSelected"] = df["percentSelected"].fillna(0.0)

    weights = pd.read_csv("data/weight_table.csv")[
        ["player", "team", "gls_p90", "ast_p90", "league_strength"]
    ].drop_duplicates(subset=["player", "team"])
    df = df.merge(weights, on=["player", "team"], how="left")

    pos_avg = df.groupby("position")[["gls_p90", "ast_p90"]].transform("mean")
    df["gls_p90"]         = df["gls_p90"].fillna(pos_avg["gls_p90"])
    df["ast_p90"]         = df["ast_p90"].fillna(pos_avg["ast_p90"])
    df["league_strength"] = df["league_strength"].fillna(1.0)

    # Shrink raw club rates toward positional mean
    SHRINKAGE = {
        "GK":  {"gls": 1.0, "ast": 1.0},
        "DEF": {"gls": 0.7, "ast": 0.5},
        "MID": {"gls": 0.4, "ast": 0.3},
        "FWD": {"gls": 0.2, "ast": 0.2},
    }
    pos_mean = df.groupby("position")[["gls_p90", "ast_p90"]].transform("mean")
    for pos, lams in SHRINKAGE.items():
        mask = df["position"] == pos
        df.loc[mask, "gls_p90"] = (1 - lams["gls"]) * df.loc[mask, "gls_p90"] + lams["gls"] * pos_mean.loc[mask, "gls_p90"]
        df.loc[mask, "ast_p90"] = (1 - lams["ast"]) * df.loc[mask, "ast_p90"] + lams["ast"] * pos_mean.loc[mask, "ast_p90"]

    # Elo → team xG (per fixture). Home-field advantage is added match-by-match to
    # the host's rating: all 2026 group games for a host are played in-country, and
    # the three hosts are drawn into separate groups so they never face each other,
    # so applying each side's own bonus is correct (it's 0 for non-hosts).
    home_adv     = df["home_field"].fillna(0.0)
    opp_home_adv = df["opp_home_field"].fillna(0.0)
    df["win_exp"]     = win_expectancy(df["elo"] + home_adv, df["opp_elo"] + opp_home_adv)
    df["xg_scored"]   = expected_goals(df["win_exp"].values) * GOALS_SCALE
    df["xg_conceded"] = expected_goals(1 - df["win_exp"].values) * GOALS_SCALE

    TILT_K = 0.25
    df["match_tilt"]    = df["tactical_tilt"] + df["opp_tactical_tilt"]
    df["xg_scored"]    *= (1 + TILT_K * df["match_tilt"])
    df["xg_conceded"]  *= (1 + TILT_K * df["match_tilt"])
    df["p_clean_sheet"] = clean_sheet_prob(df["xg_conceded"].values)

    # Per-player per-minute weight constants (multiply by xmins → goal_w / assist_w)
    is_outfield = df["position"] != "GK"
    df["goal_w_per_min"]   = np.where(is_outfield, df["gls_p90"] * df["league_strength"], 0.0)
    df["assist_w_per_min"] = np.where(is_outfield, df["ast_p90"] * df["league_strength"], 0.0)

    # Default xmins from data-driven model (international stats — see notebook 05)
    default_xmins_df = pd.read_csv(DEFAULT_XMINS_PATH)[["player", "team", "default_xmins"]]
    df = df.merge(default_xmins_df, on=["player", "team"], how="left")
    df["default_xmins"] = df["default_xmins"].fillna(0)

    # Set-piece / penalty taker flags (immutable per-player metadata)
    df["is_pen_taker"] = False
    df["is_sp_assist_taker"] = False
    if os.path.exists(SET_PIECE_TAKERS_PATH):
        takers = pd.read_csv(SET_PIECE_TAKERS_PATH)
        pen_ids = set(takers.loc[takers["role"] == "penalty", "id"].astype(int))
        sp_ids  = set(takers.loc[takers["role"] == "set_piece_assist", "id"].astype(int))
        # GK pen takers disallowed — silently drop them
        outfield_mask = df["position"] != "GK"
        df.loc[outfield_mask & df["id"].isin(pen_ids), "is_pen_taker"] = True
        df.loc[df["id"].isin(sp_ids), "is_sp_assist_taker"] = True

    # Group fixtures always happen: P(play) = 1. Knockout rows (appended below)
    # carry the modelled P(reach round); exp_pts = xpts_game * p_play folds it in.
    df["p_play"] = 1.0

    # Append knockout rounds 4..8 as additional per-(player, round) rows, reusing
    # the immutable per-player constants. Team xG/xGA/P(play) come from the cached
    # Monte Carlo aggregates (confirmed fixtures override resolved rounds). No-op if
    # the KO sim hasn't been built yet.
    meta_cols = ["id", "player", "position", "price", "team", "abbr",
                 "goal_w_per_min", "assist_w_per_min", "default_xmins",
                 "is_pen_taker", "is_sp_assist_taker", "percentSelected"]
    ko_rows = _knockout_base_rows(df.drop_duplicates("id")[meta_cols].copy())
    if not ko_rows.empty:
        df = pd.concat([df, ko_rows], ignore_index=True)

    return df


def _knockout_base_rows(meta):
    """Cross the per-player constants with KO rounds 4..8.

    Each (player, round) row gets team-level opponent-mix xG/xGA + P(play) from
    knockout_team_rounds.csv. Confirmed fixtures (knockout_fixtures.csv) override
    any resolved round with the head-to-head xG and P(play)=1, and name the real
    opponent — so the player table stays consistent with the fixture cards.
    Returns an empty frame if the KO sim hasn't been built.
    """
    if not os.path.exists(KO_TEAM_ROUNDS_PATH):
        return pd.DataFrame(columns=list(meta.columns))

    tr = pd.read_csv(KO_TEAM_ROUNDS_PATH)
    mc = {(r.abbr, int(r.round)): (r.cond_scored, r.cond_conceded, r.p_play)
          for r in tr.itertuples(index=False)}

    confirmed = {}  # (abbr, round) -> (xg_scored, xg_conceded, opp_abbr)
    if os.path.exists(KO_FIXTURES_PATH):
        fx = pd.read_csv(KO_FIXTURES_PATH)
        for f in fx.itertuples(index=False):
            confirmed[(f.home_abbr, int(f.round))] = (f.home_xg, f.away_xg, f.away_abbr)
            confirmed[(f.away_abbr, int(f.round))] = (f.away_xg, f.home_xg, f.home_abbr)

    frames = []
    for rnd in KO_ROUNDS:
        rows = meta.copy()
        rows["round_id"] = rnd
        xg_s, xg_c, pplay, opp = [], [], [], []
        for ab in rows["abbr"]:
            if (ab, rnd) in confirmed:
                cs, cc, o = confirmed[(ab, rnd)]
                xg_s.append(cs); xg_c.append(cc); pplay.append(1.0); opp.append(o)
            elif (ab, rnd) in mc:
                cs, cc, p = mc[(ab, rnd)]
                xg_s.append(cs); xg_c.append(cc); pplay.append(p); opp.append(np.nan)
            else:  # team with no KO data (shouldn't happen — all 48 are in the table)
                xg_s.append(0.0); xg_c.append(0.0); pplay.append(0.0); opp.append(np.nan)
        rows["xg_scored"]    = xg_s
        rows["xg_conceded"]  = xg_c
        rows["p_play"]       = pplay
        rows["opp_abbr"]     = opp
        rows["p_clean_sheet"] = clean_sheet_prob(np.asarray(xg_c, dtype=float))
        frames.append(rows)
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Fast recompute path
#
# Given a (possibly partial) xmins/override edit, recompute allocations and
# points for the affected teams only. Touches ~1 team × 3 rounds × ~55 players
# instead of the full ~4200-row table.
# ─────────────────────────────────────────────────────────────────────────────

def _apply_allocations_and_points(df, xmins_map, override_map, scouting_off=None):
    """In-place: compute per-row player_xg/xa across three slices (pen + SP-assist + open-play),
    apply overrides to the open-play slice only, then derive all pts columns.

    scouting_off: iterable of player ids whose scouting bonus is forced off.
    """
    scouting_off = set(int(i) for i in scouting_off) if scouting_off else set()
    # Per-round override lookup keyed by (id, round_id); any round the user hasn't
    # touched falls back to the player's per-player default_xmins.
    flat_xmins = _flatten_xmins(normalize_xmins(xmins_map))
    key = pd.MultiIndex.from_arrays([df["id"].astype(int), df["round_id"].astype(int)])
    df["xmins"] = (
        pd.Series(key.map(flat_xmins), index=df.index)
        .fillna(df["default_xmins"])
        .astype(float)
    )
    df["goal_w"]   = df["goal_w_per_min"]   * df["xmins"]
    df["assist_w"] = df["assist_w_per_min"] * df["xmins"]

    # Per-team locked slices (penalty goals, set-piece assists)
    df["locked_pen_xg"]         = 0.0
    df["locked_sp_xa"]          = 0.0
    df["open_play_goal_pool"]   = df["xg_scored"].astype(float)
    df["open_play_assist_pool"] = df["xg_scored"].astype(float) * 0.75

    for (_team, _round_id), group_idx in df.groupby(["team", "round_id"]).groups.items():
        group = df.loc[group_idx]
        team_xg = float(group["xg_scored"].iloc[0])
        team_xa = team_xg * 0.75

        pen_eligible = group[group["is_pen_taker"] & (group["xmins"] > 0)]
        if not pen_eligible.empty:
            pen_target = team_xg * PEN_LOCK_FRACTION
            weights = pen_eligible["xmins"] / 90.0
            df.loc[pen_eligible.index, "locked_pen_xg"] = pen_target * weights / weights.sum()
            df.loc[group_idx, "open_play_goal_pool"] = team_xg - pen_target

        sp_eligible = group[group["is_sp_assist_taker"] & (group["xmins"] > 0)]
        if not sp_eligible.empty:
            sp_target = team_xa * SET_PIECE_ASSIST_PROB
            weights = sp_eligible["xmins"] / 90.0
            df.loc[sp_eligible.index, "locked_sp_xa"] = sp_target * weights / weights.sum()
            df.loc[group_idx, "open_play_assist_pool"] = team_xa - sp_target

    # Open-play allocation against the remaining pools
    df["goal_w_sum"]   = df.groupby(["team", "round_id"])["goal_w"].transform("sum")
    df["assist_w_sum"] = df.groupby(["team", "round_id"])["assist_w"].transform("sum")
    safe_gws = df["goal_w_sum"].replace(0, np.nan)
    safe_aws = df["assist_w_sum"].replace(0, np.nan)
    df["open_play_xg"] = (df["open_play_goal_pool"]   * df["goal_w"]   / safe_gws).fillna(0.0)
    df["open_play_xa"] = (df["open_play_assist_pool"] * df["assist_w"] / safe_aws).fillna(0.0)

    df["player_xg"] = df["open_play_xg"] + df["locked_pen_xg"]
    df["player_xa"] = df["open_play_xa"] + df["locked_sp_xa"]

    # Model defaults (full-pool share, includes any locked slices)
    df["model_goal_share"]   = (df["player_xg"] / df["xg_scored"]).fillna(0.0)
    df["model_assist_share"] = (df["player_xa"] / (df["xg_scored"] * 0.75)).fillna(0.0)

    df["override_goal_share"]   = np.nan
    df["override_assist_share"] = np.nan

    # Overrides operate on the open-play slice only. The user's slider value is a
    # full-pool per-90 share; we convert it to an open-play target by subtracting
    # the locked amount (clamped at 0 if the override is below the lock).
    if override_map:
        ov_ids = set(override_map.keys())
        for (_team, _round_id), group_idx in df.groupby(["team", "round_id"]).groups.items():
            group = df.loc[group_idx]
            overridden = group[group["id"].isin(ov_ids)]
            if overridden.empty:
                continue
            team_xg = float(group["xg_scored"].iloc[0])
            team_xa = team_xg * 0.75
            op_goal_pool   = float(group["open_play_goal_pool"].iloc[0])
            op_assist_pool = float(group["open_play_assist_pool"].iloc[0])

            total_op_xg_locked = total_op_xa_locked = 0.0
            for idx in overridden.index:
                ov = override_map[int(df.loc[idx, "id"])]
                xmins_scale = df.loc[idx, "xmins"] / 90.0
                target_full_xg = team_xg * ov["goal_share"]   * xmins_scale
                target_full_xa = team_xa * ov["assist_share"] * xmins_scale
                op_target_xg = max(0.0, target_full_xg - df.loc[idx, "locked_pen_xg"])
                op_target_xa = max(0.0, target_full_xa - df.loc[idx, "locked_sp_xa"])
                df.loc[idx, "open_play_xg"]          = op_target_xg
                df.loc[idx, "open_play_xa"]          = op_target_xa
                df.loc[idx, "override_goal_share"]   = ov["goal_share"]
                df.loc[idx, "override_assist_share"] = ov["assist_share"]
                total_op_xg_locked += op_target_xg
                total_op_xa_locked += op_target_xa

            non_ov = group[~group["id"].isin(ov_ids) & (group["position"] != "GK")]
            if non_ov.empty:
                continue
            gw_sum = df.loc[non_ov.index, "goal_w"].sum()
            aw_sum = df.loc[non_ov.index, "assist_w"].sum()
            remaining_op_xg = max(0.0, op_goal_pool   - total_op_xg_locked)
            remaining_op_xa = max(0.0, op_assist_pool - total_op_xa_locked)
            if gw_sum > 0:
                df.loc[non_ov.index, "open_play_xg"] = remaining_op_xg * df.loc[non_ov.index, "goal_w"] / gw_sum
            if aw_sum > 0:
                df.loc[non_ov.index, "open_play_xa"] = remaining_op_xa * df.loc[non_ov.index, "assist_w"] / aw_sum

        df["player_xg"] = df["open_play_xg"] + df["locked_pen_xg"]
        df["player_xa"] = df["open_play_xa"] + df["locked_sp_xa"]

    df["goal_share"]   = (df["player_xg"] / df["xg_scored"]).fillna(0.0)
    df["assist_share"] = (df["player_xa"] / (df["xg_scored"] * 0.75)).fillna(0.0)

    conceded_rate = df["position"].map(GOALS_CONCEDED_PTS).fillna(0)
    df["app_pts"]      = _appearance_pts_vec(df["xmins"].values)
    df["goal_pts"]     = df["player_xg"] * df["position"].map(GOAL_PTS).fillna(0)
    df["assist_pts"]   = df["player_xa"] * ASSIST_PTS
    df["cs_pts"]       = df["p_clean_sheet"] * df["position"].map(CLEAN_SHEET_PTS).fillna(0) * (df["xmins"] >= 60).astype(float)
    df["conceded_pts"] = (df["xg_conceded"] - 1 + np.exp(-df["xg_conceded"])) * conceded_rate

    # Scouting bonus: +2 pts if player scores >4 in a match AND ownership <5%.
    # Closed-form P(pts>4) under independence of G/A/CS:
    #   Starter (xmins≥60), FWD/MID: 1 − exp(−(xg+xa))      (CS alone ≤3 pts, doesn't qualify)
    #   Starter, DEF/GK:             1 − exp(−(xg+xa))·(1−p_cs)
    #   Sub (0<xmins<60), all pos:   1 − exp(−xg)            (assist alone = 4, doesn't qualify)
    #   xmins=0:                     0
    xmins_arr = df["xmins"].values
    xg_arr    = df["player_xg"].values
    xa_arr    = df["player_xa"].values
    p_cs_arr  = df["p_clean_sheet"].values
    cs_helps  = df["position"].isin(["DEF", "GK"]).values

    p_no_ga = np.exp(-(xg_arr + xa_arr))
    p_starter = np.where(cs_helps, 1.0 - p_no_ga * (1.0 - p_cs_arr), 1.0 - p_no_ga)
    p_sub     = 1.0 - np.exp(-xg_arr)
    p_pts_gt_4 = np.where(
        xmins_arr == 0, 0.0,
        np.where(xmins_arr >= 60, p_starter, p_sub),
    )

    # Eligibility is a soft ramp on current ownership (we don't know deadline ownership):
    # full bonus at/below LO, zero at/above HI, linear between. A per-player toggle forces
    # eligibility to 0 ("I'm sure this player will be ≥5% owned at the deadline").
    own = df["percentSelected"].values
    p_eligible = np.clip(
        (SCOUTING_RAMP_HI - own) / (SCOUTING_RAMP_HI - SCOUTING_RAMP_LO), 0.0, 1.0
    )
    if scouting_off:
        p_eligible = np.where(df["id"].isin(scouting_off).values, 0.0, p_eligible)
    df["p_scouting_eligible"] = p_eligible
    df["p_scouting_bonus"]    = p_pts_gt_4
    df["scouting_bonus_pts"]  = p_pts_gt_4 * p_eligible * SCOUTING_BONUS_PTS

    df["xpts_game"]    = (df["goal_pts"] + df["assist_pts"] + df["cs_pts"]
                          + df["conceded_pts"] + df["app_pts"] + df["scouting_bonus_pts"])

    # Unconditional expected points: discount the per-match (conditional) points by
    # P(play). Group rows have p_play = 1, so exp_pts == xpts_game there; knockout
    # rows fold in the probability the team even reaches that round.
    p_play = df["p_play"] if "p_play" in df.columns else 1.0
    df["exp_pts"] = df["xpts_game"] * p_play
    return df


def recompute_teams(xmins_map=None, override_map=None, teams=None, scouting_off=None):
    """Fast path. Returns { player_id: { round_id: { 'Pts': ..., 'xMins': ..., ... } } }
    for players on the specified teams (or all teams if teams is None).
    """
    base = get_base_state()
    if teams is not None:
        sliced = base[base["abbr"].isin(teams)].copy()
    else:
        sliced = base.copy()
    df = _apply_allocations_and_points(sliced, xmins_map or {}, override_map or {}, scouting_off)

    out = {}
    for _, r in df.iterrows():
        pid = int(r["id"])
        rd  = int(r["round_id"])
        out.setdefault(pid, {})[rd] = {
            "Pts":                 round(float(r["exp_pts"]), 2),
            "PtsCond":             round(float(r["xpts_game"]), 2),
            "PPlay":               float(r["p_play"]),
            "xMins":               float(r["xmins"]),
            "xG":                  float(r["player_xg"]),
            "xA":                  float(r["player_xa"]),
            "GoalShare":           float(r["goal_share"]),
            "AssistShare":         float(r["assist_share"]),
            "ModelGoalShare":      float(r["model_goal_share"]),
            "ModelAssistShare":    float(r["model_assist_share"]),
            "OverrideGoalShare":   (None if pd.isna(r["override_goal_share"])   else float(r["override_goal_share"])),
            "OverrideAssistShare": (None if pd.isna(r["override_assist_share"]) else float(r["override_assist_share"])),
            "OppAbbr":             r["opp_abbr"],
            "TeamXG":              float(r["xg_scored"]),
            "LockedPenXg":         float(r["locked_pen_xg"]),
            "LockedSpXa":          float(r["locked_sp_xa"]),
            "ScoutingBonusPts":    float(r["scouting_bonus_pts"]),
            "PScoutingBonus":      float(r["p_scouting_bonus"]),
            "PScoutingEligible":   float(r["p_scouting_eligible"]),
            "PercentSelected":     float(r["percentSelected"]),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline (CSV-shaped output, used for download + run())
# ─────────────────────────────────────────────────────────────────────────────

EXPORT_COLS = {
    "exp_pts":               "Pts",
    "xpts_game":             "PtsCond",
    "p_play":                "PPlay",
    "xmins":                 "xMins",
    "player_xg":             "xG",
    "player_xa":             "xA",
    "goal_pts":              "GoalPts",
    "assist_pts":            "AssistPts",
    "cs_pts":                "CSPts",
    "conceded_pts":          "ConcededPts",
    "app_pts":               "AppPts",
    "scouting_bonus_pts":    "ScoutingBonusPts",
    "p_scouting_bonus":      "PScoutingBonus",
    "p_scouting_eligible":   "PScoutingEligible",
    "percentSelected":       "PercentSelected",
    "opp_abbr":              "OppAbbr",
    "xg_scored":             "TeamXG",
    "xg_conceded":           "TeamXGA",
    "p_clean_sheet":         "PCleanSheet",
    "goal_share":            "GoalShare",
    "assist_share":          "AssistShare",
    "model_goal_share":      "ModelGoalShare",
    "model_assist_share":    "ModelAssistShare",
    "override_goal_share":   "OverrideGoalShare",
    "override_assist_share": "OverrideAssistShare",
    "locked_pen_xg":         "LockedPenXg",
    "locked_sp_xa":          "LockedSpXa",
}


def build_full_projections(xmins_map=None, override_map=None, scouting_off=None):
    """Return the wide-format DataFrame matching the historical projections.csv schema."""
    base = get_base_state().copy()
    base = _apply_allocations_and_points(base, xmins_map or {}, override_map or {}, scouting_off)

    metadata = base[["id", "player", "position", "price", "team", "abbr"]].drop_duplicates("id")
    df_export = metadata.copy()

    rounds = sorted(base["round_id"].unique())
    for col, suffix in EXPORT_COLS.items():
        pivot = base.pivot_table(
            index="id", columns="round_id", values=col, aggfunc="first", dropna=False
        ).reset_index()
        pivot.columns = ["id"] + [f"{int(r)}_{suffix}" for r in pivot.columns[1:]]
        df_export = df_export.merge(pivot, on="id", how="left")

    col_order = ["id", "player", "position", "price", "team", "abbr"] + [
        f"{int(r)}_{suffix}" for r in rounds for suffix in EXPORT_COLS.values()
    ]
    df_export = df_export[col_order].sort_values("id").reset_index(drop=True)

    round_pts_cols = [f"{int(r)}_Pts" for r in rounds]
    for c in round_pts_cols:
        df_export[c] = df_export[c].round(2)
    return df_export


# ─────────────────────────────────────────────────────────────────────────────
# Local CSV I/O (creator workflow)
# ─────────────────────────────────────────────────────────────────────────────

def load_xmins_csv():
    """Read data/xmins.csv into nested {id: {round_id: mins}}.

    Long format (id,round_id,xmins) is the current schema; a legacy file with just
    (id,xmins) is read as a flat map and expanded to all rounds by normalize_xmins().
    """
    if not os.path.exists(XMINS_PATH):
        return {}
    df = pd.read_csv(XMINS_PATH)
    if "round_id" in df.columns:
        nested = {}
        for r in df.itertuples(index=False):
            nested.setdefault(int(r.id), {})[int(r.round_id)] = int(r.xmins)
        return nested
    return normalize_xmins(df.set_index("id")["xmins"].to_dict())


def load_overrides_csv():
    if os.path.exists(XG_OVERRIDES_PATH):
        df = pd.read_csv(XG_OVERRIDES_PATH)
        return {
            int(r["id"]): {
                "goal_share":   float(r["goal_share"]),
                "assist_share": float(r["assist_share"]),
            }
            for _, r in df.iterrows()
        }
    return {}


def load_scouting_csv():
    """Read data/scouting_overrides.csv → set of player ids with the scouting bonus forced off."""
    if os.path.exists(SCOUTING_OVERRIDES_PATH):
        return set(int(i) for i in pd.read_csv(SCOUTING_OVERRIDES_PATH)["id"])
    return set()


def run():
    """Backwards-compat entry point: read local CSVs, compute full projections, write to disk."""
    print("Loading data...")
    xmins_map = load_xmins_csv()
    override_map = load_overrides_csv()
    scouting_off = load_scouting_csv()
    print("Applying Elo model and computing allocations...")
    df_export = build_full_projections(xmins_map, override_map, scouting_off)
    df_export.to_csv("data/projections.csv", index=False)
    print(f"Exported {len(df_export)} players to data/projections.csv")

    # Spot check
    round_pts_cols = [c for c in df_export.columns if c.endswith("_Pts")]
    eng = df_export[df_export["abbr"] == "ENG"].copy()
    if len(eng):
        eng["xpts_total"] = sum(eng[c] for c in round_pts_cols)
        print("\nEngland top 10:")
        print(eng.sort_values("xpts_total", ascending=False).head(10)[
            ["player", "position", "price"] + round_pts_cols + ["xpts_total"]
        ].to_string())


if __name__ == "__main__":
    run()

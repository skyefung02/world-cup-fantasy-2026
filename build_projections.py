import os

import numpy as np
import pandas as pd

from scoring import (
    GOAL_PTS, CLEAN_SHEET_PTS, GOALS_CONCEDED_PTS, ASSIST_PTS
)

PROCESSED_DIR = "data/processed"
XMINS_PATH = "data/xmins.csv"
XG_OVERRIDES_PATH = "data/xg_overrides.csv"


# --- Model functions ---

def win_expectancy(elo_team, elo_opp):
    """Standard Elo win expectancy formula. Neutral ground."""
    return 1 / (1 + 10 ** ((elo_opp - elo_team) / 400))


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


def assign_xmins(squad_group):
    mins_map = {
        'GK':  [90, 5, 5, 5],
        'DEF': [80, 80, 80, 80, 25, 25, 5, 5, 5],
        'MID': [75, 75, 75, 75, 35, 35, 10, 10, 5],
        'FWD': [70, 70, 35, 10, 10, 5, 5],
    }
    parts = []
    for pos, pos_group in squad_group.groupby('position'):
        pos_group = pos_group.sort_values('price', ascending=False).reset_index(drop=True)
        schedule = mins_map.get(pos, [60])
        pos_group['xmins'] = [schedule[i] if i < len(schedule) else 5 for i in range(len(pos_group))]
        parts.append(pos_group)
    return pd.concat(parts)


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


def _precompute_base_state():
    """Build the immutable per-fixture × per-player table."""
    df = pd.read_csv(f"{PROCESSED_DIR}/player_fixtures.csv")

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

    # Elo → team xG (per fixture)
    df["win_exp"]     = win_expectancy(df["elo"], df["opp_elo"])
    df["xg_scored"]   = expected_goals(df["win_exp"].values)
    df["xg_conceded"] = expected_goals(1 - df["win_exp"].values)

    TILT_K = 0.25
    df["match_tilt"]    = df["tactical_tilt"] + df["opp_tactical_tilt"]
    df["xg_scored"]    *= (1 + TILT_K * df["match_tilt"])
    df["xg_conceded"]  *= (1 + TILT_K * df["match_tilt"])
    df["p_clean_sheet"] = clean_sheet_prob(df["xg_conceded"].values)

    # Per-player per-minute weight constants (multiply by xmins → goal_w / assist_w)
    is_outfield = df["position"] != "GK"
    df["goal_w_per_min"]   = np.where(is_outfield, df["gls_p90"] * df["league_strength"], 0.0)
    df["assist_w_per_min"] = np.where(is_outfield, df["ast_p90"] * df["league_strength"], 0.0)

    # Default xmins schedule per player (used when no user override)
    players_meta = df[['id', 'squadId', 'position', 'price']].drop_duplicates('id')
    default_xmins_df = (
        players_meta
        .groupby('squadId', group_keys=False)
        .apply(assign_xmins)[['id', 'xmins']]
        .rename(columns={'xmins': 'default_xmins'})
    )
    df = df.merge(default_xmins_df, on='id', how='left')

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Fast recompute path
#
# Given a (possibly partial) xmins/override edit, recompute allocations and
# points for the affected teams only. Touches ~1 team × 3 rounds × ~55 players
# instead of the full ~4200-row table.
# ─────────────────────────────────────────────────────────────────────────────

def _apply_allocations_and_points(df, xmins_map, override_map):
    """In-place: compute per-row player_xg/xa, model/override/effective shares, and all pts columns."""
    df["xmins"] = df["id"].map(xmins_map).fillna(df["default_xmins"]).astype(float)
    df["goal_w"]   = df["goal_w_per_min"]   * df["xmins"]
    df["assist_w"] = df["assist_w_per_min"] * df["xmins"]
    df["goal_w_sum"]   = df.groupby(["team", "round_id"])["goal_w"].transform("sum")
    df["assist_w_sum"] = df.groupby(["team", "round_id"])["assist_w"].transform("sum")

    safe_gws = df["goal_w_sum"].replace(0, np.nan)
    safe_aws = df["assist_w_sum"].replace(0, np.nan)
    df["player_xg"] = (df["xg_scored"] * df["goal_w"] / safe_gws).fillna(0.0)
    df["player_xa"] = (df["xg_scored"] * 0.75 * df["assist_w"] / safe_aws).fillna(0.0)

    df["model_goal_share"]   = (df["player_xg"] / df["xg_scored"]).fillna(0.0)
    df["model_assist_share"] = (df["player_xa"] / (df["xg_scored"] * 0.75)).fillna(0.0)

    df["override_goal_share"]   = np.nan
    df["override_assist_share"] = np.nan

    if override_map:
        ov_ids = set(override_map.keys())
        for (team, round_id), group_idx in df.groupby(["team", "round_id"]).groups.items():
            group = df.loc[group_idx]
            overridden = group[group["id"].isin(ov_ids)]
            if overridden.empty:
                continue
            team_xg = group["xg_scored"].iloc[0]
            team_xa = team_xg * 0.75
            total_locked_xg = total_locked_xa = 0.0
            for idx in overridden.index:
                ov = override_map[int(df.loc[idx, "id"])]
                xmins_scale = df.loc[idx, "xmins"] / 90.0
                locked_xg = team_xg * ov["goal_share"]   * xmins_scale
                locked_xa = team_xa * ov["assist_share"] * xmins_scale
                df.loc[idx, "player_xg"]             = locked_xg
                df.loc[idx, "player_xa"]             = locked_xa
                df.loc[idx, "override_goal_share"]   = ov["goal_share"]
                df.loc[idx, "override_assist_share"] = ov["assist_share"]
                total_locked_xg += locked_xg
                total_locked_xa += locked_xa
            non_ov = group[~group["id"].isin(ov_ids) & (group["position"] != "GK")]
            if non_ov.empty:
                continue
            gw_sum = df.loc[non_ov.index, "goal_w"].sum()
            aw_sum = df.loc[non_ov.index, "assist_w"].sum()
            remaining_xg = max(0.0, team_xg - total_locked_xg)
            remaining_xa = max(0.0, team_xa - total_locked_xa)
            if gw_sum > 0:
                df.loc[non_ov.index, "player_xg"] = remaining_xg * df.loc[non_ov.index, "goal_w"] / gw_sum
            if aw_sum > 0:
                df.loc[non_ov.index, "player_xa"] = remaining_xa * df.loc[non_ov.index, "assist_w"] / aw_sum

    df["goal_share"]   = (df["player_xg"] / df["xg_scored"]).fillna(0.0)
    df["assist_share"] = (df["player_xa"] / (df["xg_scored"] * 0.75)).fillna(0.0)

    conceded_rate = df["position"].map(GOALS_CONCEDED_PTS).fillna(0)
    df["app_pts"]      = _appearance_pts_vec(df["xmins"].values)
    df["goal_pts"]     = df["player_xg"] * df["position"].map(GOAL_PTS).fillna(0)
    df["assist_pts"]   = df["player_xa"] * ASSIST_PTS
    df["cs_pts"]       = df["p_clean_sheet"] * df["position"].map(CLEAN_SHEET_PTS).fillna(0) * (df["xmins"] >= 60).astype(float)
    df["conceded_pts"] = (df["xg_conceded"] - 1 + np.exp(-df["xg_conceded"])) * conceded_rate
    df["xpts_game"]    = df["goal_pts"] + df["assist_pts"] + df["cs_pts"] + df["conceded_pts"] + df["app_pts"]
    return df


def recompute_teams(xmins_map=None, override_map=None, teams=None):
    """Fast path. Returns { player_id: { round_id: { 'Pts': ..., 'xMins': ..., ... } } }
    for players on the specified teams (or all teams if teams is None).
    """
    base = get_base_state()
    if teams is not None:
        sliced = base[base["abbr"].isin(teams)].copy()
    else:
        sliced = base.copy()
    df = _apply_allocations_and_points(sliced, xmins_map or {}, override_map or {})

    out = {}
    for _, r in df.iterrows():
        pid = int(r["id"])
        rd  = int(r["round_id"])
        out.setdefault(pid, {})[rd] = {
            "Pts":                 round(float(r["xpts_game"]), 2),
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
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline (CSV-shaped output, used for download + run())
# ─────────────────────────────────────────────────────────────────────────────

EXPORT_COLS = {
    "xpts_game":             "Pts",
    "xmins":                 "xMins",
    "player_xg":             "xG",
    "player_xa":             "xA",
    "goal_pts":              "GoalPts",
    "assist_pts":            "AssistPts",
    "cs_pts":                "CSPts",
    "conceded_pts":          "ConcededPts",
    "app_pts":               "AppPts",
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
}


def build_full_projections(xmins_map=None, override_map=None):
    """Return the wide-format DataFrame matching the historical projections.csv schema."""
    base = get_base_state().copy()
    base = _apply_allocations_and_points(base, xmins_map or {}, override_map or {})

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
    if os.path.exists(XMINS_PATH):
        return pd.read_csv(XMINS_PATH).set_index("id")["xmins"].to_dict()
    return {}


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


def run():
    """Backwards-compat entry point: read local CSVs, compute full projections, write to disk."""
    print("Loading data...")
    xmins_map = load_xmins_csv()
    override_map = load_overrides_csv()
    print("Applying Elo model and computing allocations...")
    df_export = build_full_projections(xmins_map, override_map)
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

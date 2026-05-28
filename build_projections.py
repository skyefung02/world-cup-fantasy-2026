import numpy as np
import pandas as pd

from scoring import (
    GOAL_PTS, CLEAN_SHEET_PTS, GOALS_CONCEDED_PTS, ASSIST_PTS
)

PROCESSED_DIR = "data/processed"


# --- Model functions ---

def win_expectancy(elo_team, elo_opp):
    """Standard Elo win expectancy formula. Neutral ground."""
    return 1 / (1 + 10 ** ((elo_opp - elo_team) / 400))


def expected_goals(we):
    """
    Quartic polynomial: win expectancy -> xG scored, neutral ground.
    Source: football-rankings.info, fitted on ~40,000 NT matches. R²=0.976.
    Two-regime model: We < 0.9 and We >= 0.9.
    """
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
    """Poisson probability of conceding zero goals."""
    return np.exp(-xg_conceded)



def appearance_pts(xmins):
    if xmins == 0:
        return 0
    elif xmins < 60:
        return 1
    else:
        return 2


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


# --- Main build function ---

def run():
    print("Loading data...")
    df = pd.read_csv(f"{PROCESSED_DIR}/player_fixtures.csv")

    # Merge player weights
    weights = pd.read_csv("data/weight_table.csv")[
        ["player", "team", "gls_p90", "ast_p90", "league_strength"]
    ].drop_duplicates(subset=["player", "team"])
    df = df.merge(weights, on=["player", "team"], how="left")
    pos_avg = df.groupby("position")[["gls_p90", "ast_p90"]].transform("mean")
    df["gls_p90"]        = df["gls_p90"].fillna(pos_avg["gls_p90"])
    df["ast_p90"]        = df["ast_p90"].fillna(pos_avg["ast_p90"])
    df["league_strength"] = df["league_strength"].fillna(1.0)

    print("Applying Elo model...")
    df["win_exp"]       = win_expectancy(df["elo"], df["opp_elo"])
    df["xg_scored"]     = expected_goals(df["win_exp"].values)
    df["xg_conceded"]   = expected_goals(1 - df["win_exp"].values)
    df["p_clean_sheet"] = clean_sheet_prob(df["xg_conceded"].values)

    print("Computing xMins...")
    players = df[['id', 'squadId', 'position', 'price']].drop_duplicates('id')
    default_xmins = (
        players
        .groupby('squadId', group_keys=False)
        .apply(assign_xmins)[['id', 'xmins']]
    )
    df_xmins = pd.read_csv("data/xmins.csv")
    merged_xmins = default_xmins.merge(df_xmins, on='id', how='left', suffixes=('_default', '_manual'))
    merged_xmins['xmins'] = merged_xmins['xmins_manual'].combine_first(merged_xmins['xmins_default'])
    df = df.merge(merged_xmins[['id', 'xmins']], on='id', how='left')
    df["xmins"] = df["xmins"].fillna(60)

    # Distribute team xG/xA across outfield players weighted by quality × league_strength × xmins.
    # GKs are excluded from the pool (weight forced to 0).
    is_outfield = df["position"] != "GK"
    df["goal_w"]   = np.where(is_outfield, df["gls_p90"] * df["league_strength"] * df["xmins"], 0)
    df["assist_w"] = np.where(is_outfield, df["ast_p90"] * df["league_strength"] * df["xmins"], 0)
    df["goal_w_sum"]   = df.groupby(["team", "round_id"])["goal_w"].transform("sum")
    df["assist_w_sum"] = df.groupby(["team", "round_id"])["assist_w"].transform("sum")

    df["player_xg"] = df["xg_scored"] * df["goal_w"] / df["goal_w_sum"]
    df["player_xa"] = df["xg_scored"] * 0.75 * df["assist_w"] / df["assist_w_sum"]

    # Points — computed directly, no /90 × xmins scaling needed
    conceded_rate = df["position"].map(GOALS_CONCEDED_PTS).fillna(0)
    df["app_pts"]      = df["xmins"].apply(appearance_pts)
    df["goal_pts"]     = df["player_xg"] * df["position"].map(GOAL_PTS)
    df["assist_pts"]   = df["player_xa"] * ASSIST_PTS
    df["cs_pts"]       = df["p_clean_sheet"] * df["position"].map(CLEAN_SHEET_PTS).fillna(0) * (df["xmins"] >= 60)
    df["conceded_pts"] = (df["xg_conceded"] - 1 + np.exp(-df["xg_conceded"])) * conceded_rate
    df["xpts_game"]    = df["goal_pts"] + df["assist_pts"] + df["cs_pts"] + df["conceded_pts"] + df["app_pts"]

    print("Building export...")
    EXPORT_COLS = {
        "xpts_game":    "Pts",
        "xmins":        "xMins",
        "player_xg":    "xG",
        "player_xa":    "xA",
        "goal_pts":     "GoalPts",
        "assist_pts":   "AssistPts",
        "cs_pts":       "CSPts",
        "conceded_pts": "ConcededPts",
        "app_pts":      "AppPts",
    }

    metadata = df[["id", "player", "position", "price", "team", "abbr"]].drop_duplicates("id")
    df_export = metadata.copy()

    rounds = sorted(df["round_id"].unique())
    for col, suffix in EXPORT_COLS.items():
        pivot = df.pivot_table(index="id", columns="round_id", values=col, aggfunc="first").reset_index()
        pivot.columns = ["id"] + [f"{int(r)}_{suffix}" for r in pivot.columns[1:]]
        df_export = df_export.merge(pivot, on="id")

    col_order = ["id", "player", "position", "price", "team", "abbr"] + [
        f"{int(r)}_{suffix}"
        for r in rounds
        for suffix in EXPORT_COLS.values()
    ]
    df_export = df_export[col_order].sort_values("id").reset_index(drop=True)

    round_pts_cols = [f"{int(r)}_Pts" for r in rounds]
    for c in round_pts_cols:
        df_export[c] = df_export[c].round(2)

    df_export.to_csv("data/projections.csv", index=False)

    print(f"Exported {len(df_export)} players to data/projections.csv")

    # Spot check
    print("\nEngland top 10:")
    eng = df_export[df_export["abbr"] == "ENG"].copy()
    eng["xpts_total"] = sum(eng[c] for c in round_pts_cols)
    print(eng.sort_values("xpts_total", ascending=False).head(10)[
        ["player", "position", "price"] + round_pts_cols + ["xpts_total"]
    ].to_string())


if __name__ == "__main__":
    run()
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


# Position group shares of team xG — calibrated to a 4-4-2 starting formation.
# Goals:   GK 1%,  DEF 10%, MID 35%, FWD 54%  (sum = 100%)
# Assists: GK 1%,  DEF 10%, MID 45%, FWD 44%  (sum = 100%)
GROUP_GOAL_SHARE   = {"GK": 0.01, "DEF": 0.10, "MID": 0.5, "FWD": 0.39}
GROUP_ASSIST_SHARE = {"GK": 0.01, "DEF": 0.10, "MID": 0.64, "FWD": 0.25}


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
        ["player", "team", "gls_p90_pct", "ast_p90_pct", "league_strength"]
    ].drop_duplicates(subset=["player", "team"])
    df = df.merge(weights, on=["player", "team"], how="left")
    df["gls_p90_pct"]    = df["gls_p90_pct"].fillna(0.5)
    df["ast_p90_pct"]    = df["ast_p90_pct"].fillna(0.5)
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

    # Weighted share within (team, position, fixture).
    # Weight = quality × league_strength × xmins so that a player with more
    # minutes gets a proportionally larger slice of the position's xG budget.
    df["goal_w"]   = df["gls_p90_pct"] * df["league_strength"] * df["xmins"]
    df["assist_w"] = df["ast_p90_pct"] * df["league_strength"] * df["xmins"]
    df["goal_w_sum"]   = df.groupby(["team", "position", "round_id"])["goal_w"].transform("sum")
    df["assist_w_sum"] = df.groupby(["team", "position", "round_id"])["assist_w"].transform("sum")

    # Player's absolute expected goals/assists for the game
    df["player_xg"] = (
        df["xg_scored"] * df["position"].map(GROUP_GOAL_SHARE)
        * df["goal_w"] / df["goal_w_sum"]
    )
    df["player_xa"] = (
        df["xg_scored"] * df["position"].map(GROUP_ASSIST_SHARE)
        * df["assist_w"] / df["assist_w_sum"]
    )

    # Points — computed directly, no /90 × xmins scaling needed
    conceded_rate = df["position"].map(GOALS_CONCEDED_PTS).fillna(0)
    df["app_pts"]  = df["xmins"].apply(appearance_pts)
    df["xpts_game"] = (
        df["player_xg"] * df["position"].map(GOAL_PTS)
        + df["player_xa"] * ASSIST_PTS
        + df["p_clean_sheet"] * df["position"].map(CLEAN_SHEET_PTS).fillna(0)
        + np.maximum(0, df["xg_conceded"] - 1) * conceded_rate
        + df["app_pts"]
    )

    print("Building export...")
    rounds_pivot = df.pivot_table(
        index="id", columns="round_id", values="xpts_game", aggfunc="first"
    ).reset_index()
    rounds_pivot.columns = ["id"] + [f"{int(c)}_Pts" for c in rounds_pivot.columns[1:]]

    xmins_pivot = df.pivot_table(
        index="id", columns="round_id", values="xmins", aggfunc="first"
    ).reset_index()
    xmins_pivot.columns = ["id"] + [f"{int(c)}_xMins" for c in xmins_pivot.columns[1:]]

    metadata = df[["id", "player", "position", "price", "team", "abbr"]].drop_duplicates("id")

    df_export = metadata.merge(rounds_pivot, on="id").merge(xmins_pivot, on="id")

    col_order = [
        "id", "player", "position", "price", "team", "abbr",
        "1_Pts", "1_xMins",
        "2_Pts", "2_xMins",
        "3_Pts", "3_xMins",
    ]
    df_export = df_export[col_order].sort_values("id").reset_index(drop=True)

    for c in ["1_Pts", "2_Pts", "3_Pts"]:
        df_export[c] = df_export[c].round(2)

    df_export.to_csv("data/projections.csv", index=False)

    print(f"Exported {len(df_export)} players to data/projections.csv")

    # Spot check
    print("\nEngland top 10:")
    eng = df_export[df_export["abbr"] == "ENG"].copy()
    eng["xpts_total"] = eng["1_Pts"] + eng["2_Pts"] + eng["3_Pts"]
    print(eng.sort_values("xpts_total", ascending=False).head(10)[
        ["player", "position", "price", "1_Pts", "2_Pts", "3_Pts", "xpts_total"]
    ].to_string())


if __name__ == "__main__":
    run()
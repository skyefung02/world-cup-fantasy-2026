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


def xpts_per_90(position, xg_scored, xg_conceded, p_clean_sheet):
    """
    Expected Fantasy points per 90 mins for one player.
    Per-player goal shares (position group share / typical starters):
      GK=1%, DEF=2.5%, MID=8.75%, FWD=27%
    """
    goal_share = {"GK": 0.01, "DEF": 0.025, "MID": 0.0875, "FWD": 0.27}
    assist_share = {"GK": 0.01, "DEF": 0.025, "MID": 0.1125, "FWD": 0.22}

    pos_xg      = xg_scored * goal_share[position]
    pos_assists = xg_scored * assist_share[position]

    pts_goals   = pos_xg * GOAL_PTS[position]
    pts_assists = pos_assists * ASSIST_PTS
    pts_cs      = p_clean_sheet * CLEAN_SHEET_PTS[position]

    if position in ("GK", "DEF"):
        pts_conceded = max(0, xg_conceded - 1) * GOALS_CONCEDED_PTS[position]
    else:
        pts_conceded = 0

    return pts_goals + pts_assists + pts_cs + pts_conceded


def appearance_pts(xmins):
    if xmins == 0:
        return 0
    elif xmins < 60:
        return 1
    else:
        return 2


# --- Main build function ---

def run():
    print("Loading data...")
    df = pd.read_csv(f"{PROCESSED_DIR}/player_fixtures.csv")
    df_xmins = pd.read_csv("data/xmins.csv")

    print("Applying Elo model...")
    df["win_exp"]      = win_expectancy(df["elo"], df["opp_elo"])
    df["xg_scored"]    = expected_goals(df["win_exp"].values)
    df["xg_conceded"]  = expected_goals(1 - df["win_exp"].values)
    df["p_clean_sheet"] = clean_sheet_prob(df["xg_conceded"].values)

    df["xpts_p90"] = df.apply(
        lambda r: xpts_per_90(
            r["position"], r["xg_scored"], r["xg_conceded"], r["p_clean_sheet"]
        ), axis=1
    )

    print("Merging xMins...")
    df = df.merge(df_xmins, on="id", how="left")
    df["xmins"] = df["xmins"].fillna(60)
    df["app_pts"] = df["xmins"].apply(appearance_pts)

    df["xpts_game"] = (df["xpts_p90"] / 90) * df["xmins"] + df["app_pts"]

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
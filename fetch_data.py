import json
import os
import pandas as pd

CACHE_DIR = "cache"
PROCESSED_DIR = "data/processed"


def load_cache(endpoint):
    path = os.path.join(CACHE_DIR, f"{endpoint}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cache file not found: {path}\n"
            f"Please manually save https://play.fifa.com/json/fantasy/{endpoint}.json "
            f"from DevTools into {path}"
        )
    with open(path) as f:
        return json.load(f)


def build_players(players_raw, squads_raw):
    df_players = pd.DataFrame(players_raw)
    # Drop players FIFA has flagged as no longer in the squad. The raw cache keeps
    # them; we only exclude them from processed/projection data.
    df_players = df_players[df_players["status"] == "playing"].copy()
    df_squads = pd.DataFrame(squads_raw)

    # Resolve display name
    df_players["player"] = df_players["knownName"].where(
        df_players["knownName"].notna(),
        df_players["firstName"] + " " + df_players["lastName"]
    )

    # Merge squad info
    df_players = df_players.merge(
        df_squads[["id", "name", "abbr", "group"]],
        left_on="squadId",
        right_on="id",
        suffixes=("", "_squad")
    ).rename(columns={"name": "team"}).drop(columns=["id_squad"])

    return df_players[[
        "id", "player", "position", "price", "status", "percentSelected",
        "squadId", "team", "abbr", "group"
    ]]


def build_fixtures(rounds_raw):
    records = []
    for r in rounds_raw:
        for fix in r["tournaments"]:
            records.append({
                "round_id":  r["id"],
                "fixture_id": fix["id"],
                "date":      fix["date"],
                "home_id":   fix["homeSquadId"],
                "away_id":   fix["awaySquadId"],
                "home_team": fix["homeSquadName"],
                "away_team": fix["awaySquadName"],
                "home_abbr": fix["homeSquadAbbr"],
                "away_abbr": fix["awaySquadAbbr"],
            })
    return pd.DataFrame(records)


def build_team_fixtures(df_fixtures):
    df_group = df_fixtures[df_fixtures["round_id"] <= 3].copy()

    home = df_group[["round_id", "home_id", "home_abbr", "away_id", "away_abbr"]].rename(columns={
        "home_id": "team_id", "home_abbr": "team_abbr",
        "away_id": "opp_id",  "away_abbr": "opp_abbr",
    })
    away = df_group[["round_id", "away_id", "away_abbr", "home_id", "home_abbr"]].rename(columns={
        "away_id": "team_id",  "away_abbr": "team_abbr",
        "home_id": "opp_id",   "home_abbr": "opp_abbr",
    })
    return pd.concat([home, away], ignore_index=True).sort_values(["team_id", "round_id"]).reset_index(drop=True)


def build_squads_rated(df_squads, df_elo, df_tilt):
    df_elo_clean  = df_elo[["Code", "PELE"]].rename(columns={"Code": "abbr", "PELE": "elo"})
    df_tilt_clean = df_tilt[["Code", "Tactical", "Personnel", "total_tilt", "tilt_cat"]].rename(columns={"Code": "abbr"})
    df_ratings = df_elo_clean.merge(df_tilt_clean, on="abbr")
    return df_squads.merge(df_ratings, on="abbr", how="left")


def build_player_fixtures(df_players, df_squads_rated, df_team_fixtures):
    elo_lookup           = df_squads_rated.set_index("abbr")["elo"].to_dict()
    tactical_tilt_lookup = df_squads_rated.set_index("abbr")["Tactical"].to_dict()

    df = df_players.merge(
        df_squads_rated[["id", "elo", "Tactical", "total_tilt", "tilt_cat"]],
        left_on="squadId", right_on="id", suffixes=("", "_squad")
    ).drop(columns=["id_squad"]).rename(columns={"Tactical": "tactical_tilt"})

    df = df.merge(
        df_team_fixtures[["team_abbr", "round_id", "opp_abbr"]],
        left_on="abbr", right_on="team_abbr"
    ).drop(columns=["team_abbr"])

    df["opp_elo"]           = df["opp_abbr"].map(elo_lookup)
    df["opp_tactical_tilt"] = df["opp_abbr"].map(tactical_tilt_lookup)
    return df


def run():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Loading cached FIFA data...")
    players_raw = load_cache("players")
    squads_raw  = load_cache("squads")
    rounds_raw  = load_cache("rounds")

    print("Loading Elo/Tilt ratings...")
    df_elo  = pd.read_csv("data/elo_ratings.csv")
    df_tilt = pd.read_csv("data/tilt_ratings.csv")

    print("Building DataFrames...")
    df_squads        = pd.DataFrame(squads_raw)
    df_players       = build_players(players_raw, squads_raw)
    df_fixtures      = build_fixtures(rounds_raw)
    df_team_fixtures = build_team_fixtures(df_fixtures)
    df_squads_rated  = build_squads_rated(df_squads, df_elo, df_tilt)
    df_pf            = build_player_fixtures(df_players, df_squads_rated, df_team_fixtures)

    print("Saving processed data...")
    df_pf.to_csv(f"{PROCESSED_DIR}/player_fixtures.csv", index=False)
    df_squads_rated.to_csv(f"{PROCESSED_DIR}/squads_rated.csv", index=False)
    df_fixtures.to_csv(f"{PROCESSED_DIR}/fixtures.csv", index=False)

    print(f"  player_fixtures: {len(df_pf)} rows")
    print(f"  squads_rated:    {len(df_squads_rated)} rows")
    print(f"  fixtures:        {len(df_fixtures)} rows")
    print("fetch_data complete.")


if __name__ == "__main__":
    run()
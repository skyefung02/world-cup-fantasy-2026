from flask import Flask, render_template, request, jsonify
import pandas as pd
import os

import build_projections
from build_projections import assign_xmins

app = Flask(__name__)

XMINS_PATH = "data/xmins.csv"
XG_OVERRIDES_PATH = "data/xg_overrides.csv"

FLAGS = {
    "ALG": "🇩🇿", "ARG": "🇦🇷", "AUS": "🇦🇺", "AUT": "🇦🇹",
    "BEL": "🇧🇪", "BIH": "🇧🇦", "BRA": "🇧🇷", "CPV": "🇨🇻",
    "CAN": "🇨🇦", "COL": "🇨🇴", "COD": "🇨🇩", "CRO": "🇭🇷",
    "CUW": "🇨🇼", "CZE": "🇨🇿", "CIV": "🇨🇮", "ECU": "🇪🇨",
    "EGY": "🇪🇬", "ENG": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "FRA": "🇫🇷", "GER": "🇩🇪",
    "GHA": "🇬🇭", "HAI": "🇭🇹", "IRN": "🇮🇷", "IRQ": "🇮🇶",
    "JPN": "🇯🇵", "JOR": "🇯🇴", "KOR": "🇰🇷", "MEX": "🇲🇽",
    "MAR": "🇲🇦", "NED": "🇳🇱", "NZL": "🇳🇿", "NOR": "🇳🇴",
    "PAN": "🇵🇦", "PAR": "🇵🇾", "POR": "🇵🇹", "QAT": "🇶🇦",
    "KSA": "🇸🇦", "SCO": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "SEN": "🇸🇳", "RSA": "🇿🇦",
    "ESP": "🇪🇸", "SWE": "🇸🇪", "SUI": "🇨🇭", "TUN": "🇹🇳",
    "TUR": "🇹🇷", "USA": "🇺🇸", "URU": "🇺🇾", "UZB": "🇺🇿",
}

def load_players():
    players = pd.read_csv("data/processed/player_fixtures.csv")
    # Get unique players (drop fixture rows)
    players = players[["id", "player", "position", "price", "status", "squadId", "team", "abbr", "group"]].drop_duplicates()
    return players

def load_xmins():
    if os.path.exists(XMINS_PATH):
        return pd.read_csv(XMINS_PATH).set_index("id")["xmins"].to_dict()
    return {}

def save_xmins(xmins_dict):
    df = pd.DataFrame(list(xmins_dict.items()), columns=["id", "xmins"])
    df.to_csv(XMINS_PATH, index=False)

def init_xmins(players, xmins_manual):
    """Compute defaults via assign_xmins, then apply manual overrides from xmins.csv."""
    default_df = (
        players
        .groupby('squadId', group_keys=False)
        .apply(assign_xmins)[['id', 'xmins']]
    )
    result = dict(zip(default_df['id'], default_df['xmins']))
    result.update(xmins_manual)
    return result

@app.route("/")
def index():
    players = load_players()
    xmins = load_xmins()
    xmins = init_xmins(players, xmins)

    # Get sorted team list
    teams = players[["abbr", "team", "group"]].drop_duplicates().sort_values("team")
    teams = teams.sort_values(["group", "team"]).to_dict(orient="records")

    return render_template("index.html", teams=teams, flags=FLAGS)

@app.route("/team/<abbr>")
def team(abbr):
    players = load_players()
    xmins = load_xmins()
    xmins = init_xmins(players, xmins)

    team_players = players[players["abbr"] == abbr].copy()
    team_players["xmins"] = team_players["id"].map(xmins)

    # Load per-round xP from projections.csv
    proj_path = "data/projections.csv"
    if os.path.exists(proj_path):
        proj = pd.read_csv(proj_path)[["id", "1_Pts", "2_Pts", "3_Pts"]].rename(
            columns={"1_Pts": "r1_pts", "2_Pts": "r2_pts", "3_Pts": "r3_pts"}
        )
        team_players = team_players.merge(proj, on="id", how="left")
    else:
        team_players[["r1_pts", "r2_pts", "r3_pts"]] = 0.0
    for col in ["r1_pts", "r2_pts", "r3_pts"]:
        team_players[col] = team_players[col].fillna(0).round(2)

    # Group by position in order
    pos_order = ["GK", "DEF", "MID", "FWD"]
    grouped = {}
    for pos in pos_order:
        group = team_players[team_players["position"] == pos][
            ["id", "player", "price", "xmins", "r1_pts", "r2_pts", "r3_pts"]
        ]
        grouped[pos] = group.to_dict(orient="records")

    team_info = team_players[["team", "abbr", "group"]].iloc[0].to_dict()
    team_info["flag"] = FLAGS.get(abbr, "")

    # Get prev/next team for navigation
    all_teams = players[["abbr", "team"]].drop_duplicates().sort_values("team").reset_index(drop=True)
    idx = all_teams[all_teams["abbr"] == abbr].index[0]
    prev_team = all_teams.iloc[idx - 1]["abbr"] if idx > 0 else None
    next_team = all_teams.iloc[idx + 1]["abbr"] if idx < len(all_teams) - 1 else None

    return render_template(
        "team.html",
        team=team_info,
        grouped=grouped,
        pos_order=pos_order,
        prev_team=prev_team,
        next_team=next_team,
    )

@app.route("/save", methods=["POST"])
def save():
    data = request.json  # {player_id: xmins, ...}
    players = load_players()
    default_df = (
        players
        .groupby('squadId', group_keys=False)
        .apply(assign_xmins)[['id', 'xmins']]
    )
    defaults = dict(zip(default_df['id'], default_df['xmins']))
    overrides = load_xmins()
    for pid, val in data.items():
        pid, val = int(pid), int(val)
        if val == defaults.get(pid):
            overrides.pop(pid, None)
        else:
            overrides[pid] = val
    save_xmins(overrides)
    build_projections.run()
    proj = pd.read_csv("data/projections.csv")
    # Return team-wide xP so teammates redistributed shares also refresh
    changed_ids = [int(pid) for pid in data.keys()]
    affected_teams = set(players[players["id"].isin(changed_ids)]["abbr"].unique())
    team_proj = proj[proj["abbr"].isin(affected_teams)][["id", "1_Pts", "2_Pts", "3_Pts"]]
    xp_updates = {
        str(int(r["id"])): {
            "1": round(float(r["1_Pts"]), 2),
            "2": round(float(r["2_Pts"]), 2),
            "3": round(float(r["3_Pts"]), 2),
        }
        for _, r in team_proj.iterrows()
    }
    overridden_ids = []
    if os.path.exists(XG_OVERRIDES_PATH):
        overridden_ids = list(pd.read_csv(XG_OVERRIDES_PATH)["id"].astype(int))
    return jsonify({"status": "ok", "xp": xp_updates, "overridden_ids": overridden_ids})

@app.route("/projections")
def projections_page():
    proj_path = "data/projections.csv"
    if not os.path.exists(proj_path):
        return render_template("projections.html", players=[])
    proj = pd.read_csv(proj_path)
    proj["avg_pts"] = ((proj["1_Pts"] + proj["2_Pts"] + proj["3_Pts"]) / 3).round(2)
    proj["flag"] = proj["abbr"].map(FLAGS).fillna("")
    for col in ["1_Pts", "2_Pts", "3_Pts"]:
        proj[col] = proj[col].round(2)
    overridden_ids = set()
    if os.path.exists(XG_OVERRIDES_PATH):
        overridden_ids = set(pd.read_csv(XG_OVERRIDES_PATH)["id"].tolist())
    proj["has_override"] = proj["id"].isin(overridden_ids)
    for r in [1, 2, 3]:
        proj[f"{r}_OppDisplay"] = proj[f"{r}_OppAbbr"].apply(
            lambda a: f"{FLAGS.get(a, '')} {a}".strip() if pd.notna(a) else ""
        )
    component_cols = [
        "1_OppDisplay", "2_OppDisplay", "3_OppDisplay",
        "1_xMins", "2_xMins", "3_xMins",
        "1_GoalShare", "2_GoalShare", "3_GoalShare",
        "1_AssistShare", "2_AssistShare", "3_AssistShare",
        "1_ModelGoalShare", "2_ModelGoalShare", "3_ModelGoalShare",
        "1_ModelAssistShare", "2_ModelAssistShare", "3_ModelAssistShare",
        "1_OverrideGoalShare", "1_OverrideAssistShare",
        "1_TeamXG", "2_TeamXG", "3_TeamXG",
    ]
    players = (
        proj[["id", "player", "position", "price", "team", "abbr", "flag",
              "1_Pts", "2_Pts", "3_Pts", "avg_pts", "has_override"] + component_cols]
        .sort_values("avg_pts", ascending=False)
        .to_dict(orient="records")
    )
    return render_template("projections.html", players=players)


@app.route("/save_xg_override", methods=["POST"])
def save_xg_override():
    data = request.json
    player_id   = int(data["player_id"])
    goal_share   = float(data["goal_share"])
    assist_share = float(data["assist_share"])
    overrides = pd.read_csv(XG_OVERRIDES_PATH) if os.path.exists(XG_OVERRIDES_PATH) else pd.DataFrame(columns=["id", "goal_share", "assist_share"])
    overrides = overrides[overrides["id"] != player_id]
    overrides = pd.concat([overrides, pd.DataFrame([{"id": player_id, "goal_share": goal_share, "assist_share": assist_share}])], ignore_index=True)
    overrides.to_csv(XG_OVERRIDES_PATH, index=False)
    build_projections.run()
    proj = pd.read_csv("data/projections.csv")
    players = load_players()
    team_abbr = players.loc[players["id"] == player_id, "abbr"].iloc[0]
    team_proj = proj[proj["abbr"] == team_abbr][["id", "1_Pts", "2_Pts", "3_Pts"]]
    overridden_ids = list(pd.read_csv(XG_OVERRIDES_PATH)["id"].astype(int))
    xp = {str(int(r["id"])): {"1": round(r["1_Pts"], 2), "2": round(r["2_Pts"], 2), "3": round(r["3_Pts"], 2)} for _, r in team_proj.iterrows()}
    return jsonify({"status": "ok", "xp": xp, "overridden_ids": overridden_ids})


@app.route("/reset_xg_override", methods=["POST"])
def reset_xg_override():
    data = request.json
    player_id = int(data["player_id"])
    if os.path.exists(XG_OVERRIDES_PATH):
        overrides = pd.read_csv(XG_OVERRIDES_PATH)
        overrides = overrides[overrides["id"] != player_id]
        overrides.to_csv(XG_OVERRIDES_PATH, index=False)
    build_projections.run()
    proj = pd.read_csv("data/projections.csv")
    players = load_players()
    team_abbr = players.loc[players["id"] == player_id, "abbr"].iloc[0]
    team_proj = proj[proj["abbr"] == team_abbr][["id", "1_Pts", "2_Pts", "3_Pts"]]
    overridden_ids = list(pd.read_csv(XG_OVERRIDES_PATH)["id"].astype(int)) if os.path.exists(XG_OVERRIDES_PATH) else []
    xp = {str(int(r["id"])): {"1": round(r["1_Pts"], 2), "2": round(r["2_Pts"], 2), "3": round(r["3_Pts"], 2)} for _, r in team_proj.iterrows()}
    return jsonify({"status": "ok", "xp": xp, "overridden_ids": overridden_ids})


@app.route("/match-projections")
def match_projections():
    import numpy as np
    proj_path = "data/projections.csv"
    if not os.path.exists(proj_path):
        return render_template("match_projections.html", rounds=[])

    df = pd.read_csv(proj_path)
    teams_df = (
        df[["abbr", "team", "1_OppAbbr", "1_TeamXG", "2_OppAbbr", "2_TeamXG", "3_OppAbbr", "3_TeamXG"]]
        .drop_duplicates("abbr")
    )
    team_lookup = teams_df.set_index("abbr").to_dict("index")

    # Pull team → group mapping from fixtures
    fixtures = pd.read_csv("data/processed/player_fixtures.csv")
    team_group = fixtures[["abbr", "group"]].drop_duplicates().set_index("abbr")["group"].to_dict()

    rounds = []
    for r in [1, 2, 3]:
        seen = set()
        round_matches = []
        for abbr, row in team_lookup.items():
            opp = row[f"{r}_OppAbbr"]
            if pd.isna(opp) or opp not in team_lookup:
                continue
            key = tuple(sorted([abbr, opp]))
            if key in seen:
                continue
            seen.add(key)

            xg_a = row[f"{r}_TeamXG"]
            xg_b = team_lookup[opp][f"{r}_TeamXG"]
            cs_a = round(float(np.exp(-xg_b)) * 100, 1)
            cs_b = round(float(np.exp(-xg_a)) * 100, 1)
            total = xg_a + xg_b
            a_share = round(xg_a / total * 100, 1) if total > 0 else 50.0
            group = team_group.get(abbr, "")

            round_matches.append({
                "group": group,
                "a_abbr": abbr,
                "a_team": row["team"],
                "a_flag": FLAGS.get(abbr, ""),
                "b_abbr": opp,
                "b_team": team_lookup[opp]["team"],
                "b_flag": FLAGS.get(opp, ""),
                "a_xg": round(float(xg_a), 2),
                "b_xg": round(float(xg_b), 2),
                "a_cs": cs_a,
                "b_cs": cs_b,
                "a_share": a_share,
            })

        round_matches.sort(key=lambda m: (m["group"], m["a_team"]))

        # Nest into groups
        groups = []
        for m in round_matches:
            if not groups or groups[-1]["group"] != m["group"]:
                groups.append({"group": m["group"], "matches": []})
            groups[-1]["matches"].append(m)

        # Flat per-team rows for the sortable table (default: xG desc)
        team_rows = []
        for abbr, row in team_lookup.items():
            opp = row[f"{r}_OppAbbr"]
            if pd.isna(opp) or opp not in team_lookup:
                continue
            xg  = row[f"{r}_TeamXG"]
            xga = team_lookup[opp][f"{r}_TeamXG"]
            team_rows.append({
                "flag":     FLAGS.get(abbr, ""),
                "team":     row["team"],
                "abbr":     abbr,
                "group":    team_group.get(abbr, "").upper(),
                "opp_flag": FLAGS.get(opp, ""),
                "opp_abbr": opp,
                "xg":       round(float(xg),  2),
                "xga":      round(float(xga), 2),
                "cs":       round(float(np.exp(-xga)) * 100, 1),
            })
        team_rows.sort(key=lambda t: t["xg"], reverse=True)

        rounds.append({"round": r, "groups": groups, "team_rows": team_rows})

    return render_template("match_projections.html", rounds=rounds)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
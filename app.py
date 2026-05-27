from flask import Flask, render_template, request, jsonify
import pandas as pd
import os

import build_projections
from build_projections import assign_xmins

app = Flask(__name__)

XMINS_PATH = "data/xmins.csv"

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
    proj = pd.read_csv("data/projections.csv").set_index("id")
    xp_updates = {}
    for pid in data.keys():
        pid_int = int(pid)
        if pid_int in proj.index:
            xp_updates[pid] = {
                "1": round(float(proj.loc[pid_int, "1_Pts"]), 2),
                "2": round(float(proj.loc[pid_int, "2_Pts"]), 2),
                "3": round(float(proj.loc[pid_int, "3_Pts"]), 2),
            }
    return jsonify({"status": "ok", "xp": xp_updates})

if __name__ == "__main__":
    app.run(debug=True)
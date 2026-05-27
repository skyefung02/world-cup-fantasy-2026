from flask import Flask, render_template, request, jsonify
import pandas as pd
import os

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

def init_xmins(players, xmins):
    """Default to 60 for any player not yet assigned."""
    for pid in players["id"]:
        if pid not in xmins:
            xmins[pid] = 60
    return xmins

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

    # Group by position in order
    pos_order = ["GK", "DEF", "MID", "FWD"]
    grouped = {}
    for pos in pos_order:
        group = team_players[team_players["position"] == pos][["id", "player", "price", "xmins"]]
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
    xmins = load_xmins()
    for pid, val in data.items():
        xmins[int(pid)] = int(val)
    save_xmins(xmins)
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True)
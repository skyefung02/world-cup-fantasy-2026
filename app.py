from flask import Flask, render_template, request, jsonify, Response
import pandas as pd
import os
import io

import build_projections
from build_projections import assign_xmins, recompute_teams, build_full_projections

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


# ── Player metadata (read once per request from base_state) ──

def load_players():
    """Player metadata table — used by page routes."""
    base = build_projections.get_base_state()
    return base[["id", "player", "position", "price", "squadId", "team", "abbr", "group"]].drop_duplicates("id")


def load_xmins():
    if os.path.exists(XMINS_PATH):
        return pd.read_csv(XMINS_PATH).set_index("id")["xmins"].to_dict()
    return {}


def save_xmins(xmins_dict):
    df = pd.DataFrame(list(xmins_dict.items()), columns=["id", "xmins"])
    df.to_csv(XMINS_PATH, index=False)


def load_overrides():
    if os.path.exists(XG_OVERRIDES_PATH):
        df = pd.read_csv(XG_OVERRIDES_PATH)
        return {
            int(r["id"]): {"goal_share": float(r["goal_share"]), "assist_share": float(r["assist_share"])}
            for _, r in df.iterrows()
        }
    return {}


def overridden_ids_list():
    return list(load_overrides().keys())


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


def current_projections_df():
    """Wide-format projections computed in-memory from current local CSV state."""
    return build_full_projections(load_xmins(), load_overrides())


def xp_dict_from_recompute(updates):
    """Convert recompute_teams output → {pid_str: {round_str: pts}} (the wire shape JS expects)."""
    return {
        str(pid): {str(rd): round(vals["Pts"], 2) for rd, vals in rounds.items()}
        for pid, rounds in updates.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Page routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    players = load_players()
    xmins = init_xmins(players, load_xmins())

    teams = players[["abbr", "team", "group"]].drop_duplicates().sort_values("team")
    teams = teams.sort_values(["group", "team"]).to_dict(orient="records")

    return render_template("index.html", teams=teams, flags=FLAGS)


@app.route("/team/<abbr>")
def team(abbr):
    players = load_players()
    xmins = init_xmins(players, load_xmins())

    team_players = players[players["abbr"] == abbr].copy()
    team_players["xmins"] = team_players["id"].map(xmins)

    # Per-round xP from in-memory projections
    proj = current_projections_df()[["id", "1_Pts", "2_Pts", "3_Pts"]].rename(
        columns={"1_Pts": "r1_pts", "2_Pts": "r2_pts", "3_Pts": "r3_pts"}
    )
    team_players = team_players.merge(proj, on="id", how="left")
    for col in ["r1_pts", "r2_pts", "r3_pts"]:
        team_players[col] = team_players[col].fillna(0).round(2)

    pos_order = ["GK", "DEF", "MID", "FWD"]
    grouped = {
        pos: team_players[team_players["position"] == pos][
            ["id", "player", "price", "xmins", "r1_pts", "r2_pts", "r3_pts"]
        ].to_dict(orient="records")
        for pos in pos_order
    }

    team_info = team_players[["team", "abbr", "group"]].iloc[0].to_dict()
    team_info["flag"] = FLAGS.get(abbr, "")

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


@app.route("/projections")
def projections_page():
    proj = current_projections_df()
    proj["avg_pts"] = ((proj["1_Pts"] + proj["2_Pts"] + proj["3_Pts"]) / 3).round(2)
    proj["flag"] = proj["abbr"].map(FLAGS).fillna("")
    for col in ["1_Pts", "2_Pts", "3_Pts"]:
        proj[col] = proj[col].round(2)
    ov_set = set(overridden_ids_list())
    proj["has_override"] = proj["id"].isin(ov_set)
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


@app.route("/match-projections")
def match_projections():
    import numpy as np
    df = current_projections_df()
    if df.empty:
        return render_template("match_projections.html", rounds=[])

    teams_df = (
        df[["abbr", "team", "1_OppAbbr", "1_TeamXG", "2_OppAbbr", "2_TeamXG", "3_OppAbbr", "3_TeamXG"]]
        .drop_duplicates("abbr")
    )
    team_lookup = teams_df.set_index("abbr").to_dict("index")

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
                "a_abbr": abbr, "a_team": row["team"], "a_flag": FLAGS.get(abbr, ""),
                "b_abbr": opp,  "b_team": team_lookup[opp]["team"], "b_flag": FLAGS.get(opp, ""),
                "a_xg": round(float(xg_a), 2), "b_xg": round(float(xg_b), 2),
                "a_cs": cs_a, "b_cs": cs_b, "a_share": a_share,
            })

        round_matches.sort(key=lambda m: (m["group"], m["a_team"]))

        groups = []
        for m in round_matches:
            if not groups or groups[-1]["group"] != m["group"]:
                groups.append({"group": m["group"], "matches": []})
            groups[-1]["matches"].append(m)

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


# ─────────────────────────────────────────────────────────────────────────────
# Local creator workflow — writes through to CSV, uses fast recompute_teams path
# ─────────────────────────────────────────────────────────────────────────────

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

    # Fast recompute: just the affected teams
    changed_ids = [int(pid) for pid in data.keys()]
    affected_teams = list(players[players["id"].isin(changed_ids)]["abbr"].unique())
    updates = recompute_teams(overrides, load_overrides(), teams=affected_teams)

    return jsonify({
        "status": "ok",
        "xp": xp_dict_from_recompute(updates),
        "overridden_ids": overridden_ids_list(),
    })


@app.route("/save_xg_override", methods=["POST"])
def save_xg_override():
    data = request.json
    player_id    = int(data["player_id"])
    goal_share   = float(data["goal_share"])
    assist_share = float(data["assist_share"])

    overrides = pd.read_csv(XG_OVERRIDES_PATH) if os.path.exists(XG_OVERRIDES_PATH) else pd.DataFrame(columns=["id", "goal_share", "assist_share"])
    overrides = overrides[overrides["id"] != player_id]
    overrides = pd.concat(
        [overrides, pd.DataFrame([{"id": player_id, "goal_share": goal_share, "assist_share": assist_share}])],
        ignore_index=True,
    )
    overrides.to_csv(XG_OVERRIDES_PATH, index=False)

    players = load_players()
    team_abbr = players.loc[players["id"] == player_id, "abbr"].iloc[0]
    updates = recompute_teams(load_xmins(), load_overrides(), teams=[team_abbr])

    return jsonify({
        "status": "ok",
        "xp": xp_dict_from_recompute(updates),
        "overridden_ids": overridden_ids_list(),
    })


@app.route("/reset_xg_override", methods=["POST"])
def reset_xg_override():
    data = request.json
    player_id = int(data["player_id"])
    if os.path.exists(XG_OVERRIDES_PATH):
        overrides = pd.read_csv(XG_OVERRIDES_PATH)
        overrides = overrides[overrides["id"] != player_id]
        overrides.to_csv(XG_OVERRIDES_PATH, index=False)

    players = load_players()
    team_abbr = players.loc[players["id"] == player_id, "abbr"].iloc[0]
    updates = recompute_teams(load_xmins(), load_overrides(), teams=[team_abbr])

    return jsonify({
        "status": "ok",
        "xp": xp_dict_from_recompute(updates),
        "overridden_ids": overridden_ids_list(),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Stateless endpoints (deploy hooks). No reads or writes to local CSV state.
# Client sends its full xmins/override blob, server computes and returns.
# ─────────────────────────────────────────────────────────────────────────────

def _parse_state(payload):
    """Coerce JSON payload into (xmins_map, override_map) with int keys."""
    xmins_in = payload.get("xmins", {}) or {}
    over_in  = payload.get("overrides", {}) or {}
    xmins_map = {int(k): int(v) for k, v in xmins_in.items()}
    override_map = {
        int(k): {
            "goal_share":   float(v.get("goal_share", 0.0)),
            "assist_share": float(v.get("assist_share", 0.0)),
        }
        for k, v in over_in.items()
    }
    return xmins_map, override_map


@app.route("/recompute", methods=["POST"])
def recompute_endpoint():
    """Stateless team-scoped recompute. Body: {teams: ["ARG"], xmins: {...}, overrides: {...}}."""
    payload = request.json or {}
    teams = payload.get("teams")  # None → all
    xmins_map, override_map = _parse_state(payload)
    updates = recompute_teams(xmins_map, override_map, teams=teams)

    # Return the rich shape (per-player per-round, all model fields)
    serializable = {
        str(pid): {str(rd): vals for rd, vals in rounds.items()}
        for pid, rounds in updates.items()
    }
    return jsonify({"status": "ok", "players": serializable})


@app.route("/download.csv", methods=["POST"])
def download_csv():
    """Stateless full export. Body: {xmins: {...}, overrides: {...}}. Returns CSV download."""
    payload = request.json or {}
    xmins_map, override_map = _parse_state(payload)
    df = build_full_projections(xmins_map, override_map)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=projections.csv"},
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)

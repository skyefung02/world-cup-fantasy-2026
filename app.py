from flask import Flask, render_template, request, jsonify, Response
import pandas as pd
import os
import io
import threading

from dotenv import load_dotenv
load_dotenv()  # local-only convenience; Railway injects env vars directly

import build_projections
import fifa_team
import refresh_ownership
from build_projections import (
    get_default_xmins_map, recompute_teams, build_full_projections,
    load_xmins_csv, normalize_xmins, load_scouting_csv,
)

app = Flask(__name__)

# "local" → write-through CSVs, all endpoints active (creator workflow).
# "public" → no disk writes, write endpoints disabled (anonymous deploy).
DEPLOY_MODE = os.environ.get("DEPLOY_MODE", "local")
IS_PUBLIC = DEPLOY_MODE == "public"

# Token gating the /admin/refresh endpoint. If unset, the endpoint returns 503.
REFRESH_ADMIN_TOKEN = os.environ.get("REFRESH_ADMIN_TOKEN")


@app.context_processor
def _inject_deploy_mode():
    """Make DEPLOY_MODE + IS_PUBLIC available in every template."""
    return {"DEPLOY_MODE": DEPLOY_MODE, "IS_PUBLIC": IS_PUBLIC}

PROJECTIONS_CSV_PATH = "data/projections.csv"
_projections_write_lock = threading.Lock()


def _refresh_projections_csv_async():
    """Fire-and-forget background write of data/projections.csv from current local CSV state.
    Keeps the solver pipeline's input in sync with UI tweaks without blocking the response."""
    def _do():
        with _projections_write_lock:
            df = build_full_projections(load_xmins(), load_overrides(), load_scouting())
            df.to_csv(PROJECTIONS_CSV_PATH, index=False)
    threading.Thread(target=_do, daemon=True).start()

XMINS_PATH = "data/xmins.csv"
XG_OVERRIDES_PATH = "data/xg_overrides.csv"
SCOUTING_PATH = "data/scouting_overrides.csv"

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
    """Nested {id: {round_id: mins}} from data/xmins.csv (legacy flat files expanded)."""
    return load_xmins_csv()


def save_xmins(xmins_dict):
    """Persist nested {id: {round_id: mins}} as long-format CSV (id,round_id,xmins)."""
    nested = normalize_xmins(xmins_dict)
    rows = [
        {"id": pid, "round_id": rnd, "xmins": mins}
        for pid, rounds in nested.items()
        for rnd, mins in sorted(rounds.items())
    ]
    df = pd.DataFrame(rows, columns=["id", "round_id", "xmins"])
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


def load_scouting():
    """Set of player ids whose scouting bonus is forced off."""
    return load_scouting_csv()


def save_scouting(ids):
    """Persist the toggled-off ids as data/scouting_overrides.csv (single `id` column)."""
    pd.DataFrame({"id": sorted(int(i) for i in ids)}).to_csv(SCOUTING_PATH, index=False)


def scouting_off_list():
    return sorted(load_scouting())


def current_projections_df():
    """Wide-format projections computed in-memory.
    In public mode, ignore any local creator CSVs — render pure defaults.
    """
    if IS_PUBLIC:
        return build_full_projections({}, {}, set())
    return build_full_projections(load_xmins(), load_overrides(), load_scouting())


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
def home():
    return render_template("home.html")


@app.route("/squad")
def squad():
    players = load_players()
    teams = players[["abbr", "team", "group"]].drop_duplicates().sort_values("team")
    teams = teams.sort_values(["group", "team"]).to_dict(orient="records")

    return render_template("squad.html", teams=teams, flags=FLAGS)


@app.route("/team/<abbr>")
def team(abbr):
    players = load_players()
    team_players = players[players["abbr"] == abbr].copy()

    # Per-round xP and (post-override) xMins from in-memory projections. xMins seeds
    # the three per-round inputs; respects IS_PUBLIC (pure defaults on the deploy).
    proj = current_projections_df()[
        ["id", "1_Pts", "2_Pts", "3_Pts", "1_xMins", "2_xMins", "3_xMins"]
    ].rename(columns={
        "1_Pts": "r1_pts", "2_Pts": "r2_pts", "3_Pts": "r3_pts",
        "1_xMins": "r1_xmins", "2_xMins": "r2_xmins", "3_xMins": "r3_xmins",
    })
    team_players = team_players.merge(proj, on="id", how="left")
    for col in ["r1_pts", "r2_pts", "r3_pts"]:
        team_players[col] = team_players[col].fillna(0).round(2)
    for col in ["r1_xmins", "r2_xmins", "r3_xmins"]:
        team_players[col] = team_players[col].fillna(0).round().astype(int)

    pos_order = ["GK", "DEF", "MID", "FWD"]
    grouped = {
        pos: team_players[team_players["position"] == pos][
            ["id", "player", "price",
             "r1_xmins", "r2_xmins", "r3_xmins", "r1_pts", "r2_pts", "r3_pts"]
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


@app.route("/my-team")
def my_team():
    """Local-only debug page: fetch and render the creator's own FIFA fantasy team.

    Gated by _local_only() so it 404s on the public Railway deploy (DEPLOY_MODE=public)
    and is reachable only on localhost. Reads the session cookie from FIFA_SID (.env).
    """
    guard = _local_only()
    if guard:
        return guard

    sid = os.environ.get("FIFA_SID")
    if not sid:
        return render_template("my_team.html", error="FIFA_SID is not set in .env — add it and restart.")
    try:
        raw = fifa_team.fetch_team(sid)
    except Exception as e:
        return render_template("my_team.html", error=f"Couldn't fetch team: {e}")

    proj_df = current_projections_df()
    meta = load_players().set_index("id").to_dict("index")
    proj = proj_df.set_index("id")[["1_Pts", "2_Pts", "3_Pts"]].to_dict("index")
    captain_id, vice_id = raw.get("captain"), raw.get("vice")

    def resolve(pid):
        m, p = meta.get(pid, {}), proj.get(pid, {})
        abbr = m.get("abbr", "")
        return {
            "id": pid,
            "player": m.get("player", f"#{pid}"),
            "position": m.get("position", ""),
            "team": m.get("team", ""),
            "abbr": abbr,
            "flag": FLAGS.get(abbr, ""),
            "price": m.get("price"),
            "is_captain": pid == captain_id,
            "is_vice": pid == vice_id,
            "r1": round(float(p.get("1_Pts", 0)), 2),
            "r2": round(float(p.get("2_Pts", 0)), 2),
            "r3": round(float(p.get("3_Pts", 0)), 2),
        }

    pos_order = ["GK", "DEF", "MID", "FWD"]
    lineup = {pos: [resolve(pid) for pid in raw.get("lineup", {}).get(pos, [])] for pos in pos_order}
    bench = [resolve(pid) for pid in raw.get("benchOrder", [])]

    # Per-round projected totals for the starting XI, with the captain doubled.
    xi = [pl for pos in pos_order for pl in lineup[pos]]
    totals = {}
    for key in ("r1", "r2", "r3"):
        cap_bonus = next((pl[key] for pl in xi if pl["is_captain"]), 0)
        totals[key] = round(sum(pl[key] for pl in xi) + cap_bonus, 2)

    # Rolling-captaincy analysis (local-only; scipy imported lazily so the public
    # deploy never pulls it in for this debug page).
    import captain
    squad_ids = [pid for ids in raw.get("lineup", {}).values() for pid in ids] \
        + list(raw.get("benchOrder", []))
    squad_records = captain.squad_records_from_df(proj_df, squad_ids, warn=False)
    fixtures = pd.read_csv("data/processed/fixtures.csv")
    captaincy = [captain.analyze_round(squad_records, fixtures, rnd) for rnd in (1, 2, 3)]

    return render_template(
        "my_team.html",
        error=None,
        team_id=raw.get("id"),
        lineup=lineup,
        bench=bench,
        pos_order=pos_order,
        totals=totals,
        captaincy=captaincy,
        flags=FLAGS,
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
    sc_set = set(scouting_off_list())
    proj["scouting_off"] = proj["id"].isin(sc_set)
    # Scouting bonus eligibility is now graded (soft ramp): show the marker whenever the
    # player retains any bonus, i.e. eligibility > 0. Ownership is per-player, R1 representative.
    proj["has_scouting_bonus"] = proj["1_PScoutingEligible"].fillna(0) > 0
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
        "1_TeamXG",       "2_TeamXG",       "3_TeamXG",
        "1_TeamXGA",      "2_TeamXGA",      "3_TeamXGA",
        "1_PCleanSheet",  "2_PCleanSheet",  "3_PCleanSheet",
        "1_LockedPenXg",  "2_LockedPenXg",  "3_LockedPenXg",
        "1_LockedSpXa",   "2_LockedSpXa",   "3_LockedSpXa",
        "1_PercentSelected", "1_PScoutingEligible",
    ]
    players = (
        proj[["id", "player", "position", "price", "team", "abbr", "flag",
              "1_Pts", "2_Pts", "3_Pts", "avg_pts", "has_override",
              "has_scouting_bonus", "scouting_off"] + component_cols]
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

def _local_only():
    """Return a 404 response if running in public mode; else None."""
    if IS_PUBLIC:
        return jsonify({"error": "Endpoint disabled on public deploy"}), 404
    return None


@app.route("/save", methods=["POST"])
def save():
    guard = _local_only()
    if guard: return guard
    # Body is {player_id: xmins} (flat, current team UI) or {player_id: {round: xmins}}
    # (forward-compat). normalize_xmins() expands the flat form to all rounds.
    data = normalize_xmins(request.json)
    players = load_players()
    defaults = get_default_xmins_map()
    overrides = load_xmins()  # nested {id: {round: mins}}
    for pid, rounds in data.items():
        default = defaults.get(pid)
        kept = {r: m for r, m in rounds.items() if m != default}
        if kept:
            overrides[pid] = kept
        else:
            overrides.pop(pid, None)
    save_xmins(overrides)

    # Fast recompute: just the affected teams
    changed_ids = [int(pid) for pid in data.keys()]
    affected_teams = list(players[players["id"].isin(changed_ids)]["abbr"].unique())
    updates = recompute_teams(overrides, load_overrides(), teams=affected_teams, scouting_off=load_scouting())

    _refresh_projections_csv_async()
    return jsonify({
        "status": "ok",
        "xp": xp_dict_from_recompute(updates),
        "overridden_ids": overridden_ids_list(),
        "scouting_off_ids": scouting_off_list(),
    })


@app.route("/save_xg_override", methods=["POST"])
def save_xg_override():
    guard = _local_only()
    if guard: return guard
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
    updates = recompute_teams(load_xmins(), load_overrides(), teams=[team_abbr], scouting_off=load_scouting())

    _refresh_projections_csv_async()
    return jsonify({
        "status": "ok",
        "xp": xp_dict_from_recompute(updates),
        "overridden_ids": overridden_ids_list(),
        "scouting_off_ids": scouting_off_list(),
    })


@app.route("/reset_xg_override", methods=["POST"])
def reset_xg_override():
    guard = _local_only()
    if guard: return guard
    data = request.json
    player_id = int(data["player_id"])
    if os.path.exists(XG_OVERRIDES_PATH):
        overrides = pd.read_csv(XG_OVERRIDES_PATH)
        overrides = overrides[overrides["id"] != player_id]
        overrides.to_csv(XG_OVERRIDES_PATH, index=False)

    players = load_players()
    team_abbr = players.loc[players["id"] == player_id, "abbr"].iloc[0]
    updates = recompute_teams(load_xmins(), load_overrides(), teams=[team_abbr], scouting_off=load_scouting())

    _refresh_projections_csv_async()
    return jsonify({
        "status": "ok",
        "xp": xp_dict_from_recompute(updates),
        "overridden_ids": overridden_ids_list(),
        "scouting_off_ids": scouting_off_list(),
    })


@app.route("/toggle_scouting", methods=["POST"])
def toggle_scouting():
    """Local write-through: force a player's scouting bonus off (off=true) or back to the
    ramp (off=false). Body: {player_id, off}."""
    guard = _local_only()
    if guard: return guard
    data = request.json
    player_id = int(data["player_id"])
    off = bool(data.get("off"))

    ids = load_scouting()
    if off:
        ids.add(player_id)
    else:
        ids.discard(player_id)
    save_scouting(ids)

    players = load_players()
    team_abbr = players.loc[players["id"] == player_id, "abbr"].iloc[0]
    updates = recompute_teams(load_xmins(), load_overrides(), teams=[team_abbr], scouting_off=ids)

    _refresh_projections_csv_async()
    return jsonify({
        "status": "ok",
        "xp": xp_dict_from_recompute(updates),
        "overridden_ids": overridden_ids_list(),
        "scouting_off_ids": scouting_off_list(),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Stateless endpoints (deploy hooks). No reads or writes to local CSV state.
# Client sends its full xmins/override blob, server computes and returns.
# ─────────────────────────────────────────────────────────────────────────────

def _parse_state(payload):
    """Coerce JSON payload into (xmins_map, override_map, scouting_off) with int keys."""
    over_in  = payload.get("overrides", {}) or {}
    # Accept flat {id: mins} (current UI) or nested {id: {round: mins}}; normalize to nested.
    xmins_map = normalize_xmins(payload.get("xmins", {}))
    override_map = {
        int(k): {
            "goal_share":   float(v.get("goal_share", 0.0)),
            "assist_share": float(v.get("assist_share", 0.0)),
        }
        for k, v in over_in.items()
    }
    # scouting may arrive as a list of ids or an {id: true} object; both → set of ids.
    scouting_in = payload.get("scouting", []) or []
    scouting_off = {int(k) for k in (scouting_in.keys() if isinstance(scouting_in, dict) else scouting_in)}
    return xmins_map, override_map, scouting_off


@app.route("/recompute", methods=["POST"])
def recompute_endpoint():
    """Stateless team-scoped recompute. Body: {teams: ["ARG"], xmins: {...}, overrides: {...}}."""
    payload = request.json or {}
    teams = payload.get("teams")  # None → all
    xmins_map, override_map, scouting_off = _parse_state(payload)
    updates = recompute_teams(xmins_map, override_map, teams=teams, scouting_off=scouting_off)

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
    xmins_map, override_map, scouting_off = _parse_state(payload)
    df = build_full_projections(xmins_map, override_map, scouting_off)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=projections.csv"},
    )


# Pre-warm the cached base state so the first request doesn't pay the ~110ms cold cost.
# Runs once at module load (so once per gunicorn worker on Railway, once per local restart).
build_projections.get_base_state()


# ── Hourly ownership refresh (public deploy only) ──
# Assumes Procfile uses `--workers 1`. With multiple workers each one would
# start its own scheduler and stomp on the cache file. Revisit if scaling out.
if IS_PUBLIC:
    from apscheduler.schedulers.background import BackgroundScheduler
    from datetime import datetime
    import atexit

    _scheduler = BackgroundScheduler(daemon=True)
    _scheduler.add_job(
        refresh_ownership.refresh_all,
        trigger="interval",
        hours=1,
        id="hourly_refresh",
        coalesce=True,
        max_instances=1,
        next_run_time=datetime.now(),  # fire once at boot, then hourly
    )
    _scheduler.start()
    atexit.register(lambda: _scheduler.shutdown(wait=False))


@app.route("/admin/refresh", methods=["POST"])
def admin_refresh():
    """Manual trigger for the ownership refresh. Token-gated via REFRESH_ADMIN_TOKEN."""
    if not REFRESH_ADMIN_TOKEN:
        return jsonify({"error": "REFRESH_ADMIN_TOKEN not configured on server"}), 503
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if token != REFRESH_ADMIN_TOKEN:
        return jsonify({"error": "unauthorized"}), 401
    result = refresh_ownership.refresh_all()
    status = 200 if result.get("status") == "ok" else 500
    return jsonify(result), status


if __name__ == "__main__":
    app.run(debug=True, port=5001)

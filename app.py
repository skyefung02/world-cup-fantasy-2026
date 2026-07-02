from flask import Flask, render_template, request, jsonify, Response
import pandas as pd
import os
import io
import json
import threading

from dotenv import load_dotenv
load_dotenv()  # local-only convenience; Railway injects env vars directly

import build_projections
import build_knockout_projections as bk
import fifa_team
import refresh_ownership
from build_projections import (
    get_default_xmins_map, recompute_teams, build_full_projections,
    load_xmins_csv, normalize_xmins, load_scouting_csv,
)

app = Flask(__name__)
# A pasted /team blob is ~0.5 KB; cap request bodies so a giant paste can't be abused.
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024

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

    # Pure xMins editor: one input per player covering all upcoming (knockout) rounds.
    # Seed it from the earliest KO round (R32 / round 4) xMins; saving fans the single
    # value back out across all KO rounds. Respects IS_PUBLIC (pure defaults on deploy).
    proj = current_projections_df()[["id", "4_xMins"]].rename(columns={"4_xMins": "xmins"})
    team_players = team_players.merge(proj, on="id", how="left")
    team_players["xmins"] = team_players["xmins"].fillna(0).round().astype(int)

    pos_order = ["GK", "DEF", "MID", "FWD"]
    grouped = {
        pos: team_players[team_players["position"] == pos][
            ["id", "player", "price", "xmins"]
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


@app.route("/my-team", methods=["GET", "POST"])
def my_team():
    """Rolling-captaincy page.

    LOCAL (DEPLOY_MODE unset): auto-fetch the creator's own team via FIFA_SID —
      convenient, no bookmarklet; the cookie stays on the local machine.
    PUBLIC (Railway): credential-free. The visitor's team arrives as a pasted
      /team blob (bookmarklet or manual paste); no session cookie ever touches
      the server.

    The only thing that differs by environment is how the team is obtained — the
    analysis and rendering below are identical.
    """
    import captain
    import realized
    import fetch_data
    import team_import

    # ── obtain the team source (the ONLY environment-dependent step) ──
    if not IS_PUBLIC:
        sid = os.environ.get("FIFA_SID")
        if not sid:
            return render_template("my_team.html", error="FIFA_SID is not set in .env — add it and restart.")
        try:
            source = fifa_team.fetch_team(sid)
        except Exception as e:
            return render_template("my_team.html", error=f"Couldn't fetch team: {e}")
    else:
        source = (request.form.get("team_json") or "").strip()
        if not source:
            return render_template("my_team.html", awaiting_paste=True)

    # Normalise both sources (raw FIFA dict OR pasted JSON string) to one shape.
    try:
        team = team_import.parse_team(source)
    except team_import.TeamParseError as e:
        if IS_PUBLIC:
            return render_template("my_team.html", awaiting_paste=True, paste_error=str(e))
        return render_template("my_team.html", error=f"Couldn't read team: {e}")

    # ── shared analysis ──
    # Projections: locally read your CSVs; publicly honor the visitor's client-side
    # edits — the same {xmins, overrides, scouting} blob /recompute consumes, posted
    # alongside the team. Missing/malformed → defaults (identical to no edits).
    if not IS_PUBLIC:
        proj_df = current_projections_df()
    else:
        try:
            edits = _parse_state(json.loads(request.form.get("client_state") or "{}"))
            proj_df = build_full_projections(*edits)
        except Exception:
            proj_df = current_projections_df()
    # Captaincy runs on the live knockout round onward (the group stage is done):
    # each KO round from the current one up, but only once its bracket is fully
    # confirmed (all fixtures present) — a half-set round (some winners TBD) is
    # hidden until the rest of its matchups lock in.
    import squad_risk
    fixtures_df = pd.read_csv("data/processed/fixtures.csv")
    live_round = squad_risk.current_ko_round()
    fx_counts = fixtures_df["round_id"].value_counts().to_dict()
    cap_rounds = sorted(
        int(r) for r in fx_counts
        if r >= live_round and fx_counts[r] >= KO_ROUND_FIXTURES.get(r, 1))

    proj = proj_df.set_index("id")[[f"{r}_Pts" for r in cap_rounds]].to_dict("index")

    pos_order = ["GK", "DEF", "MID", "FWD"]
    xi_ids = [pid for pos in pos_order for pid in team["lineup"].get(pos, [])]
    cap_id = team["captain"]

    def _total(col):
        s = sum(float(proj.get(pid, {}).get(col, 0)) for pid in xi_ids)
        s += float(proj.get(cap_id, {}).get(col, 0)) if cap_id in xi_ids else 0  # captain doubled
        return round(s, 2)
    totals = [{"label": KO_ROUND_LABELS.get(r, f"R{r}"), "value": _total(f"{r}_Pts")}
              for r in cap_rounds]

    captaincy = captain.analyze_squad_ids(
        team["player_ids"], proj_df, fixtures_df, rounds=cap_rounds)

    # Overlay realized results so far (no-op pre-round): marks resolved blocks,
    # settles keep/twist vs the frozen thresholds, finds the live decision point.
    points, finals = realized.load_realized(
        fetch_data.load_cache("players"), fetch_data.load_cache("rounds"))
    for rd in captaincy:
        realized.overlay_realized(rd, points, finals)

    # Open on the live round — the first one whose armband decision is still open
    # ('rolling'). Completed rounds resolve to 'locked', so this auto-advances as
    # rounds finish (R1 done → opens on R2). All locked → fall back to the last round.
    active_round = next(
        (rd["round"] for rd in captaincy if rd["live_state"] == "rolling"), None)
    if active_round is None and captaincy:
        active_round = captaincy[-1]["round"]

    return render_template(
        "my_team.html",
        error=None,
        awaiting_paste=False,
        team_id=team["team_id"],
        totals=totals,
        captaincy=captaincy,
        active_round=active_round,
        round_labels=KO_ROUND_NAMES,
        flags=FLAGS,
    )


# Rounds surfaced on the projections page. The group stage is complete (R1-R3), so
# no group rounds are shipped — the page is knockout-only, showing rounds 4..8 with
# their P(play)-weighted points. KO_ROUND_LABELS drives the column headers + round pills.
KO_DISPLAY_ROUNDS = [4, 5, 6, 7, 8]
KO_ROUND_LABELS = {4: "R32", 5: "R16", 6: "QF", 7: "SF", 8: "F"}
# Full names for the captaincy round selector.
KO_ROUND_NAMES = {4: "Round of 32", 5: "Round of 16", 6: "Quarter Finals",
                  7: "Semi Finals", 8: "Final"}
# Fixtures a KO round has once its bracket is fully set — used to hide a round
# from captaincy until every matchup is confirmed (R16 stays hidden while some
# R32 winners are still TBD). Round 8 = final + third-place playoff.
KO_ROUND_FIXTURES = {4: 16, 5: 8, 6: 4, 7: 2, 8: 2}


def _counts_by_country(player_ids, proj_df):
    """Collapse a squad's player ids to {abbr: count} — all the risk math needs."""
    id2abbr = proj_df.set_index("id")["abbr"].to_dict()
    counts = {}
    for pid in player_ids:
        abbr = id2abbr.get(pid)
        if abbr:
            counts[abbr] = counts.get(abbr, 0) + 1
    return counts


@app.route("/squad-risk", methods=["GET", "POST"])
def squad_risk_page():
    """Interactive planner: edit per-country player counts and watch the chance
    of being forced past your free transfers move. Import seeds the counts from
    the real team; from there it's a client-side sandbox (risk depends only on
    country composition, so all recompute happens in the browser)."""
    import squad_risk as sr

    # Planning spans the live round through the final: the same squad can be
    # evaluated against any upcoming round's survival odds. team_ko_probs(r) is
    # the chance a team loses in round r *given it's in it* — exactly what you
    # want when modelling "the squad I'll field in the R16".
    live_round = sr.current_ko_round()
    plan_rounds = [r for r in KO_DISPLAY_ROUNDS if r >= live_round]

    proj_df = current_projections_df()
    name_by_abbr = proj_df.drop_duplicates("abbr").set_index("abbr")["team"].to_dict()

    ko_probs_by_round = {r: {a: round(p, 4) for a, p in sr.team_ko_probs(r).items()}
                         for r in plan_rounds}
    all_abbrs = set().union(*(m.keys() for m in ko_probs_by_round.values())) if ko_probs_by_round else set()
    teams_meta = {a: {"name": name_by_abbr.get(a, a), "flag": FLAGS.get(a, "")} for a in all_abbrs}
    ft_by_round = {r: sr.FREE_TRANSFERS.get(r, 4) for r in plan_rounds}
    round_options = [{"num": r, "label": KO_ROUND_LABELS.get(r, f"R{r}"),
                      "name": KO_ROUND_NAMES.get(r, f"Round {r}")} for r in plan_rounds]

    # A team's survival is "confirmed" when its opponent for that round is set —
    # i.e. it appears in the confirmed fixtures. Otherwise the odds are projected
    # (a Monte-Carlo average over the possible opponents the bracket could yield).
    conf_fx = pd.read_csv("data/processed/fixtures.csv")
    confirmed_by_round = {
        r: sorted(set(sub["home_abbr"]) | set(sub["away_abbr"]))
        for r in plan_rounds
        for sub in [conf_fx[conf_fx["round_id"] == r]]
    }

    counts, paste_error = {}, None
    start_empty = request.args.get("empty")

    # Seed counts from the imported team (local: auto-fetch; public: pasted blob).
    if request.method == "POST":
        import team_import
        source = (request.form.get("team_json") or "").strip()
        try:
            team = team_import.parse_team(source)
            counts = _counts_by_country(team["player_ids"], proj_df)
        except team_import.TeamParseError as e:
            paste_error = str(e)
    elif not IS_PUBLIC and not start_empty:
        import fifa_team
        import team_import
        sid = os.environ.get("FIFA_SID")
        if sid:
            try:
                team = team_import.parse_team(fifa_team.fetch_team(sid))
                counts = _counts_by_country(team["player_ids"], proj_df)
            except Exception:
                counts = {}

    # Keep only countries that appear in some planning round (drops group-stage
    # leftovers a stale imported squad might carry).
    counts = {a: n for a, n in counts.items() if a in all_abbrs}

    # Public + no team yet (and not explicitly starting empty) → show the import prompt.
    awaiting = IS_PUBLIC and request.method == "GET" and not start_empty
    if paste_error:
        awaiting = True

    return render_template(
        "squad_risk.html",
        awaiting_paste=awaiting,
        paste_error=paste_error,
        round_options=round_options,
        default_round=live_round,
        ko_probs_by_round=ko_probs_by_round,
        teams_meta=teams_meta,
        ft_by_round=ft_by_round,
        confirmed_by_round=confirmed_by_round,
        counts=counts,
        flags=FLAGS,
    )


@app.route("/projections")
def projections_page():
    proj = current_projections_df()
    display_rounds = KO_DISPLAY_ROUNDS

    proj["flag"] = proj["abbr"].map(FLAGS).fillna("")
    # Headline = Total xP, the sum of each KO round's already-P(play)-weighted points.
    proj["ko_total"] = proj[[f"{r}_Pts" for r in KO_DISPLAY_ROUNDS]].sum(axis=1).round(2)
    # Hide eliminated teams from the Knockout segment (no realistic path to any KO round).
    proj["ko_alive"] = proj[[f"{r}_PPlay" for r in KO_DISPLAY_ROUNDS]].max(axis=1) > 0.01
    for r in display_rounds:
        proj[f"{r}_Pts"] = proj[f"{r}_Pts"].round(2)

    ov_set = set(overridden_ids_list())
    proj["has_override"] = proj["id"].isin(ov_set)
    sc_set = set(scouting_off_list())
    proj["scouting_off"] = proj["id"].isin(sc_set)
    # Scouting bonus eligibility is graded (soft ramp): show the marker whenever the
    # player retains any bonus. Ownership is per-player, R1-representative.
    proj["has_scouting_bonus"] = proj["1_PScoutingEligible"].fillna(0) > 0

    # Opponent display per round: flag + abbr for known fixtures (all group rounds and
    # confirmed KO fixtures), blank for opponent-mix KO rounds (rendered as "vs field").
    for r in display_rounds:
        proj[f"{r}_OppDisplay"] = proj[f"{r}_OppAbbr"].apply(
            lambda a: f"{FLAGS.get(a, '')} {a}".strip() if pd.notna(a) else ""
        )

    # Per-round model components shipped for every displayed round (group + KO). The
    # modal reads these to recompute live and to show the conditional/P(play) split.
    per_round = [
        "Pts", "PtsCond", "PPlay", "OppDisplay", "xMins",
        "GoalShare", "AssistShare", "ModelGoalShare", "ModelAssistShare",
        "TeamXG", "TeamXGA", "PCleanSheet", "LockedPenXg", "LockedSpXa",
    ]
    round_cols = [f"{r}_{c}" for r in display_rounds for c in per_round]
    base_cols = [
        "id", "player", "position", "price", "team", "abbr", "flag",
        "ko_total", "ko_alive", "has_override", "has_scouting_bonus",
        "scouting_off", "1_OverrideGoalShare", "1_OverrideAssistShare",
        "1_PercentSelected", "1_PScoutingEligible",
    ]
    players = (
        proj[base_cols + round_cols]
        .sort_values("ko_total", ascending=False)
        .to_dict(orient="records")
    )
    return render_template(
        "projections.html", players=players,
        ko_rounds=KO_DISPLAY_ROUNDS, round_labels=KO_ROUND_LABELS,
    )


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
    # Group stage is complete — omit group rounds from the UI entirely (knockout is the
    # default view now). Data is untouched in the projection CSVs; this only hides it.
    for r in []:
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

    # ── Knockout projections (separate Monte Carlo pipeline) ──
    # Advancement grid (team × round reach probabilities) + per-round team tables
    # carrying P(play) and opponent-mix-averaged xG/xGA/CS. Empty if not yet built.
    import os
    grid = []
    ko_rounds = []
    KO_LABELS = {4: "R32", 5: "R16", 6: "QF", 7: "SF", 8: "Final"}
    tp_path = "data/knockout_team_probs.csv"
    kr_path = "data/knockout_team_rounds.csv"
    if os.path.exists(tp_path):
        tp = pd.read_csv(tp_path).sort_values("P_Champ", ascending=False)
        reach_cols = ("P_R32", "P_R16", "P_QF", "P_SF", "P_Final", "P_Champ")
        for _, row in tp.iterrows():
            ab = row["abbr"]
            cells = [round(row[c] * 100, 1) for c in reach_cols]
            # Show only teams still mathematically alive. Decided rounds are exactly
            # 0 or 1 (group exits are all-zero; a team knocked out in the R32 sits at
            # R32=1, everything after 0), so "alive" == some reach prob is strictly
            # between 0 and 1, i.e. its final placing is still undetermined.
            if not any(0 < row[c] < 1 for c in reach_cols):
                continue
            grid.append({
                "flag": FLAGS.get(ab, ""), "team": row["name"], "abbr": ab,
                "group": str(row["group"]).upper(),
                "cells": cells,
            })
    if os.path.exists(kr_path):
        # Team-level Monte Carlo aggregates (the same source the live player engine
        # reads, so the player table and these tables can't diverge). Long format:
        # one row per (abbr, round) with opponent-mix cond_scored/cond_conceded + p_play.
        kr = pd.read_csv(kr_path)
        ko_team = {(t.abbr, int(t.round)): (t.cond_scored, t.cond_conceded, t.p_play)
                   for t in kr.itertuples(index=False)}
        name_by_abbr = {ab: team_lookup.get(ab, {}).get("team", ab)
                        for ab in kr["abbr"].unique()}

        # Confirmed knockout fixtures (FIFA-drawn) → shown as resolved cards, and
        # their teams dropped from the opponent-uncertain table for that round.
        confirmed = {}       # round -> [card, ...]
        confirmed_abbrs = {}  # round -> {abbr, ...}
        fx_path = "data/knockout_fixtures.csv"
        if os.path.exists(fx_path):
            fx = pd.read_csv(fx_path).sort_values(["round", "match"])  # bracket order
            for _, f in fx.iterrows():
                rnd, a, b = int(f["round"]), f["home_abbr"], f["away_abbr"]
                total = float(f["home_xg"]) + float(f["away_xg"])
                confirmed.setdefault(rnd, []).append({
                    "match": int(f["match"]),
                    "a_abbr": a, "a_team": name_by_abbr.get(a, a), "a_flag": FLAGS.get(a, ""),
                    "a_xg": round(float(f["home_xg"]), 2), "a_cs": float(f["home_cs"]),
                    "b_abbr": b, "b_team": name_by_abbr.get(b, b), "b_flag": FLAGS.get(b, ""),
                    "b_xg": round(float(f["away_xg"]), 2), "b_cs": float(f["away_cs"]),
                    "a_share": round(float(f["home_xg"]) / total * 100, 1) if total > 0 else 50.0,
                    "venue": f["venue"],
                })
                confirmed_abbrs.setdefault(rnd, set()).update([a, b])

        # Bracket feeder tree: which two same-round ties feed each next-round match.
        # Lets the R32 tab show the bracket (winner of tie A meets winner of tie B).
        bracket_def = json.load(open("data/knockout_bracket.json"))["matches"]
        NEXT_STAGE = {4: "R16", 5: "QF", 6: "SF", 7: "F"}

        for r in (4, 5, 6, 7, 8):
            done = confirmed_abbrs.get(r, set())
            team_rows = []
            for ab in kr.loc[kr["round"] == r, "abbr"]:
                cond_scored, cond_conceded, pplay = ko_team[(ab, r)]
                if pplay < 0.01:  # hide <1% longshots; full picture is in the grid
                    continue
                if ab in done:    # already shown as a confirmed fixture card
                    continue
                team_rows.append({
                    "flag": FLAGS.get(ab, ""), "team": name_by_abbr.get(ab, ab), "abbr": ab,
                    "group": team_group.get(ab, "").upper(),
                    "pplay": round(float(pplay) * 100, 1),
                    "xg": round(float(cond_scored), 2),
                    "xga": round(float(cond_conceded), 2),
                    "cs": round(float(np.exp(-cond_conceded)) * 100, 1),
                })
            team_rows.sort(key=lambda t: t["xg"], reverse=True)

            # Pair up this round's confirmed ties by the next-round match their winners
            # feed into. Feed match-ids are permuted vs the template, so map each tie to
            # its TEMPLATE bracket position (by group-slot) first, then walk the template
            # feeder tree. Only pairs where BOTH feeders are confirmed are shown.
            cards = confirmed.get(r, [])
            pairs = []
            nxt = NEXT_STAGE.get(r)
            if nxt and cards:
                seeded = bk.actual_r32_in_template_space() if r == 4 else {}
                card_by_teams = {frozenset((c["a_abbr"], c["b_abbr"])): c for c in cards}
                card_by_template = {}
                for tmid, (ha, aa, _v) in seeded.items():
                    c = card_by_teams.get(frozenset((ha, aa)))
                    if c:
                        card_by_template[tmid] = c
                for m in sorted((m for m in bracket_def if m.get("stage") == nxt),
                                key=lambda m: m["match"]):
                    h, a = m.get("home", {}), m.get("away", {})
                    if h.get("type") == "winner" and a.get("type") == "winner":
                        top, bot = card_by_template.get(h["match"]), card_by_template.get(a["match"])
                        if top and bot:
                            pairs.append({"top": top, "bottom": bot, "next_label": KO_LABELS.get(r + 1, "")})

            ko_rounds.append({"round": r, "label": KO_LABELS[r],
                              "team_rows": team_rows, "confirmed": cards,
                              "bracket_pairs": pairs})

    return render_template("match_projections.html", rounds=rounds,
                           ko_rounds=ko_rounds, grid=grid)


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
        # Merge the posted rounds onto any existing edits (the modal posts only the
        # active segment's rounds, so a KO save must not drop saved group rounds).
        merged = {**overrides.get(pid, {}), **rounds}
        kept = {r: m for r, m in merged.items() if m != default}
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

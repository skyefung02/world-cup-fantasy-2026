"""
build_knockout_projections.py — Monte Carlo tournament simulator for the WC 2026
knockout stage.

Idempotent: re-run after each round. It always simulates the *results* of
not-yet-played matches forward from the current FIFA feed, but takes
*participants* as fixed wherever the data already fixes them (completed group
results; a drawn R32 bracket; etc.). As the tournament resolves, more of the
tree is known and less is simulated — the script itself never changes.

Reuses the group-stage Elo->xG engine from build_projections (win_expectancy,
expected_goals, GOALS_SCALE) so knockout match modelling is consistent with the
group-stage projections.

Outputs (all TEAM-level — per-player knockout xPts are produced live by the web
app, which crosses these aggregates with the per-player constants and routes them
through the group-stage scoring engine; see build_projections._knockout_base_rows):
    data/knockout_team_probs.csv   — per-team P(finish 1st/2nd/3rd), P(best-third),
                                     P(reach R32/R16/QF/SF/Final), P(win)
    data/knockout_team_rounds.csv  — per-(team, round 4..8) opponent-mix xG/xGA + P(play)
    data/knockout_fixtures.csv     — confirmed (FIFA-drawn) fixtures, head-to-head xG
"""

import json
import numpy as np
import pandas as pd

from build_projections import win_expectancy, expected_goals, GOALS_SCALE

# Match the tilt multiplier used in build_projections._precompute_base_state.
TILT_K = 0.25

# The 3rd-place play-off outscores what strength alone predicts: a dead rubber
# between two elite, evenly-matched semi-final losers, which the Elo->goals curve
# prices at its lowest per-side values. Play-offs since 1974 (n=13) averaged 3.46
# goals at 90' vs the 2.39 this model gives the 2026 FRA-ENG tie -> 1.45x
# (Poisson p=0.011). Calibrated against the tilt-inclusive projection, so this
# multiplies the *final* xG and TILT_K must stay applied underneath it.
THIRD_PLACE_XG_MULT = 1.45

SQUADS_RATED_PATH = "data/processed/squads_rated.csv"
FIXTURES_PATH     = "data/processed/fixtures.csv"
ROUNDS_CACHE      = "cache/rounds.json"
BRACKET_PATH      = "data/knockout_bracket.json"

GROUP_ROUNDS = (1, 2, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Inputs
# ─────────────────────────────────────────────────────────────────────────────

def load_teams():
    """Clean per-team ratings, indexed 0..n-1. Returns (df, abbr->idx)."""
    df = pd.read_csv(SQUADS_RATED_PATH)
    df = df.reset_index(drop=True)
    df["tilt"] = df["total_tilt"].fillna(0.0)
    df["home_field"] = df["home_field"].fillna(0.0)
    idx = {a: i for i, a in enumerate(df["abbr"])}
    return df, idx


def load_group_matches(idx):
    """Group-stage matches with current state.

    Each record: home_i, away_i (team indices), and either a fixed (gh, ga)
    scoreline (completed) or None (to be simulated).
    """
    fixtures = pd.read_csv(FIXTURES_PATH)
    fixtures = fixtures[fixtures["round_id"].isin(GROUP_ROUNDS)]

    # Actual results keyed by fixture_id, from the rounds feed.
    rounds = json.load(open(ROUNDS_CACHE))
    results = {}
    for r in rounds:
        if r["id"] not in GROUP_ROUNDS:
            continue
        for t in r["tournaments"]:
            if t.get("status") == "complete" and t.get("homeScore") is not None:
                results[t["id"]] = (int(t["homeScore"]), int(t["awayScore"]))

    matches = []
    for _, row in fixtures.iterrows():
        h, a = idx[row["home_abbr"]], idx[row["away_abbr"]]
        matches.append({
            "round": int(row["round_id"]),
            "home_i": h, "away_i": a,
            "result": results.get(int(row["fixture_id"])),  # (gh, ga) or None
        })
    return matches


# ─────────────────────────────────────────────────────────────────────────────
# Match model
# ─────────────────────────────────────────────────────────────────────────────

def match_xg(teams, home_i, away_i):
    """Expected goals (home, away) for a single match, Elo + home-field + tilt.

    Mirrors build_projections: each side's own home-field is added to its rating
    (non-zero only for the three hosts), tilt is a symmetric match-level multiplier.
    """
    elo = teams["elo"].values
    hf = teams["home_field"].values
    tilt = teams["tilt"].values

    eff_h = elo[home_i] + hf[home_i]
    eff_a = elo[away_i] + hf[away_i]
    we = win_expectancy(eff_h, eff_a)
    match_tilt = tilt[home_i] + tilt[away_i]
    scale = GOALS_SCALE * (1 + TILT_K * match_tilt)
    xg_h = float(expected_goals(np.array([we]))[0]) * scale
    xg_a = float(expected_goals(np.array([1 - we]))[0]) * scale
    return xg_h, xg_a


# ─────────────────────────────────────────────────────────────────────────────
# Group-stage simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_groups(n_sims=20000, seed=0):
    """Monte Carlo the remaining group matches.

    Returns a dict with arrays shape (n_teams, n_sims):
        pos            : finishing position within group (1..4)
        third_qualify  : bool, finished 3rd AND among the best 8 thirds
        advance        : bool, reaches the Round of 32
    plus the teams df and abbr->idx.
    """
    rng = np.random.default_rng(seed)
    teams, idx = load_teams()
    n_teams = len(teams)
    matches = load_group_matches(idx)

    pts = np.zeros((n_teams, n_sims))
    gf = np.zeros((n_teams, n_sims))
    ga = np.zeros((n_teams, n_sims))

    for m in matches:
        h, a = m["home_i"], m["away_i"]
        if m["result"] is not None:
            gh = np.full(n_sims, m["result"][0])
            gag = np.full(n_sims, m["result"][1])
        else:
            xg_h, xg_a = match_xg(teams, h, a)
            gh = rng.poisson(xg_h, n_sims)
            gag = rng.poisson(xg_a, n_sims)
        gf[h] += gh; ga[h] += gag
        gf[a] += gag; ga[a] += gh
        pts[h] += np.where(gh > gag, 3, np.where(gh == gag, 1, 0))
        pts[a] += np.where(gag > gh, 3, np.where(gh == gag, 1, 0))

    gd = gf - ga
    # Sortable key: pts dominates, then GD, then GF, then tiny noise to break
    # remaining ties randomly (stands in for h2h / fair-play / drawing of lots).
    noise = rng.random((n_teams, n_sims)) * 1e-3
    key = pts * 1e7 + (gd + 100) * 1e3 + gf * 1.0 + noise

    groups = teams["group"].values
    pos = np.zeros((n_teams, n_sims), dtype=int)
    third_key = {}  # group -> (key array of its 3rd-place team, team_idx array)
    # Per-group finisher identities (team index arrays, shape (n_sims,)).
    pos_team = {1: {}, 2: {}, 3: {}}
    for g in sorted(set(groups)):
        members = np.where(groups == g)[0]          # 4 team indices
        gk = key[members]                            # (4, n_sims)
        order = np.argsort(-gk, axis=0)              # best first; (4, n_sims)
        ranks = np.argsort(order, axis=0)            # inverse perm -> rank 0..3 per sim
        for li, ti in enumerate(members):
            pos[ti] = ranks[li] + 1                  # 1..4
        pos_team[1][g] = members[order[0]]           # group winner, per sim
        pos_team[2][g] = members[order[1]]           # runner-up, per sim
        pos_team[3][g] = members[order[2]]           # 3rd place, per sim
        third_team = pos_team[3][g]                  # global team idx per sim
        third_key[g] = (gk[order[2], np.arange(n_sims)], third_team)

    # Best 8 of the 12 third-placed teams qualify.
    group_list = sorted(set(groups))
    thirds_keys = np.stack([third_key[g][0] for g in group_list])   # (12, n_sims)
    thirds_team = np.stack([third_key[g][1] for g in group_list])   # (12, n_sims)
    qual_local = np.argsort(-thirds_keys, axis=0)[:8]               # (8, n_sims) local group rows
    third_qualify = np.zeros((n_teams, n_sims), dtype=bool)
    for s in range(n_sims):
        for li in qual_local[:, s]:
            third_qualify[thirds_team[li, s], s] = True

    advance = (pos == 1) | (pos == 2) | third_qualify

    return {
        "teams": teams, "idx": idx, "n_sims": n_sims,
        "pos": pos, "third_qualify": third_qualify, "advance": advance,
        "pos_team": pos_team,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Knockout bracket: resolution + walk
# ─────────────────────────────────────────────────────────────────────────────

def load_bracket():
    return json.load(open(BRACKET_PATH))["matches"]


# Knockout venue -> host nation. A host gets its home-field bonus only when the
# match is played in its own country (mirrors the group-stage host logic).
VENUE_HOST = {
    "Inglewood": "USA", "Foxborough": "USA", "Houston": "USA",
    "East Rutherford": "USA", "Arlington": "USA", "Atlanta": "USA",
    "Santa Clara": "USA", "Seattle": "USA", "Miami Gardens": "USA",
    "Kansas City": "USA", "Philadelphia": "USA",
    "Guadalupe": "MEX", "Mexico City": "MEX",
    "Toronto": "CAN", "Vancouver": "CAN",
}

# FIFA fantasy knockout rounds. Round 8 holds BOTH the final and the 3rd-place
# play-off, so every semi-finalist plays a round-8 match.
STAGE_TO_ROUND = {"R32": 4, "R16": 5, "QF": 6, "SF": 7, "F": 8, "3P": 8}


def _third_slot_eligibility(bracket):
    """{match_id: set(group_letters_lowercase)} for every group_third slot."""
    elig = {}
    for m in bracket:
        for side in ("home", "away"):
            s = m[side]
            if s["type"] == "group_third":
                elig[m["match"]] = {g.lower() for g in s["third_from"]}
    return elig


def _match_thirds(qual_groups, slot_elig):
    """Assign each qualifying third-place group to an eligible third slot.

    Bipartite perfect matching (Kuhn's algorithm): slots on the left, the 8
    qualifying groups on the right, an edge where the group is eligible for the
    slot. FIFA's eligibility sets guarantee a perfect matching exists for every
    combination of qualifying groups. Returns {match_id: group_letter}.
    """
    slots = list(slot_elig)
    adj = {s: [g for g in qual_groups if g in slot_elig[s]] for s in slots}
    match_g = {}  # group -> slot
    for s in slots:
        seen = set()
        stack_s = s
        # iterative augmenting path
        def assign(node, seen):
            for g in adj[node]:
                if g in seen:
                    continue
                seen.add(g)
                if g not in match_g or assign(match_g[g], seen):
                    match_g[g] = node
                    return True
            return False
        assign(stack_s, seen)
    slot_to_group = {s: g for g, s in match_g.items()}
    # Fallback: if any slot went unmatched, fill with leftover groups arbitrarily.
    if len(slot_to_group) < len(slots):
        used = set(slot_to_group.values())
        leftovers = [g for g in qual_groups if g not in used]
        for s in slots:
            if s not in slot_to_group and leftovers:
                slot_to_group[s] = leftovers.pop()
    return slot_to_group


def resolve_r32(sim, bracket):
    """Per sim, fill each R32 match's home/away with concrete team indices.

    Returns {match_id: {"home": arr, "away": arr}} with arrays shape (n_sims,).
    """
    n_sims = sim["n_sims"]
    pos_team = sim["pos_team"]
    third_qualify = sim["third_qualify"]
    group_letters = sorted(pos_team[1])               # ['a'..'l']
    gbit = {g: i for i, g in enumerate(group_letters)}
    arange = np.arange(n_sims)

    # Which groups produced a qualifying third, per sim -> 12-bit mask.
    qual_by_group = {
        g: third_qualify[pos_team[3][g], arange] for g in group_letters
    }
    mask = np.zeros(n_sims, dtype=int)
    for g in group_letters:
        mask |= (qual_by_group[g].astype(int) << gbit[g])

    slot_elig = _third_slot_eligibility(bracket)
    # Resolve the slot->group matching once per distinct qualifying-group combo.
    third_slot_group = {mid: np.empty(n_sims, dtype=int) for mid in slot_elig}
    for mval in np.unique(mask):
        sims = np.where(mask == mval)[0]
        qual_groups = [g for g in group_letters if mval & (1 << gbit[g])]
        slot_to_group = _match_thirds(qual_groups, slot_elig)
        for mid, g in slot_to_group.items():
            third_slot_group[mid][sims] = pos_team[3][g][sims]

    def resolve_slot(spec):
        if spec["type"] == "group_winner":
            return pos_team[1][spec["group"].lower()]
        if spec["type"] == "group_runner_up":
            return pos_team[2][spec["group"].lower()]
        raise ValueError(spec)

    r32 = {}
    for m in bracket:
        if m["stage"] != "R32":
            continue
        sides = {}
        for side in ("home", "away"):
            s = m[side]
            sides[side] = third_slot_group[m["match"]] if s["type"] == "group_third" else resolve_slot(s)
        r32[m["match"]] = sides
    return r32


def actual_r32_draw():
    """Actual drawn R32 fixtures from the live feed (rounds.json):
    {match_id: (home_abbr, away_abbr, venue_city)}. Empty until the R32 is drawn.

    Venue is normalised ("City, State" -> "City") to match VENUE_HOST. Once all 16
    ties are present, walk_knockouts seeds them directly instead of reconstructing the
    R32 from the pre-tournament template (whose group->slot routing and venues are
    stale), removing the matchup/venue ambiguity for the rest of the bracket.
    """
    squad_abbr = {s["id"]: s["abbr"] for s in json.load(open("cache/squads.json"))}
    out = {}
    for r in json.load(open(ROUNDS_CACHE)):
        if r["id"] != 4:
            continue
        for t in r["tournaments"]:
            hid, aid = t.get("homeSquadId"), t.get("awaySquadId")
            if hid and aid:
                out[t["id"]] = (squad_abbr.get(hid), squad_abbr.get(aid),
                                (t.get("venueCity") or "").split(",")[0].strip())
    return out


def completed_ko_results(idx):
    """Actual finished knockout ties from the live feed, as fixed outcomes:
    {frozenset({winner_i, loser_i}): winner_i} keyed by team indices.

    Winner is the higher scorer; on a level scoreline the penalty shootout
    (home/awayPenaltyScore) decides it. A drawn tie with no shootout data is
    skipped (can't be resolved). walk_knockouts uses this to hold played
    matches fixed instead of re-simulating them — a knocked-out team must not
    keep a live path in the tournament, exactly as group results are held fixed.
    """
    squad_abbr = {s["id"]: s["abbr"] for s in json.load(open("cache/squads.json"))}
    out = {}
    for r in json.load(open(ROUNDS_CACHE)):
        if r["id"] < 4:
            continue
        for t in r["tournaments"]:
            if t.get("status") != "complete" or t.get("homeScore") is None:
                continue
            ha, aa = squad_abbr.get(t.get("homeSquadId")), squad_abbr.get(t.get("awaySquadId"))
            if ha not in idx or aa not in idx:
                continue
            hs, as_ = int(t["homeScore"]), int(t["awayScore"])
            if hs != as_:
                win = ha if hs > as_ else aa
            else:
                hp, ap = t.get("homePenaltyScore"), t.get("awayPenaltyScore")
                if hp is None or ap is None:
                    continue  # level tie, no shootout data — can't resolve
                win = ha if int(hp) > int(ap) else aa
            out[frozenset((idx[ha], idx[aa]))] = idx[win]
    return out


def _draw_to_template_positions(sim, bracket):
    """Map the actual R32 draw onto TEMPLATE bracket positions by group-slot.

    The live feed numbers R32 ties differently from the pre-tournament template, so
    seeding by match-id would mis-pair the R16 (it did). Instead, identify each drawn
    tie by its non-third group slots (e.g. winner-of-E) and drop it into the template
    match that calls for those slots — then the template's (correct) feeder tree pairs
    the right ties. The actual third-placed team rides along with the tie, so FIFA's
    third-place routing is taken from the feed, not reconstructed.

    Returns {template_match_id: (home_abbr, away_abbr, venue)} when the R32 is fully
    drawn and maps cleanly; {} otherwise (caller falls back to reconstruction).
    """
    draw = actual_r32_draw()
    r32_ids = {m["match"] for m in bracket if m["stage"] == "R32"}
    if not r32_ids.issubset(draw.keys()):
        return {}

    teams = sim["teams"]; pos = sim["pos"][:, 0]; idx = sim["idx"]
    group_of = {ab: str(teams["group"].values[i]).upper() for ab, i in idx.items()}
    team_slot = {ab: (int(pos[i]), group_of[ab]) for ab, i in idx.items()}  # (1/2/3/4, GROUP)

    def fixed_key(m):
        out = set()
        for side in ("home", "away"):
            s = m[side]
            if s["type"] == "group_winner":
                out.add((1, str(s["group"]).upper()))
            elif s["type"] == "group_runner_up":
                out.add((2, str(s["group"]).upper()))
        return frozenset(out)

    tmpl_by_key = {fixed_key(m): m["match"] for m in bracket if m["stage"] == "R32"}

    out = {}
    for ha, aa, venue in draw.values():
        key = frozenset(s for s in (team_slot.get(ha), team_slot.get(aa)) if s and s[0] in (1, 2))
        tmid = tmpl_by_key.get(key)
        if tmid is None or tmid in out:
            return {}  # ambiguous/incomplete mapping — fall back to reconstruction
        out[tmid] = (ha, aa, venue)
    return out if len(out) == len(r32_ids) else {}


def actual_r32_in_template_space(n_sims=1, seed=0):
    """{template_match_id: (home_abbr, away_abbr, venue)} for the drawn R32, mapped onto
    template positions (see _draw_to_template_positions). Lets the web app render the
    bracket via the template feeder tree. {} if not fully drawn / unmappable."""
    return _draw_to_template_positions(simulate_groups(n_sims=n_sims, seed=seed), load_bracket())


def walk_knockouts(sim, seed=1):
    """Walk R32->Final, sampling each tie's winner from Elo win expectancy.

    Host home-field applies only when a host plays a tie in its own country (venue
    -> host nation via VENUE_HOST). Returns reach-probability arrays per stage plus
    champion / third-place participation, shape (n_teams, n_sims).
    """
    rng = np.random.default_rng(seed)
    teams = sim["teams"]; n_sims = sim["n_sims"]; n_teams = len(teams)
    elo = teams["elo"].values
    hf = teams["home_field"].values
    tilt = teams["tilt"].values
    idx = sim["idx"]
    host_idx = {c: idx[c] for c in ("USA", "MEX", "CAN") if c in idx}
    arange = np.arange(n_sims)
    bracket = load_bracket()

    # Finished knockout ties are held fixed (like completed group results) so a
    # knocked-out team keeps no simulated path forward. {frozenset(pair): winner_i}.
    results = completed_ko_results(idx)

    r32 = resolve_r32(sim, bracket)

    # Once the R32 is fully drawn, seed the actual ties + venues onto the correct
    # template bracket positions (by group-slot — feed match-ids are permuted vs the
    # template's, so seeding by id mis-pairs the R16). The template feeder tree then
    # pairs the right ties. Partial/unmappable draws fall back to the reconstruction.
    seeded = _draw_to_template_positions(sim, bracket)
    venue_override = {}
    for tmid, (ha, aa, venue) in seeded.items():
        r32[tmid] = {"home": np.full(n_sims, idx[ha]), "away": np.full(n_sims, idx[aa])}
        venue_override[tmid] = venue

    winner = {}; loser = {}
    stages = ["R32", "R16", "QF", "SF", "3P", "F"]
    played = {st: np.zeros((n_teams, n_sims), dtype=bool) for st in stages}
    champ = np.zeros((n_teams, n_sims), dtype=bool)
    third = np.zeros((n_teams, n_sims), dtype=bool)

    # Opponent-mix xG accumulators per projection round (R32->4 ... F/3P->8).
    # Summed over sims where the team played that round; divided out at the end.
    proj_rounds = (4, 5, 6, 7, 8)
    sum_scored = {r: np.zeros(n_teams) for r in proj_rounds}
    sum_conceded = {r: np.zeros(n_teams) for r in proj_rounds}
    play_count = {r: np.zeros(n_teams) for r in proj_rounds}

    for m in sorted(bracket, key=lambda x: x["match"]):
        mid = m["match"]; st = m["stage"]
        if st == "R32":
            home_i, away_i = r32[mid]["home"], r32[mid]["away"]
        else:
            def feeder(spec):
                src = winner[spec["match"]] if spec["type"] == "winner" else loser[spec["match"]]
                return src
            home_i, away_i = feeder(m["home"]), feeder(m["away"])

        played[st][home_i, arange] = True
        played[st][away_i, arange] = True

        # Host home-field applies only when the host plays in its own country. Use the
        # actual drawn venue for seeded R32 ties; template venue elsewhere (R16+ until drawn).
        venue = venue_override.get(mid, m["venue"])
        vh = host_idx.get(VENUE_HOST.get(venue))
        bonus_h = hf[home_i] * (home_i == vh) if vh is not None else 0.0
        bonus_a = hf[away_i] * (away_i == vh) if vh is not None else 0.0
        we = win_expectancy(elo[home_i] + bonus_h, elo[away_i] + bonus_a)

        # Accumulate opponent-mix xG (same Elo->xG + tilt form as the group engine).
        match_tilt = tilt[home_i] + tilt[away_i]
        scale = GOALS_SCALE * (1 + TILT_K * match_tilt)
        xg_h = expected_goals(we) * scale
        xg_a = expected_goals(1 - we) * scale
        pr = STAGE_TO_ROUND[st]
        np.add.at(sum_scored[pr], home_i, xg_h)
        np.add.at(sum_conceded[pr], home_i, xg_a)
        np.add.at(play_count[pr], home_i, 1)
        np.add.at(sum_scored[pr], away_i, xg_a)
        np.add.at(sum_conceded[pr], away_i, xg_h)
        np.add.at(play_count[pr], away_i, 1)

        home_win = rng.random(n_sims) < we
        # Override the sampled outcome wherever this exact tie has already been
        # played — force the real winner so eliminated teams advance in 0 sims.
        if results:
            for pair, win_i in results.items():
                a_i, b_i = tuple(pair)
                hit = ((home_i == a_i) & (away_i == b_i)) | ((home_i == b_i) & (away_i == a_i))
                if hit.any():
                    home_win[hit] = home_i[hit] == win_i
        winner[mid] = np.where(home_win, home_i, away_i)
        loser[mid] = np.where(home_win, away_i, home_i)

        if mid == 104:
            champ[winner[mid], arange] = True
        if mid == 103:
            third[winner[mid], arange] = True

    # Conditional (given the team plays that round) opponent-mix xG, and P(play).
    cond_scored = {}; cond_conceded = {}; p_play = {}
    for r in proj_rounds:
        cnt = play_count[r]
        safe = np.where(cnt > 0, cnt, 1)
        cond_scored[r] = sum_scored[r] / safe
        cond_conceded[r] = sum_conceded[r] / safe
        p_play[r] = cnt / n_sims

    return {
        "played": played, "champ": champ, "third": third,
        "cond_scored": cond_scored, "cond_conceded": cond_conceded, "p_play": p_play,
    }


TEAM_PROBS_PATH = "data/knockout_team_probs.csv"
TEAM_ROUNDS_PATH = "data/knockout_team_rounds.csv"
PROJ_ROUNDS = (4, 5, 6, 7, 8)


# P(progress FROM this round) for the Qualification Booster. "Progress" = reach the
# NEXT knockout stage, which is bracket-aware: from the SF it means reaching the
# final (NOT merely playing round 8, since round 8 also holds the 3rd-place playoff
# for SF losers); from the final it means winning it (champion).
ADVANCE_STAGE = {4: "R16", 5: "QF", 6: "SF", 7: "F"}  # round -> stage that = "progressed"


def build_team_rounds(n_sims=40000, seed=0):
    """Per-team, per-knockout-round (4..8) Monte Carlo aggregates.

    Long-format table the *live* projection engine consumes to build its KO base
    state — this replaces the static per-player knockout_projections.csv. Each row
    carries the opponent-mix-averaged team xG/xGA *conditional on the team playing
    that round*, P(play) (the probability the team reaches it), and p_advance
    (P(progress FROM that round) — feeds the solver's Qualification Booster).
    The web app crosses these with the immutable per-player constants and routes
    them through the same scoring engine as the group stage, so xMins / share /
    scouting edits apply live. Resolved-fixture overrides (head-to-head xG,
    P(play)=1) are layered on app-side from knockout_fixtures.csv.

    Columns: abbr, round, cond_scored, cond_conceded, p_play, p_advance.
    """
    sim = simulate_groups(n_sims=n_sims, seed=seed)
    ko = walk_knockouts(sim, seed=seed + 1)
    abbr = sim["teams"]["abbr"].values
    rows = []
    for r in PROJ_ROUNDS:
        cs, cc, pp = ko["cond_scored"][r], ko["cond_conceded"][r], ko["p_play"][r]
        # p_advance: reach the next stage (rounds 4-7) or win the final (round 8).
        padv = ko["champ"].mean(axis=1) if r == 8 else ko["played"][ADVANCE_STAGE[r]].mean(axis=1)
        for i, ab in enumerate(abbr):
            rows.append({
                "abbr": ab,
                "round": r,
                "cond_scored": round(float(cs[i]), 4),
                "cond_conceded": round(float(cc[i]), 4),
                "p_play": round(float(pp[i]), 4),
                "p_advance": round(float(padv[i]), 4),
            })
    return pd.DataFrame(rows, columns=["abbr", "round", "cond_scored",
                                       "cond_conceded", "p_play", "p_advance"])


KO_FIXTURES_PATH = "data/knockout_fixtures.csv"


def _h2h_xg(teams, idx, home_abbr, away_abbr, venue):
    """Head-to-head (home, away) xG for a known knockout matchup, venue-aware HF."""
    elo = teams["elo"].values; hf = teams["home_field"].values; tilt = teams["tilt"].values
    h, a = idx[home_abbr], idx[away_abbr]
    host = idx.get(VENUE_HOST.get(venue))
    bonus_h = hf[h] if (host is not None and h == host) else 0.0
    bonus_a = hf[a] if (host is not None and a == host) else 0.0
    we = win_expectancy(elo[h] + bonus_h, elo[a] + bonus_a)
    scale = GOALS_SCALE * (1 + TILT_K * (tilt[h] + tilt[a]))
    xg_h = float(expected_goals(np.array([we]))[0]) * scale
    xg_a = float(expected_goals(np.array([1 - we]))[0]) * scale
    return xg_h, xg_a


def confirmed_knockout_fixtures():
    """Knockout matchups FIFA has already populated in rounds.json, with
    head-to-head xG/CS. Lets the app show resolved fixtures as cards rather than
    opponent-mix projections. Empty frame if none are drawn yet."""
    cols = ["round", "stage", "match", "venue", "home_abbr", "away_abbr",
            "home_xg", "away_xg", "home_cs", "away_cs"]
    teams, idx = load_teams()
    bracket = {m["match"]: m for m in load_bracket()}
    squad_abbr = {s["id"]: s["abbr"] for s in json.load(open("cache/squads.json"))}
    rows = []
    for r in json.load(open(ROUNDS_CACHE)):
        if r["id"] < 4:
            continue
        for t in r["tournaments"]:
            hid, aid = t.get("homeSquadId"), t.get("awaySquadId")
            if not hid or not aid:
                continue
            ha, aa = squad_abbr.get(hid), squad_abbr.get(aid)
            if ha not in idx or aa not in idx:
                continue
            # Venue from the live feed (rounds.json), NOT the pre-tournament bracket
            # template — the template's venues/routing can be stale or misaligned with
            # the actual draw. Normalise "City, State" -> "City" to match VENUE_HOST
            # (e.g. "Houston, Texas" -> "Houston"), so host home-field is applied right.
            venue = (t.get("venueCity") or "").split(",")[0].strip()
            stage = bracket.get(t["id"], {}).get("stage", "")
            xg_h, xg_a = _h2h_xg(teams, idx, ha, aa, venue)
            if stage == "3P":
                xg_h *= THIRD_PLACE_XG_MULT
                xg_a *= THIRD_PLACE_XG_MULT
            rows.append({
                "round": r["id"], "stage": stage,
                "match": t["id"], "venue": venue, "home_abbr": ha, "away_abbr": aa,
                "home_xg": round(xg_h, 2), "away_xg": round(xg_a, 2),
                "home_cs": round(float(np.exp(-xg_a)) * 100, 1),
                "away_cs": round(float(np.exp(-xg_h)) * 100, 1),
            })
    return pd.DataFrame(rows, columns=cols)


def build_team_probs(n_sims=40000, seed=0):
    """Run the full tournament sim and return per-team round-reach probabilities."""
    sim = simulate_groups(n_sims=n_sims, seed=seed)
    ko = walk_knockouts(sim, seed=seed + 1)
    t = sim["teams"]
    out = pd.DataFrame({"abbr": t["abbr"], "name": t["name"], "group": t["group"], "elo": t["elo"]})
    out["P_1st"] = (sim["pos"] == 1).mean(axis=1)
    out["P_2nd"] = (sim["pos"] == 2).mean(axis=1)
    out["P_3rd"] = (sim["pos"] == 3).mean(axis=1)
    out["P_best3"] = sim["third_qualify"].mean(axis=1)
    out["P_R32"] = sim["advance"].mean(axis=1)
    out["P_R16"] = ko["played"]["R16"].mean(axis=1)
    out["P_QF"] = ko["played"]["QF"].mean(axis=1)
    out["P_SF"] = ko["played"]["SF"].mean(axis=1)
    out["P_Final"] = ko["played"]["F"].mean(axis=1)
    out["P_Champ"] = ko["champ"].mean(axis=1)
    out["P_3rdPlace"] = ko["third"].mean(axis=1)
    return out.sort_values("P_Champ", ascending=False).reset_index(drop=True)


def _report(sim):
    teams = sim["teams"]; n = sim["n_sims"]; pos = sim["pos"]
    out = teams[["abbr", "name", "group", "elo"]].copy()
    out["P_1st"] = (pos == 1).mean(axis=1)
    out["P_2nd"] = (pos == 2).mean(axis=1)
    out["P_3rd"] = (pos == 3).mean(axis=1)
    out["P_best3"] = sim["third_qualify"].mean(axis=1)
    out["P_R32"] = sim["advance"].mean(axis=1)
    out = out.sort_values(["group", "P_R32"], ascending=[True, False])
    pd.set_option("display.max_rows", None, "display.width", 140)
    for col in ["P_1st", "P_2nd", "P_3rd", "P_best3", "P_R32"]:
        out[col] = (out[col] * 100).round(1)
    print(out.to_string(index=False))
    # Sanity checks
    print("\n--- sanity ---")
    print("mean teams advancing per sim:", round(sim["advance"].sum(axis=0).mean(), 2), "(expect 32)")
    print("mean best-thirds per sim:    ", round(sim["third_qualify"].sum(axis=0).mean(), 2), "(expect 8)")


if __name__ == "__main__":
    probs = build_team_probs(n_sims=40000, seed=0)
    probs.to_csv(TEAM_PROBS_PATH, index=False)
    print(f"wrote {TEAM_PROBS_PATH}  ({len(probs)} teams)")

    team_rounds = build_team_rounds(n_sims=40000, seed=0)
    team_rounds.to_csv(TEAM_ROUNDS_PATH, index=False)
    print(f"wrote {TEAM_ROUNDS_PATH}  ({len(team_rounds)} team-rounds)")

    fixtures = confirmed_knockout_fixtures()
    fixtures.to_csv(KO_FIXTURES_PATH, index=False)
    print(f"wrote {KO_FIXTURES_PATH}  ({len(fixtures)} confirmed fixtures)")

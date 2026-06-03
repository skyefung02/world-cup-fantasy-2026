#!/usr/bin/env python
"""
build_default_xmins.py

Scrapes FBref international competition data, computes per-player expected
minutes per match, and writes data/default_xmins.csv.

Pipeline (mirrors notebooks/05_international_data_fbref.ipynb):
    1. Scrape WCQ (6 confederations) + 5 v2 competitions (cached via soccerdata)
    2. Filter to WC nations, write data/processed/international_stats_2026.csv
    3. Compute per-competition ELO-based strength, write data/processed/comp_strength.csv
    4. Aggregate per-player: mp_share, shrunken conditional_min, per_team_match
    5. Match WC roster: exact ASCII → multi-scorer fuzzy → manual overrides
    6. Select starters: algorithmic (top 10 outfield + 1 GK) → manual XI overrides
    7. Hybrid normalization (starters → conditional, non-starters → leftover)
    8. Build output with composite-position-expanded + name-only fallback merge
    9. Distribute leftover evenly to fill 990/team, write data/default_xmins.csv

Usage:
    python build_default_xmins.py

Heavy work (FBref scraping) is cached in ~/soccerdata/data/FBref/ — subsequent runs
take well under a minute.
"""

import os
import re
import sys
import unicodedata

import numpy as np
import pandas as pd
import soccerdata as sd
from lxml import etree, html
from pathlib import Path
from rapidfuzz import process, fuzz

from data.manual_overrides import MANUAL_OVERRIDES
from data.manual_starting_xi import MANUAL_STARTING_XI

# ──────────────────────────────────────────────────────────────────────────────
# Paths and constants
# ──────────────────────────────────────────────────────────────────────────────

PROCESSED_DIR  = "data/processed"
ROSTER_PATH    = f"{PROCESSED_DIR}/player_fixtures.csv"
INTL_CSV       = f"{PROCESSED_DIR}/international_stats_2026.csv"
COMP_STR_CSV   = f"{PROCESSED_DIR}/comp_strength.csv"
ELO_PATH       = "data/elo_ratings.csv"
OUT_PATH       = "data/default_xmins.csv"

CACHE_DIR = Path.home() / "soccerdata" / "data" / "FBref" / "intl_probe"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 6 WCQ confederation pages + 5 continental tournaments + Nations League
WCQ_URLS = {
    "UEFA":      "https://fbref.com/en/comps/6/stats/WCQ----UEFA-M-Stats",
    "CAF":       "https://fbref.com/en/comps/2/stats/WCQ----CAF-M-Stats",
    "CONCACAF":  "https://fbref.com/en/comps/3/stats/WCQ----CONCACAF-M-Stats",
    "CONMEBOL":  "https://fbref.com/en/comps/4/stats/WCQ----CONMEBOL-M-Stats",
    "OFC":       "https://fbref.com/en/comps/5/stats/WCQ----OFC-M-Stats",
    "AFC":       "https://fbref.com/en/comps/7/stats/WCQ----AFC-M-Stats",
}
V2_COMPS = [
    ("UEFA",     "UEFA Nations League", "https://fbref.com/en/comps/677/stats/UEFA-Nations-League-Stats"),
    ("CAF",      "AFCON 2025",          "https://fbref.com/en/comps/656/stats/Africa-Cup-of-Nations-Stats"),
    ("CONCACAF", "Gold Cup 2025",       "https://fbref.com/en/comps/681/stats/Gold-Cup-Stats"),
    ("UEFA",     "Euro 2024",           "https://fbref.com/en/comps/676/stats/UEFA-Euro-Stats"),
    ("CONMEBOL", "Copa America 2024",   "https://fbref.com/en/comps/685/stats/Copa-America-Stats"),
]

COMP_WEIGHTS = {
    "UEFA WCQ": 1.0, "CAF WCQ": 1.0, "CONMEBOL WCQ": 1.0,
    "AFC WCQ": 1.0, "CONCACAF WCQ": 1.0, "OFC WCQ": 1.0,
    "UEFA Nations League": 0.7,
    "AFCON 2025": 0.7,
    "Gold Cup 2025": 0.7,
    "Euro 2024": 0.5,
    "Copa America 2024": 0.5,
}

WC_TO_FBREF_SQUAD = {
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Cabo Verde": "Cape Verde",
    "USA": "United States",
}

FBREF_TO_ELO = {
    "Bosnia-Herzegovina":   "Bosnia/Herzegovina",
    "Côte d'Ivoire":        "Cote d'Ivoire",
    "Curaçao":              "Curacao",
    "Congo DR":             "Dem. Rep. Congo",
    "Congo":                "Rep, Congo",
    "Rep. of Ireland":      "Rep. Ireland",
    "N. Macedonia":         "North Macedonia",
    "UAE":                  "Unit. Arab Emir.",
    "China PR":             "China",
    "CAR":                  "Cent. Afr. Rep.",
    "Korea Republic":       "South Korea",
    "United States":        "United States",
    "IR Iran":              "Iran",
}

# Tunable parameters
K_COND_SHRINK        = 5     # phantom appearances toward prior in conditional_min
PRIOR_COND           = 75    # typical starter min/appearance prior
MIN_STARTER_MP_SHARE = 0.4   # eligibility threshold for algorithmic starter selection
POS_TO_FBREF = {"GK": "GK", "DEF": "DF", "MID": "MF", "FWD": "FW"}  # roster pos → FBref Pos code
POS_MAP = {"GK": "GK", "DF": "DEF", "MF": "MID", "FW": "FWD"}        # FBref Pos code → roster pos
FUZZY_SCORERS = [
    ("token_sort_ratio", 80),
    ("WRatio",           85),
    ("partial_ratio",    90),
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def to_ascii(name):
    s = str(name)
    s = s.replace("ı", "i").replace("İ", "I")
    for ch in ("'", "'", "`", "ʼ"):
        s = s.replace(ch, "")
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()


def section(title):
    print(f"\n{'═' * 78}\n {title}\n{'═' * 78}")


def fetch_comp_standard(fbref_reader, url, cache_name):
    """Pull a comp's player Standard Stats table off FBref via soccerdata's session."""
    from soccerdata.fbref import _parse_table
    cache = CACHE_DIR / cache_name
    reader = fbref_reader.get(url, cache)
    tree = html.parse(reader)
    parser = etree.HTMLParser(recover=True)

    candidates = []
    for c in tree.xpath("//comment()[contains(., 'stats_standard')]"):
        root = etree.fromstring(f"<root>{c.text}</root>", parser)
        candidates.extend(root.xpath(".//table[contains(@id, 'stats_standard')]"))
    candidates.extend(tree.xpath("//table[contains(@id, 'stats_standard')]"))

    seen, unique = set(), []
    for t in candidates:
        if t.get("id") in seen:
            continue
        seen.add(t.get("id"))
        unique.append(t)
    if not unique:
        return None

    df = _parse_table(unique[0])
    df.columns = [b if (not a or a == b or str(a).startswith("Unnamed")) else f"{a}_{b}"
                  for a, b in df.columns]
    return df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def step_1_scrape_intl():
    """Scrape WCQ + v2 comps, filter to WC nations, write international_stats_2026.csv."""
    section("Step 1: Scrape international stats (cached via soccerdata)")

    fbref = sd.FBref(leagues=["ENG-Premier League"], seasons=2025)  # session-only

    # WCQ confederation pages
    frames = []
    for confed, url in WCQ_URLS.items():
        df = fetch_comp_standard(fbref, url, f"wcq_{confed}_2026.html")
        if df is None:
            print(f"  {confed}: NO TABLE FOUND")
            continue
        df["confederation"] = confed
        df["competition"] = f"{confed} WCQ"
        print(f"  {confed} WCQ:        {len(df):4d} player rows, {df['Squad'].nunique():2d} squads")
        frames.append(df)

    # v2 comps
    for confed, comp, url in V2_COMPS:
        df = fetch_comp_standard(fbref, url, f"{comp.replace(' ', '_')}.html")
        if df is None:
            print(f"  {comp}: NO TABLE FOUND")
            continue
        df["confederation"] = confed
        df["competition"] = comp
        print(f"  {comp:25s} {len(df):4d} player rows, {df['Squad'].nunique():2d} squads")
        frames.append(df)

    intl_all = pd.concat(frames, ignore_index=True)
    print(f"\n  Combined: {len(intl_all)} rows across {intl_all['Squad'].nunique()} squads")

    # Filter to WC nations
    wc_players = pd.read_csv(ROSTER_PATH)[["player", "team"]].drop_duplicates()
    wc_squads_fbref = set(wc_players["team"].map(WC_TO_FBREF_SQUAD).fillna(wc_players["team"]))

    intl_filtered = intl_all[intl_all["Squad"].isin(wc_squads_fbref)].copy()
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    intl_filtered.to_csv(INTL_CSV, index=False)
    print(f"\n  Wrote {len(intl_filtered)} WC-team rows → {INTL_CSV}")

    return intl_filtered


def step_2_comp_strength(intl_full):
    """Compute per-competition mean ELO and normalized strength multiplier."""
    section("Step 2: Compute comp strength from ELO")

    elo = pd.read_csv(ELO_PATH)
    elo["country_clean"] = elo["Country"].apply(
        lambda s: re.sub(r":[a-z\-]+:\s*", "", str(s)).replace("🏆", "").strip()
    )
    intl_full["elo_country"] = intl_full["Squad"].map(FBREF_TO_ELO).fillna(intl_full["Squad"])
    elo_lookup = elo.set_index("country_clean")["PELE"].to_dict()

    participants = intl_full.groupby("competition")["elo_country"].unique().to_dict()
    rows, unmatched = [], set()
    for comp, squads in participants.items():
        peles = []
        for s in squads:
            if s in elo_lookup:
                peles.append(elo_lookup[s])
            else:
                unmatched.add(s)
        rows.append({"competition": comp, "n_squads": len(squads),
                     "n_matched": len(peles),
                     "mean_pele": sum(peles) / len(peles) if peles else None})

    cs = pd.DataFrame(rows).sort_values("mean_pele", ascending=False)
    cs["strength_mult"] = cs["mean_pele"] / cs["mean_pele"].max()
    cs.to_csv(COMP_STR_CSV, index=False)

    print(cs[["competition", "mean_pele", "strength_mult"]].round(3).to_string(index=False))
    if unmatched:
        print(f"\n  Unmatched ELO squads (treated as 0-PELE): {sorted(unmatched)[:10]}{'...' if len(unmatched) > 10 else ''}")
    print(f"\n  Wrote → {COMP_STR_CSV}")


def step_3_player_aggregates(intl_full):
    """Per-(Player, Squad) mp_share, shrunken conditional_min, per_team_match."""
    section("Step 3: Per-player aggregates")

    intl = intl_full.copy()
    intl["min_num"] = pd.to_numeric(intl["Playing Time_Min"].astype(str).str.replace(",", ""),
                                     errors="coerce").fillna(0)
    intl["mp_num"]  = pd.to_numeric(intl["Playing Time_MP"], errors="coerce").fillna(0)
    intl["comp_w"]  = intl["competition"].map(COMP_WEIGHTS).fillna(0.5)

    team_matches = (intl.groupby(["Squad", "competition"])["mp_num"].max().reset_index()
                    .rename(columns={"mp_num": "team_matches"}))
    team_matches["comp_w"] = team_matches["competition"].map(COMP_WEIGHTS).fillna(0.5)
    team_matches["weighted_team_mp"] = team_matches["team_matches"] * team_matches["comp_w"]
    team_avail = (team_matches.groupby("Squad")["weighted_team_mp"].sum()
                  .reset_index().rename(columns={"weighted_team_mp": "team_mp_w"}))

    intl["weighted_min"] = intl["min_num"] * intl["comp_w"]
    intl["weighted_mp"]  = intl["mp_num"]  * intl["comp_w"]

    player_agg = (intl.groupby(["Player", "Squad"], as_index=False)
                  .agg(weighted_min=("weighted_min", "sum"),
                       weighted_mp=("weighted_mp", "sum")))
    player_pos = (intl.groupby(["Player", "Squad"])["Pos"]
                  .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
                  .reset_index())
    player_agg = player_agg.merge(player_pos, on=["Player", "Squad"]).merge(team_avail, on="Squad")

    player_agg["is_gk"]    = player_agg["Pos"].str.startswith("GK")
    player_agg["mp_share"] = player_agg["weighted_mp"] / player_agg["team_mp_w"]
    player_agg["conditional_min"] = np.where(
        player_agg["weighted_mp"] > 0,
        (player_agg["weighted_min"] + K_COND_SHRINK * PRIOR_COND) /
        (player_agg["weighted_mp"] + K_COND_SHRINK),
        0
    )
    player_agg["per_team_match"] = player_agg["weighted_min"] / player_agg["team_mp_w"]
    player_agg["name_ascii"]     = player_agg["Player"].apply(to_ascii)

    print(f"  {len(player_agg)} (Player, Squad) rows across {player_agg['Squad'].nunique()} squads")
    return player_agg


def step_4_roster_match(player_agg):
    """Match WC roster to FBref intl data: exact ASCII → multi-scorer fuzzy."""
    section("Step 4: WC roster matching")

    wc_roster = pd.read_csv(ROSTER_PATH)[["player", "team", "position"]].drop_duplicates()
    wc_roster["name_ascii"] = wc_roster["player"].apply(
        lambda p: to_ascii(MANUAL_OVERRIDES.get(p, p))
    )
    wc_roster["fbref_squad"] = wc_roster["team"].map(WC_TO_FBREF_SQUAD).fillna(wc_roster["team"])

    # Pass 1: exact ASCII match
    roster_keys = set(zip(wc_roster["name_ascii"], wc_roster["fbref_squad"]))
    player_agg["roster_key"] = list(zip(player_agg["name_ascii"], player_agg["Squad"]))
    player_agg["in_roster"]  = player_agg["roster_key"].isin(roster_keys)

    print(f"  Intl player rows: {len(player_agg)}")
    print(f"    - exact match (Pass 1): {player_agg['in_roster'].sum()}")

    # Pass 2: multi-scorer fuzzy match
    matched_keys = set(zip(player_agg[player_agg["in_roster"]]["name_ascii"],
                           player_agg[player_agg["in_roster"]]["Squad"]))
    roster_unmatched = wc_roster[~wc_roster.apply(
        lambda r: (r["name_ascii"], r["fbref_squad"]) in matched_keys, axis=1
    )].copy()
    unclaimed_intl = player_agg[~player_agg["in_roster"]][["Squad", "name_ascii"]].copy()

    fuzzy_hits, match_log = [], []
    for _, r in roster_unmatched.iterrows():
        pool = unclaimed_intl[unclaimed_intl["Squad"] == r["fbref_squad"]]["name_ascii"].tolist()
        if not pool:
            continue
        best_match, best_score, best_scorer = None, -1, None
        for scorer_name, threshold in FUZZY_SCORERS:
            scorer = getattr(fuzz, scorer_name)
            result = process.extractOne(r["name_ascii"], pool, scorer=scorer, score_cutoff=threshold)
            if result and result[1] > best_score:
                best_match, best_score, best_scorer = result[0], result[1], scorer_name
        if best_match:
            fuzzy_hits.append((best_match, r["fbref_squad"]))
            match_log.append((r["player"], best_match, best_scorer, best_score))
            unclaimed_intl = unclaimed_intl[
                ~((unclaimed_intl["Squad"] == r["fbref_squad"]) &
                  (unclaimed_intl["name_ascii"] == best_match))
            ]

    roster_keys.update(fuzzy_hits)
    player_agg["in_roster"] = player_agg["roster_key"].isin(roster_keys)
    print(f"    - fuzzy match  (Pass 2): {len(fuzzy_hits)}")
    print(f"    Total matched: {player_agg['in_roster'].sum()} / {len(wc_roster)} WC roster players")

    # Validation: WC roster players still unmatched
    matched_intl_set = set(player_agg[player_agg["in_roster"]]["name_ascii"]) | {h[0] for h in fuzzy_hits}
    still_unmatched = wc_roster[~wc_roster.apply(
        lambda r: (r["name_ascii"], r["fbref_squad"]) in roster_keys, axis=1
    )]
    print(f"\n  Roster players NOT matched to intl data: {len(still_unmatched)}")
    if not still_unmatched.empty:
        by_team = still_unmatched.groupby("team").size().sort_values(ascending=False)
        print("    Top teams by unmatched count:")
        for team, n in by_team.head(10).items():
            print(f"      {team:30s} {n} players")

    return wc_roster, player_agg[player_agg["in_roster"]].copy()


def step_5_starter_selection(squad_agg, wc_roster):
    """Algorithmic starter selection + manual XI overrides."""
    section("Step 5: Starter selection (algorithmic + manual XI overrides)")

    # Algorithmic: top 1 GK + top 10 outfield by per_team_match, eligible if mp_share >= 0.4
    squad_agg["starter_eligible"] = squad_agg["mp_share"] >= MIN_STARTER_MP_SHARE
    squad_agg["rank_in_pool"] = (squad_agg[squad_agg["starter_eligible"]]
                                  .groupby(["Squad", "is_gk"])["per_team_match"]
                                  .rank(method="first", ascending=False))
    squad_agg["rank_in_pool"] = squad_agg["rank_in_pool"].fillna(999)
    squad_agg["is_starter"] = (
        squad_agg["starter_eligible"] & (
            ((squad_agg["is_gk"]) & (squad_agg["rank_in_pool"] == 1)) |
            ((~squad_agg["is_gk"]) & (squad_agg["rank_in_pool"] <= 10))
        )
    )

    # Override with MANUAL_STARTING_XI for teams in the dict
    MANUAL_XI_ASCII = {
        team: {to_ascii(MANUAL_OVERRIDES.get(p, p)) for p in players}
        for team, players in MANUAL_STARTING_XI.items()
    }
    FBREF_TO_WC = {v: k for k, v in WC_TO_FBREF_SQUAD.items()}
    squad_agg["wc_team"] = squad_agg["Squad"].map(FBREF_TO_WC).fillna(squad_agg["Squad"])

    def _apply_manual_xi(row):
        team = row["wc_team"]
        if team in MANUAL_XI_ASCII:
            return row["name_ascii"] in MANUAL_XI_ASCII[team]
        return row["is_starter"]

    squad_agg["is_starter"] = squad_agg.apply(_apply_manual_xi, axis=1)

    # Inject manual-XI starters that have no international data (uncapped backups,
    # fresh call-ups). They're absent from squad_agg because player_agg is built
    # purely from intl stats. Fall back to the mean conditional_min of *actual*
    # starters at the same position (a missing keeper inherits the typical
    # starting-GK minutes, an outfielder the typical starter at his position) so
    # the locked XI is honored instead of dropping to the ~4-min leftover floor.
    starters = squad_agg[squad_agg["is_starter"]]
    pos_norm = starters["Pos"].str.split(",").str[0].map(POS_MAP)
    pos_avg_cond = starters["conditional_min"].groupby(pos_norm).mean().to_dict()
    overall_avg_cond = starters["conditional_min"].mean()
    print("  Starter conditional_min by position (fallback for no-intl-data starters):")
    for p in ("GK", "DEF", "MID", "FWD"):
        print(f"    {p:4s} {pos_avg_cond.get(p, overall_avg_cond):.1f}")

    existing = set(zip(squad_agg["wc_team"], squad_agg["name_ascii"]))
    inj = []
    for team, xi_ascii in MANUAL_XI_ASCII.items():
        for a in xi_ascii:
            if (team, a) in existing:
                continue
            rr = wc_roster[(wc_roster["team"] == team) & (wc_roster["name_ascii"] == a)]
            if rr.empty:
                continue  # genuine name mismatch — leave it for the <11 audit below
            r = rr.iloc[0]
            inj.append({
                "Player": r["player"],
                "Squad": r["fbref_squad"],
                "name_ascii": a,
                "Pos": POS_TO_FBREF.get(r["position"], "MF"),
                "is_gk": r["position"] == "GK",
                "conditional_min": pos_avg_cond.get(r["position"], overall_avg_cond),
                "per_team_match": 0.0,
                "mp_share": 1.0,
                "wc_team": team,
                "is_starter": True,
            })
    if inj:
        squad_agg = pd.concat([squad_agg, pd.DataFrame(inj)], ignore_index=True)
        details = ", ".join(f"{d['Player']} ({d['Pos']}→{d['conditional_min']:.0f})" for d in inj)
        print(f"  Injected {len(inj)} manual-XI starter(s) with no intl data, "
              f"using position-average conditional_min: {details}")

    # Audit: manual XI coverage per team
    print(f"  Algorithmic teams (no manual XI): "
          f"{squad_agg['wc_team'].nunique() - len(MANUAL_XI_ASCII)}")
    print(f"  Manual XI teams: {len(MANUAL_XI_ASCII)}")
    print(f"\n  Manual XI match counts (anything < 11 indicates a name-matching issue):")
    incomplete = []
    for team in MANUAL_XI_ASCII:
        matched = squad_agg[(squad_agg["wc_team"] == team) & squad_agg["is_starter"]].shape[0]
        flag = "  ⚠" if matched < 11 else ""
        print(f"    {team:30s} {matched}/11{flag}")
        if matched < 11:
            incomplete.append((team, matched))
    if incomplete:
        print(f"\n  ⚠ {len(incomplete)} team(s) have incomplete manual XI matching — "
              "check name spellings or add entries to MANUAL_OVERRIDES")

    return squad_agg


def step_6_normalize_and_export(wc_roster, squad_agg):
    """Hybrid normalization, hybrid merge, leftover redistribution, export."""
    section("Step 6: Hybrid normalization + output build")

    # Starter sums per (Squad, is_gk)
    starter_sum_gk = (squad_agg[squad_agg["is_gk"] & squad_agg["is_starter"]]
                      .groupby("Squad")["conditional_min"].sum().rename("starter_sum_gk").reset_index())
    starter_sum_of = (squad_agg[~squad_agg["is_gk"] & squad_agg["is_starter"]]
                      .groupby("Squad")["conditional_min"].sum().rename("starter_sum_of").reset_index())
    squad_agg = (squad_agg.merge(starter_sum_gk, on="Squad", how="left")
                          .merge(starter_sum_of, on="Squad", how="left"))
    squad_agg["starter_sum"] = np.where(squad_agg["is_gk"], squad_agg["starter_sum_gk"], squad_agg["starter_sum_of"])
    squad_agg["pool_budget"] = np.where(squad_agg["is_gk"], 90, 900)
    squad_agg["starter_factor"] = np.minimum(1.0, squad_agg["pool_budget"] / squad_agg["starter_sum"])

    ns_sum_gk = (squad_agg[squad_agg["is_gk"] & ~squad_agg["is_starter"]]
                 .groupby("Squad")["per_team_match"].sum().rename("ns_sum_gk").reset_index())
    ns_sum_of = (squad_agg[~squad_agg["is_gk"] & ~squad_agg["is_starter"]]
                 .groupby("Squad")["per_team_match"].sum().rename("ns_sum_of").reset_index())
    squad_agg = (squad_agg.merge(ns_sum_gk, on="Squad", how="left")
                          .merge(ns_sum_of, on="Squad", how="left"))
    squad_agg["ns_sum"]    = np.where(squad_agg["is_gk"], squad_agg["ns_sum_gk"], squad_agg["ns_sum_of"])
    squad_agg["ns_sum"]    = squad_agg["ns_sum"].fillna(0)
    squad_agg["ns_budget"] = squad_agg["pool_budget"] - squad_agg["starter_sum"] * squad_agg["starter_factor"]
    squad_agg["ns_factor"] = np.where(squad_agg["ns_sum"] > 0,
                                       np.minimum(1.0, squad_agg["ns_budget"] / squad_agg["ns_sum"]), 0)
    squad_agg["xmins"] = np.where(
        squad_agg["is_starter"],
        squad_agg["conditional_min"] * squad_agg["starter_factor"],
        squad_agg["per_team_match"] * squad_agg["ns_factor"]
    )

    # Hybrid merge: name-only for unique names, position-aware for duplicates (Danilo case)
    te = squad_agg.assign(pos_tokens=squad_agg["Pos"].str.split(",")).explode("pos_tokens")
    te["pos_primary"] = te["pos_tokens"].map(POS_MAP)
    te = te.drop_duplicates(subset=["Player", "Squad", "pos_primary"])

    wc_roster["dup_count"] = wc_roster.groupby(["team", "name_ascii"])["name_ascii"].transform("count")

    te_namekey = te.drop_duplicates(subset=["Squad", "name_ascii"])
    out_uniq = wc_roster[wc_roster["dup_count"] == 1].merge(
        te_namekey[["name_ascii", "Squad", "Pos", "is_starter", "xmins"]],
        left_on=["name_ascii", "fbref_squad"], right_on=["name_ascii", "Squad"],
        how="left"
    )
    out_dup = wc_roster[wc_roster["dup_count"] > 1].merge(
        te[["name_ascii", "Squad", "pos_primary", "Pos", "is_starter", "xmins"]],
        left_on=["name_ascii", "fbref_squad", "position"],
        right_on=["name_ascii", "Squad", "pos_primary"], how="left"
    ).drop_duplicates(subset=["player", "team", "position"], keep="first")

    output = pd.concat([out_uniq, out_dup], ignore_index=True, sort=False)
    output["xmins"]      = output["xmins"].fillna(0)
    output["is_starter"] = output["is_starter"].fillna(False)

    matched = (output["xmins"] > 0).sum()
    print(f"  Roster players with non-zero algorithmic xMins: {matched} / {len(output)}")

    # Even-distribution leftover boost so each team's total = 990
    team_total = output.groupby("team")["xmins"].sum().rename("team_total").reset_index()
    team_size  = output.groupby("team").size().rename("team_size").reset_index()
    output = output.merge(team_total, on="team").merge(team_size, on="team")
    output["leftover"]   = (990 - output["team_total"]).clip(lower=0)
    output["even_boost"] = output["leftover"] / output["team_size"]
    output["xmins"]      = (output["xmins"] + output["even_boost"]).clip(upper=90)

    # Per-team totals (sanity check)
    print(f"\n  Per-team xMins totals (should all = ~990):")
    totals = output.groupby("team")["xmins"].sum().sort_values()
    print(f"    Lowest 5:")
    for t, v in totals.head().items():
        flag = "  ⚠" if v < 980 else ""
        print(f"      {t:30s} {v:6.1f}{flag}")
    print(f"    Highest 5:")
    for t, v in totals.tail().items():
        print(f"      {t:30s} {v:6.1f}")

    # Export
    out = output[["player", "team", "position", "xmins"]].rename(columns={"xmins": "default_xmins"})
    out["default_xmins"] = out["default_xmins"].round(2)
    out.to_csv(OUT_PATH, index=False)
    print(f"\n  Wrote {len(out)} rows → {OUT_PATH}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(ROSTER_PATH):
        print(f"ERROR: {ROSTER_PATH} not found. Run fetch_data.py first.", file=sys.stderr)
        sys.exit(1)

    intl_full   = step_1_scrape_intl()
    step_2_comp_strength(intl_full)
    player_agg  = step_3_player_aggregates(intl_full)
    wc_roster, squad_agg = step_4_roster_match(player_agg)
    squad_agg   = step_5_starter_selection(squad_agg, wc_roster)
    step_6_normalize_and_export(wc_roster, squad_agg)

    print(f"\n{'═' * 78}\n  DONE.\n{'═' * 78}")


if __name__ == "__main__":
    main()

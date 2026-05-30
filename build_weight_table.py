#!/usr/bin/env python
"""
build_weight_table.py

Scrapes FBref club data from 22 leagues, matches WC squad players,
blends club + international per-90 rates via shrinkage, and writes
data/weight_table.csv.

Pipeline (mirrors notebooks/03_player_weights.ipynb):
    1. Scrape player_season_stats(stat_type="standard") for 22 club leagues (cached)
    2. Match WC roster to FBref club data:
         - Pass 1: exact ASCII match
         - Pass 2: nation-constrained fuzzy match (token_sort_ratio >= 85)
         - Pass 3: MANUAL_OVERRIDES from data/manual_overrides.py
    3. Merge stats; filter to players with >= 900 club minutes
    4. International blend layer (shrinkage of intl rates → blend with club via τ=1.7)
    5. Apply league_strength multiplier and write data/weight_table.csv

Usage:
    python build_weight_table.py

Heavy work (FBref scraping) is cached in ~/soccerdata/data/FBref/ — subsequent runs
take well under a minute.

Prerequisites:
    - fetch_data.py has been run (so data/processed/player_fixtures.csv exists)
    - build_default_xmins.py has been run (so the international stats + comp_strength
      CSVs exist; weight_table builds on top of these for the intl blend)
"""

import json
import os
import sys
import unicodedata

import numpy as np
import pandas as pd
import soccerdata as sd
from rapidfuzz import process, fuzz

from data.manual_overrides import MANUAL_OVERRIDES

# ──────────────────────────────────────────────────────────────────────────────
# Paths and constants
# ──────────────────────────────────────────────────────────────────────────────

PROCESSED_DIR   = "data/processed"
ROSTER_PATH     = f"{PROCESSED_DIR}/player_fixtures.csv"
INTL_CSV        = f"{PROCESSED_DIR}/international_stats_2026.csv"
COMP_STR_CSV    = f"{PROCESSED_DIR}/comp_strength.csv"
LEAGUE_STR_PATH = "data/league_strength.json"
OUT_PATH        = "data/weight_table.csv"

ALL_LEAGUES = [
    "ENG-Premier League", "ESP-La Liga", "FRA-Ligue 1", "GER-Bundesliga", "ITA-Serie A",
    "NED-Eredivisie", "POR-Primeira Liga", "SAU-Saudi Pro League", "TUR-Super Lig",
    "BEL-First Division A", "SCO-Premiership", "GRE-Super League", "CZE-First League",
    "AUT-Bundesliga", "SUI-Super League", "BRA-Serie A", "ARG-Primera Division",
    "USA-MLS", "MEX-Liga MX", "KOR-K League 1", "JPN-J1 League", "ENG2-EFL Championship",
]

# Same comp weights as build_default_xmins.py — kept in sync
COMP_WEIGHTS = {
    "UEFA WCQ": 1.0, "CAF WCQ": 1.0, "CONMEBOL WCQ": 1.0,
    "AFC WCQ": 1.0, "CONCACAF WCQ": 1.0, "OFC WCQ": 1.0,
    "UEFA Nations League": 0.7,
    "AFCON 2025": 0.7,
    "Gold Cup 2025": 0.7,
    "Euro 2024": 0.5,
    "Copa America 2024": 0.5,
}

TEAM_TO_NATION = {
    "Algeria": "ALG", "Argentina": "ARG", "Australia": "AUS",
    "Austria": "AUT", "Belgium": "BEL", "Bosnia and Herzegovina": "BIH",
    "Brazil": "BRA", "Cabo Verde": "CPV", "Canada": "CAN", "Chile": "CHI",
    "Colombia": "COL", "Congo DR": "COD", "Costa Rica": "CRC",
    "Croatia": "CRO", "Curaçao": "CUW", "Côte d'Ivoire": "CIV",
    "Czechia": "CZE", "Ecuador": "ECU", "Egypt": "EGY", "England": "ENG",
    "France": "FRA", "Germany": "GER", "Ghana": "GHA", "Greece": "GRE",
    "Haiti": "HAI", "Honduras": "HON", "Hungary": "HUN", "IR Iran": "IRN",
    "Iraq": "IRQ", "Japan": "JPN", "Jordan": "JOR", "Kenya": "KEN",
    "Korea Republic": "KOR", "Mexico": "MEX", "Morocco": "MAR",
    "Netherlands": "NED", "New Zealand": "NZL", "Nigeria": "NGA",
    "Norway": "NOR", "Panama": "PAN", "Paraguay": "PAR", "Peru": "PER",
    "Poland": "POL", "Portugal": "POR", "Qatar": "QAT", "Romania": "ROU",
    "Saudi Arabia": "KSA", "Scotland": "SCO", "Senegal": "SEN",
    "Serbia": "SRB", "Slovakia": "SVK", "Slovenia": "SVN",
    "South Africa": "RSA", "Spain": "ESP", "Sweden": "SWE",
    "Switzerland": "SUI", "Trinidad and Tobago": "TRI", "Tunisia": "TUN",
    "Türkiye": "TUR", "Ukraine": "UKR", "Uruguay": "URU", "USA": "USA",
    "Uzbekistan": "UZB", "Venezuela": "VEN",
}

WC_TO_FBREF_SQUAD = {
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Cabo Verde": "Cape Verde",
    "USA": "United States",
}

# Tunable parameters (kept aligned with notebook 03)
MIN_MINS                = 900   # minimum club minutes to enter the blend
FUZZY_THRESHOLD         = 85    # for nation-constrained fuzzy match
KNOWN_FALSE_POSITIVES   = {("Weverton", "Brazil")}
TAU                     = 1.7   # international-minute trust multiplier
K_INTL_SHRINK           = 375   # phantom-minute prior pulling intl rate toward club rate


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def to_ascii(name):
    return unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower().strip()


def section(title):
    print(f"\n{'═' * 78}\n {title}\n{'═' * 78}")


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def step_1_scrape_clubs():
    """Scrape player_season_stats from 22 leagues; concat into one MultiIndex DataFrame."""
    section("Step 1: Scrape FBref club data (cached via soccerdata)")

    all_stats = []
    for league in ALL_LEAGUES:
        try:
            fbref = sd.FBref(leagues=[league], seasons=2025)
            s = fbref.read_player_season_stats(stat_type="standard")
            all_stats.append(s)
            print(f"  {league:30s} {s.shape[0]:5d} players")
        except Exception as e:
            print(f"  {league:30s} ERROR — {e}")

    stats = pd.concat(all_stats)
    print(f"\n  Total players across leagues: {stats.shape[0]}")
    return stats


def step_2_match_roster(stats):
    """Match WC roster to FBref club data via 3 passes (exact / nation-fuzzy / manual)."""
    section("Step 2: Match WC roster to FBref club data")

    # Flatten MultiIndex columns
    stats_flat = stats.copy()
    stats_flat.columns = [
        f"{b}" if not a or a == b else f"{a}_{b}"
        for a, b in stats_flat.columns
    ]
    stats_reset = stats_flat.reset_index()
    stats_reset["name_ascii"] = stats_reset["player"].apply(to_ascii)

    # WC players
    wc_players = pd.read_csv(ROSTER_PATH)[["player", "team", "position"]].drop_duplicates()
    wc_players["name_ascii"] = wc_players["player"].apply(to_ascii)
    print(f"  WC roster: {len(wc_players)} players")

    # Pass 1: exact ASCII match
    fbref_ascii_set = set(stats_reset["name_ascii"])
    exact_matched = wc_players[wc_players["name_ascii"].isin(fbref_ascii_set)].copy()
    print(f"\n  Pass 1 (exact ASCII):    {len(exact_matched):4d} matched")

    # Pass 2: nation-constrained fuzzy match
    fbref_by_nation = stats_reset[["player", "name_ascii", "nation_"]].drop_duplicates()
    unmatched = wc_players[~wc_players["name_ascii"].isin(fbref_ascii_set)].copy()

    fuzzy_rows = []
    for _, row in unmatched.iterrows():
        nation_code = TEAM_TO_NATION.get(row["team"])
        if not nation_code:
            continue
        candidates = fbref_by_nation[fbref_by_nation["nation_"] == nation_code]
        if candidates.empty:
            continue
        result = process.extractOne(
            row["name_ascii"], candidates["name_ascii"].tolist(),
            scorer=fuzz.token_sort_ratio, score_cutoff=FUZZY_THRESHOLD,
        )
        if result and (row["player"], row["team"]) not in KNOWN_FALSE_POSITIVES:
            matched_ascii, score, _ = result
            fuzzy_rows.append({
                "player": row["player"], "team": row["team"], "position": row["position"],
                "name_ascii": matched_ascii,
            })

    fuzzy_matched = pd.DataFrame(fuzzy_rows) if fuzzy_rows else pd.DataFrame(columns=exact_matched.columns)
    print(f"  Pass 2 (nation fuzzy):   {len(fuzzy_matched):4d} matched")

    matched_players = pd.concat([exact_matched, fuzzy_matched], ignore_index=True)

    # Pass 3: manual overrides
    still_unmatched = wc_players[~wc_players["player"].isin(matched_players["player"])].copy()
    override_rows = []
    for _, row in still_unmatched.iterrows():
        fbref_name = MANUAL_OVERRIDES.get(row["player"])
        if fbref_name:
            override_rows.append({
                "player": row["player"], "team": row["team"], "position": row["position"],
                "name_ascii": to_ascii(fbref_name),
            })
    override_matched = pd.DataFrame(override_rows) if override_rows else pd.DataFrame(columns=matched_players.columns)
    print(f"  Pass 3 (manual override):{len(override_matched):4d} matched")

    matched_players = pd.concat([matched_players, override_matched], ignore_index=True)
    print(f"\n  Total matched: {len(matched_players)} / {len(wc_players)} "
          f"({len(matched_players)/len(wc_players)*100:.1f}%)")

    # Validation: WC roster players still unmatched
    final_unmatched = wc_players[~wc_players["player"].isin(matched_players["player"])][
        ["player", "team", "position"]
    ].sort_values(["team", "position"]).reset_index(drop=True)
    print(f"\n  Roster players NOT matched to club data: {len(final_unmatched)}")
    if not final_unmatched.empty:
        by_team = final_unmatched.groupby("team").size().sort_values(ascending=False)
        print("    Top teams by unmatched count:")
        for team, n in by_team.head(10).items():
            print(f"      {team:30s} {n} players")

    return matched_players, stats_reset, wc_players


def step_3_merge_and_filter(matched_players, stats_reset):
    """Merge matched players with their stats; filter to MIN_MINS minutes."""
    section(f"Step 3: Merge club stats + filter to >= {MIN_MINS} minutes")

    merged = matched_players.merge(
        stats_reset[[
            "name_ascii", "league", "pos_",
            "Playing Time_Min",
            "Per 90 Minutes_G-PK", "Per 90 Minutes_Ast", "Per 90 Minutes_G+A"
        ]],
        on="name_ascii", how="left"
    ).rename(columns={
        "pos_":               "fbref_pos",
        "Playing Time_Min":   "minutes",
        "Per 90 Minutes_G-PK": "gls_p90",
        "Per 90 Minutes_Ast":  "ast_p90",
        "Per 90 Minutes_G+A":  "ga_p90",
    })

    merged_filtered = merged[merged["minutes"] >= MIN_MINS].copy()
    print(f"  Players after {MIN_MINS}-min filter: {len(merged_filtered)}")
    print(f"  Players dropped (below threshold or no stats): "
          f"{len(merged) - len(merged_filtered)}")

    return merged_filtered


def step_4_international_blend(merged_filtered):
    """Blend club p90 with international p90 via shrinkage formula."""
    section("Step 4: Blend club + international per-90 rates")

    if not os.path.exists(INTL_CSV) or not os.path.exists(COMP_STR_CSV):
        print(f"  ⚠ Missing {INTL_CSV} or {COMP_STR_CSV}.")
        print(f"  Run build_default_xmins.py first to generate them.")
        sys.exit(1)

    intl_raw = pd.read_csv(INTL_CSV)
    comp_strength = pd.read_csv(COMP_STR_CSV)
    strength = dict(zip(comp_strength["competition"], comp_strength["strength_mult"]))

    # Effective weight per row = recency comp_weight × ELO-derived strength_mult
    intl_raw["comp_w"]      = intl_raw["competition"].map(COMP_WEIGHTS).fillna(0.5)
    intl_raw["strength"]    = intl_raw["competition"].map(strength).fillna(0.7)
    intl_raw["effective_w"] = intl_raw["comp_w"] * intl_raw["strength"]

    intl_raw["min_num"] = pd.to_numeric(intl_raw["Playing Time_Min"].astype(str).str.replace(",", ""),
                                         errors="coerce").fillna(0)
    intl_raw["gpk"] = pd.to_numeric(intl_raw["Performance_G-PK"], errors="coerce").fillna(0)
    intl_raw["ast"] = pd.to_numeric(intl_raw["Performance_Ast"], errors="coerce").fillna(0)
    intl_raw["name_ascii"] = intl_raw["Player"].apply(to_ascii)

    intl_raw["w_min"] = intl_raw["min_num"] * intl_raw["effective_w"]
    intl_raw["w_gpk"] = intl_raw["gpk"]     * intl_raw["effective_w"]
    intl_raw["w_ast"] = intl_raw["ast"]     * intl_raw["effective_w"]

    agg = (intl_raw.groupby(["name_ascii", "Squad"], as_index=False)
                    .agg(intl_min_w=("w_min", "sum"),
                         intl_gpk_w=("w_gpk", "sum"),
                         intl_ast_w=("w_ast", "sum"),
                         intl_min_raw=("min_num", "sum"),
                         intl_comps=("competition", "nunique")))

    agg["intl_gls_p90"] = (agg["intl_gpk_w"] / agg["intl_min_w"] * 90).where(agg["intl_min_w"] > 0)
    agg["intl_ast_p90"] = (agg["intl_ast_w"] / agg["intl_min_w"] * 90).where(agg["intl_min_w"] > 0)

    # Map FBref Squad → WC team for the join
    FBREF_TO_WC = {v: k for k, v in WC_TO_FBREF_SQUAD.items()}
    agg["team"] = agg["Squad"].map(FBREF_TO_WC).fillna(agg["Squad"])

    blended = merged_filtered.merge(
        agg[["name_ascii", "team", "intl_min_w", "intl_min_raw", "intl_comps",
             "intl_gls_p90", "intl_ast_p90"]],
        on=["name_ascii", "team"], how="left"
    )
    for c in ["intl_min_w", "intl_min_raw", "intl_comps", "intl_gls_p90", "intl_ast_p90"]:
        blended[c] = blended[c].fillna(0)

    # Step 1: shrink intl rate toward club rate (Bayesian-flavored)
    shrink_denom = blended["intl_min_w"] + K_INTL_SHRINK
    blended["shrunk_intl_gls_p90"] = (
        blended["intl_min_w"] * blended["intl_gls_p90"] +
        K_INTL_SHRINK * blended["gls_p90"]
    ) / shrink_denom
    blended["shrunk_intl_ast_p90"] = (
        blended["intl_min_w"] * blended["intl_ast_p90"] +
        K_INTL_SHRINK * blended["ast_p90"]
    ) / shrink_denom

    # Step 2: blend club with shrunk intl via τ (intl-minute trust multiplier)
    denom = blended["minutes"] + TAU * blended["intl_min_w"]
    blended["blended_gls_p90"] = (
        blended["minutes"] * blended["gls_p90"] +
        TAU * blended["intl_min_w"] * blended["shrunk_intl_gls_p90"]
    ) / denom
    blended["blended_ast_p90"] = (
        blended["minutes"] * blended["ast_p90"] +
        TAU * blended["intl_min_w"] * blended["shrunk_intl_ast_p90"]
    ) / denom

    matched_intl = (blended["intl_min_w"] > 0).sum()
    print(f"  Players with non-zero intl contribution: "
          f"{matched_intl} / {len(blended)} ({matched_intl/len(blended)*100:.1f}%)")

    # Validation: top movers in blend (G+A delta)
    blended["delta_g+a_p90"] = (
        (blended["blended_gls_p90"] + blended["blended_ast_p90"]) -
        (blended["gls_p90"] + blended["ast_p90"])
    )
    movers = blended[blended["intl_min_w"] > 0].copy()
    cols = ["player", "team", "position", "minutes", "intl_min_w",
            "gls_p90", "ast_p90", "blended_gls_p90", "blended_ast_p90", "delta_g+a_p90"]

    print(f"\n  Top 10 increased (intl rate > club rate):")
    print(movers.sort_values("delta_g+a_p90", ascending=False)[cols]
                .head(10).round(2).to_string(index=False))
    print(f"\n  Top 10 decreased (intl rate < club rate):")
    print(movers.sort_values("delta_g+a_p90", ascending=True)[cols]
                .head(10).round(2).to_string(index=False))

    return blended


def step_5_export(blended):
    """Build weight_table.csv with league_strength multiplier."""
    section("Step 5: Export weight_table.csv")

    matched_weights = blended[[
        "player", "team", "position", "league",
        "blended_gls_p90", "blended_ast_p90",
        "minutes", "intl_min_w", "intl_comps",
    ]].rename(columns={
        "blended_gls_p90": "gls_p90",
        "blended_ast_p90": "ast_p90",
    }).copy()

    matched_weights["weight_source"] = matched_weights["intl_min_w"].apply(
        lambda x: "fbref+intl" if x > 0 else "fbref"
    )

    weight_table = (
        matched_weights
        .sort_values("minutes", ascending=False)
        .drop_duplicates(subset=["player", "team"], keep="first")
        .drop(columns=["minutes", "intl_min_w", "intl_comps"])
        .reset_index(drop=True)
    )

    with open(LEAGUE_STR_PATH) as f:
        league_strength = json.load(f)
    max_strength = max(league_strength.values())
    weight_table["league_strength"] = (
        weight_table["league"].map(league_strength) / max_strength
    ).fillna(1.0)

    weight_table.to_csv(OUT_PATH, index=False)
    print(f"  {len(weight_table)} unique players in weight_table")
    print(f"\n  Weight source breakdown:")
    for src, n in weight_table["weight_source"].value_counts().items():
        print(f"    {src:15s} {n:4d} ({n/len(weight_table)*100:.1f}%)")
    print(f"\n  Wrote → {OUT_PATH}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(ROSTER_PATH):
        print(f"ERROR: {ROSTER_PATH} not found. Run fetch_data.py first.", file=sys.stderr)
        sys.exit(1)

    stats = step_1_scrape_clubs()
    matched_players, stats_reset, _ = step_2_match_roster(stats)
    merged_filtered = step_3_merge_and_filter(matched_players, stats_reset)
    blended         = step_4_international_blend(merged_filtered)
    step_5_export(blended)

    print(f"\n{'═' * 78}\n  DONE.\n{'═' * 78}")


if __name__ == "__main__":
    main()

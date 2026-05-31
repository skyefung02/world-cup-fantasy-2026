#!/usr/bin/env python
"""
audit_unmatched_players.py

Lists WC roster players who couldn't be matched to FBref data — either club data
(used in weight_table) or international data (used in default_xmins). For each
unmatched player, suggests the top fuzzy candidate so you know what to add to
data/manual_overrides.py.

Run after a build_*.py run flagged coverage gaps. Add overrides for confident
matches, re-run the builds, audit again.

Outputs:
    - Console report (grouped by team, only teams with unmatched players)
    - data/unmatched_players_audit.csv  (so you can sort/filter in a spreadsheet)

Usage:
    python audit_unmatched_players.py
"""

import os
import sys
import unicodedata

import pandas as pd
import soccerdata as sd
from rapidfuzz import process, fuzz

from data.manual_overrides import MANUAL_OVERRIDES

# ──────────────────────────────────────────────────────────────────────────────
# Paths and constants
# ──────────────────────────────────────────────────────────────────────────────

PROCESSED_DIR  = "data/processed"
ROSTER_PATH    = f"{PROCESSED_DIR}/player_fixtures.csv"
INTL_CSV       = f"{PROCESSED_DIR}/international_stats_2026.csv"
OUT_PATH       = "data/unmatched_players_audit.csv"

ALL_LEAGUES = [
    "ENG-Premier League", "ESP-La Liga", "FRA-Ligue 1", "GER-Bundesliga", "ITA-Serie A",
    "NED-Eredivisie", "POR-Primeira Liga", "SAU-Saudi Pro League", "TUR-Super Lig",
    "BEL-First Division A", "SCO-Premiership", "GRE-Super League", "CZE-First League",
    "AUT-Bundesliga", "SUI-Super League", "BRA-Serie A", "ARG-Primera Division",
    "USA-MLS", "MEX-Liga MX", "KOR-K League 1", "JPN-J1 League", "ENG2-EFL Championship",
]

WC_TO_FBREF_SQUAD = {
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Cabo Verde": "Cape Verde",
    "USA": "United States",
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

SUGGESTION_THRESHOLD = 60   # broad — we want suggestions even for weak matches


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def to_ascii(name):
    """Same normalization used by both build scripts."""
    s = str(name)
    s = s.replace("ı", "i").replace("İ", "I")
    for ch in ("'", "'", "`", "ʼ"):
        s = s.replace(ch, "")
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()


def best_fuzzy_suggestion(name_ascii, pool, ascii_to_real_name):
    """Return (real_name, score) for the best fuzzy match in pool, or (None, None)."""
    if not pool:
        return None, None
    result = process.extractOne(name_ascii, pool,
                                 scorer=fuzz.WRatio, score_cutoff=SUGGESTION_THRESHOLD)
    if not result:
        return None, None
    hit_ascii, score, _ = result
    return ascii_to_real_name.get(hit_ascii), score


# ──────────────────────────────────────────────────────────────────────────────
# Loading FBref data
# ──────────────────────────────────────────────────────────────────────────────

def load_club_data():
    """Load cached FBref club stats; return (ascii→player, ascii→nation)."""
    print("Loading FBref club data (cached)...")
    frames = []
    for league in ALL_LEAGUES:
        try:
            fbref = sd.FBref(leagues=[league], seasons=2025)
            frames.append(fbref.read_player_season_stats(stat_type="standard"))
        except Exception:
            pass
    stats = pd.concat(frames)
    stats.columns = [f"{b}" if not a or a == b else f"{a}_{b}" for a, b in stats.columns]
    stats = stats.reset_index()
    stats["name_ascii"] = stats["player"].apply(to_ascii)

    by_nation = {}
    ascii_to_player = {}
    for _, r in stats.drop_duplicates(["player", "nation_"]).iterrows():
        by_nation.setdefault(r["nation_"], []).append(r["name_ascii"])
        ascii_to_player[r["name_ascii"]] = r["player"]
    return by_nation, ascii_to_player


def load_intl_data():
    """Load cached intl stats; return (squad→ascii_list, ascii→player)."""
    if not os.path.exists(INTL_CSV):
        print(f"⚠ {INTL_CSV} not found. Run build_default_xmins.py first.")
        return None, None
    print("Loading international stats CSV...")
    intl = pd.read_csv(INTL_CSV)
    intl["name_ascii"] = intl["Player"].apply(to_ascii)
    by_squad = {}
    ascii_to_player = {}
    for _, r in intl.drop_duplicates(["Player", "Squad"]).iterrows():
        by_squad.setdefault(r["Squad"], []).append(r["name_ascii"])
        ascii_to_player[r["name_ascii"]] = r["Player"]
    return by_squad, ascii_to_player


# ──────────────────────────────────────────────────────────────────────────────
# Audit
# ──────────────────────────────────────────────────────────────────────────────

def audit():
    if not os.path.exists(ROSTER_PATH):
        print(f"ERROR: {ROSTER_PATH} not found. Run fetch_data.py first.", file=sys.stderr)
        sys.exit(1)

    wc_roster = pd.read_csv(ROSTER_PATH)[["player", "team", "position"]].drop_duplicates()
    wc_roster["name_ascii_raw"]      = wc_roster["player"].apply(to_ascii)
    wc_roster["name_ascii_override"] = wc_roster["player"].apply(
        lambda p: to_ascii(MANUAL_OVERRIDES.get(p, p))
    )

    # Load FBref sources
    club_by_nation, club_ascii_to_player = load_club_data()
    intl_by_squad,  intl_ascii_to_player = load_intl_data()

    all_club_ascii_set = set(club_ascii_to_player.keys())
    all_intl_ascii_set = set(intl_ascii_to_player.keys()) if intl_ascii_to_player else set()

    print("\nAuditing each roster player...\n")
    rows = []
    for _, r in wc_roster.iterrows():
        ascii_used = r["name_ascii_override"]
        team       = r["team"]

        # Club match (any FBref player with the same ascii)
        club_matched = ascii_used in all_club_ascii_set

        # Intl match (any FBref intl player with same ascii in the right squad)
        fbref_squad   = WC_TO_FBREF_SQUAD.get(team, team)
        intl_pool     = intl_by_squad.get(fbref_squad, []) if intl_by_squad else []
        intl_matched  = ascii_used in set(intl_pool)

        if club_matched and intl_matched:
            continue   # nothing to report

        # Club suggestion (nation-scoped pool)
        club_pool = club_by_nation.get(TEAM_TO_NATION.get(team, ""), [])
        club_sug_name, club_sug_score = best_fuzzy_suggestion(
            ascii_used, club_pool, club_ascii_to_player
        )

        # Intl suggestion (squad-scoped pool)
        intl_sug_name, intl_sug_score = best_fuzzy_suggestion(
            ascii_used, intl_pool, intl_ascii_to_player
        )

        rows.append({
            "player":            r["player"],
            "team":              team,
            "position":          r["position"],
            "matched_club":      club_matched,
            "matched_intl":      intl_matched,
            "club_suggestion":   club_sug_name,
            "club_score":        round(club_sug_score, 1) if club_sug_score else None,
            "intl_suggestion":   intl_sug_name,
            "intl_score":        round(intl_sug_score, 1) if intl_sug_score else None,
        })

    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(OUT_PATH, index=False)

    # Report
    print(f"{'═' * 96}")
    print(f"  Audit Summary")
    print(f"{'═' * 96}")
    print(f"  Total WC roster players: {len(wc_roster)}")
    print(f"  Players unmatched in CLUB data: "
          f"{(~audit_df['matched_club']).sum() if not audit_df.empty else 0}")
    print(f"  Players unmatched in INTL data: "
          f"{(~audit_df['matched_intl']).sum() if not audit_df.empty else 0}")
    # Actionable = missing in BOTH. FBref uses canonical names across competitions, so a player
    # matched in either source has a working name. Missing-only-one is a data gap, not a naming
    # issue, and can't be fixed via manual_overrides.py.
    actionable_df = (audit_df[~audit_df["matched_club"] & ~audit_df["matched_intl"]]
                     if not audit_df.empty else audit_df)
    print(f"  Actionable (missing in BOTH — likely name override needed): {len(actionable_df)}")
    print(f"  Rows in audit CSV: {len(audit_df)}")
    print(f"  Wrote → {OUT_PATH}")

    if audit_df.empty:
        print("\n  All players matched. Nothing to do.")
        return

    if actionable_df.empty:
        print("\n  No actionable cases — every roster player matches FBref in at least one source.")
        print("  (Players missing only in one source are still in the CSV for reference.)")
        return

    # Per-team breakdown — only teams with actionable cases (missing in BOTH sources)
    print(f"\n{'═' * 96}")
    print(f"  Actionable cases by team (missing in BOTH club and intl)")
    print(f"{'═' * 96}")

    for team in sorted(actionable_df["team"].unique()):
        team_rows = actionable_df[actionable_df["team"] == team]
        print(f"\n  ── {team} ({len(team_rows)} actionable) ──")
        for _, r in team_rows.iterrows():
            print(f"    {r['player']:30s} [{r['position']:3s}]  missing: CLUB, INTL")
            if r["club_suggestion"]:
                strong = " ★" if r["club_score"] and r["club_score"] >= 75 else ""
                print(f"        club  → '{r['club_suggestion']}' (score {r['club_score']:.0f}){strong}")
            if r["intl_suggestion"] and r["intl_suggestion"] != r["club_suggestion"]:
                strong = " ★" if r["intl_score"] and r["intl_score"] >= 75 else ""
                print(f"        intl  → '{r['intl_suggestion']}' (score {r['intl_score']:.0f}){strong}")

    print(f"\n{'═' * 96}")
    print(f"  Legend: ★ = high-confidence suggestion (score ≥ 75)")
    print(f"  Workflow: pick confident matches → add to data/manual_overrides.py →")
    print(f"            re-run build_default_xmins.py + build_weight_table.py → audit again")
    print(f"{'═' * 96}")


if __name__ == "__main__":
    audit()

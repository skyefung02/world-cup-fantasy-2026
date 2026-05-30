"""Parse cache/set_piece_takers.txt and emit data/set_piece_takers.csv with FIFA IDs.

Re-run after editing the .txt source or when cache/players.json is refreshed.
Names that can't be resolved to a FIFA Fantasy player are skipped and logged.
"""
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

TXT_PATH  = "cache/set_piece_takers.txt"
CSV_PATH  = "data/set_piece_takers.csv"
PLAYERS_JSON = "cache/players.json"
SQUADS_JSON  = "cache/squads.json"
FUZZY_THRESHOLD = 82

# Map names as written in the .txt to FIFA squad names
TEAM_NAME_MAP = {
    "South Korea":             "Korea Republic",
    "Czechia (Czech Republic)": "Czechia",
    "Bosnia-Herzegovina":      "Bosnia and Herzegovina",
    "United States":           "USA",
    "Turkiye (Turkey)":        "Türkiye",
    "Curacao":                 "Curaçao",
    "Ivory Coast":             "Côte d'Ivoire",
    "Iran":                    "IR Iran",
    "Cape Verde":              "Cabo Verde",
    "DR Congo":                "Congo DR",
}

# Manual player overrides for tricky cases (txt-name → FIFA knownName/full name)
MANUAL_PLAYER_OVERRIDES = {
    # populate after first run reveals unmatched names
}


# NFKD doesn't decompose these — handle explicitly before stripping non-ASCII
_NON_DECOMPOSING = {
    "Ø": "O", "ø": "o", "Ł": "L", "ł": "l",
    "Ð": "D", "ð": "d", "Þ": "Th", "þ": "th",
    "Æ": "AE", "æ": "ae", "ß": "ss",
}


def to_ascii(s):
    for k, v in _NON_DECOMPOSING.items():
        s = s.replace(k, v)
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()


def parse_txt(path):
    """Yield (team, role, name) tuples from the source text file."""
    lines = [ln.rstrip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    # Skip "Group X" / "Team" / "Penalties" / "Corners and Free Kicks" header lines
    header_re = re.compile(r"^(Group\s+\w+|Team|Penalties|Corners and Free Kicks)\s*$", re.I)
    i = 0
    while i < len(lines):
        line = lines[i]
        if header_re.match(line):
            i += 1
            continue
        team = line.strip()
        if i + 2 >= len(lines):
            break
        pen_line = lines[i + 1].strip()
        sp_line  = lines[i + 2].strip()
        for name in [n.strip() for n in pen_line.split(",") if n.strip()]:
            yield (team, "penalty", name)
        for name in [n.strip() for n in sp_line.split(",") if n.strip()]:
            yield (team, "set_piece_assist", name)
        i += 3


def build_player_index(players_raw, squads_raw):
    """Per FIFA squad, a list of {id, display_name, ascii_name} for fuzzy matching."""
    squad_id_to_name = {s["id"]: s["name"] for s in squads_raw}
    by_squad_name = {}
    for p in players_raw:
        squad_name = squad_id_to_name.get(p["squadId"])
        if not squad_name:
            continue
        display = p["knownName"] or f"{p['firstName']} {p['lastName']}"
        by_squad_name.setdefault(squad_name, []).append({
            "id": p["id"],
            "display": display,
            "ascii": to_ascii(display),
        })
    return by_squad_name


def resolve_player(name, squad_name, player_index):
    """Return FIFA player id or None."""
    if name in MANUAL_PLAYER_OVERRIDES:
        name = MANUAL_PLAYER_OVERRIDES[name]
    candidates = player_index.get(squad_name, [])
    if not candidates:
        return None
    target = to_ascii(name)
    # Exact ascii match
    for c in candidates:
        if c["ascii"] == target:
            return c["id"]
    # Fuzzy match (token_sort handles re-orderings & hyphen differences)
    pool = {c["ascii"]: c["id"] for c in candidates}
    result = process.extractOne(target, list(pool.keys()), scorer=fuzz.token_sort_ratio, score_cutoff=FUZZY_THRESHOLD)
    if result:
        matched_ascii, _score, _ = result
        return pool[matched_ascii]
    return None


def main():
    players_raw = json.load(open(PLAYERS_JSON))
    squads_raw  = json.load(open(SQUADS_JSON))
    player_index = build_player_index(players_raw, squads_raw)
    squad_names = {s["name"] for s in squads_raw}

    matched_rows = []
    unmatched = []
    bad_teams = []

    for txt_team, role, name in parse_txt(TXT_PATH):
        squad_name = TEAM_NAME_MAP.get(txt_team, txt_team)
        if squad_name not in squad_names:
            bad_teams.append((txt_team, squad_name))
            continue
        pid = resolve_player(name, squad_name, player_index)
        if pid is None:
            unmatched.append((squad_name, role, name))
        else:
            matched_rows.append({"id": pid, "team": squad_name, "role": role})

    df = pd.DataFrame(matched_rows).drop_duplicates(["id", "role"])
    df.to_csv(CSV_PATH, index=False)
    print(f"Wrote {len(df)} matched rows to {CSV_PATH}")

    if bad_teams:
        print(f"\nUnknown team names ({len(bad_teams)}):")
        for txt_team, mapped in bad_teams:
            print(f"  {txt_team!r} → {mapped!r} (not in FIFA squads)")

    if unmatched:
        print(f"\nUnmatched players ({len(unmatched)}):")
        for team, role, name in unmatched:
            print(f"  {team:25s} {role:18s} {name}")


if __name__ == "__main__":
    main()

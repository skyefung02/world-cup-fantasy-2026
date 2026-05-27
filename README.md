# World Cup Fantasy 2026 — Projection Model

A lightweight points projection model for the [FIFA World Cup 2026 Fantasy](https://play.fifa.com/fantasy) game, built in Python.

## Overview

The model generates expected Fantasy points (xPts) for every player across the group stage, based primarily on **team-level Elo ratings** and **fixture difficulty**. It is intentionally simple — the goal is a principled, systematic framework rather than a highly accurate per-player model.

Two players from the same team in the same position with the same xMins will project identically in the first pass. The value comes from correctly identifying which teams (and therefore which players) have favourable fixtures, and letting an FPL-style solver handle squad selection.

## Data Sources

| Data | Source | Method |
|------|--------|--------|
| Players, prices, positions | [FIFA Fantasy API](https://play.fifa.com/json/fantasy/players.json) | Open JSON endpoint |
| Squads & groups | [FIFA Fantasy API](https://play.fifa.com/json/fantasy/squads.json) | Open JSON endpoint |
| Fixtures | [FIFA Fantasy API](https://play.fifa.com/json/fantasy/rounds.json) | Open JSON endpoint |
| Team Elo ratings | [Silver Bulletin PELE](https://www.natesilver.net/p/pele-international-football-rankings-soccer-ratings-projections) | CSV download |
| Tilt ratings | [Silver Bulletin PELE](https://www.natesilver.net/p/pele-international-football-rankings-soccer-ratings-projections) | CSV download |

## Methodology

### Step 1 — Elo → Match Outcomes
Win expectancy is computed using the standard Elo formula (neutral ground, no home advantage):

```
W_ij = 1 / (1 + 10^((Elo_j - Elo_i) / 400))
```

Win expectancy is then mapped to expected goals using an empirically fitted quartic polynomial based on ~40,000 international matches ([football-rankings.info](http://www.football-rankings.info/2020/12/simulation-of-scheduled-football-matches.html)), with two regimes (We < 0.9 and We ≥ 0.9). Clean sheet probability is derived from xG conceded via a Poisson distribution.

### Step 2 — Expected Fantasy Points per 90
Match outcomes are converted to expected Fantasy points using the official FIFA scoring system, broken down by position:
- Goals scored (per-player share of team xG)
- Assists
- Clean sheet points
- Goals conceded penalty (GK/DEF)
- Appearance points

### Step 3 — xMins
A local web UI (Flask) allows manual xMins assignment per player. xPts per game is computed as:

```
xpts_game = (xpts_p90 / 90) * xmins + appearance_pts
```

Total group stage xPts is the sum across all 3 fixtures.

## Project Structure

```
world-cup-fantasy-2026/
│
├── 01_data_fetch.ipynb        # Fetch & join all data sources
├── 02_projection_model.ipynb  # Elo model & projections
│
├── app.py                     # Flask xMins editor (localhost)
├── scoring.py                 # FIFA Fantasy scoring constants
│
├── templates/
│   ├── index.html             # Team list UI
│   └── team.html              # Per-team xMins editor
│
└── data/
    ├── elo_ratings.csv        # Silver Bulletin PELE ratings
    ├── tilt_ratings.csv       # Silver Bulletin Tilt ratings
    └── xmins.csv              # Manual xMins assignments
```

## Setup

```bash
# Create and activate conda environment
conda create -n world-cup-fantasy python=3.11 -y
conda activate world-cup-fantasy
conda install -c conda-forge requests pandas flask jupyter ipykernel -y

# Run xMins editor
python app.py
# Open http://127.0.0.1:5000

# Run notebooks in order
# 01_data_fetch.ipynb → 02_projection_model.ipynb
```

### Data setup
The FIFA Fantasy JSON endpoints are open but bot-protected. On first run, manually save the following files from your browser's Network tab into `cache/`:
- `cache/players.json`
- `cache/squads.json`
- `cache/rounds.json`

Silver Bulletin PELE/Tilt CSVs should be saved to `data/`.

## Limitations & Future Work

- **No within-position differentiation** — Bellingham and a holding midfielder project identically. A per-player club-level xG/xA adjustment is the natural next step.
- **Tilt ratings not yet integrated** — Silver Bulletin's attacking/defensive tilt ratings could adjust positional point distribution within a team.
- **Knockout rounds** — the model is designed to re-run once knockout fixtures are known, with updated xMins assumptions.
- **xMins defaults** — currently set manually. Could be partially automated using squad role signals.

## Acknowledgements

- [Silver Bulletin / Nate Silver](https://www.natesilver.net) — PELE Elo ratings
- [football-rankings.info](http://www.football-rankings.info) — Elo → xG polynomial
- [FIFA Fantasy](https://play.fifa.com/fantasy) — player/squad/fixture data

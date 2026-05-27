# Contributing to World Cup Fantasy 2026

Thanks for your interest in contributing! This project is a lightweight Fantasy points projection model for the FIFA World Cup 2026, built by and for analytically minded FPL players. Contributions of all kinds are welcome — from model improvements to bug fixes to documentation.

---

## How It Works

This is a community-driven project. The maintainer (@skyefung02) reviews and merges all contributions via Pull Requests. Nobody can merge code into `main` without review.

The general flow is:
1. **Find something to work on** — browse open [Issues](../../issues) or propose something new
2. **Fork the repo** and create a branch
3. **Make your changes**, test them locally
4. **Open a Pull Request** with a clear description of what you changed and why

---

## Getting Started

### 1. Fork and clone
```bash
git clone https://github.com/YOUR_USERNAME/world-cup-fantasy-2026.git
cd world-cup-fantasy-2026
```

### 2. Set up the environment
```bash
conda create -n world-cup-fantasy python=3.11 -y
conda activate world-cup-fantasy
conda install -c conda-forge requests pandas flask jupyter ipykernel -y
```

### 3. Set up data files
The FIFA Fantasy JSON endpoints are bot-protected and can't be fetched programmatically. You'll need to manually save three files from your browser's Network tab into `cache/`:
- `cache/players.json` — from `https://play.fifa.com/json/fantasy/players.json`
- `cache/squads.json` — from `https://play.fifa.com/json/fantasy/squads.json`
- `cache/rounds.json` — from `https://play.fifa.com/json/fantasy/rounds.json`

Silver Bulletin PELE/Tilt CSVs are already committed to `data/`.

### 4. Run the pipeline
```bash
python main.py
```

Or skip the data fetch step if you already have processed CSVs:
```bash
python main.py --skip-fetch
```

### 5. Run the xMins editor (optional)
```bash
python app.py
# Open http://127.0.0.1:5000
```

---

## What We're Looking For

Below are the known limitations and planned improvements — these are great starting points for contributors. Check the [Issues](../../issues) page to see if someone is already working on something before starting.

### High Priority
- **Tilt rating integration** — Silver Bulletin's attacking/defensive tilt ratings are already loaded but not yet used. The idea is to adjust positional point distribution within a team based on tilt (e.g. Germany +0.55 very attacking → more points flow to FWD/MID). Logic lives in `build_projections.py` → `xpts_per_90()`
- **xMins automation** — currently all unassigned players default to 60 mins. Could be improved using signals already in the data like `price` and `percentSelected` from `players.json` to seed smarter defaults

### Medium Priority
- **Per-player club-level adjustment** — add a small multiplier based on club xG+xA per 90 to differentiate within positions (e.g. Bellingham vs Rice, Yamal vs Zubimendi). Suggested data source: FBref via `soccerdata` or `worldfootballR`
- **Knockout round support** — the model is designed to re-run once knockout fixtures are confirmed. Needs updated xMins assumptions and fixture ingestion logic

### Lower Priority
- **Automated data fetching** — find a way around the FIFA endpoint bot protection so `cache/` files can be refreshed programmatically
- **Tests** — basic sanity checks (e.g. every team has 3 fixtures, xPts are non-negative, projections export has 1410 rows)
- **Documentation improvements** — better inline comments, docstrings, or a methodology writeup

---

## Project Structure

```
world-cup-fantasy-2026/
├── main.py                  # master pipeline script
├── fetch_data.py            # builds processed CSVs from cache + data/
├── build_projections.py     # Elo model → projections.csv
├── scoring.py               # FIFA Fantasy scoring constants
├── app.py                   # Flask xMins editor (localhost)
├── wc_solver.py             # FPL-style squad optimiser
├── templates/               # Flask UI templates
├── notebooks/               # Prototyping notebooks (not production)
├── cache/                   # Raw FIFA JSON (gitignored)
└── data/                    # CSVs: Elo ratings, xMins, projections
```

---

## Guidelines

- **Keep it simple** — the model is intentionally lightweight. Contributions should add value without dramatically increasing complexity or data requirements
- **Prototype in a notebook first** — add a new notebook to `notebooks/` to develop and test your idea, then port the logic into the relevant script
- **One thing per PR** — keep PRs focused. A PR that adds Tilt integration is great. A PR that adds Tilt integration + per-player adjustments + tests is harder to review
- **Describe your changes** — in your PR, explain what you changed, why, and include a before/after spot check (e.g. England top 10 projections)
- **Don't commit data files** — `cache/` and `data/processed/` are gitignored for a reason

---

## Questions?

Open an [Issue](../../issues) and tag it as a question. Happy to discuss ideas before you start writing code.

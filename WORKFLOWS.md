# Operational Workflows

Quick reference for the day-to-day commands. See `DEPLOY.md` for the deploy architecture.

---

## What runs automatically

On Railway (when `DEPLOY_MODE=public` is set):
- An in-process scheduler fires `refresh_ownership.refresh_all()` **once at boot, then hourly**.
- Each refresh: `curl players.json` → write `cache/players.json` → run `fetch_data` → invalidate the cached projection base state.
- Next page request rebuilds projections with the fresh `percentSelected`, `price`, `status`, and roster.
- **Fails gracefully** — on FIFA outage or 5xx, logs the error and keeps serving stale data until the next attempt.

No git push needed for hourly ownership refreshes — they happen entirely server-side.

---

## Daily refresh (tournament in progress)

Once scores start coming in, the day-to-day update is a single command:

```bash
python refresh.py          # new scores only
python refresh.py --elo    # new scores AND a fresh SilverBulletin Elo export
```

`refresh.py` runs the chain in the required order:

1. `python main.py --live` — pull fresh scores/ownership from FIFA, rebuild processed CSVs + group `projections.csv`.
2. *(only with `--elo`)* `python sync_elo.py --yes` — install the newest SilverBulletin export from `~/Downloads`, then rebuild. Must come **after** step 1, since it rebuilds from the cached (now fresh) `rounds.json` but does **not** pull scores itself.
3. `python build_knockout_projections.py` — re-run the knockout Monte Carlo, regenerating `knockout_team_probs.csv` (advancement grid), `knockout_projections.csv` (per-round player xPts / opponent-uncertain tables), and `knockout_fixtures.csv` (confirmed-fixture cards).

Then reload `http://localhost:5001/match-projections` — confirmed fixtures, advancement probabilities, and per-round tables all reflect the update.

Use `--elo` **only** when you've downloaded a fresh SilverBulletin CSV that day; otherwise `sync_elo.py` errors (no new export to install).

---

## When you need to push code/data changes

Three situations require a manual `git push` to redeploy:

1. **WC roster changed** (new player added, replacement for injury) — need to re-match FBref data
2. **Model logic changed** (scoring rules, projection math, UI)
3. **Set-piece / penalty takers updated** (`data/set_piece_takers.csv` changed)

For situation 1, run the full manual chain:

```bash
python main.py --live              # refresh players.json + rebuild player_fixtures.csv
python audit_unmatched_players.py  # see if any new player needs a manual override
python build_default_xmins.py      # regenerate default_xmins.csv (FBref intl data — cached)
python build_weight_table.py       # regenerate weight_table.csv (FBref club data — cached)
python main.py                     # rebuild projections from new CSVs (no --live, data is fresh)

git add data/default_xmins.csv data/weight_table.csv data/processed/
git commit -m "..."
git push
```

If `audit_unmatched_players.py` flags a player whose name doesn't match FBref (eg. "Rúben" vs "Rubén"), add the mapping to `data/manual_overrides.py`, then re-run the two `build_*.py` scripts.

**Expected cadence:** a handful of times before June 3 (squads finalising), then basically never until the tournament ends — modulo injury replacements.

---

## Local viewing / iteration

```bash
python app.py                       # local Flask server on :5001
python main.py --live               # one-shot: pull fresh ownership + rebuild projections.csv
python main.py                      # same, but uses cached players.json (faster, possibly stale)
python main.py --skip-fetch         # only rebuild projections from existing processed CSVs
```

Visit `http://localhost:5001/projections` to see the current model output.

Players with **<5% ownership** are highlighted with a violet row tint + dot — those have the +2pt scouting bonus expected value baked into their `Pts`.

---

## Manual refresh on the live site (Railway)

If you ever need to force a refresh outside the hourly cadence (eg. just deployed something, or testing):

```bash
TOKEN=<your-REFRESH_ADMIN_TOKEN>
curl -X POST https://<your-app>.railway.app/admin/refresh \
  -H "Authorization: Bearer $TOKEN"
```

Expected response: `{"bytes": ~1076000, "elapsed_s": ~0.2, "status": "ok"}`.

The same endpoint works locally on `http://localhost:5001/admin/refresh` using the token in your `.env`.

---

## Environment variables

| Variable | Where set | Purpose |
|---|---|---|
| `DEPLOY_MODE` | Railway dashboard (`public`); unset locally | Toggles scheduler + disables write-through CSV endpoints |
| `REFRESH_ADMIN_TOKEN` | Railway dashboard + local `.env` | Auth token for `POST /admin/refresh` |

`.env` is gitignored. Never commit secrets.

---

## Troubleshooting

| Symptom | Likely cause | Check |
|---|---|---|
| Projections look stale on Railway | Scheduler not firing | Railway logs for `refresh_all OK:` lines; check `DEPLOY_MODE=public` is set |
| `/admin/refresh` returns 503 | `REFRESH_ADMIN_TOKEN` not set in Railway | Variables tab in Railway dashboard |
| `/admin/refresh` returns 401 | Wrong token in the curl `Authorization` header | Re-check your password manager |
| `refresh_all` returns `status: error` with 403 | FIFA blocked the request UA | Already mitigated with browser UA in `refresh_ownership.py`; if it recurs, FIFA changed their gating |
| New player shows positional-mean projections | `weight_table.csv` doesn't have them yet | Run the manual chain above |

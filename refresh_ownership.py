"""
refresh_ownership.py

Pulls fresh players.json from FIFA's public endpoint, replaces cache/players.json,
re-runs fetch_data (which regenerates data/processed/player_fixtures.csv with
updated percentSelected / price / status), and invalidates the projection
base state so subsequent requests pick up the new numbers.

Called by the in-app APScheduler (hourly in DEPLOY_MODE=public) and by
POST /admin/refresh. Always returns a dict; never raises — fetch failures
are logged and the app keeps serving stale data until the next attempt.
"""

import logging
import os
import tempfile
import time

import requests

import build_projections
import fetch_data

PLAYERS_URL = "https://play.fifa.com/json/fantasy/players.json"
CACHE_PATH = "cache/players.json"
REQUEST_TIMEOUT = 30  # seconds

# FIFA's endpoint 403s the default python-requests UA. A browser UA is accepted.
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

logger = logging.getLogger(__name__)


def _fetch_and_save_players():
    """Download players.json and atomically replace cache/players.json."""
    response = requests.get(PLAYERS_URL, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    payload = response.content

    dir_ = os.path.dirname(CACHE_PATH) or "."
    os.makedirs(dir_, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".players.", suffix=".json.tmp", dir=dir_)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
        os.replace(tmp, CACHE_PATH)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return len(payload)


def refresh_all():
    """Full hourly refresh: fetch → rebuild processed CSVs → invalidate base state.

    Returns: {"status": "ok"|"error", "elapsed_s": float, ...}
    """
    started = time.time()
    try:
        n_bytes = _fetch_and_save_players()
        fetch_data.run()
        build_projections.invalidate_base_state()
        elapsed = round(time.time() - started, 2)
        logger.info("refresh_all OK: %d bytes, %.2fs", n_bytes, elapsed)
        return {"status": "ok", "bytes": n_bytes, "elapsed_s": elapsed}
    except Exception as e:
        elapsed = round(time.time() - started, 2)
        logger.exception("refresh_all FAILED after %.2fs", elapsed)
        return {"status": "error", "error": str(e), "elapsed_s": elapsed}

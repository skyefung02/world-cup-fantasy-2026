"""
fifa_team.py

Fetch the session owner's FIFA fantasy team from play.fifa.com.

The /api/en/fantasy/team endpoint returns ONLY the team belonging to the
session in the X-SID cookie — there is no public team-by-id lookup, and the
id/teamId/entryId query params are silently ignored. So this can only ever
fetch *your own* team, and only with a valid session cookie.

Used by the local-only /my-team debug page. The cookie lives in FIFA_SID
(.env, gitignored) — never hardcode it in this file, which IS committed.
"""

import requests

TEAM_URL = "https://play.fifa.com/api/en/fantasy/team"
REQUEST_TIMEOUT = 20  # seconds

# FIFA's edge 403s a default python-requests UA; a browser UA is accepted.
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
    "Accept": "application/json",
}


def fetch_team(sid):
    """GET the session owner's team. Returns the `success` dict.

    Raises RuntimeError with the API's message on a reported error (e.g. an
    expired cookie → "Invalid credentials"), or the underlying HTTPError for
    other transport failures.
    """
    resp = requests.get(
        TEAM_URL,
        headers=REQUEST_HEADERS,
        cookies={"X-SID": sid},
        timeout=REQUEST_TIMEOUT,
    )
    # Parse the body first so we can surface FIFA's own error message (the
    # 403 "Invalid credentials" payload is more useful than a bare HTTPError).
    try:
        payload = resp.json()
    except ValueError:
        resp.raise_for_status()
        raise

    errors = payload.get("errors") or []
    if errors:
        raise RuntimeError(errors[0].get("message", "FIFA API error"))
    resp.raise_for_status()
    return payload["success"]

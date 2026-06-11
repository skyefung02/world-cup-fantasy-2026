"""
realized.py — overlay realized results onto the (fixed) captaincy policy.

The backward-induction thresholds in captain.analyze_round() are a precomputed
policy: realized scores never change them, they only tell you which branch you
took. This module walks a round's blocks against the results so far and reports
the live state — what resolved, where the armband locked (if anywhere), and the
current decision point — without recomputing any math.

Inputs are the two cache files the hourly refresh already maintains:
  * players.json -> stats.roundPoints  (realized points per player per round)
  * rounds.json  -> fixture status     ("complete" = that kickoff is final)

Both are global/objective — realized points don't depend on whose team it is.
"""


def load_realized(players_raw, rounds_raw):
    """Return (points, final_kickoffs):
        points         {player_id: {round_int: realized_pts}}
        final_kickoffs {round_int: {kickoff_iso, ...}}  # fixtures that are complete

    Note: FIFA serializes an empty roundPoints map as a list ([]), so we
    type-check and skip those — a player with no results simply isn't in `points`,
    and the overlay defaults their realized score to 0 once their game is final.
    """
    points = {}
    for p in players_raw:
        rp = (p.get("stats") or {}).get("roundPoints")
        if isinstance(rp, dict):
            points[p["id"]] = {int(k): v for k, v in rp.items()}

    final_kickoffs = {
        r["id"]: {fx["date"] for fx in r["tournaments"] if fx.get("status") == "complete"}
        for r in rounds_raw
    }
    return points, final_kickoffs


def overlay_realized(rd, points, final_kickoffs):
    """Mutate + return an analyze_round() result with the live state overlaid.

    Adds per block: resolved (bool), realized (pts | None), decision ('keep'|'roll'|None).
    Adds per round: live_state ('rolling'|'locked'), locked_block, final_captain_pts,
                    current_block (first unresolved while still rolling), forward_ev.

    Thresholds are never touched — we only walk the realized outcomes against them.
    """
    rnd = rd["round"]
    final = final_kickoffs.get(rnd, set())
    rd.update(live_state="rolling", locked_block=None,
              final_captain_pts=None, current_block=None, forward_ev=None)

    # Mark resolution + the candidate's realized score (0 if final but no entry).
    for b in rd["blocks"]:
        b["resolved"] = b["kickoff"] in final
        b["realized"] = points.get(b["id"], {}).get(rnd, 0) if b["resolved"] else None
        b["decision"] = None

    # Walk the policy in kickoff order until we keep (lock) or hit the live block.
    for b in rd["blocks"]:
        if rd["live_state"] != "rolling":
            break
        if not b["resolved"]:
            rd["current_block"] = b["index"]   # decision window opens here
            rd["forward_ev"] = b["value"]      # U[k] = expected captain pts from here
            break
        keep = b["is_last"] or (b["threshold"] is not None and b["realized"] >= b["threshold"])
        b["decision"] = "keep" if keep else "roll"
        if keep:
            rd.update(live_state="locked", locked_block=b["index"],
                      final_captain_pts=b["realized"])

    return rd

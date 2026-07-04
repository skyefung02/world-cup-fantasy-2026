"""Squad knockout-risk: how many of your 15 players are likely to be eliminated
in the *upcoming* knockout round, and the chance that outnumbers your free
transfers.

The key subtlety is correlation: players on the same country are perfectly
correlated — France's four players all survive together or all leave together.
So this is NOT 15 independent coin flips. It's one Bernoulli per *country*, each
contributing all of that country's players at once. Convolving those per-country
spikes gives the exact distribution of "how many players get knocked out"
(a grouped Poisson-binomial), from which the mean and P(> free transfers) fall out.

Survival is taken conditional on the team actually being in the round:
    p_survive = p_advance / p_play          (== 1 - (1 - p_advance) when p_play==1)
    p_ko      = 1 - p_survive
For the current live round every squad team has p_play == 1, so p_ko == 1 - p_advance.
"""

from __future__ import annotations

import pandas as pd

ROUNDS_CSV = "data/knockout_team_rounds.csv"

# Free transfers granted in the window *after* a round finishes, to fix players
# knocked out that round. Keyed by the round whose knockouts they cover
# (4=R32, 5=R16, 6=QF, 7=SF). The window opens before the next round starts:
# after R32→4 (before R16), after R16→4 (before QF), after QF→5 (before SF),
# after SF→6 (before Final). The Final (8) has no subsequent transfer window.
FREE_TRANSFERS = {4: 4, 5: 4, 6: 5, 7: 6, 8: 0}

# Max players allowed from one country, by round (mirrors wc_settings
# country_limit_by_round). Drives the "ideal squad" floor on the planner: the
# lowest-risk 15 is the safest teams filled up to this cap. Rises as the field
# shrinks so 15 players stay feasible.
COUNTRY_LIMITS = {4: 3, 5: 4, 6: 5, 7: 6, 8: 8}


def current_ko_round(rounds_csv=ROUNDS_CSV):
    """The live knockout round: the lowest round that still has an unresolved
    team (p_advance not pinned to 0 or 1). A fully-resolved round is finished, so
    we advance to the next. Falls back to the last round once all are decided."""
    r = pd.read_csv(rounds_csv)
    for rnd in sorted(r["round"].unique()):
        adv = r[r["round"] == rnd]["p_advance"]
        if ((adv > 0) & (adv < 1)).any():
            return int(rnd)
    return int(r["round"].max())


def team_ko_probs(round_num, rounds_csv=ROUNDS_CSV):
    """Map abbr -> P(knocked out in `round_num` | in the round) for that round."""
    r = pd.read_csv(rounds_csv)
    r = r[r["round"] == round_num]
    out = {}
    for _, row in r.iterrows():
        p_play = float(row["p_play"])
        if p_play <= 0:
            out[row["abbr"]] = 1.0            # can't reach the round -> already gone
        else:
            p_survive = float(row["p_advance"]) / p_play
            out[row["abbr"]] = max(0.0, min(1.0, 1.0 - p_survive))
    return out


def squad_risk(player_ids, proj_df, round_num, free_transfers=None,
               rounds_csv=ROUNDS_CSV):
    """Knockout-risk summary for a 15-man squad going into `round_num`.

    Returns a dict:
        expected      float   expected # players knocked out
        dist          list    dist[k] = P(exactly k knocked out), len == len(squad)+1
        p_over_ft     float    P(knockouts > free_transfers)  (None if FT not given)
        by_country    list of {abbr, n, p_ko, expected} sorted by expected desc
        unmatched     list of player_ids with no team/prob (excluded from the math)
    """
    id2abbr = proj_df.set_index("id")["abbr"].to_dict()
    ko = team_ko_probs(round_num, rounds_csv)

    # Count players per country; track ids we can't resolve.
    counts, unmatched = {}, []
    for pid in player_ids:
        abbr = id2abbr.get(pid)
        if abbr is None or abbr not in ko:
            unmatched.append(pid)
            continue
        counts[abbr] = counts.get(abbr, 0) + 1

    # Grouped Poisson-binomial: convolve one spike per country.
    # Each country: survives (0 knocked out) w.p. 1-p_ko, else all n knocked out.
    dist = [1.0]
    for abbr, n in counts.items():
        p = ko[abbr]
        new = [0.0] * (len(dist) + n)
        for k, pk in enumerate(dist):
            new[k] += pk * (1.0 - p)      # country survives
            new[k + n] += pk * p          # country knocked out -> +n players
        dist = new

    expected = sum(k * pk for k, pk in enumerate(dist))
    by_country = sorted(
        ({"abbr": a, "n": n, "p_ko": ko[a], "expected": n * ko[a]}
         for a, n in counts.items()),
        key=lambda d: d["expected"], reverse=True)

    p_over_ft = None
    if free_transfers is not None:
        p_over_ft = sum(pk for k, pk in enumerate(dist) if k > free_transfers)

    return {
        "expected": expected,
        "dist": dist,
        "p_over_ft": p_over_ft,
        "by_country": by_country,
        "unmatched": unmatched,
    }


# ── CLI sanity check ─────────────────────────────────────────────────────────
# Usage:
#   python squad_risk.py                      # fetch your own team via FIFA_SID
#   python squad_risk.py FRA:4 ARG:2 ESP:3 …  # ad-hoc country:count breakdown
if __name__ == "__main__":
    import sys

    proj_df = pd.read_csv("data/projections.csv")

    round_num = current_ko_round()
    ft = FREE_TRANSFERS.get(round_num)

    args = [a for a in sys.argv[1:] if ":" in a]
    if args:
        # Ad-hoc mode: build a fake proj_df + id list from ABBR:count pairs.
        counts = {a.split(":")[0]: int(a.split(":")[1]) for a in args}
        rows, pids, nid = [], [], 1
        for abbr, n in counts.items():
            for _ in range(n):
                rows.append({"id": nid, "abbr": abbr})
                pids.append(nid)
                nid += 1
        proj_df = pd.DataFrame(rows)
        player_ids = pids
        print(f"Ad-hoc squad: {counts}")
    else:
        import os
        import fifa_team
        import team_import
        sid = os.environ.get("FIFA_SID")
        if not sid:
            sys.exit("FIFA_SID not set and no ABBR:count args given.")
        team = team_import.parse_team(fifa_team.fetch_team(sid))
        player_ids = team["player_ids"]

    res = squad_risk(player_ids, proj_df, round_num, free_transfers=ft)

    print(f"\nRound {round_num}  |  {len(player_ids)} players  |  free transfers: {ft}")
    print(f"Expected knocked out: {res['expected']:.2f}")
    if res["p_over_ft"] is not None:
        print(f"P(forced transfers > {ft}): {res['p_over_ft']:.1%}")
    print("\nBy country:")
    for c in res["by_country"]:
        print(f"  {c['abbr']}  ×{c['n']}  survive {1-c['p_ko']:.0%}  "
              f"(E[out]={c['expected']:.2f})")
    print("\nP(exactly k knocked out):")
    for k, pk in enumerate(res["dist"]):
        if pk >= 0.005:
            bar = "█" * round(pk * 40)
            print(f"  {k:2d}: {pk:5.1%} {bar}")
    if res["unmatched"]:
        print(f"\nUnmatched player ids (excluded): {res['unmatched']}")

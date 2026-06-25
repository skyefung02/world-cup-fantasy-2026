"""
refresh.py — one-command daily refresh.

Runs the update chain in the correct order so the site (group projections,
knockout advancement grid, confirmed fixtures, per-round tables) reflects the
latest scores and Elo:

    1. main.py --live                  pull fresh scores/ownership from FIFA,
                                       rebuild processed CSVs + group projections
    2. sync_elo.py --yes  (--elo only) install the latest SilverBulletin Elo
                                       export from ~/Downloads, rebuild again
    3. build_knockout_projections.py   re-run the knockout Monte Carlo
                                       (advancement, player xPts, confirmed fixtures)

Usage:
    python refresh.py            # scores only (no new Elo download today)
    python refresh.py --elo      # also install a fresh SilverBulletin Elo export

Step 2 is skipped without --elo because sync_elo.py requires a new export in
~/Downloads and errors if there isn't one.
"""

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    print(f"\n=== {' '.join(cmd[1:])} ===", flush=True)
    result = subprocess.run([sys.executable, *cmd], cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"\nFAILED: {' '.join(cmd)} (exit {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Daily WC fantasy refresh (scores → elo → knockouts).")
    parser.add_argument("--elo", action="store_true",
                        help="Also install the latest SilverBulletin Elo export from ~/Downloads.")
    args = parser.parse_args()

    run(["main.py", "--live"])                  # fresh scores + group projections
    if args.elo:
        run(["sync_elo.py", "--yes"])           # new Elo + rebuild (uses cached fresh scores)
    run(["build_knockout_projections.py"])      # knockout Monte Carlo

    print("\n=== Done. Reload /match-projections to see the update. ===")


if __name__ == "__main__":
    main()

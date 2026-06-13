"""
sync_elo.py — one-command ELO refresh from a Silver Bulletin download.

Workflow:
    1. Download the WC ratings CSV from Silver Bulletin (lands in ~/Downloads).
    2. Run `python sync_elo.py`.

This script finds the newest matching CSV in ~/Downloads, shows it to you for
confirmation, installs it as data/elo_wc_adjusted.csv (backing up the old one),
and rebuilds projections by running main.py.

It does NOT touch git — review the diff and commit/push yourself.

Flags:
    --yes            Skip the confirmation prompt (for unattended/scheduled runs).
    --dry-run        Show what would happen without copying or rebuilding.
    --downloads DIR  Override the download folder (default: ~/Downloads).
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime

import pandas as pd

REPO_ROOT  = os.path.dirname(os.path.abspath(__file__))
TARGET     = os.path.join(REPO_ROOT, "data", "elo_wc_adjusted.csv")
BACKUP     = TARGET + ".bak"

# Columns the projection model actually reads (fetch_data.build_squads_rated).
# We use these as the signature to tell a real SB export from any other CSV.
REQUIRED_COLS = {"Code", "WC PELE", "Home field"}


def pipeline_python():
    """The interpreter used to run main.py — prefer the world-cup-fantasy env."""
    candidates = [
        os.environ.get("WC_FANTASY_PYTHON"),
        "/opt/miniconda3/envs/world-cup-fantasy/bin/python",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return sys.executable  # fall back to whatever ran this script


def is_sb_export(path):
    """True if `path` is a CSV whose header carries the SB ratings columns."""
    try:
        header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
    except Exception:
        return False
    return REQUIRED_COLS.issubset(set(header.columns))


def find_latest_export(downloads_dir):
    """Newest CSV in `downloads_dir` that looks like an SB ratings export."""
    if not os.path.isdir(downloads_dir):
        return None
    csvs = [
        os.path.join(downloads_dir, f)
        for f in os.listdir(downloads_dir)
        if f.lower().endswith(".csv")
    ]
    matches = [p for p in csvs if is_sb_export(p)]
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def preview(path):
    """Print a short summary of the candidate file so the user can sanity-check."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
    print(f"\n  File:     {path}")
    print(f"  Modified: {mtime}")
    print(f"  Teams:    {len(df)}")
    n_perf = df["Perf. adj."].notna().sum() if "Perf. adj." in df.columns else 0
    print(f"  Perf. adj. populated: {n_perf}")
    cols = ["Code", "WC PELE"] + (["Home field"] if "Home field" in df.columns else [])
    top = df.sort_values("WC PELE", ascending=False).head(5)[cols]
    print("\n  Top 5 by WC PELE:")
    for line in top.to_string(index=False).splitlines():
        print(f"    {line}")
    print()


def diff_against_current(src, limit=20):
    """Print how WC PELE changes per team vs the currently-installed file."""
    if not os.path.exists(TARGET):
        print("  No existing elo_wc_adjusted.csv to compare — this will be the first install.\n")
        return

    old = pd.read_csv(TARGET, encoding="utf-8-sig")[["Code", "WC PELE"]].rename(columns={"WC PELE": "old"})
    new = pd.read_csv(src, encoding="utf-8-sig")[["Code", "WC PELE"]].rename(columns={"WC PELE": "new"})
    merged = new.merge(old, on="Code", how="outer", indicator=True)

    both = merged[merged["_merge"] == "both"].copy()
    both["delta"] = both["new"] - both["old"]
    movers = both[both["delta"].round(1) != 0].sort_values(
        "delta", key=lambda s: s.abs(), ascending=False
    )

    if movers.empty:
        print("  WC PELE changes vs current file: none.")
    else:
        print(f"  WC PELE changes vs current file ({len(movers)} teams moved):")
        for _, r in movers.head(limit).iterrows():
            print(f"    {r['Code']:<4} {int(round(r['old']))} -> {int(round(r['new']))} ({r['delta']:+.0f})")
        if len(movers) > limit:
            print(f"    ... and {len(movers) - limit} more")

    added   = merged.loc[merged["_merge"] == "left_only",  "Code"].tolist()
    removed = merged.loc[merged["_merge"] == "right_only", "Code"].tolist()
    if added:
        print(f"  New teams (not in current file): {', '.join(added)}")
    if removed:
        print(f"  Dropped teams (in current, not in new): {', '.join(removed)}")
    print()


def confirm(prompt):
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def run_pipeline(dry_run):
    py = pipeline_python()
    print(f"Rebuilding projections: {py} main.py")
    if dry_run:
        print("  [dry-run] skipped.")
        return 0
    result = subprocess.run([py, "main.py"], cwd=REPO_ROOT)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Sync Silver Bulletin ELO and rebuild projections.")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    parser.add_argument("--dry-run", action="store_true", help="Show actions without applying them.")
    parser.add_argument(
        "--downloads",
        default=os.path.expanduser("~/Downloads"),
        help="Folder to scan for the SB export (default: ~/Downloads).",
    )
    args = parser.parse_args()

    src = find_latest_export(args.downloads)
    if src is None:
        print(f"No Silver Bulletin export found in {args.downloads}.", file=sys.stderr)
        print(f"  (Looking for a .csv containing columns: {sorted(REQUIRED_COLS)})", file=sys.stderr)
        sys.exit(1)

    preview(src)
    diff_against_current(src)

    if not args.yes and not confirm(f"Install as {os.path.relpath(TARGET, REPO_ROOT)} and rebuild projections? [y/N] "):
        print("Aborted. Nothing changed.")
        sys.exit(0)

    if args.dry_run:
        print(f"[dry-run] would back up {os.path.relpath(TARGET, REPO_ROOT)} -> elo_wc_adjusted.csv.bak")
        print(f"[dry-run] would copy {src} -> {os.path.relpath(TARGET, REPO_ROOT)}")
    else:
        if os.path.exists(TARGET):
            shutil.copy2(TARGET, BACKUP)
            print(f"Backed up old ELO -> {os.path.relpath(BACKUP, REPO_ROOT)}")
        shutil.copy2(src, TARGET)
        print(f"Installed new ELO -> {os.path.relpath(TARGET, REPO_ROOT)}")

    print()
    rc = run_pipeline(args.dry_run)
    if rc != 0:
        print(f"\nPipeline failed (exit {rc}). New ELO is in place; old one is at "
              f"{os.path.relpath(BACKUP, REPO_ROOT)} if you need to revert.", file=sys.stderr)
        sys.exit(rc)

    print("\nDone. Review the diff, then commit/push when ready:")
    print("  git add data/elo_wc_adjusted.csv data/processed/ data/projections.csv")
    print('  git commit -m "Update ELO ratings"')
    print("  git push")


if __name__ == "__main__":
    main()

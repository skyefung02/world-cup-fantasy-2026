import argparse
import sys

import fetch_data
import build_projections
import refresh_ownership


def main():
    parser = argparse.ArgumentParser(description="World Cup Fantasy 2026 — Projection Pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--live",
        action="store_true",
        help="Pull fresh players.json from FIFA before processing (refreshes ownership/price/status)"
    )
    group.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip data fetch step (use existing processed CSVs)"
    )
    args = parser.parse_args()

    if args.live:
        print("=== Step 1: Live fetch from FIFA + process data ===")
        result = refresh_ownership.refresh_all()
        if result.get("status") != "ok":
            print(f"  FAILED: {result.get('error')}", file=sys.stderr)
            sys.exit(1)
        print(f"  OK: {result['bytes']} bytes, {result['elapsed_s']}s")
        print()
    elif not args.skip_fetch:
        print("=== Step 1: Fetch & process data (from cached players.json) ===")
        fetch_data.run()
        print()

    print("=== Step 2: Build projections ===")
    build_projections.run()
    print()

    print("=== Done. Projections saved to data/projections.csv ===")


if __name__ == "__main__":
    main()
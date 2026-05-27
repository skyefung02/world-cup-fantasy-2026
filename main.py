import argparse
import fetch_data
import build_projections


def main():
    parser = argparse.ArgumentParser(description="World Cup Fantasy 2026 — Projection Pipeline")
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip data fetch step (use existing processed CSVs)"
    )
    args = parser.parse_args()

    if not args.skip_fetch:
        print("=== Step 1: Fetch & process data ===")
        fetch_data.run()
        print()

    print("=== Step 2: Build projections ===")
    build_projections.run()
    print()

    print("=== Done. Projections saved to data/projections.csv ===")


if __name__ == "__main__":
    main()
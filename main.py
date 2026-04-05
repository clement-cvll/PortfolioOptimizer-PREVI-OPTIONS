"""Full pipeline: refresh Parquet data, then run analysis.

    uv run main.py            — update data + analyse
    uv run main.py --rebuild  — re-scrape + rebuild from scratch
"""

import argparse
import os
import sys


def _ensure_src_on_path() -> None:
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update Parquet data, then run Markowitz analysis.",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Re-scrape tickers and rebuild the Parquet dataset from scratch",
    )
    args = parser.parse_args()
    _ensure_src_on_path()

    import build_database
    build_database.main(rebuild=args.rebuild)

    import run_analysis
    run_analysis.main()


if __name__ == "__main__":
    main()

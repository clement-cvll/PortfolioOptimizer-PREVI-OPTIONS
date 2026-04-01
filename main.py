"""Full pipeline: refresh Parquet data, then run analysis.

From the repository root, after ``uv sync``:

- ``uv run main.py`` — update or build data, then optimise and plot
- ``uv run main.py --rebuild`` — re-scrape tickers and rebuild Parquet, then analyse

Same behaviour: ``uv run previ-options`` (console script).
"""

from __future__ import annotations

import argparse
import os
import sys


def _ensure_src_on_path() -> None:
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(repo_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update Parquet data, then run Markowitz analysis.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Re-scrape tickers and rebuild the Parquet dataset from scratch",
    )
    args = parser.parse_args()

    _ensure_src_on_path()

    import build_database

    old_argv = sys.argv[:]
    try:
        sys.argv = ["build_database.py"] + (["--rebuild"] if args.rebuild else [])
        build_database.main()
    finally:
        sys.argv = old_argv

    import run_analysis

    run_analysis.main()


if __name__ == "__main__":
    main()

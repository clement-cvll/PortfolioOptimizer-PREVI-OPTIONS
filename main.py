"""Project entrypoint: build/update data then run analysis.

Usage examples:
    uv run python main.py                 # update data (if present) + run analysis
    uv run python main.py --rebuild       # full rebuild + run analysis
    uv run python main.py --skip-ingest   # run analysis only (expects data exists)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Re-scrape tickers and rebuild the Parquet dataset from scratch",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip data ingestion and only run the analysis",
    )
    args = parser.parse_args()

    _ensure_src_on_path()

    if not args.skip_ingest:
        import build_database

        # Reuse the module's CLI behavior by calling its main().
        # It reads sys.argv via argparse, so we call it with a minimal argv tweak.
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

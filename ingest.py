#!/usr/bin/env python3
"""
ingest.py
CLI to run or re-run the ingestion pipeline.

Usage:
    python3 ingest.py           # skip if already indexed
    python3 ingest.py --force   # force full re-index
"""

import argparse
import sys
import logging

import structlog
from dotenv import load_dotenv

load_dotenv()

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    processors=[structlog.dev.ConsoleRenderer(colors=True)],
)

from src.ingestion.pipeline import run_ingestion_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Nexla MCP — PDF ingestion pipeline"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-index all documents even if Weaviate already has data",
    )
    args = parser.parse_args()

    print("\n── Nexla MCP · Ingestion Pipeline ──────────────────────")

    try:
        summary = run_ingestion_pipeline(force=args.force)

        print("\n✓ Done")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print()
        sys.exit(0)

    except FileNotFoundError as e:
        print(f"\n✗ PDF not found: {e}")
        print("  Place PDF files in data/raw/ and retry.\n")
        sys.exit(1)

    except EnvironmentError as e:
        print(f"\n✗ Missing environment variable: {e}")
        print("  Check your .env file against .env.example\n")
        sys.exit(1)

    except Exception as e:
        print(f"\n✗ Ingestion failed: {e}\n")
        raise


if __name__ == "__main__":
    main()
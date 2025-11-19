#!/usr/bin/env python3
"""
Main entry point for Chilean Growth Slowdown analysis.

This module serves as the primary entry point for the replication code.
It delegates to replicate.py which contains the full analysis pipeline.

Usage:
    uv run main.py                  # Run full analysis
    uv run main.py --data-only      # Only fetch and save data
    uv run main.py --scm-only       # Only run SCM (skip BSTS)
    uv run main.py --skip-robustness # Skip robustness tests
    uv run main.py --refresh-data   # Force refresh data from sources
"""

from replicate import main as run_analysis


def main():
    """Run the Chilean Growth Slowdown replication analysis."""
    run_analysis()


if __name__ == "__main__":
    main()

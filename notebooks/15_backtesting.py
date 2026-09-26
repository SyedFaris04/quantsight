"""Compatibility entry point for the audited backtesting package.

Run from any directory: python notebooks/15_backtesting.py --random-runs 1000
Results: backend/data/backtests/report.json and per-run daily/trade audit CSVs.
Legacy backtest_results.csv is preserved but not used by the new report.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from backtesting.report import main

if __name__ == "__main__":
    main()

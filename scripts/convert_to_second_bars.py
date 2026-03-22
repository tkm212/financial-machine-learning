#!/usr/bin/env python3
"""
Convert Bitcoin tick/trade data to 1-second OHLCV bars.

Reads a CSV with columns: time, Price, Quantity (and optional Id, IsBuyerMaker, etc.)
and outputs resampled second-level bars with open, high, low, close, volume.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_tick_data(path: Path, min_time: float = 1e9) -> pd.DataFrame:
    """Load and clean tick data CSV. Expects 'time', 'Price', 'Quantity' columns."""
    logger.info("Loading tick data from %s", path)
    df = pd.read_csv(path)
    required = {"time", "Price", "Quantity"}
    missing = required - set(df.columns)
    if missing:
        msg = f"Input file missing columns: {missing}. Found: {list(df.columns)}"
        logger.error(msg)
        raise ValueError(msg)

    # Drop rows with invalid/malformed timestamps (e.g. truncated)
    n_before = len(df)
    df = df.dropna(subset=["time", "Price", "Quantity"])
    df = df[(df["time"] >= min_time) & (df["Price"] > 0) & (df["Quantity"] > 0)]
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.info("Dropped %d invalid rows, %d ticks remaining", n_dropped, len(df))
    return df


def convert_to_second_bars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample tick data to 1-second OHLCV bars.

    Uses groupby instead of resample to avoid OOM: only allocates for seconds
    that have trades instead of the full time range.

    - open: first price in the second
    - high: max price in the second
    - low: min price in the second
    - close: last price in the second
    - volume: sum of quantity in the second
    - count: number of trades in the second
    """
    logger.info("Converting %d ticks to 1-second bars", len(df))
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.floor("1s")

    bars = (
        df
        .groupby("datetime", sort=True)
        .agg(
            open=("Price", "first"),
            high=("Price", "max"),
            low=("Price", "min"),
            close=("Price", "last"),
            volume=("Quantity", "sum"),
            count=("Price", "count"),
        )
        .reset_index()
    )

    bars["count"] = bars["count"].astype(int)
    logger.info("Produced %d second bars", len(bars))

    return bars


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    default_input = project_root / "inputs" / "btc_bid_ask_data.csv"
    default_output = project_root / "outputs" / "btc_bid_ask_1s.csv"

    parser = argparse.ArgumentParser(description="Convert Bitcoin tick data to 1-second OHLCV bars.")
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=default_input,
        help=f"Path to input CSV (default: {default_input})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"Output path (default: {default_output})",
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Do not write the datetime index to the output CSV",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        return 1

    output = args.output or (project_root / "outputs" / f"{args.input.stem}_1s.csv")

    logger.info("Input: %s -> Output: %s", args.input, output)
    logger.info("Project root: %s", project_root)

    try:
        df = load_tick_data(args.input)
    except ValueError:
        logger.exception("Invalid input data")
        return 1
    except Exception:
        logger.exception("Failed to load data")
        return 1

    try:
        bars = convert_to_second_bars(df)
    except Exception:
        logger.exception("Failed to convert")
        return 1
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        bars.to_csv(output, index=not args.no_index)
        logger.info("Wrote %d second bars to %s", len(bars), output)
    except Exception:
        logger.exception("Failed to write output")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

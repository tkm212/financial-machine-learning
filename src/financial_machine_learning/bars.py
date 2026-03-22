"""
Alternative sampling methods for financial data.

Based on López de Prado, M. (2018). Advances in Financial Machine Learning.
Wiley. Chapter 2: Financial Data Structures.
"""

from __future__ import annotations

import pandas as pd


def time_bars(df: pd.DataFrame, freq: str = "1s") -> pd.DataFrame:
    """
    Build time bars: OHLCV at fixed time intervals.

    Uses groupby for memory efficiency (no full-range allocation).
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.floor(freq)

    bars = (
        df.groupby("datetime", sort=True)
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
    return bars


def tick_bars(df: pd.DataFrame, threshold: int = 100) -> pd.DataFrame:
    """
    Build tick bars: one bar per fixed number of ticks.

    Mandelbrot and Taylor (1967) showed price changes over a fixed number
    of transactions are closer to Gaussian (IID) than over fixed time.
    """
    df = df.sort_values("time").reset_index(drop=True)
    df = df.copy()
    df["cum_ticks"] = (df.index + 1)
    df["bar_id"] = (df["cum_ticks"] - 1) // threshold

    bars = (
        df.groupby("bar_id")
        .agg(
            datetime=("time", "last"),
            open=("Price", "first"),
            high=("Price", "max"),
            low=("Price", "min"),
            close=("Price", "last"),
            volume=("Quantity", "sum"),
            count=("Price", "count"),
        )
        .reset_index(drop=True)
    )
    bars["datetime"] = pd.to_datetime(bars["datetime"], unit="s", utc=True)
    return bars


def volume_bars(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Build volume bars: one bar per fixed traded volume.

    Volume bars sample when a fixed amount of volume has been traded,
    normalizing for varying activity across time.
    """
    df = df.sort_values("time").reset_index(drop=True)
    df = df.copy()
    df["cum_vol"] = df["Quantity"].cumsum()
    df["bar_id"] = (df["cum_vol"] // threshold).astype(int)

    bars = (
        df.groupby("bar_id")
        .agg(
            datetime=("time", "last"),
            open=("Price", "first"),
            high=("Price", "max"),
            low=("Price", "min"),
            close=("Price", "last"),
            volume=("Quantity", "sum"),
            count=("Price", "count"),
        )
        .reset_index(drop=True)
    )
    bars["datetime"] = pd.to_datetime(bars["datetime"], unit="s", utc=True)
    return bars


def dollar_bars(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Build dollar (information) bars: one bar per fixed dollar value traded.

    Dollar value = Price * Quantity. These bars sample by economic
    information flow rather than time or tick count.
    """
    df = df.sort_values("time").reset_index(drop=True)
    df = df.copy()
    df["dollar_value"] = df["Price"] * df["Quantity"]
    df["cum_dollar"] = df["dollar_value"].cumsum()
    df["bar_id"] = (df["cum_dollar"] // threshold).astype(int)

    bars = (
        df.groupby("bar_id")
        .agg(
            datetime=("time", "last"),
            open=("Price", "first"),
            high=("Price", "max"),
            low=("Price", "min"),
            close=("Price", "last"),
            volume=("Quantity", "sum"),
            count=("Price", "count"),
        )
        .reset_index(drop=True)
    )
    bars["datetime"] = pd.to_datetime(bars["datetime"], unit="s", utc=True)
    return bars

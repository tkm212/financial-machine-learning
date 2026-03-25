"""Unit tests for financial_machine_learning.labeling."""

import pandas as pd
import pytest

from financial_machine_learning.labeling import triple_barrier_labels


@pytest.fixture
def ohlc_df() -> pd.DataFrame:
    """Synthetic OHLC: flat then up, down, flat."""
    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-01 09:00", periods=20, freq="1min"),
        "open": [100.0] * 20,
        "high": [100.5, 101.0, 102.0, 102.5, 103.0, 99.0, 98.5, 98.0, 100.0, 100.2] + [100.3] * 10,
        "low": [99.5, 99.8, 100.5, 101.5, 102.0, 98.5, 97.5, 97.0, 99.5, 99.8] + [100.0] * 10,
        "close": [100.0, 100.5, 101.5, 102.0, 102.5, 98.5, 98.0, 97.5, 100.0, 100.0] + [100.2] * 10,
    })


def test_triple_barrier_labels_columns(ohlc_df: pd.DataFrame) -> None:
    events = pd.DatetimeIndex([ohlc_df["datetime"].iloc[0]])
    out = triple_barrier_labels(ohlc_df, events, pt=0.02, sl=0.02, num_bars=5)
    assert list(out.columns) == ["datetime", "label", "exit_time", "exit_price"]
    assert len(out) == 1


def test_triple_barrier_labels_profit_take(ohlc_df: pd.DataFrame) -> None:
    # Bar 1: close=100, upper=102. High at bar 2 is 102 -> hit upper
    events = pd.DatetimeIndex([ohlc_df["datetime"].iloc[1]])
    out = triple_barrier_labels(ohlc_df, events, pt=0.02, sl=0.02, num_bars=10)
    assert len(out) == 1
    assert out["label"].iloc[0] == 1


def test_triple_barrier_labels_stop_loss(ohlc_df: pd.DataFrame) -> None:
    # Bar 4: close=102.5, lower=100.45. Low at bar 6 is 98.5 -> hit lower
    events = pd.DatetimeIndex([ohlc_df["datetime"].iloc[4]])
    out = triple_barrier_labels(ohlc_df, events, pt=0.01, sl=0.02, num_bars=10)
    assert len(out) == 1
    assert out["label"].iloc[0] == -1


def test_triple_barrier_labels_vertical_barrier(ohlc_df: pd.DataFrame) -> None:
    # Flat prices: high=100.3, low=100.0. With pt=0.02, sl=0.02 from close 100.2,
    # upper=102.2, lower=98.2. Neither hit in 3 bars -> label 0
    events = pd.DatetimeIndex([ohlc_df["datetime"].iloc[10]])
    out = triple_barrier_labels(ohlc_df, events, pt=0.02, sl=0.02, num_bars=3)
    assert len(out) == 1
    assert out["label"].iloc[0] == 0


def test_triple_barrier_labels_empty_events(ohlc_df: pd.DataFrame) -> None:
    events = pd.DatetimeIndex([])
    out = triple_barrier_labels(ohlc_df, events, pt=0.01, sl=0.01, num_bars=5)
    assert len(out) == 0
    assert list(out.columns) == ["datetime", "label", "exit_time", "exit_price"]


def test_triple_barrier_labels_skips_missing_events(ohlc_df: pd.DataFrame) -> None:
    # Event timestamp not in OHLC
    events = pd.DatetimeIndex(["2020-01-01 12:00"])  # not in ohlc
    out = triple_barrier_labels(ohlc_df, events, pt=0.01, sl=0.01, num_bars=5)
    assert len(out) == 0


def test_triple_barrier_labels_custom_columns() -> None:
    df = pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=5, freq="1h"),
        "o": [100.0] * 5,
        "h": [101.0, 102.0, 103.0, 104.0, 105.0],
        "l": [99.0] * 5,
        "c": [100.5, 101.5, 102.5, 103.5, 104.5],
    })
    events = pd.DatetimeIndex([df["ts"].iloc[0]])
    out = triple_barrier_labels(
        df,
        events,
        pt=0.02,
        sl=0.02,
        num_bars=5,
        open_col="o",
        high_col="h",
        low_col="l",
        close_col="c",
        datetime_col="ts",
    )
    assert len(out) == 1
    assert (
        out["label"].iloc[0] == 1
    )  # h=101 >= 100.5*1.02=102.51? No. h goes to 105. At bar 1 h=101, at bar 2 h=102... 100.5*1.02=102.51, so bar 3 h=103 hits it. Yes.

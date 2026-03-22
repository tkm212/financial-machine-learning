"""Unit tests for financial_machine_learning.filters."""

import pandas as pd
import pytest

from financial_machine_learning.filters import cusum_filter


def test_cusum_filter_returns_datetime_index(price_series: pd.Series) -> None:
    events = cusum_filter(price_series, threshold=0.02, time_stamps=True)
    assert isinstance(events, pd.DatetimeIndex)


def test_cusum_filter_returns_list_when_requested(price_series: pd.Series) -> None:
    events = cusum_filter(price_series, threshold=0.02, time_stamps=False)
    assert isinstance(events, list)


def test_cusum_filter_triggers_on_drop(price_series: pd.Series) -> None:
    # ~5% drop from 101 to 96 should trigger with threshold 0.02
    events = cusum_filter(price_series, threshold=0.02)
    assert len(events) >= 1


def test_cusum_filter_no_events_small_threshold() -> None:
    # Flat prices: no divergence
    flat = pd.Series([100.0] * 10, index=pd.date_range("2024-01-01", periods=10, freq="1h"))
    events = cusum_filter(flat, threshold=0.5)
    assert len(events) == 0


def test_cusum_filter_empty_series() -> None:
    empty = pd.Series(dtype=float)
    events = cusum_filter(empty, threshold=0.02)
    assert len(events) == 0


def test_cusum_filter_single_price() -> None:
    single = pd.Series([100.0], index=pd.DatetimeIndex(["2024-01-01"]))
    events = cusum_filter(single, threshold=0.02)
    assert len(events) == 0  # No returns = no events

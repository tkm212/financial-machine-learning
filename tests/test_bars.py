"""Unit tests for financial_machine_learning.bars."""

import pandas as pd
import pytest

from financial_machine_learning.bars import dollar_bars, tick_bars, time_bars, volume_bars


def test_time_bars(tick_df: pd.DataFrame) -> None:
    bars = time_bars(tick_df)
    assert list(bars.columns) == ["datetime", "open", "high", "low", "close", "volume", "count"]
    assert len(bars) >= 1
    assert bars["open"].iloc[0] == 100.0
    assert bars["close"].iloc[-1] == 107.0
    assert bars["high"].max() >= bars["close"].max()
    assert bars["low"].min() <= bars["open"].min()
    assert bars["count"].dtype == int


def test_tick_bars(tick_df: pd.DataFrame) -> None:
    bars = tick_bars(tick_df, threshold=3)
    # 10 ticks / 3 per bar = 4 bars (last bar may have 1 tick)
    assert len(bars) == 4
    assert bars["count"].iloc[0] == 3
    assert bars["open"].iloc[0] == 100.0
    assert bars["close"].iloc[0] == 102.0


def test_tick_bars_single_bar(tick_df: pd.DataFrame) -> None:
    bars = tick_bars(tick_df, threshold=100)
    assert len(bars) == 1
    assert bars["count"].iloc[0] == 10


def test_volume_bars(tick_df: pd.DataFrame) -> None:
    total_vol = tick_df["Quantity"].sum()
    bars = volume_bars(tick_df, threshold=total_vol / 2)
    assert len(bars) >= 2
    assert bars["volume"].sum() == pytest.approx(total_vol)


def test_dollar_bars(tick_df: pd.DataFrame) -> None:
    total_dollar = (tick_df["Price"] * tick_df["Quantity"]).sum()
    bars = dollar_bars(tick_df, threshold=total_dollar / 2)
    assert len(bars) >= 2
    assert (bars["high"] >= bars["low"]).all()


def test_dollar_bars_ohlc_invariants(tick_df: pd.DataFrame) -> None:
    bars = dollar_bars(tick_df, threshold=100.0)
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["open"] >= 0).all()
    assert (bars["close"] >= 0).all()

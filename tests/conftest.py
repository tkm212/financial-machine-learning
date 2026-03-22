"""Pytest fixtures for financial_machine_learning tests."""

import pandas as pd
import pytest


@pytest.fixture
def tick_df() -> pd.DataFrame:
    """Synthetic tick data: time, Price, Quantity."""
    # 10 ticks over 5 seconds, prices 100-109
    return pd.DataFrame({
        "time": [1000.0, 1000.5, 1001.0, 1001.2, 1002.0, 1002.5, 1003.0, 1003.1, 1004.0, 1004.5],
        "Price": [100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 106.0, 107.0],
        "Quantity": [1.0, 2.0, 1.5, 1.0, 2.0, 1.0, 1.5, 1.0, 2.0, 1.0],
    })


@pytest.fixture
def price_series() -> pd.Series:
    """Price series for CUSUM: monotonic then drop."""
    # Log returns: small up, small up, then -5% drop (triggers CUSUM)
    prices = [100.0, 100.5, 101.0, 96.0, 96.2, 96.4]
    return pd.Series(prices, index=pd.date_range("2024-01-01", periods=6, freq="1h"))

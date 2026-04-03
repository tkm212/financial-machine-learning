"""Unit tests for financial_machine_learning.weights."""

import pandas as pd
import pytest

from financial_machine_learning.weights import (
    average_uniqueness,
    concurrent_labels_per_bar,
    time_decay_weights,
)


def test_concurrent_labels_per_bar() -> None:
    ev = pd.DataFrame({
        "datetime": [
            pd.Timestamp("2024-01-01 08:00"),
            pd.Timestamp("2024-01-01 09:00"),
            pd.Timestamp("2024-01-01 10:00"),
        ],
        "exit_time": [
            pd.Timestamp("2024-01-01 12:00"),
            pd.Timestamp("2024-01-01 12:00"),
            pd.Timestamp("2024-01-01 12:00"),
        ],
    })
    bars = pd.date_range("2024-01-01 09:00", "2024-01-01 13:00", freq="h")
    c = concurrent_labels_per_bar(ev, bars)
    assert int(c.loc[pd.Timestamp("2024-01-01 10:00")]) == 3
    assert int(c.loc[pd.Timestamp("2024-01-01 12:00")]) == 3


def test_average_uniqueness_single_event() -> None:
    ev = pd.DataFrame({
        "datetime": [pd.Timestamp("2024-01-01 10:00")],
        "exit_time": [pd.Timestamp("2024-01-01 12:00")],
    })
    bars = pd.date_range("2024-01-01 09:00", "2024-01-01 13:00", freq="h")
    u = average_uniqueness(ev, bars)
    assert len(u) == 1
    assert u.iloc[0] == pytest.approx(1.0)


def test_average_uniqueness_full_overlap() -> None:
    ev = pd.DataFrame({
        "datetime": [pd.Timestamp("2024-01-01 10:00")] * 2,
        "exit_time": [pd.Timestamp("2024-01-01 12:00")] * 2,
    })
    bars = pd.date_range("2024-01-01 09:00", "2024-01-01 13:00", freq="h")
    u = average_uniqueness(ev, bars)
    assert u.iloc[0] == pytest.approx(0.5)
    assert u.iloc[1] == pytest.approx(0.5)


def test_time_decay_weights_order() -> None:
    t = pd.Series(pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"))
    ref = pd.Timestamp("2024-01-02", tz="UTC")
    w = time_decay_weights(t, ref_time=ref, decay_span=pd.Timedelta(hours=6))
    assert w.iloc[2] > w.iloc[1] > w.iloc[0]


def test_time_decay_weights_normalized_span() -> None:
    t = pd.Series([pd.Timestamp("2024-01-01 12:00", tz="UTC")])
    ref = pd.Timestamp("2024-01-01 12:00", tz="UTC")
    w = time_decay_weights(t, ref_time=ref, decay_span=pd.Timedelta(days=1))
    assert w.iloc[0] == pytest.approx(1.0)

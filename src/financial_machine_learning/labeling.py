"""
Labeling methods for financial time series (López de Prado, Ch. 3).

Triple-barrier method: assign labels based on which barrier is hit first:
- Upper (profit take): +1
- Lower (stop loss): -1
- Vertical (max holding period): 0
"""

from __future__ import annotations

import pandas as pd


def triple_barrier_labels(
    ohlc: pd.DataFrame,
    events: pd.DatetimeIndex,
    pt: float = 0.01,
    sl: float = 0.01,
    num_bars: int = 20,
    *,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    datetime_col: str = "datetime",
) -> pd.DataFrame:
    """
    Triple-barrier labeling (López de Prado, Snippet 3.2, p.45).

    For each event time, track forward until the first barrier is hit:
    - Upper: high >= close_at_event * (1 + pt)  -> label 1
    - Lower: low <= close_at_event * (1 - sl)   -> label -1
    - Vertical: num_bars bars elapse             -> label 0

    Parameters
    ----------
    ohlc : pd.DataFrame
        OHLC data with datetime and price columns.
    events : pd.DatetimeIndex
        Event timestamps (e.g. from CUSUM filter) at which to start each barrier.
    pt : float
        Profit-taking threshold (fraction, e.g. 0.01 = 1%).
    sl : float
        Stop-loss threshold (fraction).
    num_bars : int
        Max holding period (bars). Vertical barrier.
    open_col, high_col, low_col, close_col, datetime_col : str
        Column names in ohlc.

    Returns
    -------
    pd.DataFrame
        Columns: datetime (event time), label (1, -1, 0), exit_time, exit_price.
    """
    df = ohlc.copy()
    if datetime_col in df.columns:
        df = df.set_index(datetime_col)
    df = df.sort_index()

    rows = []
    for t in events:
        if t not in df.index:
            continue
        idx = df.index.get_loc(t)
        window = df.iloc[idx : idx + num_bars + 1]
        if window.empty:
            continue

        p0 = window.iloc[0][close_col]
        upper = p0 * (1 + pt)
        lower = p0 * (1 - sl)

        label = 0
        exit_time = window.index[-1]
        exit_price = window.iloc[-1][close_col]

        for i in range(1, len(window)):
            row = window.iloc[i]
            if row[high_col] >= upper:
                label = 1
                exit_time = window.index[i]
                exit_price = upper
                break
            if row[low_col] <= lower:
                label = -1
                exit_time = window.index[i]
                exit_price = lower
                break
            if i == num_bars:
                label = 0
                exit_time = window.index[i]
                exit_price = row[close_col]
                break

        rows.append({"datetime": t, "label": label, "exit_time": exit_time, "exit_price": exit_price})

    return pd.DataFrame(rows, columns=["datetime", "label", "exit_time", "exit_price"])

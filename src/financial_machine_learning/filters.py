"""
Event filters for financial time series.

Based on López de Prado, M. (2018). Advances in Financial Machine Learning.
Wiley. Chapter 2, Snippet 2.4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cusum_filter(
    raw_time_series: pd.Series,
    threshold: float,
    time_stamps: bool = True,
) -> pd.DatetimeIndex | list:
    """
    Symmetric CUSUM filter (López de Prado, Snippet 2.4, p.39).

    Quality-control method to detect shifts in mean away from a target.
    Identifies upside/downside divergences from reset level zero. Sample bar t
    if and only if S_t >= threshold, then reset S_t to 0.

    Advantage over Bollinger Bands: avoids multiple triggers when price hovers
    near threshold. Requires a full run of cumulative divergence to trigger.
    """
    t_events: list = []
    s_pos = 0.0
    s_neg = 0.0

    prices = raw_time_series.dropna()
    log_ret = np.log(prices).diff().dropna()

    for t, r in log_ret.items():
        pos = s_pos + r
        neg = s_neg + r
        s_pos = max(0.0, pos)
        s_neg = min(0.0, neg)

        if s_neg < -threshold:
            s_neg = 0.0
            t_events.append(t)
        elif s_pos > threshold:
            s_pos = 0.0
            t_events.append(t)

    return pd.DatetimeIndex(t_events) if time_stamps else t_events

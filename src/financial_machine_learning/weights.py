"""
Sample weights for financial ML (López de Prado, Ch. 4).

Overlapping labels break IID assumptions. Uniqueness and time decay down-weight
redundant or stale observations when fitting models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def concurrent_labels_per_bar(
    events: pd.DataFrame,
    bar_index: pd.DatetimeIndex,
    *,
    start_col: str = "datetime",
    end_col: str = "exit_time",
) -> pd.Series:
    """
    Count how many labels are active at each bar (concurrency).

    A label is active on bar t if start_col <= t <= end_col.
    """
    counts = np.zeros(len(bar_index), dtype=np.int64)
    for s, e in zip(events[start_col], events[end_col], strict=True):
        mask = (bar_index >= s) & (bar_index <= e)
        counts += np.asarray(mask, dtype=np.int64)
    return pd.Series(counts, index=bar_index, dtype=np.int64)


def average_uniqueness(
    events: pd.DataFrame,
    bar_index: pd.DatetimeIndex,
    *,
    start_col: str = "datetime",
    end_col: str = "exit_time",
) -> pd.Series:
    """
    Average uniqueness per event (López de Prado, Ch. 4).

    For each event, take the mean of 1/c(t) over bars t where the label is active,
    and c(t) is the concurrency at t. A fully unique event has uniqueness 1.0;
    k fully overlapping events each have average uniqueness 1/k.
    """
    c = concurrent_labels_per_bar(events, bar_index, start_col=start_col, end_col=end_col)
    out: list[float] = []
    for s, e in zip(events[start_col], events[end_col], strict=True):
        mask = (bar_index >= s) & (bar_index <= e)
        conc = c.loc[mask]
        if conc.empty or (conc == 0).any():
            out.append(float("nan"))
        else:
            out.append(float((1.0 / conc).mean()))
    return pd.Series(out, index=events.index, dtype=float)


def time_decay_weights(
    event_starts: pd.DatetimeIndex | pd.Series,
    *,
    ref_time: pd.Timestamp,
    decay_span: pd.Timedelta,
) -> pd.Series:
    """
    Exponential time-decay weights: newer events (closer to ref_time) weigh more.

    weight_i = exp(-age_i / decay_span) where age_i = ref_time - start_i (non-negative).
    """
    starts = pd.Series(event_starts) if isinstance(event_starts, pd.DatetimeIndex) else event_starts
    idx = starts.index
    t = pd.to_datetime(starts, utc=False)
    ref = pd.Timestamp(ref_time)
    if t.dt.tz is not None and ref.tzinfo is None:
        ref = ref.tz_localize("UTC").tz_convert(t.dt.tz)
    elif t.dt.tz is None and ref.tzinfo is not None:
        t = t.dt.tz_localize(ref.tzinfo)
    age_s = (ref - t).dt.total_seconds().clip(lower=0.0)
    span_s = max(decay_span.total_seconds(), 1e-12)
    return pd.Series(np.exp(-age_s / span_s), index=idx, dtype=float)

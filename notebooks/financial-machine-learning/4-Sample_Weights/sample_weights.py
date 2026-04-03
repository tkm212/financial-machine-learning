import marimo

__generated_with = "0.22.3"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sample Weights: López de Prado

    *Advances in Financial Machine Learning (2018), Chapter 4.*

    Chapter 4 is about **weighting training samples**, not feature coefficients. Financial labels overlap in time, so observations are not IID. Standard ML assumes IID; without correction, models over-emphasize redundant periods and under-use scarce information. López de Prado defines **concurrency**, **average uniqueness**, **time decay**, and **sequential bootstrapping** to address this.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The IID Problem

    | Issue | Why it matters |
    |-------|----------------|
    | **Overlapping labels** | Two triple-barrier events can span the same bars; both "see" the same price path. |
    | **Redundancy** | Treating them as independent inflates effective sample size and biases cross-validation. |
    | **Goal** | Down-weight samples that share information with many concurrent labels; up-weight unique or recent ones. |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Concurrency and Average Uniqueness

    At each bar \(t\), let \(c(t)\) be the **number of labels active** (event started and not yet closed).

    For an event spanning \([t_0, t_1]\), **average uniqueness** is the mean of \(1/c(t)\) over bars where the label is alive. If you are alone, \(c=1\) and uniqueness is 1. If \(k\) identical-length events fully overlap, each has average uniqueness about \(1/k\).

    These values feed **sample weights** (e.g. proportional to return magnitude divided by uniqueness, or used directly as sklearn `sample_weight`).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Time Decay

    Older events may matter less for a model that must predict **now**. A common form is exponential decay: weight \(\propto \exp(-\text{age}/\tau)\) with reference time at the end of the sample and decay span \(\tau\) (e.g. one day of bars).

    Decay can be **multiplied** with uniqueness-based weights.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sequential Bootstrapping (concept)

    Classical bagging draws rows **with replacement** uniformly. **Sequential bootstrapping** prefers draws that are more **unique** (lower overlap with already sampled events), reducing redundancy in each bootstrap replicate. It changes how ensembles are built, not only per-row weights in a single fit.

    Implementation of the full sequential bootstrap is omitted here; see the book and packages such as `mlfinlab` for production code.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ROOT = Path.cwd().resolve()
    for _ in range(5):
        if (ROOT / "pyproject.toml").exists():
            break
        ROOT = ROOT.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    OUTPUTS = ROOT / "outputs"
    return OUTPUTS, go, make_subplots, pd


@app.cell
def _(OUTPUTS, pd):
    # 1-second bars (same pipeline as Ch.3 labeling notebook)
    bars = pd.read_csv(OUTPUTS / "btc_bid_ask_data_1s.csv")
    bars["datetime"] = pd.to_datetime(bars["datetime"])
    print(f"Loaded {len(bars):,} bars")
    return (bars,)


@app.cell
def _(bars):
    from financial_machine_learning.filters import cusum_filter
    from financial_machine_learning.labeling import triple_barrier_labels
    from financial_machine_learning.weights import (
        average_uniqueness,
        concurrent_labels_per_bar,
        time_decay_weights,
    )

    close = bars.set_index("datetime")["close"].dropna()
    cusum_events = cusum_filter(close, threshold=0.0002)
    labels = triple_barrier_labels(
        bars,
        cusum_events,
        pt=0.001,
        sl=0.001,
        num_bars=30,
    )
    print(f"Labeled events: {len(labels):,}")
    return (
        average_uniqueness,
        concurrent_labels_per_bar,
        labels,
        time_decay_weights,
    )


@app.cell
def _(bars, concurrent_labels_per_bar, labels, pd):
    # Bar index only where labels overlap (faster than full history)
    t0 = labels["datetime"].min()
    t1 = labels["exit_time"].max()
    bar_index = pd.DatetimeIndex(bars.loc[(bars["datetime"] >= t0) & (bars["datetime"] <= t1), "datetime"].unique())
    bar_index = bar_index.sort_values()

    conc = concurrent_labels_per_bar(labels, bar_index)
    print(f"Concurrency at each bar (max concurrent labels): {conc.max()}")
    return bar_index, conc


@app.cell
def _(average_uniqueness, bar_index, labels):
    uniq = average_uniqueness(labels, bar_index)
    labels_1 = labels.assign(avg_uniqueness=uniq.values)
    print("Average uniqueness (describe):")
    print(labels_1["avg_uniqueness"].describe())
    return (labels_1,)


@app.cell
def _(bars, labels_1, pd, time_decay_weights):
    # Time-decay vs end of sample (use last bar time as reference)
    ref = bars["datetime"].max()
    decay_span = pd.Timedelta(hours=1)
    td = time_decay_weights(labels_1["datetime"], ref_time=ref, decay_span=decay_span)
    labels_2 = labels_1.assign(time_decay=td.values)
    raw = labels_2["avg_uniqueness"] * labels_2["time_decay"]
    # Combined weight (normalize to mean 1 for intuition)
    labels_2["sample_weight"] = raw / raw.mean()
    print(labels_2[["avg_uniqueness", "time_decay", "sample_weight"]].describe())
    return (labels_2,)


@app.cell
def _(bars, conc, go, make_subplots):
    # Concurrency through time (subset for readability)
    plot_conc = conc.iloc[: min(5000, len(conc))]
    _fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.08)
    price_win = bars[(bars["datetime"] >= plot_conc.index.min()) & (bars["datetime"] <= plot_conc.index.max())]
    _fig.add_trace(
        go.Scatter(x=price_win["datetime"], y=price_win["close"], name="Close", line={"width": 1}), row=1, col=1
    )
    _fig.add_trace(
        go.Scatter(x=plot_conc.index, y=plot_conc.values, name="Concurrency", fill="tozeroy", line={"width": 0}),
        row=2,
        col=1,
    )
    _fig.update_layout(height=520, title_text="BTC close and label concurrency (first segment)")
    _fig.update_yaxes(title_text="Price", row=1, col=1)
    _fig.update_yaxes(title_text="Active labels", row=2, col=1)
    _fig.show()
    return


@app.cell
def _(go, labels_2):
    _fig = go.Figure()
    _fig.add_trace(go.Histogram(x=labels_2["avg_uniqueness"], nbinsx=50, name="Avg uniqueness"))
    _fig.update_layout(
        title="Distribution of average uniqueness per event", xaxis_title="Uniqueness", yaxis_title="Count", height=400
    )
    _fig.show()
    return


@app.cell
def _(go, labels_2):
    _fig = go.Figure()
    _fig.add_trace(go.Histogram(x=labels_2["sample_weight"], nbinsx=50, name="Combined weight"))
    _fig.update_layout(
        title="Combined sample weights (uniqueness * time decay, mean-normalized)",
        xaxis_title="Weight",
        yaxis_title="Count",
        height=400,
    )
    _fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    - **Concurrency** \(c(t)\): how many triple-barrier events are open at bar \(t\).
    - **Average uniqueness**: mean \(1/c(t)\) over an event's life; use to down-weight crowded overlaps.
    - **Time decay**: discount stale events relative to a reference time (e.g. now).
    - **Sequential bootstrap**: ensemble method that prefers unique draws; complements explicit `sample_weight`.

    Pass `sample_weight` (or the product of uniqueness and decay) into sklearn estimators that support it when training on labeled events.
    """)
    return


if __name__ == "__main__":
    app.run()

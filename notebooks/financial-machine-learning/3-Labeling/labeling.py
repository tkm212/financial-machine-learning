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
    # Labeling Returns for ML: López de Prado

    *Advances in Financial Machine Learning (2018), Chapter 3.*

    This chapter addresses how to **create target variables** for supervised learning in finance. Standard approaches (e.g. "return over next N bars") suffer from look-ahead bias, path-dependence, and arbitrary horizons. López de Prado introduces the **triple-barrier method** and **meta-labeling** as robust alternatives.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why Standard Labels Fail

    | Problem | Description |
    |---------|-------------|
    | **Fixed-horizon returns** | "Return over next 5 bars" ignores path: price may hit stop-loss before bar 5. The label is not the actual P&L. |
    | **Look-ahead bias** | Using future information when deciding entry/exit. |
    | **Arbitrary horizons** | Why 5 bars? Holding periods should reflect volatility and opportunity cost. |
    | **Overlapping samples** | Sequential labels overlap, causing leakage and redundant training samples. |

    **Insight**: Real trading has *path-dependent* outcomes. A bet at time t may be closed by profit-target, stop-loss, or time expiry. The label should reflect which barrier is hit first.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Triple-Barrier Method (Snippet 3.2, p.45)

    For each observation/event time \(t\):

    1. **Upper barrier (profit take)**: Price reaches \(P_t \cdot (1 + \mathit{pt})\)
    2. **Lower barrier (stop loss)**: Price reaches \(P_t \cdot (1 - \mathit{sl})\)
    3. **Vertical barrier**: Max holding period (e.g. \(N\) bars)

    **Label** = which barrier is hit first:
    - **+1**: Upper barrier (profit target hit)
    - **-1**: Lower barrier (stop loss hit)
    - **0**: Vertical barrier (time expired, neither target hit)

    Barriers can be set in absolute terms or scaled by volatility (e.g. ATR) for adaptive thresholds.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Meta-Labeling (p.50)

    **Two-stage approach**:

    1. **Primary model**: Decides *when* to bet and *which side* (long/short). Often a traditional signal or heuristic.
    2. **Secondary model (meta-label)**: Decides *whether to take the bet* and *how much*. Binary: take bet (1) or pass (0).

    **Benefits**:
    - Improves precision and recall; reduces false positives.
    - Separates "direction" from "size/confidence."
    - Meta-label probability can derive bet size: higher confidence → larger position.

    **Workflow**: Primary model generates events; triple-barrier labels outcomes; meta-model predicts if the bet would have been profitable (1) or not (0).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Other Chapter 3 Concepts

    - **Sample weights**: Down-weight redundant/overlapping samples (e.g. by uniqueness).
    - **Drop rare labels**: Recursively drop labels that occur &lt; min_pct (Snippet 3.8, p.54) to avoid imbalanced classes.
    - **Vertical barrier**: Can be fixed (N bars) or volatility-scaled (e.g. higher vol → shorter max hold).
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    import pandas as pd
    import plotly.graph_objects as go

    ROOT = Path.cwd().resolve()
    for _ in range(5):
        if (ROOT / "pyproject.toml").exists():
            break
        ROOT = ROOT.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    OUTPUTS = ROOT / "outputs"
    return OUTPUTS, go, pd


@app.cell
def _(OUTPUTS, pd):
    # Load 1-second bar data (run information_bars.ipynb or generate_all_bars.py first)
    bars = pd.read_csv(OUTPUTS / "btc_bid_ask_data_1s.csv")
    bars["datetime"] = pd.to_datetime(bars["datetime"])
    print(f"Loaded {len(bars):,} bars")
    return (bars,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### CUSUM events as labeling seeds

    Use CUSUM filter (Ch. 2) to select *when* to label. This avoids labeling every bar and reduces overlap.
    """)
    return


@app.cell
def _(bars):
    from financial_machine_learning.filters import cusum_filter
    from financial_machine_learning.labeling import triple_barrier_labels

    close = bars.set_index("datetime")["close"].dropna()
    events = cusum_filter(close, threshold=0.0002)
    print(f"CUSUM events: {len(events):,}")
    return events, triple_barrier_labels


@app.cell
def _(bars, events, triple_barrier_labels):
    # Triple-barrier labels: pt=0.1%, sl=0.1%, max 30 bars
    labels = triple_barrier_labels(
        bars,
        events,
        pt=0.001,
        sl=0.001,
        num_bars=30,
    )
    print(f"Labeled events: {len(labels):,}")
    print(labels["label"].value_counts().sort_index())
    return (labels,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualize triple-barrier outcomes
    """)
    return


@app.cell
def _(bars, go, labels, pd):
    # Sample a short window with a few labeled events
    sample = labels.head(50)
    start = sample['datetime'].min()
    end = sample['exit_time'].max()
    window_bars = bars[(bars['datetime'] >= start) & (bars['datetime'] <= end)]
    bars_idx = bars.set_index('datetime')
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=window_bars['datetime'], y=window_bars['close'], name='Close', mode='lines'))
    colors = {1: 'green', -1: 'red', 0: 'gray'}
    for _, row in sample.iterrows():
        try:
            entry_price = bars_idx.loc[row['datetime'], 'close']
            if isinstance(entry_price, pd.Series):
                entry_price = entry_price.iloc[0]
        except KeyError:
            continue
        _fig.add_trace(go.Scatter(x=[row['datetime'], row['exit_time']], y=[entry_price, row['exit_price']], mode='lines+markers', line={'dash': 'dot', 'width': 1}, marker={'size': 6, 'color': colors[row['label']]}, name=f"Label {int(row['label'])}", showlegend=False))
    _fig.update_layout(title='Triple-Barrier Labels on BTC 1s Bars (sample)', xaxis_title='Time', yaxis_title='Price', height=450)
    _fig.show()
    return


@app.cell
def _(go, labels):
    # Label distribution (full run)
    _fig = go.Figure(data=[go.Bar(x=labels['label'].value_counts().sort_index().index.astype(str), y=labels['label'].value_counts().sort_index().values, text=labels['label'].value_counts().sort_index().values, textposition='auto')])
    _fig.update_layout(title='Triple-Barrier Label Distribution', xaxis_title='Label', yaxis_title='Count', xaxis={'tickvals': ['-1', '0', '1'], 'ticktext': ['Stop loss', 'Time out', 'Profit take']}, height=400)
    _fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    - **Triple-barrier method**: Labels reflect which barrier (profit target, stop loss, or time) is hit first, producing path-dependent, realistic targets.
    - **Meta-labeling**: Secondary model refines bet size and filters false positives from a primary signal.
    - **Event sampling**: Use CUSUM or similar filters to reduce overlap and focus on meaningful events.
    """)
    return


if __name__ == "__main__":
    app.run()

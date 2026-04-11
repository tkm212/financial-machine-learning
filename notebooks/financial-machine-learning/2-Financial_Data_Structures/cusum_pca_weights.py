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
    # CUSUM Filter & PCA Weights: López de Prado

    *Advances in Financial Machine Learning (2018), Chapter 2.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CUSUM Filter (Snippet 2.4, p.39)

    The CUSUM filter is a quality-control method to detect shifts in the mean of a measured quantity away from a target. It identifies sequences of upside or downside divergences from reset level zero.

    **Rule**: Sample bar t if and only if S_t ≥ threshold, at which point S_t resets to 0.

    **Advantage**: Unlike Bollinger Bands, it does not trigger multiple events when price hovers around a threshold. A full cumulative run is required to trigger.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PCA Weights

    Principal component weights derived from the correlation matrix of returns across bar types.
    Eigenvectors (loadings) capture the directions of maximum variance; the first PC represents
    the dominant shared signal across bar types and can serve as a composite feature.

    Note: López de Prado (Ch. 2) discusses Marchenko-Pastur denoising to separate signal from
    noise eigenvalues. That technique requires fitting the Marchenko-Pastur distribution to the
    eigenvalue spectrum - this notebook implements plain PCA without the denoising step.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    import numpy as np
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
    return OUTPUTS, go, make_subplots, np, pd


@app.cell
def _(OUTPUTS, pd):
    # Load bar data (run information_bars.ipynb or generate_all_bars.py first)
    time_bars = pd.read_csv(OUTPUTS / "btc_bid_ask_data_1s.csv")
    time_bars["datetime"] = pd.to_datetime(time_bars["datetime"])

    # Use close prices for CUSUM
    close = time_bars.set_index("datetime")["close"]
    close = close.dropna()
    print(f"Loaded {len(close):,} bars")
    return close, time_bars


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### CUSUM filter: event timestamps
    """)
    return


@app.cell
def _(close):
    from financial_machine_learning.filters import cusum_filter

    threshold = 0.0002  # 2% cumulative log-return divergence
    events = cusum_filter(close, threshold=threshold)
    print(f"CUSUM events (threshold={threshold}): {len(events)}")
    return events, threshold


@app.cell
def _(close, events, go, make_subplots, np, threshold):
    log_ret = np.log(close).diff().dropna()
    _fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    _fig.add_trace(
        go.Scatter(x=close.index, y=close.values, name="Close", mode="lines", line={"width": 1}), row=1, col=1
    )
    _fig.add_trace(
        go.Scatter(
            x=events,
            y=close.reindex(events).values,
            name="CUSUM events",
            mode="markers",
            marker={"size": 8, "color": "red"},
        ),
        row=1,
        col=1,
    )
    _fig.add_trace(
        go.Scatter(x=log_ret.index, y=log_ret.values, fill="tozeroy", name="Log return", line={"width": 0}),
        row=2,
        col=1,
    )
    _fig.add_hline(y=threshold, line_dash="dash", line_color="red", opacity=0.7, row=2, col=1)
    _fig.add_hline(y=-threshold, line_dash="dash", line_color="red", opacity=0.7, row=2, col=1)
    _fig.update_layout(height=500, title_text="BTC Close with CUSUM Events (López de Prado, Snippet 2.4)")
    _fig.update_yaxes(title_text="Price", row=1, col=1)
    _fig.update_yaxes(title_text="Log return", row=2, col=1)
    _fig.update_xaxes(title_text="Time", row=2, col=1)
    _fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PCA weights

    PCA on the correlation matrix of returns yields eigenvectors (loadings). We use multi-horizon returns (1, 5, 10, 30 bars) as our "assets." The first PC captures shared variation; its weights show how each horizon contributes.
    """)
    return


@app.cell
def _(pd, time_bars):
    # PCA on multi-horizon returns (same asset, different lookbacks)
    # periods must be integers (number of bars to shift)
    # For 1s bars: 1, 5, 10, 30 = 1s, 5s, 10s, 30s horizons
    time_bars_1 = time_bars.sort_values("datetime").reset_index(drop=True)
    close_1 = time_bars_1["close"].astype(float)
    rets = pd.DataFrame({
        "1-bar": close_1.pct_change(1),
        "5-bar": close_1.pct_change(5),
        "10-bar": close_1.pct_change(10),
        "30-bar": close_1.pct_change(30),
    })
    rets.index = time_bars_1["datetime"]
    rets = rets.dropna()
    print(f"Return matrix: {len(rets):,} rows")
    return (rets,)


@app.cell
def _(pd, rets):
    # PCA on correlation matrix
    from numpy.linalg import eigh

    corr = rets.corr()
    eigenvalues, eigenvectors = eigh(corr)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # First PC weights (loadings)
    weights = pd.Series(eigenvectors[:, 0], index=corr.columns)
    weights = weights / weights.abs().sum()  # normalize for display
    print("First PC weights (normalized):")
    print(weights)
    return eigenvalues, weights


@app.cell
def _(eigenvalues, go, make_subplots, weights):
    _fig = make_subplots(
        rows=1, cols=2, subplot_titles=("PCA Weights (First Principal Component)", "Eigenvalues (explained variance)")
    )
    _fig.add_trace(go.Bar(x=weights.index, y=weights.values, name="Weights", showlegend=False), row=1, col=1)
    _fig.add_trace(
        go.Bar(x=list(range(len(eigenvalues))), y=eigenvalues, name="Eigenvalues", showlegend=False), row=1, col=2
    )
    _fig.add_hline(y=0, line_color="black", row=1, col=1)
    _fig.update_layout(height=400)
    _fig.update_xaxes(title_text="Component", row=1, col=2)
    _fig.update_yaxes(title_text="Weight", row=1, col=1)
    _fig.update_yaxes(title_text="Eigenvalue", row=1, col=2)
    _fig.show()
    return


if __name__ == "__main__":
    app.run()

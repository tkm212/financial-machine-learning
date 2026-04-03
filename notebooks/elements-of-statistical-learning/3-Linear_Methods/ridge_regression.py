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
    # Ridge regression: ESL Ch. 3

    *Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning. Chapter 3.*

    Ridge adds an L2 penalty $\lambda \sum_j \beta_j^2$ to OLS, shrinking all coefficients toward zero but never exactly to zero. As $\lambda$ grows the solution becomes increasingly biased but lower variance — the bias-variance trade-off in action on TMDB revenue.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    import ch3_helpers as helpers

    _root, INPUTS, _outputs = helpers.init_paths()
    return INPUTS, helpers


@app.cell
def _(INPUTS, helpers):
    X, y, _ = helpers.load_tmdb_xy(INPUTS)
    data = helpers.scale_split(X, y)
    print(f"Train: {len(data['X_train']):,} | Test: {len(data['X_test']):,}")
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bias-variance trade-off across alpha (§3.4.1)

    Ridge solves:

    $$\hat{\beta}^{\text{ridge}} = \arg\min_\beta \|y - X\beta\|^2 + \alpha\|\beta\|^2$$

    - **Small alpha**: near-OLS — low bias, potentially high variance.
    - **Large alpha**: heavy shrinkage — high bias, low variance.

    Train MSE rises monotonically with alpha (the model is increasingly constrained). Test MSE first falls as variance is reduced, then rises as bias dominates. The dashed line marks the optimal alpha.
    """)
    return


@app.cell
def _(data, helpers):
    fig, summary = helpers.ridge_alpha_path_figure(data["X_train_s"], data["X_test_s"], data["y_train"], data["y_test"])
    fig.show()
    print(f"Best alpha: {summary['best_alpha']:.2f} | test MSE: {summary['min_test_mse']:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Coefficient shrinkage paths

    Each line traces one coefficient as alpha increases. Ridge shrinks all coefficients proportionally toward zero, but **none reaches exactly zero** — every feature remains in the model regardless of alpha. Features with the weakest signal (small OLS values) shrink fastest. Those that stay large under heavy regularisation are the most robustly correlated with revenue.
    """)
    return


@app.cell
def _(data, helpers):
    helpers.ridge_coef_figure(data["X_train_s"], data["y_train"], data["feat_names"]).show()
    return


if __name__ == "__main__":
    app.run()

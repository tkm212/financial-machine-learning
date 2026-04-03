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
    # Least squares regression: ESL Ch. 2 §2.3

    *Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning.*

    OLS finds the coefficient vector $\hat{\beta}$ that minimises the residual sum of squares:

    $$\hat{\beta} = \arg\min_\beta \|y - X\beta\|^2 = (X^\top X)^{-1} X^\top y$$

    The solution is closed-form — no tuning parameter, all features included. TMDB movie **revenue** is predicted from budget, popularity, votes, runtime, and release year (`inputs/tmdb-movie-metadata/`, download with `uv run python scripts/download_tmdb_movie_metadata.py`).
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    import helpers

    _root, INPUTS, _outputs = helpers.init_paths()
    return INPUTS, helpers


@app.cell
def _(INPUTS, helpers):
    X, y, target = helpers.load_tmdb_xy(INPUTS)
    print(f"Loaded {len(X):,} rows, {X.shape[1]} numeric features, target={target!r}")
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ordinary least squares

    The predicted vs actual scatter shows fit quality — a perfect model puts every point on the diagonal. The gap between train MSE and test MSE measures **optimism**: how much the model has overfit to the training sample. With OLS and no regularisation this gap widens as the number of features grows relative to sample size.
    """)
    return


@app.cell
def _(X, helpers, y):
    out = helpers.fit_linear_train_test_mse(X, y)
    print(
        "Train MSE:",
        out["train_mse"],
        "| Test MSE:",
        out["test_mse"],
    )

    y_pred = out["model"].predict(out["X_test"])
    helpers.plot_predicted_vs_actual(
        out["y_test"], y_pred, title="Linear regression: predicted vs actual revenue"
    ).show()
    return


if __name__ == "__main__":
    app.run()

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
    # $k$-nearest neighbors & local regression: ESL

    *Based on Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning. Chapter 2.*

    TMDB movie data (`inputs/tmdb-movie-metadata/`): predict **revenue** from budget, popularity, votes, runtime, release year. Download with `uv run python scripts/download_tmdb_movie_metadata.py`. Same data as `least_squares_regression.ipynb`. Neighbor count $k$ vs train/test MSE, then linear vs $k$-NN on one input (e.g. `budget`).
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
    return X, target, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bias-variance trade-off as a function of $k$

    The $k$-NN prediction at $x_0$ is the average of the $k$ nearest training responses:

    $$\hat{f}(x_0) = \frac{1}{k} \sum_{x_i \in \mathcal{N}_k(x_0)} y_i$$

    - **Small $k$**: low bias (the model can hug the data), high variance (sensitive to individual points).
    - **Large $k$**: high bias (over-smoothed), low variance.

    Train MSE falls monotonically as $k$ decreases; test MSE has a minimum — the sweet spot between the two extremes.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig, summary = helpers.knn_train_test_mse_figure(X, y, max_rows=40_000)
    fig.show()

    ks = summary["ks"]
    print(
        f"Best k ({int(ks.min())}..{int(ks.max())}) by test MSE: {summary['k_best']}, "
        f"test MSE = {summary['min_test_mse']:.4f}"
    )
    print(f"Linear regression test MSE: {summary['linear_test_mse']:.4f}")
    return (summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Linear vs $k$-NN on a single feature

    Projecting onto one feature (budget) makes the comparison visual. The linear fit is a global straight line; $k$-NN traces the local average of nearby training points. In sparse regions $k$-NN falls back toward the global mean, while the linear model extrapolates indefinitely — each approach has its failure mode.
    """)
    return


@app.cell
def _(helpers, summary, target):
    fig2, _feat = helpers.linear_vs_knn_single_feature_figure(summary["X_train"], summary["y_train"], target)
    fig2.show()
    return


if __name__ == "__main__":
    app.run()

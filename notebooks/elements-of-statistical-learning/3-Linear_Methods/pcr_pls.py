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
    # Principal components & partial least squares regression: ESL Ch. 3

    *Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning. §3.5.*

    Both methods replace the original features with a small set of **derived directions** before regressing.

    - **PCR (§3.5.1)**: directions are the PCA eigenvectors — maximise variance in X, ignoring y.
    - **PLS (§3.5.2)**: directions are chosen to maximise covariance with y, so each component is more directly predictive.

    In low-signal settings PLS often needs fewer components than PCR because it orients the subspace toward the response.
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
    print(f"Train: {len(data['X_train']):,} | Test: {len(data['X_test']):,} | Features: {data['feat_names']}")
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test MSE vs number of components

    Both methods project the $p$ features into $M \ll p$ directions, then run OLS in that reduced space.

    - **PCR**: directions are PCA eigenvectors — chosen to maximise variance in $X$, ignoring $y$. A high-variance direction in $X$ may have little bearing on $y$.
    - **PLS**: at each step, the direction maximises the covariance between a linear combination of $X$ and $y$, so it is directly oriented toward the response.

    In practice PLS often reaches its test-MSE minimum with fewer components than PCR, because every direction it adds is at least partially aligned with $y$. Beyond the optimal $M$, both methods start recovering noise directions and test MSE rises.
    """)
    return


@app.cell
def _(data, helpers):
    fig, summary = helpers.pcr_pls_figure(data["X_train_s"], data["X_test_s"], data["y_train"], data["y_test"])
    fig.show()
    print(
        f"Best PCR: {summary['best_pcr_n']} components | test MSE: {summary['min_pcr_mse']:.4e}\n"
        f"Best PLS: {summary['best_pls_n']} components | test MSE: {summary['min_pls_mse']:.4e}"
    )
    return (summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## All methods compared (§3.6)

    OLS, Ridge, Lasso, PCR, and PLS on the same train/test split.
    """)
    return


@app.cell
def _(data, helpers, summary):
    helpers.compare_models(
        data["X_train_s"],
        data["X_test_s"],
        data["y_train"],
        data["y_test"],
        alpha_ridge=100.0,
        alpha_lasso=1e6,
        n_pcr=summary["best_pcr_n"],
        n_pls=summary["best_pls_n"],
    )
    return


if __name__ == "__main__":
    app.run()

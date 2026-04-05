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
    # Additive Models (GAMs) — ESL Ch. 9

    *Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning. §9.1.*

    A **Generalised Additive Model** replaces the linear predictor $X^\top \beta$ with a
    sum of smooth functions, one per feature:

    $$\hat{y} = \alpha + \sum_{j=1}^p f_j(X_j)$$

    Each $f_j$ is estimated as a smooth function (spline, kernel, etc.) of a single predictor,
    while all other functions are held fixed.  This retains the interpretability of linear
    models — the **partial plot** of $f_j$ shows the isolated effect of $X_j$ — while
    allowing flexible nonlinear relationships.

    ## Backfitting algorithm (§9.1.1)

    Estimation proceeds via the **backfitting** algorithm:

    1. Initialise $\hat{\alpha} = \bar{y}$, $\hat{f}_j \equiv 0$ for all $j$.
    2. Repeat until convergence:
        - For each $j = 1, \ldots, p$:
            $$r_i^{(j)} = y_i - \hat{\alpha} - \sum_{k \neq j} \hat{f}_k(x_{ik})$$
            $$\hat{f}_j \leftarrow \mathcal{S}_j[r^{(j)}] - \frac{1}{N}\sum_i \mathcal{S}_j[r^{(j)}(x_{ij})]$$

    where $\mathcal{S}_j$ is any smoother (here cubic splines via `SplineTransformer`).
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    import ch9_helpers as helpers

    _root, INPUTS, _outputs = helpers.init_paths()
    return INPUTS, helpers


@app.cell
def _(INPUTS, helpers):
    X, y, target = helpers.load_tmdb_xy(INPUTS)
    print(f"Loaded {len(X):,} rows | target: {target!r}")
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Partial effect plots (§9.1)

    Each panel shows $\hat{f}_j(x_j)$ — the estimated contribution of one predictor to the
    response after adjusting for all others.  Steeper slopes indicate stronger marginal
    effects; nonlinearity indicates that a linear term would be insufficient.

    - **f(budget)**: typically a strong, roughly linear effect in log-space.
    - **f(popularity)**: often nonlinear — diminishing returns at high popularity.
    - **f(runtime)**: may show a non-monotone pattern (very short and very long films both
      tend to have lower revenue).
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_gam, gam_summary = helpers.gam_partial_plots_figure(X, y, feats=["budget", "popularity", "runtime"], n_knots=6)
    fig_gam.show()
    print(f"Backfitting converged in {gam_summary['n_iter']} iterations | train MSE: {gam_summary['train_mse']:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GAM vs linear model (§9.1)

    Cross-validated MSE comparison between a standard linear model and additive spline models
    of increasing degree.

    The GAM with cubic splines (degree=3) captures nonlinear feature-response relationships
    that a linear model misses, typically yielding lower CV error.  However, if the true
    relationship is near-linear in log-space, the improvement may be modest — linear models
    are already a reasonable approximation after the log₁p feature transformation.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_cmp, cmp_summary = helpers.gam_vs_linear_figure(X, y, feats=["budget", "popularity", "runtime"])
    fig_cmp.show()
    print(
        f"Best model: {cmp_summary['best_model']} | "
        f"linear MSE: {cmp_summary['linear_mse']:.4f} | "
        f"best MSE: {cmp_summary['best_mse']:.4f}"
    )
    return


if __name__ == "__main__":
    app.run()

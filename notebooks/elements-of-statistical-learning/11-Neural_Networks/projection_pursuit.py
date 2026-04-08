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
    # Projection Pursuit Regression — ESL Ch. 11 §11.2

    *Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning. §11.2.*

    **Projection Pursuit Regression** (Friedman & Stuetzle 1981) models the response as a
    sum of $M$ **ridge functions** — each a smooth nonlinear function of a single linear
    combination of inputs:

    $$f(X) = \sum_{m=1}^{M} g_m\!\left(\omega_m^\top X\right)$$

    where $\omega_m \in \mathbb{R}^p$ are unit-norm **projection directions** and $g_m$ are
    smooth **ridge functions** estimated nonparametrically.

    ## Why "projection pursuit"?

    The algorithm iteratively *pursues* projections $\omega_m^\top X$ that best reduce the
    current residuals.  Each term adds one direction of structured variation — analogous to
    principal components regression, but with nonlinear $g_m$ and learned (not PCA) directions.

    ## Relationship to neural networks (§11.3)

    A **1-hidden-layer MLP** with $M$ units and sigmoid activations is a restricted PPR:

    $$f(X) = \beta_0 + \sum_{m=1}^{M} \beta_m \sigma\!\left(\alpha_{m0} + \alpha_m^\top X\right)$$

    Here $g_m(v) = \beta_m \sigma(v)$ is constrained to be scaled sigmoid rather than an
    arbitrary smooth function.  PPR is more flexible but harder to fit; neural networks
    trade flexibility for scalability and simplicity of the backpropagation algorithm.

    ## Fitting PPR: backfitting

    At each step $m$, the algorithm:
    1. Computes partial residuals $r_i = y_i - \sum_{k < m} g_k(\omega_k^\top x_i)$.
    2. Optimises over $\omega_m$ and $g_m$ jointly to minimise $\sum_i (r_i - g_m(\omega_m^\top x_i))^2$.
    3. Fits $g_m$ by a smoothing spline on the 1D projections.

    We approximate this with 1-hidden-layer `MLPRegressor` (tanh activations) since
    scikit-learn does not provide a native PPR implementation.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    import ch11_helpers as helpers

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
    ## PPR vs OLS: cross-validated MSE as M increases (§11.2)

    We compare ordinary least squares (equivalent to $M = 0$ ridge terms — no nonlinearity)
    against 1-hidden-layer MLPs of increasing width $M \in \{1, 2, 5, 10, 20, 50\}$.

    The key question: at what $M$ does the nonlinear PPR model outperform OLS?

    - **OLS**: fits a hyperplane — zero bias only if the true function is linear in
      the (log-transformed) features.
    - **PPR M=1**: single-index model — all variation captured through one projection.
    - **PPR M large**: sufficient capacity to approximate any continuous function.

    5-fold CV MSE (log₁p-space) is used to compare fairly across model complexity.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_ppr, ppr_summary = helpers.ppr_vs_linear_figure(X, y, M_values=[1, 2, 5, 10, 20, 50])
    fig_ppr.show()
    print(
        f"Best model: {ppr_summary['best_model']} | "
        f"best MSE: {ppr_summary['best_mse']:.4f} | "
        f"OLS MSE: {ppr_summary['ols_mse']:.4f} | "
        f"improvement: {ppr_summary['ppr_improvement_pct']:.1f}%"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ridge function visualisation (§11.2)

    After fitting a PPR model with $M = 3$ terms, we plot the $M$ learned ridge functions
    $\beta_m \sigma(\omega_m^\top x)$ as functions of their projection values $\omega_m^\top x$.

    Each ridge function reveals a different direction of nonlinear variation:
    - A steep S-curve indicates the projection strongly discriminates high/low response.
    - A flat function suggests that term contributes little — the network might discard it
      with stronger regularisation.
    - The output weight $\beta_m$ (shown in the legend) scales each function's contribution
      to the final prediction.

    These plots are the PPR equivalent of GAM partial plots (§9.1): each panel shows
    the isolated, nonlinear effect of one learned projection direction.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_ridge = helpers.ppr_ridge_functions_figure(X, y, M=3)
    fig_ridge.show()
    return


if __name__ == "__main__":
    app.run()

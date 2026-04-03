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
    # Smoothing Splines: ESL Ch. 5

    *Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning. §5.4-5.5.*

    Rather than choosing a fixed set of knots, a **smoothing spline** places a knot at every unique data point and then penalises roughness via the integrated squared second derivative:

    $$\min_{f \in \mathcal{H}} \sum_{i=1}^n (y_i - f(x_i))^2 + \lambda \int [f''(t)]^2 \, dt$$

    The solution is a **natural cubic spline** with knots at the data.  The penalty $\lambda$ controls the bias-variance tradeoff:

    - $\lambda \to 0$: interpolating spline — zero bias, potentially enormous variance.
    - $\lambda \to \infty$: global linear fit — maximum bias, minimum variance.

    The **effective degrees of freedom** $\text{df}_\lambda = \text{tr}(S_\lambda)$ — the trace of the hat matrix — summarises how much freedom the model retains as $\lambda$ grows.  Automatic selection of $\lambda$ via **generalised cross-validation** (§5.5) removes the need to pick it manually.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    import ch5_helpers as helpers

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
    ## Effect of the smoothing parameter $\lambda$ (§5.4)

    scipy's `UnivariateSpline` minimises a penalised sum of squares; its `s` parameter controls the allowed residual — smaller `s` forces the spline to fit the data more tightly (less smoothing, smaller effective $\lambda$).

    The five curves range from a global linear fit (very large `s`) down to interpolation (`s = 0`).  In between lies the sweet spot where the spline captures real structure without chasing noise.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_lambda = helpers.smoothing_spline_lambda_figure(X, y, feat="budget")
    fig_lambda.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Effective degrees of freedom (§5.4.1)

    Every smoothing spline can be written as $\hat{f} = S_\lambda y$ where $S_\lambda$ is the $n \times n$ smoother matrix.  The **effective degrees of freedom** $\text{df}_\lambda = \text{tr}(S_\lambda)$ measures model complexity:

    $$\text{df}_\lambda \in [2,\ n]$$

    For fixed knots and linear fit $\text{df} = 2$; for the interpolating spline $\text{df} = n$.  Choosing $\lambda$ to target a specific $\text{df}$ is a convenient alternative to cross-validation and makes comparisons across datasets easier (§5.4.1).

    Below each curve is labelled by its approximate $\text{df}$, estimated from the number of interior knots selected by the spline.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_df = helpers.smoothing_spline_df_figure(X, y, feat="budget", df_values=[4, 6, 9, 15])
    fig_df.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bias-variance tradeoff (§5.5.2)

    To make the bias-variance curve precise, we approximate the smoothing spline as a ridge-penalised B-spline with many basis functions.  The ridge penalty $\alpha$ plays the role of $\lambda$:

    $$\min_\beta \|y - B\beta\|^2 + \alpha \|\beta\|^2$$

    As $\alpha$ increases: train MSE rises monotonically (the model is increasingly constrained). Test MSE first falls as variance is controlled, then rises as bias dominates. The optimal $\alpha$ (dashed line) minimises test MSE.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_bv, bv_summary = helpers.smoothing_spline_bias_variance_figure(X, y, feat="budget")
    fig_bv.show()
    print(f"Best lambda: 10^{bv_summary['best_alpha']:.2e} | min test MSE: {bv_summary['min_test_mse']:.4e}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Automatic selection via GCV (§5.5)

    **Generalised cross-validation** approximates leave-one-out CV without refitting the model $n$ times:

    $$\text{GCV}(\lambda) = \frac{1}{n} \sum_{i=1}^n \left[\frac{y_i - \hat{f}_\lambda(x_i)}{1 - S_{\lambda,ii}}\right]^2 \approx \frac{\text{RSS}(\lambda)/n}{\left(1 - \text{df}(\lambda)/n\right)^2}$$

    The approximation (right-hand side) replaces the diagonal hat-matrix entries with their average $\text{df}/n$.  Minimising GCV over a grid of $\lambda$ values selects the smoothing parameter without a held-out set — it uses all $n$ observations for both fitting and evaluation.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_gcv, gcv_summary = helpers.gcv_lambda_figure(X, y, feat="budget")
    fig_gcv.show()
    print(f"GCV-optimal lambda: {gcv_summary['best_alpha']:.2e} | GCV score: {gcv_summary['min_gcv']:.4e}")
    return


if __name__ == "__main__":
    app.run()

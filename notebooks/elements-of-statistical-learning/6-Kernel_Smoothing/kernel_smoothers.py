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
    # Kernel Smoothing Methods: ESL Ch. 6

    *Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning. §6.1-6.2.*

    **Kernel smoothers** estimate $f(x_0)$ by fitting a locally-weighted model in a neighbourhood of $x_0$.  The neighbourhood is defined by a kernel $K_\lambda$ that assigns weight to each observation based on its distance from $x_0$.

    The **Nadaraya-Watson** estimator (§6.1) is the simplest — it takes a locally-weighted average:

    $$\hat{f}(x_0) = \frac{\sum_{i=1}^n K_\lambda(x_0, x_i) \, y_i}{\sum_{i=1}^n K_\lambda(x_0, x_i)}$$

    **Local polynomial regression** (§6.1.2) generalises this by fitting a degree-$d$ polynomial at each query point via weighted least squares:

    $$\min_{\alpha, \beta} \sum_{i=1}^n K_\lambda(x_0, x_i) \left[y_i - \alpha(x_0) - \sum_{j=1}^d \beta_j(x_0)(x_i - x_0)^j\right]^2$$

    Degree 0 recovers NW; degree 1 gives **local linear regression** (§6.1.1), which provably reduces boundary bias.

    The **bandwidth** $\lambda$ controls the bias-variance tradeoff.  Section 6.2 shows how to select it via leave-one-out cross-validation, exploiting the linear smoother structure for computational efficiency.

    All visualisations use TMDB movie budget as the single predictor for revenue (log₁p-transformed for stability).
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    import ch6_helpers as helpers

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
    ## Nadaraya-Watson estimator (§6.1)

    The Gaussian kernel $K_\lambda(x_0, x) = \phi(|x - x_0|/\lambda)$ assigns weights that decay smoothly with distance.  The bandwidth $\lambda$ directly controls the amount of smoothing:

    - **Very small** $\lambda$: the smoother only averages a few nearby points — low bias but high variance (wiggly curve).
    - **Very large** $\lambda$: nearly all points receive equal weight — the estimate converges to the global mean (high bias, near-zero variance).

    Both axes are in **log₁p-space** (the space in which the fit is computed).  This makes the scatter data legible across the full range and shows where each bandwidth begins to under- or over-smooth.  Plotting in original (linear) scale compresses all data near the origin and makes every curve look like the same power-law shape, hiding the differences that matter.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_nw = helpers.nadaraya_watson_figure(X, y, feat="budget")
    fig_nw.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Local linear vs Nadaraya-Watson: boundary bias (§6.1.1)

    The NW estimator is equivalent to fitting a **degree-0 local polynomial** (local constant).  At the boundaries of the data, the kernel window extends beyond the observed range, and NW systematically pulls estimates toward zero — this is **boundary bias**.

    **Local linear regression** (degree-1 local polynomial) corrects this.  At each $x_0$ it fits a locally-weighted line rather than a constant, so the estimate extrapolates naturally instead of shrinking.  The highlighted regions mark the lower and upper 5% of the feature range where boundary effects are strongest.

    The correction matters most when the true function has non-zero slope at the boundaries — exactly the case for budget-revenue, where revenue rises steeply at the low end.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_ll = helpers.local_linear_vs_nw_figure(X, y, feat="budget", bw=0.5)
    fig_ll.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Local polynomial regression: comparing degrees (§6.1.2)

    Fitting a degree-$d$ polynomial locally generalises the bias-variance tradeoff within each neighbourhood:

    | Degree | Name | Boundary bias | Asymptotic bias order |
    |---|---|---|---|
    | 0 | Nadaraya-Watson | Yes | $O(\lambda^2)$ |
    | 1 | Local linear | No | $O(\lambda^2)$ |
    | 2 | Local quadratic | No | $O(\lambda^4)$ |
    | 3 | Local cubic | No | $O(\lambda^4)$ |

    Odd-degree fits reduce boundary bias one order beyond even-degree fits of the same order.  In practice, local linear (degree 1) is the most common choice — it eliminates boundary bias at minimal cost in variance.

    Plotted in **log₁p-space** with a narrower bandwidth ($h=0.3$) so the curves separate.  With a wide bandwidth or in back-transformed (power-law) scale, all degrees collapse onto essentially the same smooth curve because the underlying log-log relationship is already close to linear.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_poly = helpers.local_poly_figure(X, y, feat="budget", bw=0.3, degrees=[0, 1, 2, 3])
    fig_poly.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bandwidth selection via LOO cross-validation (§6.2)

    Because the NW estimator is a **linear smoother** ($\hat{y} = S_\lambda y$), the leave-one-out prediction can be computed without re-fitting $n$ times.  The diagonal hat-matrix entries are:

    $$S_{\lambda,ii} = \frac{K_\lambda(0)}{\sum_j K_\lambda(x_i - x_j)}$$

    and the LOO residual shortcut gives:

    $$\text{CV}(\lambda) = \frac{1}{n} \sum_{i=1}^n \left[\frac{y_i - \hat{f}_\lambda(x_i)}{1 - S_{\lambda,ii}}\right]^2$$

    The CV curve has a clear minimum — the bandwidth that achieves the best predictive balance.  Left of the minimum, the smoother is too wiggly (high variance); right of it, too flat (high bias).
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_cv, cv_summary = helpers.bandwidth_loocv_figure(X, y, feat="budget")
    fig_cv.show()
    print(f"CV-optimal bandwidth: {cv_summary['best_bw']:.4f} | min LOO-CV score: {cv_summary['min_cv']:.4e}")
    return


if __name__ == "__main__":
    app.run()

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
    # Model Inference: Bootstrap and Bagging — ESL Ch. 8

    *Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning. §8.2, §8.7.*

    ## Bootstrap (§8.2)

    The **bootstrap** estimates the sampling distribution of a statistic by drawing $B$ resamples
    of size $N$ with replacement from the observed data.  For a prediction function $\hat{f}$
    fit on training set $\mathcal{T}$, the bootstrap gives us $B$ perturbed versions
    $\hat{f}^{*b}$, $b = 1, \ldots, B$, each fit on a bootstrap resample $\mathcal{T}^{*b}$.

    **Pointwise confidence bands** can then be constructed as quantiles across the $B$ bootstrap
    predictions at each input point $x_0$:

    $$[\hat{f}^{[\alpha/2]}(x_0),\; \hat{f}^{[1 - \alpha/2]}(x_0)]$$

    ## Bagging (§8.7)

    **Bootstrap AGGregatING** (bagging) uses the same bootstrap resamples to build an ensemble:

    $$\hat{f}_{\text{bag}}(x) = \frac{1}{B} \sum_{b=1}^B \hat{f}^{*b}(x)$$

    Averaging over resamples reduces the variance of the individual predictor without changing
    its bias.  For a base learner with variance $\sigma^2$ and zero correlation between
    resamples, the ensemble variance is $\sigma^2 / B$.  In practice the resamples are
    correlated (they share ~63% of training points) so the reduction is smaller but still
    substantial — particularly for high-variance learners like deep decision trees.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    import ch8_helpers as helpers

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
    ## Bootstrap confidence bands (§8.2)

    We fit a degree-3 polynomial of log₁p(budget) → log₁p(revenue) and bootstrap
    $B = 200$ times to obtain pointwise 95% confidence bands.

    The shaded band shows the uncertainty due to sampling variability — it widens in regions
    where data are sparse (large log₁p(budget)) because fewer points constrain the fit there.
    This directly illustrates the bootstrap as a way to "plug in" the empirical distribution
    in place of the unknown population distribution (§8.2.1).
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_ci = helpers.bootstrap_confidence_bands_figure(X, y, feat="budget", degree=3, n_boot=200)
    fig_ci.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bagging variance reduction (§8.7)

    A single decision tree is a **high-variance** estimator: small changes in the training data
    produce very different trees (and predictions).  Bagging averages over $B$ such trees, each
    fit on a bootstrap resample.

    The plot shows test MSE as a function of ensemble size $B$, averaged over 10 random
    train/test splits to reduce noise.  The red dashed line marks the single-tree baseline.

    Key observations:
    - Test MSE **drops sharply** as $B$ increases from 1 to ~10 — most of the variance reduction
      occurs early.
    - Beyond $B \approx 20$-30, gains become negligible — the bootstrap correlation floor limits
      further improvement.
    - The bagged ensemble **never performs worse** than the single tree on average — bagging
      cannot increase expected test error.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_bag, bag_summary = helpers.bagging_figure(X, y, feat="budget", max_bags=50, tree_depth=5)
    fig_bag.show()
    print(
        f"Single tree MSE: {bag_summary['single_tree_mse']:.4f} | "
        f"Best bagged ({bag_summary['best_b']} trees): {bag_summary['best_bagged_mse']:.4f}"
    )
    return


if __name__ == "__main__":
    app.run()

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
    # Kernel Density Estimation and Classification: ESL Ch. 6

    *Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning. §6.6.*

    **Kernel density estimation** (KDE) is a non-parametric method to estimate the probability density function of a random variable from samples.  Given observations $x_1, \ldots, x_n$ the KDE is:

    $$\hat{f}(x) = \frac{1}{n} \sum_{i=1}^n K_\lambda(x, x_i) = \frac{1}{n\lambda} \sum_{i=1}^n K\!\left(\frac{x - x_i}{\lambda}\right)$$

    Each observation "contributes" a kernel bump of width $\lambda$ centred at $x_i$; the estimate is the average of these bumps.  As with regression smoothers:

    - Small $\lambda$ → spiky density that interpolates the data.
    - Large $\lambda$ → over-smoothed density that misses structure.

    **Kernel density classification** (§6.6.2) uses Bayes' rule with class-conditional KDEs:

    $$\hat{P}(Y = k \mid X = x) \propto \hat{\pi}_k \, \hat{f}_k(x)$$

    The **Naive Bayes** classifier (§6.6.3) applies this idea feature-by-feature, assuming conditional independence of inputs given the class label.
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
    X_reg, y_reg, target_reg = helpers.load_tmdb_xy(INPUTS)
    print(f"Regression data: {len(X_reg):,} rows | target: {target_reg!r}")
    return X_reg, y_reg


@app.cell
def _(INPUTS, helpers):
    X_cls, y_cls, target_cls = helpers.load_tmdb_cls(INPUTS)
    print(f"Classification data: {len(X_cls):,} rows | target: {target_cls!r} | classes: {sorted(y_cls.unique())}")
    return X_cls, y_cls


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Kernel density estimation of movie budget (§6.6.1)

    The rug plot at the bottom marks individual data points (subsampled for clarity).  The KDE curves show how the estimated density changes with bandwidth $\lambda$:

    - **h = 0.05**: highly irregular — the estimator essentially memorises the data.
    - **h = 0.15**: reveals the bimodal structure of the budget distribution (many low-budget films, a cluster of big productions).
    - **h = 0.4**: the two modes begin to merge; the curve is much smoother.
    - **h = 1.0**: nearly Gaussian — most distributional detail is lost.

    The densities are estimated in log₁p-space (where the data is more symmetric), so the x-axis is back-transformed to the original dollar scale for interpretability.
    """)
    return


@app.cell
def _(X_reg, helpers):
    fig_kde = helpers.kde_figure(X_reg, feat="budget", bandwidths=[0.05, 0.15, 0.4, 1.0])
    fig_kde.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Naive Bayes: class-conditional densities (§6.6.3)

    For binary classification (high vs low revenue), the Naive Bayes classifier estimates a separate KDE $\hat{f}_k(x)$ for each class $k \in \{0, 1\}$.  Classification uses the posterior:

    $$\hat{P}(Y=1 \mid x) = \frac{\hat{\pi}_1 \hat{f}_1(x)}{\hat{\pi}_0 \hat{f}_0(x) + \hat{\pi}_1 \hat{f}_1(x)}$$

    The **dashed black curve** (right axis) shows this posterior — it rises steeply once budget exceeds a threshold, reflecting that high-budget films are more likely to generate high revenue.

    Note that the class-conditional densities overlap substantially in the low-budget region, consistent with the noisy budget-revenue relationship there.  Despite its simplicity, Naive Bayes often performs well when class-conditional densities are well-separated.
    """)
    return


@app.cell
def _(X_cls, helpers, y_cls):
    fig_nb, nb_summary = helpers.naive_bayes_figure(X_cls, y_cls, feat="budget", bw=0.3)
    fig_nb.show()
    print(f"Classes: {nb_summary['classes']} | Priors: {nb_summary['priors']}")
    return


if __name__ == "__main__":
    app.run()

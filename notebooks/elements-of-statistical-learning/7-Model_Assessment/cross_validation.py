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
    # Model Selection: Cp, AIC, BIC, Cross-Validation and Bootstrap — ESL Ch. 7

    *Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning. §7.5-7.11.*

    When we cannot afford a large hold-out test set, we need **analytic or resampling-based** estimates of prediction error to select among competing models.

    **In-sample estimates** (§7.5-7.7) correct the training error by adding a penalty for model complexity:

    | Criterion | Formula | Penalty |
    |---|---|---|
    | Mallows' $C_p$ | $\overline{err} + 2d\hat\sigma^2/N$ | linear in $d$ |
    | AIC | $N\log(\overline{err}) + 2d$ | linear in $d$ |
    | BIC | $N\log(\overline{err}) + d\log N$ | log-linear in $d$ |

    BIC grows faster with $d$ than AIC, so it more strongly penalises complexity and tends to select sparser models.

    **Cross-validation** (§7.10) directly estimates test error by rotating through $K$ held-out folds.  It makes minimal distributional assumptions and is widely applicable.

    **Bootstrap** (§7.11) resamples the training set with replacement.  The **.632 estimator** corrects the upward bias of out-of-bag error:

    $$\widehat{\text{Err}}^{.632} = 0.368 \cdot \overline{err} + 0.632 \cdot \widehat{\text{Err}}^{(1)}$$

    where $\widehat{\text{Err}}^{(1)}$ is the average OOB prediction error across all bootstrap samples.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    import ch7_helpers as helpers

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
    ## Mallows' Cₚ, AIC and BIC (§7.5-7.7)

    All three criteria add a complexity penalty to the training RSS.  They are plotted here **normalised** to [0, 1] so their shapes can be compared directly.

    Key differences:
    - **Cₚ** requires an estimate of $\sigma^2$ from a reference model — sensitive to that estimate.
    - **AIC** is derived from maximum likelihood and uses the log of the training RSS.
    - **BIC** has a $\log N$ penalty instead of a constant 2, so it grows faster with $d$ and selects **simpler** models as $N$ increases.

    All three will generally agree for moderate $N$ and well-separated model errors, but BIC's stronger penalty can select a lower degree, especially with large datasets.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_ic, ic_summary = helpers.model_selection_criteria_figure(X, y, feat="budget", max_degree=12)
    fig_ic.show()
    print(
        f"Best degree — Cₚ: {ic_summary['best_cp_d']} | AIC: {ic_summary['best_aic_d']} | BIC: {ic_summary['best_bic_d']}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## K-fold cross-validation (§7.10.1)

    $K$-fold CV partitions the data into $K$ equal folds, trains on $K-1$ folds, and tests on the held-out fold — repeated $K$ times.  The CV estimate of test error is:

    $$\text{CV}(K) = \frac{1}{K} \sum_{k=1}^K \text{Err}_k$$

    Trade-offs:
    - **Large $K$** (towards LOO): low bias (each model trained on nearly all data), high variance (test sets are small and correlated).
    - **Small $K$** ($K=5$): higher bias but lower variance; faster to compute.

    Comparing $K=5$ and $K=10$ shows how the selected model degree can differ slightly depending on $K$ — particularly in the flat region around the minimum where multiple degrees have similar error.  The training error (grey dashed) is included to visualise the optimism gap.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_cv, cv_summary = helpers.kfold_cv_figure(X, y, feat="budget", max_degree=12, k_values=[5, 10])
    fig_cv.show()
    print(f"Best degree by CV: {cv_summary['best_degrees']}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bootstrap .632 estimator (§7.11)

    In each bootstrap iteration, approximately $1 - e^{-1} \approx 63.2\%$ of training observations appear in the bootstrap resample, leaving ~37% as **out-of-bag (OOB)** observations.

    - **Train MSE** (blue): underestimates test error — always optimistically biased.
    - **OOB (Err⁽¹⁾)** (orange): trained on ~63% of data, so it **overestimates** test error for a full-$N$ fit.
    - **.632 bootstrap** (green): the weighted combination $0.368 \cdot \overline{err} + 0.632 \cdot \widehat{\text{Err}}^{(1)}$ balances these two biases, providing a better-calibrated estimate.

    For very complex models (high polynomial degree), the OOB error can explode as the model extrapolates wildly to unseen points — the .632 estimator dampens this via the training error weight.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_boot, boot_summary = helpers.bootstrap_632_figure(X, y, feat="budget", max_degree=8, n_boot=50)
    fig_boot.show()
    print(f".632 best degree: {boot_summary['best_632_d']} | min .632 error: {boot_summary['min_632']:.4e}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

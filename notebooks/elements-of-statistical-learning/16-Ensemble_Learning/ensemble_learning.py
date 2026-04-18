import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ensemble Learning — ESL Ch. 16

    *Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning. §16.1-16.2.*

    Chapter 16 synthesises **bagging**, **random forests**, and **boosting** under one view:
    ensembles reduce variance (bagging, RF) or bias (boosting) by combining many learners.

    ## Stacking (§16.2)

    **Stacking** (Wolpert 1992) learns a **second-level** model on the outputs of
    level-0 models.  Let $Z_m(x)$ be the prediction of model $m$ at $x$.  The meta-model
    $g$ combines these:

    $$\hat{g}(x) = h\bigl(Z_1(x), \ldots, Z_M(x)\bigr)$$

    In practice, level-0 predictions on each training fold are used to fit $h$ so the
    meta-learner does not see in-sample base predictions (stacking with cross-validation).

    This is strictly more flexible than **voting**, which uses fixed weights (often uniform).

    ## Voting ensembles (§16.1)

    **Majority vote** (classification) or **average** (regression) combines base models
    with equal weight.  **Soft voting** averages predicted class probabilities — usually
    preferable when calibrated probabilities are available.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    import ch16_helpers as helpers

    _root, INPUTS, _outputs = helpers.init_paths()
    return INPUTS, helpers


@app.cell
def _(INPUTS, helpers):
    X, y, target = helpers.load_tmdb_classification_xy(INPUTS)
    print(f"Loaded {len(X):,} rows | target: {target!r} | class balance: {y.mean():.2%} positive")
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Stacking vs soft voting vs base learners (§16.2)

    We fit three diverse level-0 classifiers — **logistic regression**, **random forest**,
    and **gradient boosting** — then compare:

    - **Soft voting**: average of predicted probabilities (uniform weights).
    - **Stacking**: logistic regression meta-learner on out-of-fold predicted probabilities
      from each base model (`stack_method='predict_proba'`).

    Expected pattern: stacking often matches or slightly beats voting when bases make
    complementary errors; gains are modest when bases are highly correlated.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_ens, ens_summary = helpers.ensemble_comparison_figure(X, y, n_cv=5)
    fig_ens.show()
    print(f"Best method: {ens_summary['best_method']} | CV accuracy: {ens_summary['best_cv_accuracy']:.3%}")
    for name, acc in ens_summary["results"].items():
        print(f"  {name:<35}: {acc:.3%}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Meta-learner regularisation (§16.2)

    The meta-features $(Z_1(x), \ldots, Z_M(x))$ can be nearly collinear when base models
    are similar.  A small **L2 penalty** on the meta-logistic regression (`C < 1`) can
    stabilise weights; very strong penalty pulls toward uniform blending.

    Here we swap one base for a **calibrated linear SVM** to add a linear decision boundary
    distinct from tree ensembles, then sweep the meta-learner's `C`.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_meta, meta_summary = helpers.meta_learner_sweep_figure(X, y, n_cv=5)
    fig_meta.show()
    print(f"Best meta setting: {meta_summary['best_meta']} | CV accuracy: {meta_summary['best_cv_accuracy']:.3%}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Diversity of base errors (§16.1)

    For each base learner we form **out-of-fold** class predictions and indicator mistakes
    $e_m(i) = \mathbf{1}[\hat{y}_m(i) \neq y_i]$.  The correlation matrix of
    $(e_1, \ldots, e_M)$ measures whether models fail on the **same** examples.

    Lower off-diagonal correlation implies more **complementary** errors — the regime where
    ensembles and stacking have the most to gain over any single model.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_div, div_info = helpers.base_error_correlation_figure(X, y, n_cv=5)
    fig_div.show()
    print(f"Mean |off-diagonal| correlation of mistake indicators: {div_info['mean_abs_offdiag']:.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regression: voting vs stacking (§16.2)

    The same ensemble ideas apply to **regression**: `VotingRegressor` averages base
    predictions; `StackingRegressor` learns a ridge meta-model on out-of-fold base outputs.
    Target: TMDB **log revenue** vs the usual numeric features (log-scaled).
    """)
    return


@app.cell
def _(INPUTS, helpers):
    X_reg, y_reg, target_reg = helpers.load_tmdb_regression_xy(INPUTS)
    print(f"Regression sample: {len(X_reg):,} rows | target: {target_reg!r}")
    return X_reg, y_reg


@app.cell
def _(X_reg, helpers, y_reg):
    fig_reg, reg_stack = helpers.regression_stacking_figure(X_reg, y_reg, n_cv=5)
    fig_reg.show()
    print(f"Best: {reg_stack['best_method']} | CV R² = {reg_stack['best_r2']:.4f}")
    for name, r2 in reg_stack["results"].items():
        print(f"  {name:<28}: {r2:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Relation to earlier chapters

    | Idea | ESL reference | Role in ensembles |
    |---|---|---|
    | Bagging / RF | Ch. 8, 15 | Level-0 variance reduction; decorrelated trees |
    | Boosting | Ch. 10 | Sequential bias reduction; different error profile than bagging |
    | Stacking | Ch. 16 | Learned combination of level-0 models |

    Stacking does not replace diversity among bases — it assumes you already trained
    somewhat different models; the meta-learner then reweights their mistakes.
    """)
    return


if __name__ == "__main__":
    app.run()

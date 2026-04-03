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
    # Logistic Regression: ESL Ch. 4

    *Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning. §4.4.*

    Logistic regression models the log-odds of class membership as a linear function of the inputs:

    $$\log \frac{p(x;\theta)}{1-p(x;\theta)} = \beta_0 + \beta^T x$$

    The log-likelihood is concave and is maximised via **iteratively reweighted least squares** (IRLS, §4.4.1), which is Newton-Raphson applied to the log-likelihood.  An L1 penalty (§4.4.4) drives some $\beta_j$ to exactly zero — exactly like the Lasso in regression — enabling automatic feature selection alongside classification.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    import ch4_helpers as helpers

    _root, INPUTS, _outputs = helpers.init_paths()
    return INPUTS, helpers


@app.cell
def _(INPUTS, helpers):
    X, y, target = helpers.load_tmdb_xy(INPUTS)
    data = helpers.scale_split(X, y)
    print(f"Train: {len(data['X_train']):,} | Test: {len(data['X_test']):,} | features: {data['feat_names']}")
    return data, target


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## L1 Regularised Logistic Regression: coefficient paths (§4.4.4)

    Adding an L1 penalty $\lambda\|\beta\|_1$ to the negative log-likelihood shrinks coefficients and — like the Lasso — creates exact zeros.  The regularisation strength is parameterised as $C = 1/\lambda$:

    - **Small $C$ (left)**: heavy penalty, most coefficients zero — a sparse, high-bias model.
    - **Large $C$ (right)**: near-unregularised — all features active, possibly high variance.

    The path shows which movie features survive as the penalty is relaxed.  Features that enter early are the most robustly predictive of revenue class.
    """)
    return


@app.cell
def _(data, helpers):
    fig_paths = helpers.logistic_l1_coef_path_figure(data["X_train_s"], data["y_train"], data["feat_names"])
    fig_paths.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## L1 vs L2: accuracy across the regularisation path (§4.4.4)

    Both penalties shrink coefficients, but L2 keeps all features in the model (coefficients approach zero asymptotically) while L1 zeros them out exactly.

    - For **L2** (ridge-logistic) the optimal $C$ is where bias-variance is balanced: too small → high bias, too large → high variance.
    - For **L1** (lasso-logistic) the optimal $C$ also drives model sparsity — the best point is often sparser than the L2 optimum.

    On TMDB revenue prediction the two penalties achieve similar peak accuracy, illustrating ESL §4.4.5: logistic regression is flexible enough that the choice of penalty matters more for interpretability than raw accuracy.
    """)
    return


@app.cell
def _(data, helpers):
    fig_acc, acc_summary = helpers.logistic_l1_vs_l2_accuracy_figure(
        data["X_train_s"], data["X_test_s"], data["y_train"], data["y_test"]
    )
    fig_acc.show()
    print(
        f"Best L1: C={acc_summary['best_l1_C']:.3f}, accuracy={acc_summary['best_l1_acc']:.4f}\n"
        f"Best L2: C={acc_summary['best_l2_C']:.3f}, accuracy={acc_summary['best_l2_acc']:.4f}"
    )
    return


if __name__ == "__main__":
    app.run()

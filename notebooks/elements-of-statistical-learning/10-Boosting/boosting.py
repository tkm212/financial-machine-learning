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
    # Boosting and Additive Trees: AdaBoost — ESL Ch. 10

    *Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning. §10.1-10.4.*

    **AdaBoost.M1** (Freund & Schapire 1997) is the canonical boosting algorithm.  It builds
    a sequence of weak classifiers $G_m(x)$ by iteratively reweighting observations —
    misclassified examples receive higher weight in the next round:

    1. Initialise $w_i = 1/N$.
    2. For $m = 1, \ldots, M$:
        - Fit $G_m$ to the weighted training data.
        - Compute weighted error $\text{err}_m = \sum_i w_i \mathbf{1}[y_i \neq G_m(x_i)] / \sum_i w_i$.
        - $\alpha_m = \log\!\left(\frac{1 - \text{err}_m}{\text{err}_m}\right)$.
        - Update $w_i \leftarrow w_i \exp\!\bigl[\alpha_m \mathbf{1}[y_i \neq G_m(x_i)]\bigr]$.
    3. Output $C(x) = \operatorname{sign}\!\bigl[\sum_m \alpha_m G_m(x)\bigr]$.

    ## AdaBoost as forward stagewise additive modelling (§10.2-10.3)

    ESL shows that AdaBoost is equivalent to **forward stagewise additive modelling** with
    the **exponential loss** $L(y, F) = e^{-yF}$.  At each step, we add one basis function
    (weak learner) that best reduces the exponential loss on the current residuals.

    This connection to a specific loss function reveals both strengths (resistance to
    underfitting over rounds — exponential loss penalises negative margins exponentially)
    and weaknesses (sensitivity to outliers — unlike log-loss, the gradient never saturates).
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    import ch10_helpers as helpers

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
    ## AdaBoost training curve (§10.1)

    Train and test error rates are evaluated at every boosting round using the staged predict
    API.  The base learner is a **decision stump** (depth-1 tree) — the weakest possible
    classifier that is better than random guessing.

    Key observations:
    - **Training error** decreases monotonically — AdaBoost can drive train error to zero
      given enough rounds (it directly minimises exponential loss on training data).
    - **Test error** initially decreases rapidly then flattens.  Unlike test MSE in regression,
      boosting often continues to improve test error after train error reaches zero
      — a puzzle ESL explains via the margin theory (§10.4).
    - The vertical dashed line marks the round of minimum test error.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_ada, ada_summary = helpers.adaboost_training_curve_figure(X, y, n_estimators=200)
    fig_ada.show()
    print(
        f"Best round: {ada_summary['best_round']} | "
        f"test error: {ada_summary['best_test_error']:.3%} | "
        f"final train error: {ada_summary['final_train_error']:.3%}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Margin distribution (§10.4, §10.6)

    The **margin** of an observation is $y_i F(x_i) / \sum_m |\alpha_m|$ — positive for
    correct classifications, negative for errors.  A larger positive margin indicates more
    confident correct classification.

    The connection to exponential loss (§10.4):
    $$L(y, F) = e^{-yF} \implies \frac{\partial L}{\partial F} = -y e^{-yF}$$

    Observations with small or negative margins (near the decision boundary or misclassified)
    receive exponentially large gradients, causing AdaBoost to focus heavily on hard examples.

    As more rounds are added, the margin distribution shifts **rightward** — AdaBoost pushes
    examples away from the decision boundary even after train error reaches zero.  This
    explains why test error can continue decreasing long after training error saturates.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_margin = helpers.margin_distribution_figure(X, y, n_estimators_list=[10, 50, 200])
    fig_margin.show()
    return


if __name__ == "__main__":
    app.run()

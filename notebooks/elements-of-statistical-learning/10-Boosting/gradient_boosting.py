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
    # Boosting and Additive Trees: Gradient Boosting — ESL Ch. 10

    *Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning. §10.9-10.13.*

    **Gradient Boosting** (Friedman 2001) generalises AdaBoost to arbitrary differentiable
    loss functions by framing boosting as **gradient descent in function space**.

    At each step $m$, we fit a tree $h(x; a_m)$ to the **negative gradient** (pseudo-residuals)
    of the loss with respect to the current prediction $F_{m-1}(x)$:

    $$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F = F_{m-1}}$$

    For squared-error loss $L = \frac{1}{2}(y - F)^2$, these are just ordinary residuals
    $r_{im} = y_i - F_{m-1}(x_i)$.

    ## Regularisation (§10.12)

    Two key regularisers prevent overfitting:

    | Technique | Parameter | Effect |
    |---|---|---|
    | **Shrinkage** (learning rate) | $\nu \in (0, 1]$ | Scales each tree: $F_m = F_{m-1} + \nu \gamma_m h_m$ |
    | **Tree depth** | `max_depth` | Controls interaction order; depth-1 = main effects only |

    Smaller $\nu$ requires more trees but typically gives better generalisation:
    the regularisation path is smoother and the optimal $M$ is easier to find.

    ## Variable importance (§10.13)

    The importance of feature $j$ is:
    $$\hat{\mathcal{J}}_j^2 = \frac{1}{M} \sum_{m=1}^M \sum_{t=1}^{J_m - 1} \hat{i}_t^2 \cdot \mathbf{1}[v(t) = j]$$

    where $\hat{i}_t^2$ is the squared improvement in split criterion at node $t$ and $v(t)$ is
    the splitting variable.  This is averaged over all $M$ trees.
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
    X, y, target = helpers.load_tmdb_xy(INPUTS)
    print(f"Loaded {len(X):,} rows | target: {target!r}")
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GBM: number of trees vs train/test MSE (§10.9)

    We fit a `GradientBoostingRegressor` on TMDB features and track train/test MSE at each
    boosting stage via `staged_predict`.

    - **Training MSE** decreases monotonically — each additional tree reduces the in-sample
      loss.
    - **Test MSE** follows a U-shaped curve: the model improves initially then overfits as
      trees start to fit training noise.
    - The vertical dashed line marks the optimal number of trees $M^*$ on the test set.
      In practice, $M^*$ is estimated by cross-validation or early stopping on a validation set.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_gbm, gbm_summary = helpers.gbm_n_estimators_figure(X, y, n_estimators=200, learning_rate=0.1, max_depth=3)
    fig_gbm.show()
    print(f"Best M: {gbm_summary['best_round']} | best test MSE: {gbm_summary['best_test_mse']:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Shrinkage: effect of learning rate (§10.12)

    Three learning rates are compared: $\nu \in \{0.5, 0.1, 0.01\}$.

    - **Large $\nu$ (0.5)**: each tree makes a large step — the test error valley is sharp and
      narrow; overfitting occurs quickly.
    - **Small $\nu$ (0.01)**: the path is smooth and the optimal $M$ is much larger but the
      minimum test error is typically lower — more conservative steps explore the loss surface
      more thoroughly.
    - **$\nu = 0.1$**: a common default that balances computation and accuracy.

    The practical implication: **always use small $\nu$ and select $M$ by cross-validation** (§10.12.1).
    The additional computation (more trees) is usually worthwhile.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_shrink = helpers.gbm_shrinkage_figure(X, y, n_estimators=300, learning_rates=[0.5, 0.1, 0.01], max_depth=3)
    fig_shrink.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Variable importance (§10.13)

    Feature importances from the fitted GBM show which predictors are most influential
    for predicting log-revenue.

    - **Budget** typically dominates: higher production budgets strongly predict revenue.
    - **Popularity** captures social media buzz and marketing reach.
    - **Vote count / vote average** are proxies for critical reception and audience engagement.
    - **Runtime** is often a weaker predictor once budget and popularity are accounted for.

    Importantly, these importances capture **nonlinear and interaction effects** (since
    tree splits can exploit them), unlike linear regression coefficients.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_imp, imp_summary = helpers.gbm_feature_importance_figure(X, y, n_estimators=200, learning_rate=0.1, max_depth=3)
    fig_imp.show()
    print(f"Top feature: {imp_summary['top_feature']} (importance: {imp_summary['top_importance']:.3f})")
    return


if __name__ == "__main__":
    app.run()

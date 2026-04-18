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
    # Random Forests — ESL Ch. 15

    *Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning. §15.1-15.4.*

    ## Definition of random forests (§15.2)

    A **random forest** is an ensemble of $B$ decorrelated decision trees.
    Each tree $T_b$ is grown on a bootstrap sample $Z^{*b}$ of the training data.
    At each split, only $m$ randomly chosen features are considered as split candidates.

    The ensemble prediction is:
    $$\hat{f}_{RF}(x) = \frac{1}{B} \sum_{b=1}^B T_b(x)$$

    The key innovation over bagging is the **random feature subsampling**: using only
    $m < p$ features at each split reduces the correlation between trees, which is
    essential for variance reduction.

    ## Variance reduction through decorrelation (§15.4.1)

    For $B$ identically distributed models with pairwise correlation $\rho$ and
    variance $\sigma^2$, the variance of the average is:

    $$\text{Var}\left(\frac{1}{B}\sum_b T_b(x)\right) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$$

    As $B \to \infty$, this approaches $\rho \sigma^2$.  Reducing $\rho$ (via random
    feature selection) is therefore the primary lever for variance reduction —
    not just increasing $B$.

    Breiman's bound (§15.4.1):

    $$PE^* \leq \bar{\rho} \cdot \frac{1 - s^2}{s^2}$$

    where $\bar{\rho}$ is mean pairwise tree correlation and $s$ is mean tree strength
    (accuracy measured by the margin).  Random forests improve on bagging by reducing
    $\bar{\rho}$ at the cost of slightly lower $s$.

    ## Out-of-bag error (§15.3.1)

    Each tree is fit on approximately $0.632 N$ distinct training points; the remaining
    $0.368 N$ are **out-of-bag** (OOB).  The OOB prediction for $x_i$ is the average
    of only the trees for which $i$ was OOB.  OOB error converges to leave-one-out
    cross-validation error as $B \to \infty$, making it a computationally free
    generalisation estimate.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    import ch15_helpers as helpers

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
    ## OOB error vs number of trees (§15.3.1)

    We grow random forests with $B \in \{10, 25, 50, 100, 200, 300\}$ trees and plot
    the OOB error rate for three values of ``max_features`` ($m$):
    - `sqrt`: $m = \lfloor\sqrt{p}\rfloor$ — ESL's default for classification.
    - `log2`: $m = \lfloor\log_2 p\rfloor$ — further decorrelation.
    - `m=1`: extreme decorrelation — each split uses a single random feature
      (essentially a completely random forest).

    Expected observations:
    1. OOB error decreases monotonically and **stabilises** — forests don't overfit with more trees.
    2. Smaller $m$ gives lower error at small $B$ (more decorrelation) but potentially higher
       asymptotic error if individual trees become too weak (too few features per split).
    3. The plateau occurs around $B = 100$-$200$; additional trees rarely improve further.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_oob, oob_summary = helpers.rf_oob_figure(
        X,
        y,
        n_estimators_values=[10, 25, 50, 100, 200, 300],
        max_features_options=["sqrt", "log2", 1],
    )
    fig_oob.show()
    for label, info in oob_summary.items():
        print(f"  {label}: final OOB error={info['final_oob_error']:.3%}, min OOB error={info['min_oob_error']:.3%}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Variable importance: mean decrease in impurity (§15.3.2)

    For each feature $j$, MDI sums the weighted impurity decreases at all nodes where $j$
    was the split variable, averaged over all $B$ trees:

    $$\mathcal{I}(j) = \frac{1}{B} \sum_{b=1}^B \sum_{\substack{t \in T_b \\ j_t = j}}
      \frac{N_t}{N} \left[ i(t) - \frac{N_{t_L}}{N_t} i(t_L) - \frac{N_{t_R}}{N_t} i(t_R) \right]$$

    where $i(t)$ is the Gini impurity at node $t$ and $N_t$ is the number of training
    observations reaching $t$.

    Variable importance is useful for:
    - **Feature selection**: eliminate low-importance features without much accuracy loss.
    - **Interpretation**: understand which features drive predictions.
    - **Debugging**: check that known-informative features rank high.

    Note: MDI can be biased toward features with many unique values (continuous features
    or high-cardinality categoricals).  Permutation importance (§15.3.2) is less biased
    but slower.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_imp, imp_summary = helpers.rf_variable_importance_figure(X, y, n_estimators=200)
    fig_imp.show()
    print(f"OOB accuracy: {imp_summary['oob_accuracy']:.3%}")
    print(f"Top feature: {imp_summary['top_feature']!r}")
    print("Variable importances (MDI):")
    for feat, imp in imp_summary["importances"].items():
        print(f"  {feat:<20}: {imp:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Effect of max_features (m) (§15.4.1)

    The number of candidate features $m$ at each split is the most important tuning
    parameter for random forests.

    From Breiman's bound, the optimal $m$ minimises $\bar{\rho} \cdot (1 - s^2)/s^2$:
    - Decreasing $m$: lowers tree correlation $\bar{\rho}$ (good) but also lowers
      strength $s$ (bad, as trees become weaker with fewer candidate features).
    - Increasing $m$ toward $p$ (bagging): maximises individual tree strength but
      also maximises correlation.

    ESL's recommended defaults:
    - **Classification**: $m = \lfloor\sqrt{p}\rfloor$
    - **Regression**: $m = \lfloor p/3 \rfloor$

    In practice, CV over $\{1, \sqrt{p}, p/3, p/2, p\}$ often yields marginal improvements.
    The OOB estimate (diamonds) closely tracks CV accuracy at a fraction of the cost.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_mf, mf_summary = helpers.rf_max_features_figure(
        X,
        y,
        n_estimators=200,
        max_features_options=[1, 2, "sqrt", "log2", None],
    )
    fig_mf.show()
    print(f"Best max_features: {mf_summary['best_max_features']} | CV accuracy: {mf_summary['best_cv_accuracy']:.3%}")
    for setting, mf_acc in mf_summary["results"].items():
        print(f"  {setting:<22}: {mf_acc:.3%}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tree depth effect (§15.3)

    ESL recommends growing trees to **full depth** (no pruning) in random forests.
    The intuition: each deep tree has low bias; the ensemble average reduces variance.

    Shallow trees (stumps, depth 2-3) have high bias — even perfect de-correlation
    cannot compensate for individual-tree underfitting.

    This contrasts with **boosting** (§10.3), where shallow trees (stumps or depth ≤ 3)
    are preferred as weak learners to prevent overfitting in the sequential setting.

    The OOB estimates (diamonds) track CV accuracy closely, confirming OOB's reliability
    as a free proxy for CV even across tree-depth configurations.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_depth, depth_summary = helpers.rf_tree_depth_figure(
        X,
        y,
        n_estimators=200,
        max_depth_values=[1, 2, 3, 5, 8, None],
    )
    fig_depth.show()
    print(f"Best depth: {depth_summary['best_depth']} | CV accuracy: {depth_summary['best_cv_accuracy']:.3%}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ensemble comparison: tree, bagging, RF, GBM (§15.3)

    Contextualising random forests within ESL's ensemble taxonomy:

    | Method | Variance | Bias | Notes |
    |---|---|---|---|
    | **Single tree** | High | Low | Overfits without pruning |
    | **Bagging** ($m=p$) | Lower | Same | Reduces variance; trees still correlated |
    | **Random forest** ($m=\sqrt{p}$) | Lowest | Slightly higher | Best variance reduction |
    | **Gradient boosting** | Moderate | Lowest | Sequential; needs careful regularisation |

    Random forests typically outperform bagging because de-correlation is more powerful
    than just averaging correlated trees.  Gradient boosting can surpass random forests
    when carefully tuned, but is more sensitive to hyperparameters and training data noise.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_cmp, cmp_summary = helpers.rf_comparison_figure(X, y, n_estimators=200)
    fig_cmp.show()
    print(f"Best method: {cmp_summary['best_method']} | CV accuracy: {cmp_summary['best_cv_accuracy']:.3%}")
    for method, cmp_acc in cmp_summary["results"].items():
        print(f"  {method:<30}: {cmp_acc:.3%}")
    return


if __name__ == "__main__":
    app.run()

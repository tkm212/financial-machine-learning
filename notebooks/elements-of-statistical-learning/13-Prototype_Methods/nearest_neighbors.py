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
    # K-Nearest Neighbors — ESL Ch. 13

    *Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning. §13.3.*

    ## The k-NN classifier (§13.3)

    The **k-nearest-neighbor** classifier assigns the class by majority vote among the
    $k$ training points nearest to $x$ in input space:

    $$\hat{C}(x) = \underset{g}{\arg\max} \; \frac{1}{k} \sum_{i \in \mathcal{N}_k(x)} \mathbf{1}(y_i = g)$$

    where $\mathcal{N}_k(x) = \{i : x_i \in k\text{-nearest-neighbours of }x\}$ under a
    chosen distance metric (default: Euclidean $\|x - x_i\|_2$).

    ## Bias-variance trade-off (§13.3)

    The **effective number of parameters** for k-NN is approximately $N/k$:
    - $k = 1$: zero training error (each point is its own neighbour); highly variable.
    - $k = N$: constant classifier (global majority); high bias, zero variance.
    - Optimal $k$ minimises test error at the bottom of the U-shaped curve.

    **Cover-Hart theorem**: as $N \to \infty$ and $k/N \to 0$, the 1-NN error rate $R_{1\text{NN}}$
    satisfies:

    $$R^* \leq R_{1\text{NN}} \leq R^* \left(2 - \frac{K}{K-1} R^* \right) \leq 2R^*$$

    where $R^*$ is the Bayes error.  1-NN is never more than twice the Bayes error.

    ## Curse of dimensionality (§2.5)

    In high dimensions, the $k$-NN neighbourhood degenerates: the nearest neighbours of
    a test point may be far away in Euclidean distance, even with large $N$.  The fraction
    of the unit hypercube that must be covered to capture a fraction $r$ of the data grows
    as $r^{1/p}$ — for $p = 10$ and $r = 0.1$, this is $0.1^{1/10} \approx 0.80$ of the
    full input range.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    import ch13_helpers as helpers

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
    ## Selecting k by cross-validation (§13.3)

    We sweep $k \in \{1, 3, 5, 7, 10, 15, 20, 30, 50\}$ and record 5-fold CV accuracy.

    Standard practice (ESL §13.3): use a **log-spaced** grid for $k$ and choose by CV.
    Very small $k$ values (especially $k=1$) often overfit; very large $k$ values
    underfits by smoothing over class boundaries.

    The error bar shows $\pm 1$ SD across folds — a reasonable $k$ is the **simplest**
    model within one standard error of the minimum (the "1-SE rule").
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_k, k_summary = helpers.knn_k_selection_figure(X, y, k_values=[1, 3, 5, 7, 10, 15, 20, 30, 50])
    fig_k.show()
    print(f"Best k: {k_summary['best_k']} | CV accuracy: {k_summary['best_cv_accuracy']:.3%}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Distance metric comparison (§13.3)

    The choice of distance metric encodes assumptions about feature-space geometry:

    | Metric | Formula | Notes |
    |---|---|---|
    | **Euclidean** ($L_2$) | $\sqrt{\sum_j (x_j - x_j')^2}$ | Sensitive to scale; most common |
    | **Manhattan** ($L_1$) | $\sum_j |x_j - x_j'|$ | Robust to outliers in individual dims |
    | **Chebyshev** ($L_\infty$) | $\max_j |x_j - x_j'|$ | Dominated by worst dimension |
    | **Cosine** | $1 - \frac{x \cdot x'}{\|x\|\|x'\|}$ | Scale-invariant; useful for high-$p$ |

    All features are standardised before computing distances (otherwise higher-variance
    features dominate Euclidean distances).

    For well-scaled features, Euclidean and Manhattan are usually comparable.
    Cosine similarity can outperform both when the magnitude of the feature vector is
    less informative than its direction (e.g. text/sparse data).
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_metric, metric_summary = helpers.knn_metric_figure(
        X, y, k=5, metrics=["euclidean", "manhattan", "chebyshev", "cosine"]
    )
    fig_metric.show()
    print(f"Best metric: {metric_summary['best_metric']} | CV accuracy: {metric_summary['best_cv_accuracy']:.3%}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train vs test error curve (§13.3)

    The held-out train/test split reveals the bias-variance structure that CV averages over.

    Key observations:
    - **$k = 1$**: training error is exactly 0 (every point is its own nearest neighbour).
    - **Test error** has a clear minimum — this is the sweet spot between memorisation and
      over-smoothing.
    - The gap between train and test error **shrinks** as $k$ increases: larger $k$ models
      are more stable (lower variance) but less flexible (higher bias).

    This mirrors the classical ESL Figure 2.4 for a regression setting.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_tt = helpers.knn_train_test_figure(X, y, k_values=list(range(1, 51)))
    fig_tt.show()
    return


if __name__ == "__main__":
    app.run()

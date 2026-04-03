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
    # Linear Discriminant Analysis: ESL Ch. 4

    *Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning. Chapter 4.*

    TMDB movie data (`inputs/tmdb-movie-metadata/`): binary target — **high revenue** (1) if revenue ≥ median, **low revenue** (0) otherwise.  LDA assumes each class has a Gaussian distribution with a **shared** covariance matrix $\Sigma$; the log-ratio of class posteriors is linear in $x$.  QDA allows **class-specific** covariance matrices, giving quadratic boundaries.  Regularized LDA (RDA, §4.3.1) shrinks class covariances toward a common pooled estimate, interpolating between LDA and QDA.
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
    print(
        f"Loaded {len(X):,} rows | features: {data['feat_names']} | target: {target!r} "
        f"| class balance: {y.mean():.2%} positive"
    )
    return data, target


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LDA decision boundary in 2D (§4.3)

    LDA finds the linear combination of features that best separates the classes.  The decision boundary is the set of points where the two class posteriors are equal:

    $$\log \frac{\Pr(G=k \mid x)}{\Pr(G=\ell \mid x)} = \log\frac{\pi_k}{\pi_\ell} - \frac{1}{2}(\mu_k+\mu_\ell)^T \Sigma^{-1}(\mu_k-\mu_\ell) + x^T \Sigma^{-1}(\mu_k-\mu_\ell) = 0$$

    Because $\Sigma$ is shared, this is **linear** in $x$.  PCA reduces the feature space to two dimensions for visualisation — the boundary appears as a straight line.
    """)
    return


@app.cell
def _(data, helpers):
    fig_boundary, _lda, _pca = helpers.lda_2d_boundary_figure(data["X_train_s"], data["y_train"])
    fig_boundary.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LDA vs QDA vs Logistic Regression (§4.3 / §4.4 / §4.4.5)

    - **LDA** assumes equal class covariances — requires estimating $p(p+1)/2$ parameters for $\Sigma$.
    - **QDA** allows $\Sigma_k$ per class — richer but higher variance.
    - **Logistic Regression** makes no distributional assumption on $X$; it directly models $\Pr(G=1\mid x)$ via the log-odds.

    When the Gaussian assumption holds, LDA is the optimal Bayes classifier.  When it is violated, logistic regression is more robust.  ESL §4.4.5 shows that LDA and logistic regression often give similar decision boundaries in practice.
    """)
    return


@app.cell
def _(data, helpers):
    fig_compare, df_compare = helpers.lda_vs_qda_logistic_figure(
        data["X_train_s"], data["X_test_s"], data["y_train"], data["y_test"]
    )
    fig_compare.show()
    print(df_compare.to_string(index=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regularized Discriminant Analysis (§4.3.1)

    RDA shrinks the class-specific covariance matrices toward a common pooled estimate:

    $$\hat\Sigma_k(\alpha) = (1-\alpha)\hat\Sigma_k + \alpha\hat\Sigma$$

    $\alpha = 0$ recovers QDA; $\alpha = 1$ recovers LDA.  A further shrinkage toward a scaled identity matrix is sometimes applied.  Here we use sklearn's `shrinkage` parameter in `LinearDiscriminantAnalysis(solver="lsqr")`, which interpolates the pooled covariance toward the diagonal.  The optimal shrinkage is where test accuracy peaks.
    """)
    return


@app.cell
def _(data, helpers):
    fig_rda, rda_summary = helpers.rda_shrinkage_figure(
        data["X_train_s"], data["X_test_s"], data["y_train"], data["y_test"]
    )
    fig_rda.show()
    print(f"Best shrinkage: {rda_summary['best_shrinkage']:.2f} | test accuracy: {rda_summary['best_test_acc']:.4f}")
    return


if __name__ == "__main__":
    app.run()

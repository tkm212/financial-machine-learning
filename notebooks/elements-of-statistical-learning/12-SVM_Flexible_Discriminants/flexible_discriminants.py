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
    # Flexible Discriminant Analysis — ESL Ch. 12

    *Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning. §12.4-12.6.*

    ## Generalising LDA (§12.4)

    Classical **Linear Discriminant Analysis** (§4.3) finds the linear combinations
    $Z = a^\top X$ that best separate the $K$ classes, assuming within-class Gaussian
    distributions with common covariance $\Sigma$:

    $$\delta_k(x) = x^\top \Sigma^{-1} \mu_k - \tfrac{1}{2}\mu_k^\top \Sigma^{-1}\mu_k + \log \pi_k$$

    The $k$-th discriminant direction solves:
    $$\max_a \frac{a^\top B a}{a^\top W a}, \quad B = \text{between-class scatter}, \quad W = \text{within-class scatter}$$

    LDA is optimal when the Gaussian equal-covariance assumption holds.  When it fails,
    the linear decision boundary is misspecified and **more flexible** approaches are needed.

    ## Flexible Discriminant Analysis (§12.5)

    **FDA** (Hastie, Tibshirani & Buja 1994) generalises LDA by replacing the linear
    regression step with an arbitrary regression method:

    1. Encode classes as dummy variables $Y = [Y_1, \ldots, Y_K]$ via optimal scoring:
       find scores $\theta_k$ for each class that maximise the ratio of between-class
       to within-class variance in the regression $\theta(g) \sim \eta(X)$.
    2. Apply any regression method $\eta$ (GAM, MARS, etc.) instead of linear regression.

    We approximate FDA by applying polynomial feature expansion before LDA — equivalent
    to using polynomial regression as the $\eta$ in the FDA framework.

    ## Penalised Discriminant Analysis (§12.6)

    **PDA** (Hastie, Buja & Tibshirani 1995) adds a roughness penalty to the FDA criterion,
    shrinking discriminant directions toward a structured null:

    $$\text{ASR}(c) + \lambda \cdot c^\top \Omega c$$

    This is most useful when:
    - $p > N$ (more features than observations) — standard LDA is singular.
    - Features have natural smoothness (e.g. wavelets, splines) — $\Omega$ can encode
      smoothness across adjacent coefficients.
    - Class separation is driven by a small number of directions — $\lambda$ zeros out
      irrelevant discriminant components.

    We approximate PDA via sklearn's `shrinkage` parameter, which regularises the
    within-class covariance estimate:
    $$\hat{\Sigma}_{shrunk} = (1 - t)\hat{\Sigma} + t \cdot \frac{\text{tr}(\hat{\Sigma})}{p} \cdot I$$
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    import ch12_helpers as helpers

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
    ## LDA vs FDA: polynomial feature expansion (§12.5)

    We compare standard LDA (degree=1) against FDA approximated by feeding degree-2 and
    degree-3 polynomial features into LDA.

    This is equivalent to using polynomial regression as the $\eta$ function in FDA's
    optimal scoring framework.  The polynomial expansion generates interaction terms like
    $x_j x_k$ and $x_j^2$, allowing LDA to find **curved** rather than linear class boundaries.

    Trade-offs as degree increases:
    - **Higher accuracy potential** — can capture nonlinear class structure.
    - **Higher variance** — many more features, larger chance of overfitting.
    - **Computational cost** — number of features grows as $O(p^d)$.

    A second `StandardScaler` after expansion prevents numerically extreme polynomial terms
    from dominating the covariance estimate.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_fda, fda_summary = helpers.fda_vs_lda_figure(X, y, poly_degrees=[1, 2, 3])
    fig_fda.show()
    print(
        f"Best model: {fda_summary['best_model']} | "
        f"CV accuracy: {fda_summary['best_cv_accuracy']:.3%} | "
        f"LDA baseline: {fda_summary['lda_accuracy']:.3%}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PDA: covariance shrinkage (§12.6)

    The shrinkage parameter $t \in [0, 1]$ interpolates between:
    - $t = 0$: sample covariance (MLE) — unbiased but noisy when $p$ is large.
    - $t = 1$: scaled identity — maximum regularisation, forces isotropic assumption.
    - `Ledoit-Wolf`: analytically optimal $t$ minimising Frobenius distance to the true
      covariance (Oracle approximating shrinkage).

    Key observations:
    - Mild shrinkage often improves test accuracy by stabilising the within-class
      covariance estimate — especially beneficial when features are correlated.
    - Extreme shrinkage ($t \to 1$) forces the model toward nearest-centroid classification.
    - The optimal $t$ can be determined analytically (Ledoit-Wolf) or by cross-validation.

    The dashed line shows MLE accuracy as the baseline for comparison.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_pda, pda_summary = helpers.pda_shrinkage_figure(X, y)
    fig_pda.show()
    print(
        f"Best shrinkage: {pda_summary['best_shrinkage']} | "
        f"CV accuracy: {pda_summary['best_cv_accuracy']:.3%} | "
        f"MLE baseline: {pda_summary['mle_accuracy']:.3%}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary: SVM vs LDA, FDA, PDA (§12.4-12.6)

    All methods are evaluated by 5-fold CV on the same standardised features.

    **Theoretical relationships** (ESL §12.4):
    - **LDA** and **Linear SVM** both find linear decision boundaries, but via different
      criteria: LDA maximises the between/within-class ratio; SVM maximises the margin.
    - When classes are Gaussian with equal covariance, LDA is Bayes-optimal.
    - SVM is more robust to outliers (margin only depends on support vectors).
    - **FDA** subsumes LDA; **RBF SVM** and FDA with a kernel regression are closely related
      via the reproducing kernel Hilbert space framework (§12.3.3).

    The bar chart gives an empirical comparison on the TMDB movie revenue classification task.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_cmp, cmp_summary = helpers.method_comparison_figure(X, y)
    fig_cmp.show()
    print(f"Best method: {cmp_summary['best_method']} | CV accuracy: {cmp_summary['best_cv_accuracy']:.3%}")
    for method, acc in cmp_summary["results"].items():
        print(f"  {method:<22}: {acc:.3%}")
    return


if __name__ == "__main__":
    app.run()

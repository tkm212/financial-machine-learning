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
    # Support Vector Machines — ESL Ch. 12

    *Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning. §12.2-12.3.*

    ## The support vector classifier (§12.2)

    For separable classes, the **hard-margin SVM** finds the hyperplane $\{x : f(x) = x^\top \beta + \beta_0 = 0\}$
    that maximises the **geometric margin** $2/\|\beta\|$:

    $$\min_{\beta, \beta_0} \tfrac{1}{2}\|\beta\|^2 \quad \text{s.t.} \quad y_i(x_i^\top \beta + \beta_0) \geq 1 \; \forall i$$

    For non-separable data, **slack variables** $\xi_i \geq 0$ allow margin violations:

    $$\min_{\beta, \beta_0, \xi} \tfrac{1}{2}\|\beta\|^2 + C \sum_{i=1}^N \xi_i \quad \text{s.t.} \quad y_i(x_i^\top \beta + \beta_0) \geq 1 - \xi_i$$

    The **cost parameter** $C$ controls the bias-variance trade-off:
    - $C \to \infty$: hard margin, zero training error (high variance).
    - $C \to 0$: wide margin, many violations tolerated (high bias).

    ## The dual and support vectors (§12.2.1)

    Via Lagrange duality, the solution is:
    $$\hat{f}(x) = \sum_{i=1}^N \hat{\alpha}_i y_i \langle x_i, x \rangle + \hat{\beta}_0$$

    Only observations with $\hat{\alpha}_i > 0$ contribute — these are the **support vectors**.
    On the margin: $0 < \hat{\alpha}_i < C$; margin violations: $\hat{\alpha}_i = C$.
    The decision boundary depends only on inner products — the key to the kernel trick (§12.3).

    ## The kernel trick (§12.3)

    Replacing $\langle x_i, x_j \rangle$ with a kernel $K(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle$
    implicitly maps inputs to a high-dimensional feature space without computing $\phi(x)$ explicitly:

    $$\hat{f}(x) = \sum_{i=1}^N \hat{\alpha}_i y_i K(x_i, x) + \hat{\beta}_0$$

    Common kernels and their implicit feature spaces:
    - **Linear**: $K(x, x') = x^\top x'$ — original space.
    - **Polynomial**: $K(x, x') = (1 + x^\top x')^d$ — all monomials up to degree $d$.
    - **RBF**: $K(x, x') = \exp(-\gamma \|x - x'\|^2)$ — infinite-dimensional Gaussian RKHS.
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
    ## Cost parameter C: accuracy vs support vector fraction (§12.2)

    We sweep $C$ over several orders of magnitude and record:
    1. **Cross-validated accuracy** (left axis) — how well the model generalises.
    2. **Fraction of training points that are support vectors** (right axis, dotted) —
       a proxy for model complexity.

    As $C$ increases:
    - Fewer margin violations are allowed → the margin shrinks → more support vectors
      lie exactly on the margin boundary.
    - The model fits the training data more tightly.

    The optimal $C$ sits at the balance point where test accuracy peaks.  ESL recommends
    using 10-fold CV to select $C$ from a log-spaced grid (§12.2).
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_cost, cost_summary = helpers.svm_cost_figure(X, y, C_values=[0.01, 0.1, 1.0, 10.0, 100.0], kernel="rbf")
    fig_cost.show()
    print(
        f"Best C: {cost_summary['best_C']} | "
        f"kernel: {cost_summary['kernel']} | "
        f"CV accuracy: {cost_summary['best_cv_accuracy']:.3%}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Kernel comparison (§12.3)

    Different kernels encode different assumptions about the decision boundary's geometry:

    - **Linear** ($K = x^\top x'$): The original hard-margin SVM.  Appropriate when classes are
      approximately linearly separable — or when $p \gg N$ (high-dimensional input space).
    - **Polynomial** ($K = (1 + x^\top x')^d$, $d = 3$): Allows polynomial decision boundaries.
      Can overfit with high degree; sensitive to feature scaling.
    - **RBF / Gaussian** ($K = e^{-\gamma\|x - x'\|^2}$): The most popular kernel.  Acts as a
      similarity measure; $\gamma$ controls the width (high $\gamma$ = narrow Gaussians = more
      local decisions).
    - **Sigmoid**: Mimics the hidden-layer activation of a neural network.  Not always positive
      semi-definite so technically may not satisfy Mercer's theorem for all parameters.

    All kernels are evaluated at $C = 1$ with standardised features.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_kernel, kernel_summary = helpers.svm_kernel_figure(X, y, kernels=["linear", "poly", "rbf", "sigmoid"], C=1.0)
    fig_kernel.show()
    print(f"Best kernel: {kernel_summary['best_kernel']} | CV accuracy: {kernel_summary['best_cv_accuracy']:.3%}")
    return


if __name__ == "__main__":
    app.run()

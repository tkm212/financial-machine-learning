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
    # Model Assessment: Bias, Variance and Optimism — ESL Ch. 7

    *Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning. §7.2-7.4.*

    A central challenge in supervised learning is **generalisation**: models trained on finite data must predict well on unseen examples.  The training error $\overline{err} = \frac{1}{N}\sum_i L(y_i, \hat{f}(x_i))$ is an optimistically biased estimate of the true prediction error, because the model has already seen these observations.

    The **test error** (generalisation error) for a fixed training set $\mathcal{T}$ is:

    $$\text{Err}_\mathcal{T} = \mathbb{E}_{(X,Y) \sim P}\bigl[L(Y, \hat{f}(X)) \mid \mathcal{T}\bigr]$$

    Under squared error loss the expected test error decomposes as:

    $$\text{Err}(x_0) = \underbrace{\sigma^2}_{\text{irreducible noise}} + \underbrace{\text{Bias}^2[\hat{f}(x_0)]}_{\text{systematic error}} + \underbrace{\text{Var}[\hat{f}(x_0)]}_{\text{estimation noise}}$$

    This **bias-variance tradeoff** (§7.3) determines the optimal model complexity:
    - Simple models: high bias, low variance.
    - Complex models: low bias, high variance.

    The gap between test and training error is the **optimism** (§7.4), which grows with model complexity.
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
    ## Train vs test error as a function of complexity (§7.2)

    We fit polynomial regression of increasing degree on log₁p(budget) → log₁p(revenue), evaluated on a held-out 25% test set.

    The characteristic **U-shaped** test error curve illustrates the bias-variance tradeoff:

    - **Low degree**: both errors are high and similar — high bias dominates.
    - **Intermediate degree**: test error reaches its minimum — the sweet spot.
    - **High degree**: training error keeps falling while test error rises — variance dominates.

    The gap between train and test error at each complexity level is precisely the **optimism**.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_tt = helpers.train_test_error_figure(X, y, feat="budget", max_degree=12)
    fig_tt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bias-variance decomposition (§7.3)

    To see the decomposition precisely, we use synthetic data $y = \sin(2\pi x) + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, \sigma^2)$, where the true function is known.

    For each polynomial degree we independently fit 100 models on fresh training sets of size 60.  At each test point $x_0$:

    $$\text{Bias}^2[\hat{f}(x_0)] = \Bigl(\mathbb{E}_\mathcal{T}[\hat{f}(x_0)] - f(x_0)\Bigr)^2, \qquad \text{Var}[\hat{f}(x_0)] = \mathbb{E}_\mathcal{T}\Bigl[\bigl(\hat{f}(x_0) - \mathbb{E}_\mathcal{T}[\hat{f}(x_0)]\bigr)^2\Bigr]$$

    The curves are then averaged over the test grid.  Note how:

    - **Bias²** (blue) decreases monotonically — more complex models fit the true function better.
    - **Variance** (orange) increases monotonically — high-degree polynomials are sensitive to which training set they saw.
    - **Total error** (red) has a clear minimum at the optimal degree.
    - **Noise floor** (grey dashed) is irreducible — no model can do better than $\sigma^2$ on average.
    """)
    return


@app.cell
def _(helpers):
    fig_bv = helpers.bias_variance_decomposition_figure(n_train=60, n_test=200, noise_std=0.3, max_degree=8, n_boot=100)
    fig_bv.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Optimism of the training error (§7.4)

    The **optimism** of the training error is defined as:

    $$\text{op} = \text{Err}_\text{in} - \overline{err} = \frac{2}{N} \sum_{i=1}^N \text{Cov}(\hat{y}_i, y_i)$$

    For linear fits with $d$ parameters under squared loss, this simplifies to $2d\sigma^2/N$ — the basis of Mallows' $C_p$.

    Here we estimate optimism empirically via bootstrap: for each bootstrap resample, we compute the in-bag (training) error and the OOB (out-of-bag) error.  Their difference estimates the optimism at that complexity level.

    Observe that optimism grows steadily with model complexity — exactly what $C_p$ and AIC correct for analytically.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_op = helpers.optimism_figure(X, y, feat="budget", max_degree=8, n_boot=60)
    fig_op.show()
    return


if __name__ == "__main__":
    app.run()

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
    # Undirected Graphical Models — ESL Ch. 17

    *Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning. §17.1-17.3.*

    An **undirected graphical model** (Markov random field) encodes conditional independence
    structure in a multivariate distribution via an **interaction graph**: missing edges
    correspond to conditional independences.

    ## Gaussian graphical models (§17.3)

    For $X \sim \mathcal{N}(0, \Sigma)$, the **precision matrix** $\Theta = \Sigma^{-1}$
    satisfies:

    $$\Theta_{jk} = 0 \quad \Leftrightarrow \quad X_j \perp X_k \mid X_{\setminus \{j,k\}}$$

    Estimating a **sparse** $\Theta$ under an $\ell_1$ penalty is the **graphical lasso**
    (Friedman et al. 2008), implemented in scikit-learn as `GraphicalLassoCV`.

    ## Partial correlation (§17.3.2)

    The scaled entries of $\Theta$ relate to **partial correlations** — correlation between
    two variables after removing the linear effect of all others.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    import ch17_helpers as helpers

    _root, _inputs, _outputs = helpers.init_paths()
    return helpers


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Graphical lasso: recovering sparse precision (§17.3.1)

    We simulate $n$ observations from a multivariate Gaussian with a **sparse precision**
    matrix $\Omega$ (random symmetric edges, then PD adjustment).  `GraphicalLassoCV`
    selects the penalty $\rho$ by cross-validation and returns $\hat{\Theta}$.

    Compare the heatmaps: off-diagonal structure in $\hat{\Theta}$ should resemble $\Omega$,
    though finite-sample recovery depends on $n$, $p$, and edge strength.
    """)
    return


@app.cell
def _(helpers):
    fig_gl, gl_info = helpers.graphical_lasso_demo_figure(p=20, n=120, cv=5, random_state=0)
    fig_gl.show()
    print(f"Selected alpha: {gl_info['alpha_selected']:.4f}")
    print(f"Frobenius error (off-diagonal vs true): {gl_info['frobenius_offdiag_error']:.3f}")
    print(
        f"Nonzero off-diagonals — true: {gl_info['n_nonzero_offdiag_true']}, "
        f"estimated: {gl_info['n_nonzero_offdiag_hat']}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Partial correlations (§17.3.2)

    The **partial correlation** between $X_j$ and $X_k$ given the remaining variables is
    a signed measure in $[-1, 1]$ derived from $\Theta_{jk}$ and the diagonal entries.
    Zeros in $\Theta$ imply zero partial correlation — no direct association after
    conditioning on others.
    """)
    return


@app.cell
def _(helpers):
    fig_pc, pc_info = helpers.partial_correlation_figure(p=12, n=200, random_state=1)
    fig_pc.show()
    print(f"Graphical lasso alpha used: {pc_info['alpha']:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Discrete and mixed data (§17.4)

    ESL also discusses **log-linear models** for discrete Markov networks and **copulas**
    for mixed discrete/continuous data.  Those require specialised structure learning;
    the Gaussian graphical model above is the continuous analogue most directly tied to
    covariance / precision estimation in earlier chapters.
    """)
    return


if __name__ == "__main__":
    app.run()

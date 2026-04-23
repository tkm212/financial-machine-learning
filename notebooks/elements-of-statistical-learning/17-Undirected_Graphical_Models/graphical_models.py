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

    An **undirected graphical model** (Markov random field) encodes **conditional
    independences** in a multivariate distribution.  The **graph** $G$ has a node for each
    variable; if $(j,k) \\notin E$ then (under mild positivity conditions) $X_j$ and $X_k$
    are **conditionally independent** given all other coordinates — the local / global
    **Markov** properties.  The **Hammersley-Clifford theorem** links such factorisations
    to **clique potentials** for strictly positive mass functions (ESL §17.2-17.2.2; here
    we focus on the **Gaussian** case).

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

    _root, INPUTS, _outputs = helpers.init_paths()
    return INPUTS, helpers


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
    ## Sparsity of $\hat{\Theta}$ vs penalty (§17.3.1)

    Like the lasso for regression, the **graphical lasso** trades off fit against how many
    **non-zeros** appear in the estimated precision.  The curve below (synthetic $p$-vector
    data) shows how the number of off-diagonals in $\hat{\Theta}(\alpha)$ with
    $|\cdot|>\varepsilon$ grows as $\alpha$ sweeps.  The dashed line marks the
    **cross-validated** $\alpha$ from `GraphicalLassoCV` on the same draw.
    """)
    return


@app.cell
def _(helpers):
    fig_sp, sp = helpers.graphical_lasso_edge_count_vs_alpha_figure(p=20, n=120, n_alphas=45, random_state=0)
    fig_sp.show()
    print(f"At CV alpha: {sp['alpha_cv']:.4f} | selected edges: {sp['n_edges_at_cv']}")
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
    ## Marginal vs **partial** association on real data (§17.3.2)

    **Correlation** is easy to read but conflates **direct** and **indirect** links.
    **Partial** correlations, derived from $\hat{\Theta}$, approximate **correlation
    after** conditioning on the other variables — a building block of the GMRF interpretation.
    The left panel is the usual sample matrix; the right is from the same TMDB
    `StandardScaler`'d matrix with CV-tuned `GraphicalLassoCV` (a single focused partial
    heatmap is the right column here).  With only a few numeric features, $p \ll n$ and a
    Gaussian model is more plausible than in genomics-scale $p$; still, movie variables are
    not exactly normal — treat the plot as **exploratory conditioning structure**.
    """)
    return


@app.cell
def _(INPUTS, helpers):
    fig_pair, pr = helpers.tmdb_correlation_and_partial_panels_figure(INPUTS, max_rows=900, cv=5, random_state=0)
    fig_pair.show()
    print(f"alpha_hat = {pr['alpha']:.4f} | n = {pr['n']}, p = {pr['p']}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Edge stability under bootstrap (§17.3)

    We fix the penalty at the CV-chosen $\\alpha$ and refit **graphical lasso** on many
    **bootstrap** samples of the same synthetic data.  The heatmap shows how often each
    off-diagonal edge is selected (nonzero $|\hat{\Theta}_{jk}|$).  Stable edges are
    bright; unstable ones are artefacts of sampling noise — the same stability idea used
    in structure learning more broadly.
    """)
    return


@app.cell
def _(helpers):
    fig_stab, stab_info = helpers.graphical_lasso_stability_figure(p=18, n=100, n_bootstrap=45, cv=5, random_state=0)
    fig_stab.show()
    print(f"Fixed alpha={stab_info['alpha_fixed']:.4f} | mean edge selection freq: {stab_info['mean_edge_freq']:.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Network sketch: edges from $|\hat{\Theta}_{jk}|$ (TMDB) (§17.1)

    A **graph** is not only a heatmap.  The figure places variables on a circle; line
    segments connect $(j,k)$ when $|\hat{\Theta}_{jk}|$ (off-diagonal precision) is
    among the larger quantiles.  The layout is a **visual** aid, not a unique embedding; use
    the stability and CV machinery above to judge **which** edges to trust.
    """)
    return


@app.cell
def _(INPUTS, helpers):
    fig_g, ginfo = helpers.network_sketch_from_precision_figure(INPUTS, max_rows=900, cv=5, edge_weight_quantile=0.5)
    fig_g.show()
    print(
        f"Edges drawn: {ginfo['n_edges_drawn']} | |Theta| threshold = {ginfo['threshold']:.4f} | alpha_hat = {ginfo['alpha']:.4f}"
    )
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

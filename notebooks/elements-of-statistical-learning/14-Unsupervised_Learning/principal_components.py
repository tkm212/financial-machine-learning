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
    # Principal Components and NMF — ESL Ch. 14

    *Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning. §14.5-14.6.*

    ## Principal Component Analysis (§14.5)

    **PCA** finds a low-dimensional linear subspace that captures the maximum variance
    in the data.  The $m$-th principal component direction $v_m \in \mathbb{R}^p$ solves:

    $$v_m = \underset{\|v\|=1,\; v \perp v_1,\ldots,v_{m-1}}{\arg\max} \; \text{Var}(Xv)$$

    Equivalently, $\{v_m\}_{m=1}^M$ are the top $M$ eigenvectors of the sample covariance
    $S = \frac{1}{N-1}X^\top X$.  The $m$-th **principal component score** is $z_m = Xv_m$.

    ## The SVD connection (§14.5.1)

    The SVD of the centred data matrix $X = UDV^\top$ gives:
    - **Right singular vectors** $V$: the principal component directions.
    - **Singular values** $d_m$: $\lambda_m = d_m^2/(N-1)$ are the eigenvalues of $S$.
    - **Left singular vectors** $U$: the normalised principal component scores.

    The rank-$M$ approximation $\hat{X} = U_M D_M V_M^\top$ is the best rank-$M$ reconstruction
    of $X$ in Frobenius norm (Eckart-Young theorem).

    ## Non-negative Matrix Factorization (§14.6)

    **NMF** (Lee & Seung 1999) factorises $X \approx WH$ with the constraint $W, H \geq 0$.

    Unlike PCA, non-negativity forces a **parts-based** representation:
    each observation $x_i \approx \sum_r w_{ir} h_r$ is a non-negative combination of
    $r$ non-negative basis vectors $\{h_r\}$.  Subtractive cancellation is not allowed,
    leading to localised, interpretable parts (e.g. facial features, topics in text).

    NMF solves:

    $$\min_{W \geq 0,\, H \geq 0} \|X - WH\|_F^2$$

    via alternating multiplicative updates:
    - $H \leftarrow H \circ \frac{W^\top X}{W^\top W H}$
    - $W \leftarrow W \circ \frac{X H^\top}{W H H^\top}$
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    import ch14_helpers as helpers

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
    ## PCA scree plot (§14.5)

    The **scree plot** shows the proportion of variance explained (PVE) by each principal
    component and the cumulative PVE.

    - The **elbow** in the individual PVE bars indicates where components stop accounting
      for meaningful variance.
    - The **90% line** on the cumulative plot shows how many components are needed to
      preserve 90% of the total variance.

    For a well-structured dataset, a small number of components captures most variance.
    For noisy or high-dimensional data, the scree plot is flat — more components are needed.
    """)
    return


@app.cell
def _(X, helpers):
    fig_scree, scree_summary = helpers.pca_variance_figure(X)
    fig_scree.show()
    print(f"Components for 90% variance: {scree_summary['n_components_for_90pct']}")
    for i, (pve, cum) in enumerate(zip(scree_summary["pve"], scree_summary["cumulative_pve"], strict=True), 1):
        print(f"  PC{i}: PVE={pve:.3%}  cumulative={cum:.3%}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PCA biplot: scores and loadings (§14.5.1)

    The **biplot** projects observations onto the first two principal components and
    overlays the **loading vectors** (arrows) showing how each original feature contributes.

    Interpreting the biplot:
    - **Observations** (blue points): similar movies cluster together in PCA space.
    - **Loading arrows** (red): features with large arrows strongly influence the PCs.
    - **Angle between arrows** ≈ $\cos^{-1}(r)$ where $r$ is the Pearson correlation
      between the two features.
    - **Arrow direction**: features pointing in the same direction are positively correlated;
      opposing directions indicate negative correlation.
    - **Arrow length**: longer arrows explain more variance in that PC direction.

    The biplot is one of the most useful exploratory tools in unsupervised learning,
    simultaneously showing observation similarity and feature relationships.
    """)
    return


@app.cell
def _(X, helpers):
    fig_biplot = helpers.pca_biplot_figure(X, pc_x=1, pc_y=2, max_rows=400)
    fig_biplot.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## NMF: reconstruction error vs rank (§14.6)

    We fit NMF at ranks $r \in \{1, 2, 3, 4, 5\}$ and plot the Frobenius reconstruction
    error $\|X - WH\|_F$ vs $r$.

    As $r$ increases:
    - Error decreases monotonically (more flexibility to reconstruct $X$).
    - The rate of decrease slows — an "elbow" signals a reasonable rank.

    Compared to PCA's reconstruction error (Eckart-Young):
    - PCA gives the **optimal** rank-$r$ reconstruction (minimum Frobenius error).
    - NMF is sub-optimal in Frobenius norm but imposes interpretability through
      non-negativity.

    Use NMF when the features are non-negative (counts, intensities, budgets) and
    interpretability of components matters more than minimal reconstruction error.
    """)
    return


@app.cell
def _(X, helpers):
    fig_nmf, nmf_summary = helpers.nmf_rank_figure(X, ranks=list(range(1, 6)))
    fig_nmf.show()
    print("NMF reconstruction errors:")
    for r, err in zip(nmf_summary["ranks"], nmf_summary["errors"], strict=True):
        print(f"  r={r}: ||X - WH||_F = {err:.4f}")
    return


if __name__ == "__main__":
    app.run()

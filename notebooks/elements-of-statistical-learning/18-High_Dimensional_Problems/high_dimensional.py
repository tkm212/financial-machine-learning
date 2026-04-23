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
    # High-Dimensional Problems ($p \gg N$) — ESL Ch. 18

    *Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning. §18.1-18.4.*

    When the number of features $p$ exceeds the sample size $N$, classical least squares
    is **not identifiable** (singular $X^\top X$).  **Regularisation** — ridge, lasso,
    elastic net — is required to obtain stable estimators and to encode sparsity or
    shrinkage priors.

    ## Curse of dimensionality (§18.1)

    In high dimensions, data are sparse: local neighbourhoods are empty unless $N$ grows
    exponentially with dimension.  Global methods that pool information (regularised
    linear models, ensembles) are often more reliable than naive nearest-neighbour
    regression in the $p \gg N$ regime.

    ## Lasso vs ridge (§18.2-18.3)

    **Ridge** shrinks all coefficients but sets none to exactly zero.  **Lasso** performs
    **subset selection** via the $\ell_1$ penalty; in the linear Gaussian model with $p > N$,
    at most $N$ predictors can have nonzero coefficients at any penalty level (genericity).

    ## Elastic net (§18.4)

    The elastic net mixes $\ell_1$ and $\ell_2$, stabilising variable selection when
    predictors are correlated — a compromise between ridge and lasso.  The simulation and
    TMDB sections below are chosen so you can *see* sparsity, multiple testing, and
    high-$p$ noise in plots — not to tune a single best score.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    import ch18_helpers as helpers

    _root, INPUTS, _outputs = helpers.init_paths()
    return INPUTS, helpers


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Geometry: the curse of dimensionality (§18.1)

    In a $d$-dimensional unit box $[0,1]^d$, a slightly smaller concentric box of side
    $1-\varepsilon$ captures **fraction** $(1-\varepsilon)^d$ of the volume, which
    decays to $0$ as $d$ increases.  That is one face of the **empty neighbourhood**
    phenomenon: with fixed sample size, local information becomes scarce unless
    the effective dimension is controlled (regularisation, feature selection, or manifold
    structure).
    """)
    return


@app.cell
def _(helpers):
    fig_vol, _vol = helpers.curse_of_dimensionality_volume_figure(d_max=50)
    fig_vol.show()
    print(f"ε = {_vol['epsilon']}; (1-ε)³⁰ ≈ {_vol['at_d_30']:.4f} (very little mass in a 0.95-cube in d=30)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simulation: $N \ll p$ with sparse $\beta$ (§18.2-18.3)

    We draw $X_{ij} \sim \mathcal{N}(0,1)$, fix only a handful of nonzero $\beta_j$, and
    compare **RidgeCV**, **LassoCV**, and **ElasticNetCV** with $R^2$ from out-of-fold
    predictions (`cross_val_predict`).  Inspect how many coefficients each method leaves
    nonzero after refitting on all data.
    """)
    return


@app.cell
def _(helpers):
    fig_reg, reg_info = helpers.highdim_regularization_comparison_figure(
        n=45, p=180, n_nonzero=12, n_cv=5, random_state=0
    )
    fig_reg.show()
    print(f"Best: {reg_info['best_method']} | CV R² = {reg_info['best_r2']:.4f}")
    print(f"True active set size: {reg_info['true_nonzeros']}")
    for name, nz in reg_info["nonzeros"].items():
        print(f"  Nonzeros after fit — {name}: {nz}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Lasso path: active set vs penalty (§18.4)

    Along the **lasso path**, the number of nonzero coefficients grows as $\lambda$ decreases.
    It can never exceed $\min(N, p)$ in the standard linear model — a useful sanity check
    in high dimensions.
    """)
    return


@app.cell
def _(helpers):
    fig_path, path_info = helpers.lasso_path_sparsity_figure(n=40, p=120, n_nonzero=8, random_state=1)
    fig_path.show()
    print(f"Max active coefficients along path: {path_info['max_active']} (n={path_info['n']})")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recovery of $\beta$ (one draw) (§18.2-18.3)

    Cross-validated $R^2$ is an integration over penalties.  A complementary view is
    **parameter recovery**: in one sparse simulation, we plot true $\beta_j$ against the
    **lasso** $\hat\beta_j$ (after `StandardScaler`).  Perfect recovery would lie on the
    45° line; shrinkage and selection pull points toward the origin, and *false* non-zeros
    are possible under correlation.
    """)
    return


@app.cell
def _(helpers):
    fig_tf, _tf = helpers.lasso_true_vs_fitted_figure(n=60, p=120, n_nonzero=10, random_state=0)
    fig_tf.show()
    print(f"Correlation corr(β, β̂) (all p): {_tf['lasso_r_correlation_true_hat']:.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Shrinkage shape: **ridge** vs **lasso** (§18.2-18.3)

    **Ridge** spreads the fitted vector across (almost) all $p$ coordinates; **lasso** puts
    mass on a small subset.  The line plot shows the top magnitudes in one simulation — a
    qualitative contrast, not a proof of one method's superiority in every problem.
    """)
    return


@app.cell
def _(helpers):
    fig_mag, mag = helpers.ridge_vs_lasso_coefficient_magnitude_figure(
        n=50, p=100, n_nonzero=8, random_state=1
    )
    fig_mag.show()
    print(f"Lasso estimated nonzeros: {mag['n_nonzero_lasso']}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Marginal screening + lasso (§18)

    **Independence / correlation screening**: rank features by marginal $|\mathrm{corr}(X_j, y)|$,
    keep the top $k \approx 2N$ (Fan & Lv-style sure screening), then run **lasso** only on that
    submatrix.  When the truth is sparse and marginal signal is informative, this two-stage
    procedure can match or beat lasso on all $p$ columns with lower optimization cost.
    """)
    return


@app.cell
def _(helpers):
    fig_scr, scr_info = helpers.marginal_screening_lasso_figure(n=90, p=400, n_nonzero=15, n_cv=5, random_state=0)
    fig_scr.show()
    print(
        f"CV R² — lasso all features: {scr_info['r2_full']:.4f} | "
        f"after screen k={scr_info['screen_k']}: {scr_info['r2_screened']:.4f}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multiple testing: Bonferroni vs Benjamini-Hochberg (§18)

    With $p$ marginal tests of $H_{0j}: \beta_j = 0$, uncorrected $\alpha$ yields many
    **false positives** among null coordinates.  **Bonferroni** controls family-wise error;
    **Benjamini-Hochberg** controls **false discovery rate** - often more power at the cost
    of a different guarantee.  The simulation averages true/false positive counts over Monte
    Carlo replicates.
    """)
    return


@app.cell
def _(helpers):
    fig_fdr, fdr_info = helpers.fdr_vs_bonferroni_figure(
        n_rep=40, n=100, p=220, n_signal=10, alpha=0.05, random_state=0
    )
    fig_fdr.show()
    print("Mean TP (signals found):", fdr_info["mean_tp"])
    print("Mean FP (nulls selected):", fdr_info["mean_fp"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Real data + noise features (§18)

    TMDB movie regression (log revenue vs numeric features) is augmented with **Gaussian
    noise columns** so that $p \gg n$.  Signal is confined to the original features;
    lasso typically drives most noise coefficients to zero, whereas ridge spreads weight
    across all dimensions.
    """)
    return


@app.cell
def _(INPUTS, helpers):
    fig_tm, tm_info = helpers.tmdb_with_noise_features_figure(INPUTS, n_noise=200, max_rows=600, n_cv=5, random_state=0)
    fig_tm.show()
    print(
        f"n={tm_info['n']}, p={tm_info['p']} | CV R² ridge={tm_info['r2_ridge']:.3f}, lasso={tm_info['r2_lasso']:.3f}"
    )
    print(
        "Share of |coef| on noise columns — "
        f"ridge: {tm_info['fraction_abs_coef_noise_ridge']:.1%}, "
        f"lasso: {tm_info['fraction_abs_coef_noise_lasso']:.1%}"
    )
    return


if __name__ == "__main__":
    app.run()

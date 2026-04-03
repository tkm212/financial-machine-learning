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
    # Lasso: ESL Ch. 3

    *Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning. Chapter 3.*

    Lasso adds an L1 penalty $\lambda \sum_j |\beta_j|$. Unlike Ridge, it drives some coefficients **exactly to zero**, performing automatic feature selection. The path plot shows which movie features survive as we relax the penalty.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    import ch3_helpers as helpers

    _root, INPUTS, _outputs = helpers.init_paths()
    return INPUTS, helpers


@app.cell
def _(INPUTS, helpers):
    X, y, _ = helpers.load_tmdb_xy(INPUTS)
    data = helpers.scale_split(X, y)
    print(f"Train: {len(data['X_train']):,} | Test: {len(data['X_test']):,}")
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Coefficient paths (§3.4.4 / §3.8.6)

    The L1 penalty $\alpha\sum_j|\beta_j|$ creates a diamond-shaped constraint region whose corners lie on the axes — this is why coefficients hit zero exactly rather than just approaching it.

    The path is traced by decreasing alpha from its maximum (all coefficients zero) down to zero (OLS). **Read left to right**: as the penalty relaxes, features enter one by one. The order reflects marginal predictive power given what is already in the model.
    """)
    return


@app.cell
def _(data, helpers):
    helpers.lasso_coef_path_figure(data["X_train_s"], data["y_train"], data["feat_names"]).show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Choosing alpha: train vs test MSE

    The MSE plot has a clear minimum on the test set — the alpha where regularisation is just right. Left of the minimum (strong penalty) the model is too restricted; right of it (weak penalty) it overfits. The dashed line marks the optimal alpha and the print statement reports how many features survive at that point — the effective model size.
    """)
    return


@app.cell
def _(data, helpers):
    fig, summary = helpers.lasso_alpha_path_figure(data["X_train_s"], data["X_test_s"], data["y_train"], data["y_test"])
    fig.show()
    print(
        f"Best alpha: {summary['best_alpha']:.2f} | "
        f"test MSE: {summary['min_test_mse']:.4f} | "
        f"non-zero coefficients: {summary['n_nonzero_at_best']}"
    )
    return (summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Features selected at optimal alpha

    At the best alpha, Lasso has zeroed out low-signal features entirely. The bar chart shows only the survivors and their shrunk magnitudes — positive coefficients push revenue up, negative pull it down, holding all other features fixed. Shrinkage means even the selected coefficients are smaller in magnitude than their OLS counterparts.
    """)
    return


@app.cell
def _(data, helpers, summary):
    helpers.lasso_selected_coef_figure(
        data["X_train_s"], data["y_train"], data["feat_names"], summary["best_alpha"]
    ).show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## All methods compared (§3.6)

    OLS, Ridge, Lasso, PCR, and PLS on the same split — see `pcr_pls.py` for the component-count plots.
    """)
    return


@app.cell
def _(data, helpers, summary):
    helpers.compare_models(
        data["X_train_s"],
        data["X_test_s"],
        data["y_train"],
        data["y_test"],
        alpha_ridge=summary["best_alpha"],
        alpha_lasso=summary["best_alpha"],
        n_pcr=None,
        n_pls=None,
    )
    return


if __name__ == "__main__":
    app.run()

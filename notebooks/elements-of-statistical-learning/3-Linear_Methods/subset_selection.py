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
    # Subset selection: ESL Ch. 3

    *Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning. Chapter 3.*

    **Forward stepwise selection** on TMDB movie revenue: at each step the single feature that most reduces test MSE is added. Shows how test error first falls then plateaus (or rises) as more features enter.
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
    X, y, target = helpers.load_tmdb_xy(INPUTS)
    print(f"Loaded {len(X):,} rows | features: {list(X.columns)} | target: {target!r}")
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Forward stepwise selection (§3.3.2)

    Starting from the null model, at each step the single feature that most reduces test MSE is added permanently:

    1. Try every remaining feature in turn; score by test MSE.
    2. Add the best one to the active set.
    3. Repeat until all features are included.

    The plot shows how test MSE evolves as the model grows. The minimum marks the subset size with the best generalisation — adding more features beyond that point only incorporates noise. The order features enter reveals their marginal predictive contribution given what is already in the model.
    """)
    return


@app.cell
def _(X, helpers, y):
    data = helpers.scale_split(X, y)

    fig, selected_order = helpers.subset_selection_figure(
        data["X_train"], data["X_test"], data["y_train"], data["y_test"]
    )
    fig.show()

    print("Feature entry order:", selected_order)
    return


if __name__ == "__main__":
    app.run()

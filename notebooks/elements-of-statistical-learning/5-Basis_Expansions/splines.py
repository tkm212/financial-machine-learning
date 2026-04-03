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
    # Basis Expansions and Splines: ESL Ch. 5

    *Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning. §5.1-5.2.*

    Linear models are powerful but rigid — they can only fit straight hyperplanes.  **Basis expansions** replace $x$ with a vector of derived features $h(x) = (h_1(x), \ldots, h_M(x))$ and then apply a linear model in that richer space:

    $$f(x) = \sum_{m=1}^M \theta_m h_m(x)$$

    **Piecewise polynomials** (§5.2) split the domain at *knots* and fit a separate polynomial in each region.  **Splines** add *continuity constraints* across knot boundaries — a cubic spline is continuous in $f$, $f'$, and $f''$.  **Natural cubic splines** (§5.2.1) additionally require $f'' = 0$ beyond the boundary knots, reducing the effective degrees of freedom and preventing wild extrapolation.

    All visualisations use TMDB movie budget as the single predictor for revenue.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    import ch5_helpers as helpers

    _root, INPUTS, _outputs = helpers.init_paths()
    return INPUTS, helpers


@app.cell
def _(INPUTS, helpers):
    X, y, target = helpers.load_tmdb_xy(INPUTS)
    print(f"Loaded {len(X):,} rows | features: {list(X.columns)} | target: {target!r}")
    return X, target, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Piecewise polynomials vs cubic spline (§5.2)

    Three models on budget → revenue (subsampled for clarity):

    | Model | Constraints | DoF |
    |---|---|---|
    | Global linear | continuous, global | 2 |
    | Piecewise constant | none between regions | knots + 1 |
    | Cubic spline | $C^2$ continuity at knots | knots + 4 |

    The vertical dotted lines mark the knot locations (quantile-spaced).  The step function is maximally flexible within regions but discontinuous; the cubic spline imposes smoothness while still adapting to local structure.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_piecewise = helpers.piecewise_poly_figure(X, y, feat="budget", n_knots=4)
    fig_piecewise.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Natural cubic splines with varying knot count (§5.2.1)

    A **natural cubic spline** with $K$ knots has $K$ basis functions and $K$ degrees of freedom (the natural boundary condition removes the 4 extra parameters of a free cubic spline).

    More knots → more flexibility → lower training error but higher variance.  The curves span from near-linear (few knots) to wiggly (many knots).  On TMDB data the gain from extra knots plateaus quickly because the budget-revenue relationship is already well-captured by a handful of breakpoints.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_ncs, _ncs_summary = helpers.natural_cubic_spline_figure(X, y, feat="budget", knot_counts=[2, 4, 6, 10, 16])
    fig_ncs.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bias-variance tradeoff: test MSE vs number of knots

    Scanning the number of spline knots traces out the familiar bias-variance curve:

    - **Too few knots**: high bias — the curve cannot capture the true shape.
    - **Too many knots**: high variance — the model fits noise in the training set.

    The dashed line marks the knot count minimising test MSE.  Beyond that point, adding knots only increases variance without meaningfully reducing bias on new data.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_bv = helpers.spline_knot_bias_variance_figure(X, y, feat="budget", max_knots=18)
    fig_bv.show()
    return


if __name__ == "__main__":
    app.run()

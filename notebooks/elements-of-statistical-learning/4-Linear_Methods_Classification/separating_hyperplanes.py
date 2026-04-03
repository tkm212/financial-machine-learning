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
    # Separating Hyperplanes: ESL Ch. 4

    *Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning. §4.5.*

    When two classes are **linearly separable** there are infinitely many hyperplanes that correctly classify all training points.  ESL §4.5 introduces two historical approaches to choosing one:

    1. **Rosenblatt's Perceptron** (§4.5.1) — finds *some* separating hyperplane by gradient descent on misclassified points; converges but the solution depends on the initial weights and update order.
    2. **Optimal Separating Hyperplane** (§4.5.2) — the unique hyperplane that maximises the margin between the two classes; found by quadratic programming (= the linear SVM without slack).

    A synthetic 2D dataset is used so the boundary can be plotted directly.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    import ch4_helpers as helpers

    return (helpers,)


@app.cell
def _(helpers):
    X, y = helpers.make_separable_2d(n=200, random_state=42, margin=1.5)
    print(f"Dataset: {X.shape[0]} points, 2 features | class balance: {y.mean():.0%} class 1")
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rosenblatt's Perceptron (§4.5.1)

    The perceptron update rule corrects one misclassified point at a time:

    $$\beta \leftarrow \beta + y_i x_i, \quad \beta_0 \leftarrow \beta_0 + y_i \qquad \text{if } y_i(\beta^T x_i + \beta_0) \leq 0$$

    By the **perceptron convergence theorem**, if the data are linearly separable the algorithm terminates in a finite number of steps.  The convergence curve shows the number of misclassified training points per pass through the data (epoch) — it should reach zero once the boundary is found.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_conv, fig_perc, _tracer = helpers.perceptron_convergence_figure(X, y)
    fig_conv.show()
    fig_perc.show()
    print(f"Converged in {len(_tracer.misclassified)} epochs")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Optimal Separating Hyperplane (§4.5.2)

    Among all separating hyperplanes, the optimal one maximises the **margin** — the perpendicular distance between the boundary and the nearest points of each class:

    $$\max_{\beta, \beta_0, \|\beta\|=1}\ M \quad \text{subject to}\ y_i(\beta^T x_i + \beta_0) \geq M\ \forall i$$

    Equivalently, $\min \|\beta\|^2$ subject to $y_i(\beta^T x_i + \beta_0) \geq 1$.  The solution depends only on the **support vectors** — the points that lie exactly on the margin boundary (circled below).  The dashed lines show the $\pm 1$ margin hyperplanes; the margin width is $2/\|\beta\|$.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_svm = helpers.svm_margin_figure(X, y)
    fig_svm.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Perceptron vs optimal hyperplane

    The perceptron is guaranteed to converge to **a** separating hyperplane, but not necessarily the maximum-margin one.  Depending on the initial weights and the order in which misclassifications are processed, it can produce any valid separator.  The SVM always finds the unique maximum-margin boundary, which generalises better on unseen data because it sits furthest from both classes.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_compare = helpers.perceptron_vs_svm_figure(X, y)
    fig_compare.show()
    return


if __name__ == "__main__":
    app.run()

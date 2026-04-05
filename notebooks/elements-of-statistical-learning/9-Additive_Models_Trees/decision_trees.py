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
    # Decision Trees (CART) — ESL Ch. 9

    *Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning. §9.2.*

    **CART** (Classification and Regression Trees; Breiman et al. 1984) builds a binary tree
    by recursively partitioning the feature space.  At each node, it finds the split
    $(j, s)$ that minimises the within-region sum of squares:

    $$\min_{j,\, s} \left[
      \min_{c_1} \sum_{x_i \in R_1(j,s)} (y_i - c_1)^2 +
      \min_{c_2} \sum_{x_i \in R_2(j,s)} (y_i - c_2)^2
    \right]$$

    where $R_1(j,s) = \{x \mid x_j \le s\}$ and $R_2(j,s) = \{x \mid x_j > s\}$.

    The optimal constants are the within-region means $\hat{c}_m = \text{ave}(y_i \mid x_i \in R_m)$.

    ## Tree size and complexity

    A fully grown tree memorises the training data (variance $\to \infty$).  Two strategies
    control complexity:

    1. **Limiting depth** (`max_depth`): stops splitting once the tree reaches a preset depth.
    2. **Cost-complexity pruning** (§9.2.2): grow a large tree then prune back by minimising
       $R_\alpha(T) = R(T) + \alpha |\tilde{T}|$,
       where $|\tilde{T}|$ is the number of terminal nodes and $\alpha \ge 0$ is a regularisation
       parameter.  As $\alpha$ increases, more nodes are collapsed.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    import ch9_helpers as helpers

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
    ## Tree depth vs train/test error (§9.2)

    We fit CART trees of increasing depth on five numeric TMDB features and evaluate on a
    held-out 25% test set.

    - **Shallow trees** (depth 1-2): high train and test error — the model underfits.  A depth-1
      tree (stump) uses only a single split.
    - **Intermediate depth**: test error reaches its minimum — the tree is complex enough to
      capture structure without memorising noise.
    - **Deep trees**: training error approaches zero while test error rises — classic variance
      dominance.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_depth, depth_summary = helpers.tree_depth_error_figure(X, y, max_depth_range=15)
    fig_depth.show()
    print(f"Best depth: {depth_summary['best_depth']} | best test MSE: {depth_summary['best_test_mse']:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cost-complexity pruning path (§9.2.2)

    Starting from a fully-grown tree, we trace the Breiman *minimal cost-complexity pruning*
    sequence: a series of trees $T_0 \supset T_1 \supset \cdots \supset T_{\text{root}}$
    obtained by successively collapsing the internal node whose removal causes the smallest
    increase in $R_\alpha(T)$.

    Plotting test MSE against the **number of terminal nodes**:
    - Moving left (fewer leaves): aggressive pruning eventually removes genuinely useful splits,
      increasing bias.
    - The **minimum test MSE** identifies the Goldilocks tree size — enough leaves to
      capture the signal, but not so many that noise is overfit.

    In practice, one typically selects $\alpha$ by cross-validation and then applies that
    pruning level to a tree fit on all training data.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_prune, prune_summary = helpers.cost_complexity_pruning_figure(X, y)
    fig_prune.show()
    print(
        f"Best alpha: {prune_summary['best_alpha']:.5f} | "
        f"leaves: {prune_summary['best_n_leaves']} | "
        f"test MSE: {prune_summary['best_test_mse']:.4f}"
    )
    return


if __name__ == "__main__":
    app.run()

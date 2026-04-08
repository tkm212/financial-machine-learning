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
    # Neural Networks — ESL Ch. 11

    *Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning. §11.3-11.5.*

    ## Network architecture (§11.3)

    A **single hidden layer neural network** with $M$ hidden units models:

    $$f_k(X) = g_k\!\left(\beta_{k0} + \sum_{m=1}^{M} \beta_{km} \sigma\!\left(\alpha_{m0} + \alpha_m^\top X\right)\right)$$

    where:
    - $\sigma(v) = 1 / (1 + e^{-v})$ is the logistic sigmoid activation.
    - $g_k$ is the output activation: softmax for multiclass, logistic for binary, identity for regression.
    - $Z_m = \sigma(\alpha_{m0} + \alpha_m^\top X)$ are the derived features (hidden units).

    The model is a **two-stage regression**: first project inputs through $M$ nonlinear features,
    then fit a generalised linear model on those features.

    ## Fitting neural networks: backpropagation (§11.4)

    Parameters $\theta = \{\alpha_{0m}, \alpha_m, \beta_{0k}, \beta_{km}\}$ are estimated by
    minimising the regularised cross-entropy (binary classification):

    $$R(\theta) = -\sum_{i=1}^N \left[y_i \log \hat{f}(x_i) + (1-y_i)\log(1 - \hat{f}(x_i))\right] + \frac{\lambda}{2} \|\theta\|^2$$

    **Backpropagation** (Rumelhart et al. 1986) computes $\nabla_\theta R$ efficiently
    via the chain rule, propagating error signals from output to input:

    $$\delta_m = \sigma'(\alpha_m^\top x) \sum_k \beta_{km} \delta_k$$

    This avoids redundant computations by reusing intermediate activations.
    Weights are then updated by (stochastic) gradient descent:
    $$\theta \leftarrow \theta - \eta \nabla_\theta R$$
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    import ch11_helpers as helpers

    _root, INPUTS, _outputs = helpers.init_paths()
    return INPUTS, helpers


@app.cell
def _(INPUTS, helpers):
    X, y, target = helpers.load_tmdb_classification_xy(INPUTS)
    print(f"Loaded {len(X):,} rows | target: {target!r} | class balance: {y.mean():.2%} positive")
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training curve: loss and test error vs epoch (§11.4)

    We train a 1-hidden-layer network with 50 sigmoid units by tracking the cross-entropy
    training loss and the test misclassification rate at each epoch.

    Key observations:
    - **Training loss** decreases monotonically — SGD directly minimises it.
    - **Test error** typically improves early then flattens or rises (overfitting).
    - The dashed line marks the epoch of minimum test error.

    **Early stopping** (§11.5.2) terminates training at this point, using the epoch index
    as an implicit regularisation parameter: fewer epochs means less capacity to overfit.
    This is equivalent to $L^2$ weight decay when training with gradient descent —
    both shrink weights away from the unregularised minimum.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_curve, curve_summary = helpers.nn_training_curve_figure(X, y, hidden_layer_sizes=(50,), max_epochs=150)
    fig_curve.show()
    print(
        f"Best epoch: {curve_summary['best_epoch']} | "
        f"test error: {curve_summary['best_test_error']:.3%} | "
        f"final train loss: {curve_summary['final_train_loss']:.4f}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Weight decay: regularisation by penalising large weights (§11.5.2)

    Adding an $L^2$ penalty $\frac{\lambda}{2}\|\theta\|^2$ to the loss prevents the
    network from assigning arbitrarily large weights to individual inputs.

    The trade-off:
    - $\lambda = 0$: no regularisation — model fits all training noise.
    - $\lambda$ small (e.g. 0.001): mild shrinkage, minimal bias added.
    - $\lambda$ large (e.g. 0.1): strong shrinkage — weights approach zero and the
      network converges to a near-linear model.

    In sklearn's `MLPClassifier`, the `alpha` parameter is precisely this $\lambda$.
    Optimal $\lambda$ is typically selected by cross-validation (§7.3).
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_wd = helpers.nn_weight_decay_figure(
        X, y, hidden_layer_sizes=(50,), alphas=[0.0, 0.001, 0.01, 0.1], max_epochs=150
    )
    fig_wd.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Architecture: number of hidden units and layers (§11.5.4)

    The network architecture — number of hidden layers and units per layer — is a
    **structural hyperparameter** that determines model capacity.

    Guidelines from ESL §11.5.4:
    - **A single hidden layer** suffices for most problems in practice.
    - **More units** increase capacity but require stronger regularisation.
    - **Deeper networks** can represent hierarchical features more efficiently,
      but are harder to train (vanishing/exploding gradients).

    We compare several architectures by 5-fold cross-validated accuracy.
    Single-tuple architectures like `(50,)` are one hidden layer with 50 units;
    `(100, 50)` is two layers.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_arch, arch_summary = helpers.nn_architecture_figure(
        X,
        y,
        architectures=[(5,), (20,), (50,), (100,), (200,), (50, 20), (100, 50)],
        n_cv=5,
    )
    fig_arch.show()
    print(f"Best architecture: {arch_summary['best_arch']} | CV accuracy: {arch_summary['best_cv_accuracy']:.3%}")
    return


if __name__ == "__main__":
    app.run()

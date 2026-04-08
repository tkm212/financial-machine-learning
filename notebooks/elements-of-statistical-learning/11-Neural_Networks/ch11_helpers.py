"""Shared helpers for ESL Chapter 11 (Neural Networks) notebooks."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

# Suppress the per-epoch ConvergenceWarning that fires when max_iter=1 is used
# intentionally for warm-start epoch-by-epoch training curve tracking.
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def find_project_root(max_up: int = 12) -> Path:
    p = Path.cwd().resolve()
    for _ in range(max_up):
        if (p / "pyproject.toml").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    msg = "Could not find project root (pyproject.toml)."
    raise RuntimeError(msg)


def init_paths() -> tuple[Path, Path, Path]:
    """Return ``(project_root, inputs_dir, outputs_dir)`` and put the package on ``sys.path``."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root, root / "inputs", root / "outputs"


def load_tmdb_classification_xy(inputs_dir: Path) -> tuple[pd.DataFrame, pd.Series, str]:
    from financial_machine_learning.esl_loaders import load_tmdb_revenue_classification

    return load_tmdb_revenue_classification(inputs_dir)


def load_tmdb_xy(inputs_dir: Path) -> tuple[pd.DataFrame, pd.Series, str]:
    from financial_machine_learning.esl_loaders import load_tmdb_revenue_regression

    return load_tmdb_revenue_regression(inputs_dir)


def _prepare_cls_arrays(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    max_rows: int = 2000,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if feats is None:
        feats = ["budget", "popularity", "runtime", "vote_average", "vote_count"]
    feats = [f for f in feats if f in X.columns]

    if len(X) > max_rows:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=max_rows, replace=False)
        x_sub = X.iloc[idx][feats].values.astype(float)
        y_sub = np.asarray(y, dtype=int).ravel()[idx]
    else:
        x_sub = X[feats].values.astype(float)
        y_sub = np.asarray(y, dtype=int).ravel()

    x_sub = np.log1p(np.maximum(x_sub, 0))
    return x_sub, y_sub


def _prepare_reg_arrays(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    max_rows: int = 2000,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if feats is None:
        feats = ["budget", "popularity", "runtime", "vote_average", "vote_count"]
    feats = [f for f in feats if f in X.columns]

    if len(X) > max_rows:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=max_rows, replace=False)
        x_sub = X.iloc[idx][feats].values.astype(float)
        y_sub = np.asarray(y, dtype=float).ravel()[idx]
    else:
        x_sub = X[feats].values.astype(float)
        y_sub = np.asarray(y, dtype=float).ravel()

    x_sub = np.log1p(np.maximum(x_sub, 0))
    y_sub = np.log1p(np.maximum(y_sub, 0))
    return x_sub, y_sub


# ---------------------------------------------------------------------------
# Neural network training curve (§11.4)
# ---------------------------------------------------------------------------


def nn_training_curve_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    hidden_layer_sizes: tuple[int, ...] = (50,),
    max_epochs: int = 150,
    max_rows: int = 2000,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    Track cross-entropy loss and test error over training epochs for an MLP (§11.4).

    Uses warm_start=True with max_iter=1 per step to expose the epoch-by-epoch dynamics.
    Backpropagation minimises cross-entropy via SGD:
        w <- w - eta * dL/dw
    where gradients are computed by the chain rule through all hidden layers.

    Key observations from the plot:
    - Train loss decreases monotonically — the network is directly minimising it.
    - Test error follows a U-shape if max_epochs is large enough: early stopping
      (§11.5.2) truncates training at the epoch of minimum test error.
    """
    x_arr, y_cls = _prepare_cls_arrays(X, y, feats, max_rows=max_rows)
    scaler = StandardScaler()
    x_arr = scaler.fit_transform(x_arr)
    X_tr, X_te, y_tr, y_te = train_test_split(x_arr, y_cls, test_size=0.25, random_state=0)

    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="logistic",
        max_iter=1,
        warm_start=True,
        alpha=0.0,
        random_state=0,
        learning_rate_init=0.01,
    )

    train_losses: list[float] = []
    test_errors: list[float] = []
    for _ in range(max_epochs):
        clf.fit(X_tr, y_tr)
        train_losses.append(float(clf.loss_))
        test_errors.append(float(1 - accuracy_score(y_te, clf.predict(X_te))))

    epochs = list(range(1, max_epochs + 1))
    best_epoch = epochs[int(np.argmin(test_errors))]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=train_losses,
            mode="lines",
            name="Train loss (cross-entropy)",
            line={"color": "steelblue"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=test_errors,
            mode="lines",
            name="Test error rate",
            line={"color": "tomato"},
            yaxis="y2",
        )
    )
    fig.add_vline(x=best_epoch, line_dash="dash", line_color="grey", annotation_text=f"best epoch={best_epoch}")
    fig.update_layout(
        title=f"Neural network training curve (hidden={hidden_layer_sizes}) — §11.4",
        xaxis_title="epoch",
        yaxis_title="cross-entropy loss",
        yaxis2={"title": "test error rate", "overlaying": "y", "side": "right", "showgrid": False},
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig, {
        "best_epoch": best_epoch,
        "best_test_error": min(test_errors),
        "final_train_loss": train_losses[-1],
    }


# ---------------------------------------------------------------------------
# Weight decay (§11.5.2)
# ---------------------------------------------------------------------------


def nn_weight_decay_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    alphas: list[float] | None = None,
    hidden_layer_sizes: tuple[int, ...] = (50,),
    max_epochs: int = 150,
    max_rows: int = 2000,
) -> go.Figure:
    """
    Effect of L2 weight decay (alpha) on neural network generalisation (§11.5.2).

    Weight decay augments the cross-entropy loss with a penalty on weight magnitudes:
        L_reg = L + (alpha/2) * ||W||_F^2

    This has the same effect as a Gaussian prior on weights in the Bayesian interpretation.
    Large alpha shrinks weights toward zero — effectively reducing the model to a simpler
    function (fewer effective parameters).  Small alpha gives unregularised training.

    The optimal alpha lies in a Goldilocks zone: large enough to prevent overfitting,
    small enough not to underfit.  In practice alpha is chosen by cross-validation.
    """
    if alphas is None:
        alphas = [0.0, 0.001, 0.01, 0.1]
    x_arr, y_cls = _prepare_cls_arrays(X, y, feats, max_rows=max_rows)
    scaler = StandardScaler()
    x_arr = scaler.fit_transform(x_arr)
    X_tr, X_te, y_tr, y_te = train_test_split(x_arr, y_cls, test_size=0.25, random_state=0)

    colors = ["grey", "steelblue", "tomato", "green", "purple"]
    fig = go.Figure()

    for i, alpha in enumerate(alphas):
        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="logistic",
            max_iter=1,
            warm_start=True,
            alpha=alpha,
            random_state=0,
            learning_rate_init=0.01,
        )
        test_errors: list[float] = []
        for _ in range(max_epochs):
            clf.fit(X_tr, y_tr)
            test_errors.append(float(1 - accuracy_score(y_te, clf.predict(X_te))))

        epochs = list(range(1, max_epochs + 1))
        best_e = min(test_errors)
        label = "no decay (alpha=0)" if alpha == 0 else f"alpha={alpha}"
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=test_errors,
                mode="lines",
                name=f"{label}  (best={best_e:.3f})",
                line={"color": colors[i % len(colors)]},
            )
        )

    fig.update_layout(
        title=f"Weight decay: test error vs epoch (hidden={hidden_layer_sizes}) — §11.5.2",
        xaxis_title="epoch",
        yaxis_title="test error rate",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


# ---------------------------------------------------------------------------
# Architecture comparison: hidden units and layers (§11.5.4)
# ---------------------------------------------------------------------------


def nn_architecture_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    architectures: list[tuple[int, ...]] | None = None,
    max_rows: int = 2000,
    n_cv: int = 5,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    Compare neural network architectures via cross-validated accuracy (§11.5.4).

    The number of hidden units M controls capacity:
    - Too few (M small): underfitting — cannot represent complex decision surfaces.
    - Too many (M large): overfitting — memorises training noise.
    - More layers allow hierarchical feature extraction but are harder to train.

    The universal approximation theorem guarantees that a single hidden layer with
    sufficiently many sigmoid units can approximate any continuous function on a compact
    set — but it gives no bound on M, and multiple layers are often more efficient in
    practice.  CV accuracy is used to select M without a held-out test set.
    """
    if architectures is None:
        architectures = [(5,), (20,), (50,), (100,), (200,), (50, 20), (100, 50)]
    x_arr, y_cls = _prepare_cls_arrays(X, y, feats, max_rows=max_rows)
    scaler = StandardScaler()
    x_arr = scaler.fit_transform(x_arr)

    arch_labels: list[str] = []
    mean_accs: list[float] = []
    std_accs: list[float] = []

    for arch in architectures:
        clf = MLPClassifier(
            hidden_layer_sizes=arch,
            activation="logistic",
            max_iter=500,
            alpha=0.01,
            random_state=0,
        )
        scores = cross_val_score(clf, x_arr, y_cls, cv=n_cv, scoring="accuracy")
        arch_labels.append(str(arch))
        mean_accs.append(float(scores.mean()))
        std_accs.append(float(scores.std()))

    best_idx = int(np.argmax(mean_accs))

    fig = go.Figure(
        go.Bar(
            x=arch_labels,
            y=mean_accs,
            error_y={"type": "data", "array": std_accs, "visible": True},
            marker_color="steelblue",
            name=f"{n_cv}-fold CV accuracy (±1 SD)",
        )
    )
    fig.update_layout(
        title=f"Architecture comparison: {n_cv}-fold CV accuracy — §11.5.4",
        xaxis_title="hidden layer sizes",
        yaxis_title="CV accuracy",
        yaxis={"range": [max(0.0, min(mean_accs) - 0.05), 1.0]},
        template="plotly_white",
    )
    return fig, {
        "best_arch": arch_labels[best_idx],
        "best_cv_accuracy": mean_accs[best_idx],
    }


# ---------------------------------------------------------------------------
# Projection Pursuit Regression (§11.2)
# ---------------------------------------------------------------------------


def ppr_vs_linear_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    M_values: list[int] | None = None,
    max_rows: int = 2000,
    n_cv: int = 5,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    Approximate PPR via 1-hidden-layer MLPs and compare to OLS (§11.2).

    PPR models:
        f(X) = sum_{m=1}^{M} g_m(w_m^T X)

    Each term is a ridge function — nonlinear in a single linear projection of X.
    A 1-hidden-layer MLP with M hidden units and tanh activations is exactly this form
    (with the additional constraint that g_m = tanh for all m).  The PPR algorithm instead
    estimates each g_m independently by backfitting, giving more flexibility per term.

    This approximation lets us ask: how many ridge terms M are needed before MSE
    stops improving relative to OLS?
    """
    if M_values is None:
        M_values = [1, 2, 5, 10, 20, 50]
    x_arr, y_arr = _prepare_reg_arrays(X, y, feats, max_rows=max_rows)
    scaler = StandardScaler()
    x_arr = scaler.fit_transform(x_arr)

    labels = ["OLS (M=0)"] + [f"PPR  M={m}" for m in M_values]
    mse_means: list[float] = []
    mse_stds: list[float] = []

    lr_scores = cross_val_score(LinearRegression(), x_arr, y_arr, cv=n_cv, scoring="neg_mean_squared_error")
    mse_means.append(float(-lr_scores.mean()))
    mse_stds.append(float(lr_scores.std()))

    for m in M_values:
        mlp = MLPRegressor(
            hidden_layer_sizes=(m,),
            activation="tanh",
            max_iter=500,
            alpha=0.01,
            random_state=0,
        )
        scores = cross_val_score(mlp, x_arr, y_arr, cv=n_cv, scoring="neg_mean_squared_error")
        mse_means.append(float(-scores.mean()))
        mse_stds.append(float(scores.std()))

    best_idx = int(np.argmin(mse_means))

    bar_colors = ["tomato"] + ["steelblue"] * len(M_values)
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=mse_means,
            error_y={"type": "data", "array": mse_stds, "visible": True},
            marker_color=bar_colors,
            name=f"{n_cv}-fold CV MSE (±1 SD)",
        )
    )
    fig.update_layout(
        title=f"PPR (1-hidden-layer MLP) vs OLS: {n_cv}-fold CV MSE — §11.2",
        xaxis_title="model / number of ridge terms M",
        yaxis_title="CV MSE (log₁p-space)",
        template="plotly_white",
    )
    return fig, {
        "best_model": labels[best_idx],
        "best_mse": mse_means[best_idx],
        "ols_mse": mse_means[0],
        "ppr_improvement_pct": 100 * (mse_means[0] - min(mse_means[1:])) / mse_means[0],
    }


def ppr_ridge_functions_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    M: int = 3,
    max_rows: int = 2000,
) -> go.Figure:
    """
    Visualise the M ridge functions learned by a 1-hidden-layer MLP (§11.2).

    After fitting, each hidden unit h_m computes:
        h_m(x) = sigma(w_m^T x + b_m)

    The output layer is:
        f(x) = sum_m beta_m * h_m(x) + b_out

    We plot each h_m evaluated over a grid of its projection w_m^T x on the training data,
    showing how the network decomposes the prediction into M univariate ridge functions.
    """
    if feats is None:
        feats = ["budget", "popularity", "runtime", "vote_average", "vote_count"]
    feats = [f for f in feats if f in X.columns]

    x_arr, y_arr = _prepare_reg_arrays(X, y, feats, max_rows=max_rows)
    scaler = StandardScaler()
    x_arr = scaler.fit_transform(x_arr)

    mlp = MLPRegressor(
        hidden_layer_sizes=(M,),
        activation="tanh",
        max_iter=500,
        alpha=0.01,
        random_state=0,
    )
    mlp.fit(x_arr, y_arr)

    W = mlp.coefs_[0]  # shape (n_features, M)
    b = mlp.intercepts_[0]  # shape (M,)
    beta = mlp.coefs_[1].ravel()  # shape (M,)

    projections = x_arr @ W + b  # shape (n_samples, M)
    hidden_acts = np.tanh(projections)  # shape (n_samples, M)

    colors = ["steelblue", "tomato", "green", "purple", "orange"]
    fig = go.Figure()
    for m in range(M):
        proj_m = projections[:, m]
        act_m = hidden_acts[:, m] * beta[m]
        order = np.argsort(proj_m)
        label = f"ridge {m + 1}  (β={beta[m]:.2f})"
        fig.add_trace(
            go.Scatter(
                x=proj_m[order],
                y=act_m[order],
                mode="lines",
                name=label,
                line={"color": colors[m % len(colors)]},
            )
        )

    fig.update_layout(
        title=f"PPR ridge functions: M={M} hidden units (weighted activations) — §11.2",
        xaxis_title="projection  wₘᵀx + bₘ",
        yaxis_title="βₘ · σ(wₘᵀx + bₘ)",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig

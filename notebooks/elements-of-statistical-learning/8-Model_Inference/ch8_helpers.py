"""Shared helpers for ESL Chapter 8 (Model Inference and Averaging) notebooks."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor


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


def load_tmdb_xy(inputs_dir: Path) -> tuple[pd.DataFrame, pd.Series, str]:
    from financial_machine_learning.esl_loaders import load_tmdb_revenue_regression

    return load_tmdb_revenue_regression(inputs_dir)


def _log_feature(
    X: pd.DataFrame,
    y: pd.Series,
    feat: str,
    *,
    max_rows: int = 3000,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if len(X) > max_rows:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=max_rows, replace=False)
        X, y = X.iloc[idx], y.iloc[idx]
    x = np.log1p(X[feat].values.astype(float)).reshape(-1, 1)
    y_log = np.log1p(np.asarray(y, dtype=float))
    return x, y_log


# ---------------------------------------------------------------------------
# EM algorithm internals (§8.5)
# ---------------------------------------------------------------------------


def _gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def _em_1d(
    x: np.ndarray,
    K: int = 2,
    max_iter: int = 100,
    tol: float = 1e-6,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Fit K-component 1D Gaussian mixture via EM; return (means, stds, weights, log_likelihoods)."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(x)
    idx = rng.choice(n, K, replace=False)
    means = x[idx].copy().astype(float)
    stds = np.full(K, float(x.std()))
    weights = np.full(K, 1.0 / K)

    log_liks: list[float] = []
    for _ in range(max_iter):
        # E-step: responsibilities r[i,k] = pi_k N(x_i; mu_k, sig_k) / sum_j pi_j N(x_i; mu_j, sig_j)
        resp = np.column_stack([weights[k] * _gaussian_pdf(x, means[k], stds[k]) for k in range(K)])
        resp_sum = resp.sum(axis=1, keepdims=True)
        log_liks.append(float(np.sum(np.log(resp_sum.ravel() + 1e-300))))
        resp = resp / (resp_sum + 1e-300)

        # M-step: update parameters to maximise Q(θ | θ_old)
        nk = resp.sum(axis=0)
        weights = nk / n
        means = (resp * x[:, None]).sum(axis=0) / nk
        stds = np.sqrt((resp * (x[:, None] - means[None, :]) ** 2).sum(axis=0) / nk)
        stds = np.maximum(stds, 1e-6)

        if len(log_liks) > 1 and abs(log_liks[-1] - log_liks[-2]) < tol:
            break

    return means, stds, weights, log_liks


# ---------------------------------------------------------------------------
# EM visualisation figures (§8.5)
# ---------------------------------------------------------------------------


def em_1d_figure(
    n_samples: int = 500,
    K: int = 2,
    random_state: int = 0,
) -> go.Figure:
    """
    EM on synthetic 1D bimodal data (§8.5).

    True mixture: 0.4 * N(-2, 1^2) + 0.6 * N(2, 1.5^2).
    Shows fitted component densities vs the true density.
    """
    rng = np.random.default_rng(random_state)
    true_weights = np.array([0.4, 0.6])
    true_means = np.array([-2.0, 2.0])
    true_stds = np.array([1.0, 1.5])

    n0 = int(true_weights[0] * n_samples)
    x = np.concatenate([
        rng.normal(true_means[0], true_stds[0], n0),
        rng.normal(true_means[1], true_stds[1], n_samples - n0),
    ])
    rng.shuffle(x)

    means, stds, weights, _ = _em_1d(x, K=K, rng=rng)
    x_grid = np.linspace(float(x.min()) - 1.0, float(x.max()) + 1.0, 300)

    true_pdf = np.zeros(len(x_grid))
    for k in range(K):
        true_pdf += true_weights[k] * _gaussian_pdf(x_grid, true_means[k], true_stds[k])

    mixture_pdf = np.zeros(len(x_grid))
    for k in range(K):
        mixture_pdf += weights[k] * _gaussian_pdf(x_grid, means[k], stds[k])

    component_colors = ["steelblue", "tomato", "green", "purple"]
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=x,
            name="Observed data",
            histnorm="probability density",
            marker_color="lightblue",
            opacity=0.5,
            nbinsx=40,
        )
    )
    for k in range(K):
        comp = weights[k] * _gaussian_pdf(x_grid, means[k], stds[k])
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=comp,
                mode="lines",
                name=f"Component {k + 1}: mean={means[k]:.2f}, std={stds[k]:.2f}",
                line={"dash": "dash", "color": component_colors[k % len(component_colors)]},
            )
        )
    fig.add_trace(
        go.Scatter(x=x_grid, y=mixture_pdf, mode="lines", name="Fitted mixture", line={"color": "black", "width": 2})
    )
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=true_pdf,
            mode="lines",
            name="True density",
            line={"dash": "dot", "color": "grey"},
        )
    )
    fig.update_layout(
        title=f"EM Gaussian mixture (K={K}, n={n_samples}) — §8.5",
        xaxis_title="x",
        yaxis_title="density",
        barmode="overlay",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


def em_convergence_figure(
    n_samples: int = 500,
    K: int = 2,
    n_restarts: int = 5,
    random_state: int = 0,
) -> go.Figure:
    """
    Log-likelihood per EM iteration for multiple random initialisations (§8.5).

    Different initialisations may converge to different local maxima, illustrating
    that EM is not guaranteed to find the global maximum.
    """
    rng = np.random.default_rng(random_state)
    n0 = int(0.4 * n_samples)
    x = np.concatenate([
        rng.normal(-2.0, 1.0, n0),
        rng.normal(2.0, 1.5, n_samples - n0),
    ])
    rng.shuffle(x)

    fig = go.Figure()
    for restart in range(n_restarts):
        seed_rng = np.random.default_rng(random_state + restart * 17)
        _, _, _, log_liks = _em_1d(x, K=K, max_iter=80, rng=seed_rng)
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(log_liks) + 1)),
                y=log_liks,
                mode="lines+markers",
                name=f"Init {restart + 1}",
                marker={"size": 4},
            )
        )
    fig.update_layout(
        title=f"EM log-likelihood convergence (K={K}, {n_restarts} restarts) — §8.5",
        xaxis_title="iteration",
        yaxis_title="log-likelihood",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


# ---------------------------------------------------------------------------
# Bootstrap confidence bands (§8.2)
# ---------------------------------------------------------------------------


def bootstrap_confidence_bands_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feat: str = "budget",
    *,
    degree: int = 3,
    n_boot: int = 200,
    alpha: float = 0.05,
    max_rows: int = 2000,
) -> go.Figure:
    """
    Nonparametric bootstrap confidence bands for a polynomial fit (§8.2).

    For each bootstrap resample, we refit the polynomial and evaluate on a grid.
    The pointwise (1-alpha) confidence band is the alpha/2 and 1-alpha/2 quantiles across resamples.
    """
    x, y_log = _log_feature(X, y, feat, max_rows=max_rows)
    n = len(x)
    pipe = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        LinearRegression(),
    )
    pipe.fit(x, y_log)

    x_grid = np.linspace(float(x.min()), float(x.max()), 200).reshape(-1, 1)
    y_hat = pipe.predict(x_grid)

    rng = np.random.default_rng(42)
    boot_preds = np.zeros((n_boot, len(x_grid)))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        pipe_b = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),
            StandardScaler(),
            LinearRegression(),
        )
        pipe_b.fit(x[idx], y_log[idx])
        boot_preds[b] = pipe_b.predict(x_grid)

    lo = np.quantile(boot_preds, alpha / 2, axis=0)
    hi = np.quantile(boot_preds, 1 - alpha / 2, axis=0)

    x_flat = x_grid.ravel()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x.ravel(),
            y=y_log,
            mode="markers",
            name="Data",
            marker={"size": 3, "color": "lightblue", "opacity": 0.5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x_flat, x_flat[::-1]]),
            y=np.concatenate([hi, lo[::-1]]),
            fill="toself",
            fillcolor="rgba(31,119,180,0.15)",
            line={"color": "rgba(0,0,0,0)"},
            name=f"{int((1 - alpha) * 100)}% bootstrap CI",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_flat, y=y_hat, mode="lines", name=f"Degree-{degree} fit", line={"color": "steelblue", "width": 2}
        )
    )
    fig.update_layout(
        title=f"Bootstrap confidence bands on `{feat}` (degree={degree}, B={n_boot}) — §8.2",
        xaxis_title=f"log₁p({feat})",
        yaxis_title="log₁p(revenue)",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


# ---------------------------------------------------------------------------
# Bagging (§8.7)
# ---------------------------------------------------------------------------


def bagging_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feat: str = "budget",
    *,
    max_bags: int = 50,
    tree_depth: int = 5,
    max_rows: int = 2000,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    Bagging variance reduction (§8.7).

    Compares a single decision tree to a bagged ensemble of increasing size B.
    Test MSE is averaged over 10 random train/test splits to reduce noise.
    """
    x, y_log = _log_feature(X, y, feat, max_rows=max_rows)
    n_splits = 10
    bag_sizes = [1, 2, 5, 10, 20, 30, 40, 50]
    bag_sizes = [b for b in bag_sizes if b <= max_bags]

    single_mse_runs: list[float] = []
    bagged_mse: dict[int, list[float]] = {b: [] for b in bag_sizes}

    for split_seed in range(n_splits):
        X_tr, X_te, y_tr, y_te = train_test_split(x, y_log, test_size=0.25, random_state=split_seed)

        single = DecisionTreeRegressor(max_depth=tree_depth, random_state=0)
        single.fit(X_tr, y_tr)
        single_mse_runs.append(float(mean_squared_error(y_te, single.predict(X_te))))

        for b in bag_sizes:
            bag = BaggingRegressor(
                estimator=DecisionTreeRegressor(max_depth=tree_depth),
                n_estimators=b,
                random_state=split_seed,
            )
            bag.fit(X_tr, y_tr)
            bagged_mse[b].append(float(mean_squared_error(y_te, bag.predict(X_te))))

    single_avg = float(np.mean(single_mse_runs))
    bagged_avg = [float(np.mean(bagged_mse[b])) for b in bag_sizes]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bag_sizes,
            y=bagged_avg,
            mode="lines+markers",
            name="Bagged ensemble (test MSE)",
        )
    )
    fig.add_hline(
        y=single_avg,
        line_dash="dash",
        line_color="tomato",
        annotation_text=f"Single tree MSE = {single_avg:.4f}",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title=f"Bagging variance reduction on `{feat}` (tree depth={tree_depth}) — §8.7",
        xaxis_title="number of bootstrap trees B",
        yaxis_title="test MSE (log₁p-space)",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    best_b = bag_sizes[int(np.argmin(bagged_avg))]
    return fig, {"single_tree_mse": single_avg, "best_b": best_b, "best_bagged_mse": min(bagged_avg)}

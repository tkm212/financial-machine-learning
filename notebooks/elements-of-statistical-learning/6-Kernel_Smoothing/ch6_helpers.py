"""Shared helpers for ESL Chapter 6 (Kernel Smoothing Methods) notebooks."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.neighbors import KernelDensity


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


def load_tmdb_cls(inputs_dir: Path) -> tuple[pd.DataFrame, pd.Series, str]:
    """Binary classification: high_revenue = 1 if revenue >= median (else 0)."""
    from financial_machine_learning.esl_loaders import load_tmdb_revenue_classification

    return load_tmdb_revenue_classification(inputs_dir)


# ---------------------------------------------------------------------------
# Kernel functions (§6.1)
# ---------------------------------------------------------------------------


def _gaussian_kernel(u: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)


def _epanechnikov_kernel(u: np.ndarray) -> np.ndarray:
    return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)


def _uniform_kernel(u: np.ndarray) -> np.ndarray:
    return np.where(np.abs(u) <= 1, 0.5, 0.0)


_KERNELS: dict[str, Any] = {
    "gaussian": _gaussian_kernel,
    "epanechnikov": _epanechnikov_kernel,
    "uniform": _uniform_kernel,
}


# ---------------------------------------------------------------------------
# Core smoothers
# ---------------------------------------------------------------------------


def _nw_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_grid: np.ndarray,
    bw: float,
    kernel: str = "gaussian",
) -> np.ndarray:
    """Nadaraya-Watson estimator (§6.1), vectorised over the grid."""
    K = _KERNELS[kernel]
    u = (x_grid[:, None] - x_train[None, :]) / bw  # (n_grid, n_train)
    W = K(u)
    W_sum = W.sum(axis=1, keepdims=True)
    W_norm = W / np.where(W_sum > 0, W_sum, 1.0)
    return W_norm @ y_train


def _nw_loo_cv(
    x_train: np.ndarray,
    y_train: np.ndarray,
    bw: float,
    kernel: str = "gaussian",
) -> float:
    """
    LOO cross-validation score for NW via the linear smoother shortcut (§6.2).

    For a linear smoother hat{y} = Hy, the LOO residual is:
        (y_i - hat{f}(x_i)) / (1 - H_{ii})
    For NW: H_{ii} = K(0) / sum_j K((x_i - x_j)/h).
    """
    K = _KERNELS[kernel]
    u = (x_train[:, None] - x_train[None, :]) / bw  # (n, n)
    W = K(u)
    W_sum = W.sum(axis=1)
    f_hat = (W @ y_train) / W_sum
    k0 = float(K(np.array([0.0]))[0])
    l_ii = k0 / W_sum  # diagonal of smoother matrix
    resid = y_train - f_hat
    denom = 1.0 - l_ii
    safe_denom = np.where(np.abs(denom) > 1e-10, denom, np.nan)
    loo_resid = resid / safe_denom
    return float(np.nanmean(loo_resid**2))


def _local_linear_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_grid: np.ndarray,
    bw: float,
) -> np.ndarray:
    """
    Local linear regression (§6.1.1) with Gaussian kernel — closed-form, vectorised.

    Minimises sum_i K_h(x_0, x_i) [y_i - alpha(x_0) - beta(x_0)(x_i - x_0)]^2.
    The intercept alpha(x_0) = hat{f}(x_0) via the 2x2 normal equations.
    """
    xc = x_grid[:, None] - x_train[None, :]  # centred differences (n_grid, n_train)
    w = _gaussian_kernel(xc / bw)
    y = y_train[None, :]
    s0 = w.sum(axis=1)
    s1 = (w * xc).sum(axis=1)
    s2 = (w * xc**2).sum(axis=1)
    t0 = (w * y).sum(axis=1)
    t1 = (w * xc * y).sum(axis=1)
    denom = s0 * s2 - s1**2
    safe_denom = np.where(np.abs(denom) > 1e-10, denom, np.nan)
    return (s2 * t0 - s1 * t1) / safe_denom


def _local_poly_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_grid: np.ndarray,
    bw: float,
    degree: int,
) -> np.ndarray:
    """
    Local polynomial regression (§6.1.2) of arbitrary degree with Gaussian kernel.

    Solved via the sqrt-weighted least squares system  (X*sqrt(w), y*sqrt(w))  rather
    than the normal equations  XtWX @ beta = XtWy.  This avoids the squaring of the
    condition number that makes np.linalg.solve return astronomically wrong answers
    near boundaries where the kernel window has little support.
    """
    if degree == 0:
        return _nw_predict(x_train, y_train, x_grid, bw)
    if degree == 1:
        return _local_linear_predict(x_train, y_train, x_grid, bw)
    preds = np.full(len(x_grid), np.nan)
    for i, x0 in enumerate(x_grid):
        xc = x_train - x0
        sqrt_w = np.sqrt(_gaussian_kernel(xc / bw))
        if sqrt_w.sum() < 1e-8:
            continue
        X_loc = np.column_stack([xc**d for d in range(degree + 1)])
        beta, _, _, _ = np.linalg.lstsq(X_loc * sqrt_w[:, None], y_train * sqrt_w, rcond=1e-8)
        preds[i] = beta[0]
    return preds


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _subsample_log1p(
    X: pd.DataFrame,
    y: pd.Series,
    feat: str,
    *,
    max_rows: int = 2000,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Subsample, sort, and log1p-transform (x, y) for a single feature."""
    if len(X) > max_rows:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=max_rows, replace=False)
        X = X.iloc[idx]
        y = y.iloc[idx]
    x_raw = X[feat].values.astype(float)
    y_raw = np.asarray(y, dtype=float)
    order = np.argsort(x_raw)
    x_raw, y_raw = x_raw[order], y_raw[order]
    return x_raw, y_raw, np.log1p(x_raw), np.log1p(y_raw)


# ---------------------------------------------------------------------------
# Figure: Nadaraya-Watson (§6.1)
# ---------------------------------------------------------------------------


def nadaraya_watson_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feat: str = "budget",
    *,
    bandwidths: list[float] | None = None,
    kernel: str = "gaussian",
    max_rows: int = 2000,
) -> go.Figure:
    """NW estimator with varying bandwidth — fitted in log1p-space, displayed in original scale."""
    if bandwidths is None:
        bandwidths = [0.1, 0.3, 0.6, 1.2, 2.5]

    _x_raw, _y_raw, x_fit, y_fit = _subsample_log1p(X, y, feat, max_rows=max_rows)
    grid_fit = np.linspace(float(np.percentile(x_fit, 1)), float(np.percentile(x_fit, 99)), 300)
    y_lo, y_hi = float(y_fit.min()), float(y_fit.max())

    labels = ["very narrow", "narrow", "medium", "wide", "very wide"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_fit,
            y=y_fit,
            mode="markers",
            name="data (log1p scale)",
            marker={"size": 4, "opacity": 0.2, "color": "grey"},
        )
    )
    for bw, label in zip(bandwidths, labels, strict=False):
        preds_log = np.clip(_nw_predict(x_fit, y_fit, grid_fit, bw, kernel=kernel), y_lo, y_hi)
        fig.add_trace(go.Scatter(x=grid_fit, y=preds_log, mode="lines", name=f"h={bw} ({label})"))

    fig.update_layout(
        title=f"Nadaraya-Watson ({kernel} kernel) on `{feat}` (§6.1, log1p axes)",
        xaxis_title=f"log1p({feat})",
        yaxis_title="log1p(revenue)",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


# ---------------------------------------------------------------------------
# Figure: Local linear vs NW (§6.1.1)
# ---------------------------------------------------------------------------


def local_linear_vs_nw_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feat: str = "budget",
    *,
    bw: float = 0.5,
    max_rows: int = 2000,
) -> go.Figure:
    """
    Compare NW vs local linear — highlights boundary bias correction (§6.1.1).

    Displayed in log1p-space (the space where the fit is computed) so boundary
    divergence between NW and local linear is visible rather than hidden by
    the back-transform compression in linear scale.
    """
    _x_raw, _y_raw, x_fit, y_fit = _subsample_log1p(X, y, feat, max_rows=max_rows)
    grid_fit = np.linspace(float(np.percentile(x_fit, 1)), float(np.percentile(x_fit, 99)), 300)
    y_lo, y_hi = float(y_fit.min()), float(y_fit.max())

    nw_preds = np.clip(_nw_predict(x_fit, y_fit, grid_fit, bw), y_lo, y_hi)
    ll_preds = np.clip(_local_linear_predict(x_fit, y_fit, grid_fit, bw), y_lo, y_hi)

    q5_log = float(np.percentile(x_fit, 5))
    q95_log = float(np.percentile(x_fit, 95))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_fit,
            y=y_fit,
            mode="markers",
            name="data (log1p scale)",
            marker={"size": 4, "opacity": 0.2, "color": "grey"},
        )
    )
    fig.add_trace(go.Scatter(x=grid_fit, y=nw_preds, mode="lines", name=f"Nadaraya-Watson (h={bw})"))
    fig.add_trace(go.Scatter(x=grid_fit, y=ll_preds, mode="lines", name=f"Local Linear (h={bw})"))
    fig.add_vrect(x0=float(x_fit.min()), x1=q5_log, fillcolor="lightyellow", opacity=0.3, layer="below", line_width=0)
    fig.add_vrect(x0=q95_log, x1=float(x_fit.max()), fillcolor="lightyellow", opacity=0.3, layer="below", line_width=0)
    fig.add_annotation(
        x=(float(x_fit.min()) + q5_log) / 2,
        y=0.95,
        yref="paper",
        text="boundary",
        showarrow=False,
        font={"size": 10, "color": "goldenrod"},
    )

    fig.update_layout(
        title=f"NW vs Local Linear on `{feat}` — boundary bias (§6.1.1, h={bw}, log1p axes)",
        xaxis_title=f"log1p({feat})",
        yaxis_title="log1p(revenue)",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


# ---------------------------------------------------------------------------
# Figure: Local polynomial by degree (§6.1.2)
# ---------------------------------------------------------------------------


def local_poly_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feat: str = "budget",
    *,
    bw: float = 0.3,
    degrees: list[int] | None = None,
    max_rows: int = 1500,
    n_grid: int = 150,
) -> go.Figure:
    """
    Local polynomial regression of degrees 0-3 (§6.1.2).

    Displayed in log-log scale (the space in which the fit is computed) so that:
      - the scatter data is legible across the full range, and
      - differences between polynomial degrees are visible near the boundaries.
    """
    if degrees is None:
        degrees = [0, 1, 2, 3]

    _x_raw, _y_raw, x_fit, y_fit = _subsample_log1p(X, y, feat, max_rows=max_rows)

    # Trim grid to the interior of the data (1st-99th percentile) so the kernel
    # window always overlaps enough training points.  Boundary predictions are
    # unreliable for degree ≥ 2 regardless of numerical solver.
    x_lo = float(np.percentile(x_fit, 1))
    x_hi = float(np.percentile(x_fit, 99))
    grid_fit = np.linspace(x_lo, x_hi, n_grid)

    # Hard clip bounds: no valid prediction should leave the observed response range
    y_lo = float(y_fit.min())
    y_hi = float(y_fit.max())

    degree_names = {0: "degree 0 (NW)", 1: "degree 1 (local linear)", 2: "degree 2", 3: "degree 3"}

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_fit,
            y=y_fit,
            mode="markers",
            name="data (log1p scale)",
            marker={"size": 4, "opacity": 0.2, "color": "grey"},
        )
    )
    for d in degrees:
        preds_log = _local_poly_predict(x_fit, y_fit, grid_fit, bw, d)
        preds_log = np.clip(preds_log, y_lo, y_hi)
        fig.add_trace(
            go.Scatter(x=grid_fit, y=preds_log, mode="lines", name=degree_names.get(d, f"degree {d}"))
        )

    fig.update_layout(
        title=f"Local polynomial regression on `{feat}` (§6.1.2, h={bw}, log1p axes)",
        xaxis_title=f"log1p({feat})",
        yaxis_title="log1p(revenue)",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


# ---------------------------------------------------------------------------
# Figure: Bandwidth selection via LOO-CV (§6.2)
# ---------------------------------------------------------------------------


def bandwidth_loocv_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feat: str = "budget",
    *,
    n_bw: int = 30,
    max_rows: int = 800,
    kernel: str = "gaussian",
) -> tuple[go.Figure, dict[str, Any]]:
    """LOO-CV score vs bandwidth for NW estimator (§6.2)."""
    _x_raw, _y_raw, x_fit, y_fit = _subsample_log1p(X, y, feat, max_rows=max_rows)
    bws = np.logspace(-2, 1, n_bw)
    cv_scores = [_nw_loo_cv(x_fit, y_fit, float(bw), kernel=kernel) for bw in bws]

    best_idx = int(np.argmin(cv_scores))
    best_bw = float(bws[best_idx])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.log10(bws), y=cv_scores, mode="lines+markers", name="LOO-CV"))
    fig.add_vline(
        x=float(np.log10(best_bw)),
        line_dash="dash",
        line_color="grey",
        annotation_text=f"best h={best_bw:.3f}",
    )
    fig.update_layout(
        title=f"Bandwidth selection via LOO-CV on `{feat}` ({kernel} kernel, §6.2)",
        xaxis_title="log₁₀(bandwidth h)",
        yaxis_title="LOO-CV score (log1p-space)",
        template="plotly_white",
    )
    return fig, {"best_bw": best_bw, "min_cv": float(min(cv_scores))}


# ---------------------------------------------------------------------------
# Figure: Kernel density estimation (§6.6.1)
# ---------------------------------------------------------------------------


def kde_figure(
    X: pd.DataFrame,
    feat: str = "budget",
    *,
    bandwidths: list[float] | None = None,
    max_rows: int = 2000,
) -> go.Figure:
    """KDE with Gaussian kernel for various bandwidths (§6.6.1)."""
    if bandwidths is None:
        bandwidths = [0.05, 0.15, 0.4, 1.0]

    x_raw = X[feat].values.astype(float)
    if len(x_raw) > max_rows:
        rng = np.random.default_rng(0)
        x_raw = x_raw[rng.choice(len(x_raw), size=max_rows, replace=False)]

    x_log = np.log1p(x_raw)
    grid_log = np.linspace(float(x_log.min()), float(x_log.max()), 400).reshape(-1, 1)
    grid_orig = np.expm1(grid_log.ravel())

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.expm1(x_log[:300]),
            y=np.zeros(min(300, len(x_log))),
            mode="markers",
            name="data (rug)",
            marker={"symbol": "line-ns", "size": 8, "opacity": 0.15, "color": "grey"},
        )
    )
    for bw in bandwidths:
        kde = KernelDensity(bandwidth=bw, kernel="gaussian")
        kde.fit(x_log.reshape(-1, 1))
        dens = np.exp(kde.score_samples(grid_log))
        fig.add_trace(go.Scatter(x=grid_orig, y=dens, mode="lines", name=f"h={bw}"))

    fig.update_layout(
        title=f"Kernel density estimation of `{feat}` (§6.6.1)",
        xaxis_title=feat,
        yaxis_title="density (log-space Gaussian kernel)",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


# ---------------------------------------------------------------------------
# Figure: Naive Bayes class-conditional densities (§6.6.3)
# ---------------------------------------------------------------------------


def naive_bayes_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feat: str = "budget",
    *,
    bw: float = 0.3,
    max_rows: int = 2000,
) -> tuple[go.Figure, dict[str, Any]]:
    """Class-conditional KDEs and Naive Bayes posterior (§6.6.3)."""
    if len(X) > max_rows:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X), size=max_rows, replace=False)
        X, y = X.iloc[idx], y.iloc[idx]

    x_log = np.log1p(X[feat].values.astype(float))
    y_arr = np.asarray(y)
    classes = np.unique(y_arr)
    grid_log = np.linspace(float(x_log.min()), float(x_log.max()), 400).reshape(-1, 1)
    grid_orig = np.expm1(grid_log.ravel())

    colors = ["steelblue", "darkorange"]
    fig = go.Figure()

    log_densities = []
    priors = []
    for k, cls in enumerate(classes):
        mask = y_arr == cls
        x_cls = x_log[mask].reshape(-1, 1)
        prior = float(mask.sum()) / len(y_arr)
        priors.append(prior)
        kde = KernelDensity(bandwidth=bw, kernel="gaussian")
        kde.fit(x_cls)
        log_dens = kde.score_samples(grid_log)
        log_densities.append(log_dens)
        dens = np.exp(log_dens)
        fig.add_trace(
            go.Scatter(
                x=grid_orig,
                y=dens,
                mode="lines",
                name=f"class {cls} (prior={prior:.2f})",
                line={"color": colors[k % len(colors)]},
            )
        )

    # Bayes posterior P(Y=1 | x)
    log_post_0 = np.log(priors[0]) + log_densities[0]
    log_post_1 = np.log(priors[1]) + log_densities[1]
    log_sum = np.logaddexp(log_post_0, log_post_1)
    posterior_1 = np.exp(log_post_1 - log_sum)
    fig.add_trace(
        go.Scatter(
            x=grid_orig,
            y=posterior_1,
            mode="lines",
            name="P(class=1 | x)  [Bayes posterior]",
            line={"dash": "dash", "color": "black"},
            yaxis="y2",
        )
    )

    fig.update_layout(
        title=f"Naive Bayes: class-conditional densities of `{feat}` (§6.6.3, h={bw})",
        xaxis_title=feat,
        yaxis_title="p(x | class)",
        yaxis2={"title": "posterior P(class=1 | x)", "overlaying": "y", "side": "right", "range": [0, 1]},
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    class_labels = list(classes)
    return fig, {"classes": class_labels, "priors": {int(c): p for c, p in zip(class_labels, priors, strict=False)}}

"""Shared helpers for ESL Chapter 7 (Model Assessment and Selection) notebooks."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


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


def _poly_pipe(degree: int) -> Any:
    """
    Polynomial regression pipeline with per-feature scaling for numerical stability.

    Scaling AFTER PolynomialFeatures ensures each basis function x^k is independently
    normalised.  Scaling before (then raising to a power) leaves high-degree terms
    in [-3^k, 3^k], causing OLS to explode on OOB / held-out extrapolation points.
    """
    return make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        LinearRegression(),
    )


def _log_feature(
    X: pd.DataFrame,
    y: pd.Series,
    feat: str,
    *,
    max_rows: int = 3000,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample and log1p-transform a single feature and the target."""
    if len(X) > max_rows:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=max_rows, replace=False)
        X, y = X.iloc[idx], y.iloc[idx]
    x = np.log1p(X[feat].values.astype(float)).reshape(-1, 1)
    y_log = np.log1p(np.asarray(y, dtype=float))
    return x, y_log


# ---------------------------------------------------------------------------
# Train/test error vs complexity (§7.2)
# ---------------------------------------------------------------------------


def train_test_error_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feat: str = "budget",
    *,
    max_degree: int = 8,
    max_rows: int = 3000,
) -> go.Figure:
    """Train vs test MSE across polynomial degree — the empirical bias-variance tradeoff (§7.2)."""
    x, y_log = _log_feature(X, y, feat, max_rows=max_rows)
    X_tr, X_te, y_tr, y_te = train_test_split(x, y_log, test_size=0.25, random_state=0)

    degrees = list(range(1, max_degree + 1))
    train_mse, test_mse = [], []
    for d in degrees:
        pipe = _poly_pipe(d)
        pipe.fit(X_tr, y_tr)
        train_mse.append(float(mean_squared_error(y_tr, pipe.predict(X_tr))))
        test_mse.append(float(mean_squared_error(y_te, pipe.predict(X_te))))

    best_d = degrees[int(np.argmin(test_mse))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=degrees, y=train_mse, mode="lines+markers", name="Train MSE"))
    fig.add_trace(go.Scatter(x=degrees, y=test_mse, mode="lines+markers", name="Test MSE"))
    fig.add_vline(x=best_d, line_dash="dash", line_color="grey", annotation_text=f"best d={best_d}")
    fig.update_layout(
        title=f"Train vs test error on `{feat}` (§7.2)",
        xaxis_title="polynomial degree (model complexity)",
        yaxis_title="MSE (log₁p-space)",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


# ---------------------------------------------------------------------------
# Bias-variance decomposition on synthetic data (§7.3)
# ---------------------------------------------------------------------------


def bias_variance_decomposition_figure(
    *,
    n_train: int = 60,
    n_test: int = 200,
    noise_std: float = 0.3,
    max_degree: int = 8,
    n_boot: int = 100,
    random_state: int = 0,
) -> go.Figure:
    """
    Bias-variance decomposition (§7.3) on synthetic data y = sin(2πx) + ε.

    For each polynomial degree d, fit n_boot models on independent training sets drawn from
    the same distribution.  The expected test error decomposes as:

        E[(Y - hat{f}(x))^2] = Bias^2(hat{f}(x)) + Var(hat{f}(x)) + sigma^2

    averaged over the test grid.
    """
    rng = np.random.default_rng(random_state)
    x_test = np.linspace(0, 1, n_test).reshape(-1, 1)
    true_f = np.sin(2 * np.pi * x_test.ravel())

    degrees = list(range(1, max_degree + 1))
    bias2_list, var_list, total_list = [], [], []

    for d in degrees:
        all_preds = np.zeros((n_boot, n_test))
        for b in range(n_boot):
            x_tr = rng.uniform(0, 1, n_train).reshape(-1, 1)
            y_tr = np.sin(2 * np.pi * x_tr.ravel()) + rng.normal(0, noise_std, n_train)
            pipe = _poly_pipe(d)
            pipe.fit(x_tr, y_tr)
            all_preds[b] = pipe.predict(x_test)

        mean_pred = all_preds.mean(axis=0)
        bias2 = float(np.mean((mean_pred - true_f) ** 2))
        var = float(np.mean(all_preds.var(axis=0)))
        bias2_list.append(bias2)
        var_list.append(var)
        total_list.append(bias2 + var + noise_std**2)

    noise_level = noise_std**2

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=degrees, y=bias2_list, mode="lines+markers", name="Bias²"))
    fig.add_trace(go.Scatter(x=degrees, y=var_list, mode="lines+markers", name="Variance"))
    fig.add_trace(
        go.Scatter(
            x=degrees,
            y=[noise_level] * len(degrees),
            mode="lines",
            name=f"Irreducible noise σ²={noise_level:.2f}",
            line={"dash": "dot", "color": "grey"},
        )
    )
    fig.add_trace(go.Scatter(x=degrees, y=total_list, mode="lines+markers", name="Total = Bias²+Var+σ²"))
    best_d = degrees[int(np.argmin(total_list))]
    fig.add_vline(x=best_d, line_dash="dash", line_color="grey", annotation_text=f"best d={best_d}")
    fig.update_layout(
        title=f"Bias-variance decomposition: y=sin(2πx)+ε, n_train={n_train}, n_boot={n_boot} (§7.3)",
        xaxis_title="polynomial degree",
        yaxis_title="expected squared error",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


# ---------------------------------------------------------------------------
# Optimism of the training error (§7.4)
# ---------------------------------------------------------------------------


def optimism_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feat: str = "budget",
    *,
    max_degree: int = 8,
    max_rows: int = 2000,
    n_boot: int = 60,
) -> go.Figure:
    """
    Optimism of training error estimated by bootstrap (§7.4).

    Optimism = E[test err] - E[train err].  Here estimated as mean(OOB err) - mean(in-bag err)
    over n_boot bootstrap resamples, for each polynomial degree.
    """
    x, y_log = _log_feature(X, y, feat, max_rows=max_rows)
    degrees = list(range(1, max_degree + 1))

    train_mse_avg, oob_mse_avg, optimism_list = [], [], []
    rng = np.random.default_rng(42)

    for d in degrees:
        tr_errs, oob_errs = [], []
        for _ in range(n_boot):
            idx_boot = rng.integers(0, len(x), size=len(x))
            idx_oob = np.where(np.bincount(idx_boot, minlength=len(x)) == 0)[0]
            if len(idx_oob) < 10:
                continue
            pipe = _poly_pipe(d)
            pipe.fit(x[idx_boot], y_log[idx_boot])
            tr_errs.append(float(mean_squared_error(y_log[idx_boot], pipe.predict(x[idx_boot]))))
            oob_errs.append(float(mean_squared_error(y_log[idx_oob], pipe.predict(x[idx_oob]))))

        mu_tr = float(np.mean(tr_errs))
        mu_oob = float(np.mean(oob_errs))
        train_mse_avg.append(mu_tr)
        oob_mse_avg.append(mu_oob)
        optimism_list.append(mu_oob - mu_tr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=degrees, y=train_mse_avg, mode="lines+markers", name="In-bag (train) MSE"))
    fig.add_trace(go.Scatter(x=degrees, y=oob_mse_avg, mode="lines+markers", name="OOB (test) MSE"))
    fig.add_trace(
        go.Scatter(
            x=degrees,
            y=optimism_list,
            mode="lines+markers",
            name="Optimism (OOB - in-bag)",
            line={"dash": "dot"},
        )
    )
    fig.update_layout(
        title=f"Training error optimism on `{feat}` (§7.4, bootstrap estimation)",
        xaxis_title="polynomial degree",
        yaxis_title="MSE (log₁p-space)",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


# ---------------------------------------------------------------------------
# In-sample estimates: Cp, AIC, BIC (§7.5-7.7)
# ---------------------------------------------------------------------------


def model_selection_criteria_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feat: str = "budget",
    *,
    max_degree: int = 8,
    max_rows: int = 3000,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    Mallows' Cp, AIC, and BIC vs polynomial degree (§7.5-7.7).

    Cp  = RSS/n + 2*d*sigma^2/n         (in-sample error estimate)
    AIC = n*log(RSS/n) + 2*d             (information-theoretic, Gaussian)
    BIC = n*log(RSS/n) + d*log(n)        (Bayesian, penalises complexity harder)

    sigma^2 is estimated from the highest-degree fit as RSS/(n-d_max).
    Criteria are min-max normalised across degrees for visual comparison.
    """
    x, y_log = _log_feature(X, y, feat, max_rows=max_rows)
    n = len(x)

    # Estimate sigma^2 from the highest-degree model (most flexible = least biased noise estimate)
    pipe_full = _poly_pipe(max_degree)
    pipe_full.fit(x, y_log)
    rss_full = float(np.sum((y_log - pipe_full.predict(x)) ** 2))
    d_full = max_degree + 1
    sigma2_hat = rss_full / max(n - d_full, 1)

    degrees = list(range(1, max_degree + 1))
    cp_list, aic_list, bic_list = [], [], []

    for d in degrees:
        pipe = _poly_pipe(d)
        pipe.fit(x, y_log)
        rss = float(np.sum((y_log - pipe.predict(x)) ** 2))
        d_eff = d + 1  # polynomial terms + intercept
        cp_list.append(rss / n + 2 * d_eff * sigma2_hat / n)
        aic_list.append(n * float(np.log(rss / n)) + 2 * d_eff)
        bic_list.append(n * float(np.log(rss / n)) + d_eff * float(np.log(n)))

    def _norm(v: list[float]) -> list[float]:
        lo, hi = min(v), max(v)
        span = hi - lo
        return [(vi - lo) / span if span > 0 else 0.0 for vi in v]

    best_cp = degrees[int(np.argmin(cp_list))]
    best_aic = degrees[int(np.argmin(aic_list))]
    best_bic = degrees[int(np.argmin(bic_list))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=degrees, y=_norm(cp_list), mode="lines+markers", name="Cₚ (normalised)"))
    fig.add_trace(go.Scatter(x=degrees, y=_norm(aic_list), mode="lines+markers", name="AIC (normalised)"))
    fig.add_trace(go.Scatter(x=degrees, y=_norm(bic_list), mode="lines+markers", name="BIC (normalised)"))
    fig.add_vline(x=best_cp, line_dash="dot", line_color="steelblue", annotation_text=f"Cₚ: d={best_cp}")
    fig.add_vline(x=best_aic, line_dash="dot", line_color="orange", annotation_text=f"AIC: d={best_aic}")
    fig.add_vline(x=best_bic, line_dash="dot", line_color="green", annotation_text=f"BIC: d={best_bic}")
    fig.update_layout(
        title=f"Model selection criteria on `{feat}` (§7.5-7.7)",
        xaxis_title="polynomial degree",
        yaxis_title="criterion (min-max normalised)",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig, {"best_cp_d": best_cp, "best_aic_d": best_aic, "best_bic_d": best_bic}


# ---------------------------------------------------------------------------
# K-fold cross-validation (§7.10)
# ---------------------------------------------------------------------------


def kfold_cv_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feat: str = "budget",
    *,
    max_degree: int = 8,
    k_values: list[int] | None = None,
    max_rows: int = 3000,
) -> tuple[go.Figure, dict[str, Any]]:
    """K-fold cross-validation error vs polynomial degree (§7.10.1)."""
    if k_values is None:
        k_values = [5, 10]

    x, y_log = _log_feature(X, y, feat, max_rows=max_rows)
    degrees = list(range(1, max_degree + 1))

    fig = go.Figure()
    best_degrees: dict[str, int] = {}

    for k in k_values:
        cv_scores = []
        for d in degrees:
            pipe = _poly_pipe(d)
            scores = cross_val_score(pipe, x, y_log, cv=k, scoring="neg_mean_squared_error")
            cv_scores.append(float(-scores.mean()))
        best_d = degrees[int(np.argmin(cv_scores))]
        best_degrees[f"{k}-fold"] = best_d
        fig.add_trace(go.Scatter(x=degrees, y=cv_scores, mode="lines+markers", name=f"{k}-fold CV"))

    # Training error for reference
    train_mse = []
    for d in degrees:
        pipe = _poly_pipe(d)
        pipe.fit(x, y_log)
        train_mse.append(float(mean_squared_error(y_log, pipe.predict(x))))
    fig.add_trace(
        go.Scatter(
            x=degrees,
            y=train_mse,
            mode="lines+markers",
            name="Train MSE",
            line={"dash": "dot", "color": "grey"},
        )
    )

    fig.update_layout(
        title=f"K-fold cross-validation on `{feat}` (§7.10.1)",
        xaxis_title="polynomial degree",
        yaxis_title="MSE (log₁p-space)",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig, {"best_degrees": best_degrees}


# ---------------------------------------------------------------------------
# Bootstrap .632 estimator (§7.11)
# ---------------------------------------------------------------------------


def bootstrap_632_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feat: str = "budget",
    *,
    max_degree: int = 8,
    n_boot: int = 50,
    max_rows: int = 2000,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    Bootstrap .632 estimator of prediction error vs polynomial degree (§7.11).

    Err^{(1)} = (1/n) sum_i mean_{b: i ∉ B^b} L(y_i, hat{f}^{*b}(x_i))  [OOB error]
    Err^{.632} = 0.368 * train_err + 0.632 * Err^{(1)}

    The .632 estimator corrects the upward bias of Err^{(1)} (which uses ~0.632n
    training points) toward the apparent (training) error.
    """
    x, y_log = _log_feature(X, y, feat, max_rows=max_rows)
    n = len(x)
    degrees = list(range(1, max_degree + 1))

    train_mse_list, boot_oob_list, boot_632_list = [], [], []
    rng = np.random.default_rng(42)

    for d in degrees:
        pipe = _poly_pipe(d)
        pipe.fit(x, y_log)
        train_err = float(mean_squared_error(y_log, pipe.predict(x)))
        train_mse_list.append(train_err)

        oob_losses: list[list[float]] = [[] for _ in range(n)]
        for _ in range(n_boot):
            idx_boot = rng.integers(0, n, size=n)
            idx_oob = np.where(np.bincount(idx_boot, minlength=n) == 0)[0]
            if len(idx_oob) < 5:
                continue
            pipe_b = _poly_pipe(d)
            pipe_b.fit(x[idx_boot], y_log[idx_boot])
            preds_oob = pipe_b.predict(x[idx_oob])
            for ii, pred in zip(idx_oob, preds_oob, strict=False):
                oob_losses[ii].append((y_log[ii] - pred) ** 2)

        per_sample = [float(np.mean(losses)) for losses in oob_losses if losses]
        err1 = float(np.mean(per_sample)) if per_sample else float("nan")
        boot_oob_list.append(err1)
        boot_632_list.append(0.368 * train_err + 0.632 * err1)

    best_d = degrees[int(np.nanargmin(boot_632_list))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=degrees, y=train_mse_list, mode="lines+markers", name="Train MSE"))
    fig.add_trace(go.Scatter(x=degrees, y=boot_oob_list, mode="lines+markers", name="Bootstrap Err⁽¹⁾ (OOB)"))
    fig.add_trace(go.Scatter(x=degrees, y=boot_632_list, mode="lines+markers", name=".632 Bootstrap"))
    fig.add_vline(x=best_d, line_dash="dash", line_color="grey", annotation_text=f".632 best d={best_d}")
    fig.update_layout(
        title=f"Bootstrap .632 estimator on `{feat}` (§7.11, n_boot={n_boot})",
        xaxis_title="polynomial degree",
        yaxis_title="MSE (log₁p-space)",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig, {"best_632_d": best_d, "min_632": float(np.nanmin(boot_632_list))}

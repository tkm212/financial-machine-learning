"""Shared helpers for ESL Chapter 5 (Basis Expansions and Regularization) notebooks."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler


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


def scale_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.25,
    random_state: int = 0,
) -> dict[str, Any]:
    """Train/test split then standard-scale X. Returns a dict with all pieces."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_s": X_train_s,
        "X_test_s": X_test_s,
        "scaler": scaler,
        "feat_names": list(X_train.columns),
    }


def _single_feature_subsample(
    X: pd.DataFrame,
    y: pd.Series,
    feat: str,
    *,
    max_rows: int = 3000,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted (x, y) arrays for a single feature, subsampled for speed."""
    if len(X) > max_rows:
        idx = np.random.default_rng(random_state).choice(len(X), size=max_rows, replace=False)
        X = X.iloc[idx]
        y = y.iloc[idx]
    x = X[feat].values.astype(float)
    order = np.argsort(x)
    return x[order], np.asarray(y)[order]


# ---------------------------------------------------------------------------
# Piecewise polynomials and splines (§5.2)
# ---------------------------------------------------------------------------


def piecewise_poly_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feat: str = "budget",
    *,
    n_knots: int = 4,
    max_rows: int = 3000,
) -> go.Figure:
    """
    Compare three fits on a single feature:
      - global linear regression
      - piecewise-constant (step function)
      - cubic spline (SplineTransformer, degree=3)
    """
    x_raw, y_raw = _single_feature_subsample(X, y, feat, max_rows=max_rows)
    x2d = x_raw.reshape(-1, 1)

    grid = np.linspace(float(x_raw.min()), float(x_raw.max()), 400).reshape(-1, 1)

    lin = LinearRegression().fit(x2d, y_raw)

    # piecewise constant: bin the feature
    knot_vals = np.quantile(x_raw, np.linspace(0, 1, n_knots + 2)[1:-1])
    bins = np.searchsorted(knot_vals, x_raw)
    step_means = np.array([y_raw[bins == b].mean() if (bins == b).any() else 0.0 for b in range(n_knots + 1)])
    grid_bins = np.searchsorted(knot_vals, grid.ravel())
    step_pred = step_means[grid_bins]

    spline_pipe = make_pipeline(SplineTransformer(n_knots=n_knots, degree=3, knots="quantile"), LinearRegression())
    spline_pipe.fit(x2d, y_raw)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_raw,
            y=y_raw,
            mode="markers",
            name="data",
            marker={"size": 3, "opacity": 0.3},
        )
    )
    fig.add_trace(go.Scatter(x=grid.ravel(), y=lin.predict(grid), mode="lines", name="linear"))
    fig.add_trace(go.Scatter(x=grid.ravel(), y=step_pred, mode="lines", name=f"piecewise constant ({n_knots} knots)"))
    fig.add_trace(
        go.Scatter(x=grid.ravel(), y=spline_pipe.predict(grid), mode="lines", name=f"cubic spline ({n_knots} knots)")
    )

    for kv in knot_vals:
        fig.add_vline(x=float(kv), line_dash="dot", line_color="grey", opacity=0.5)

    fig.update_layout(
        title=f"Piecewise polynomials vs cubic spline on `{feat}` (§5.2)",
        xaxis_title=feat,
        yaxis_title="revenue",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


def natural_cubic_spline_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feat: str = "budget",
    *,
    knot_counts: list[int] | None = None,
    max_rows: int = 3000,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    Natural cubic splines (NCS) for a range of knot counts.
    Uses SplineTransformer(degree=3, extrapolation='linear') as an NCS approximation.
    Returns the figure and train/test MSE for each knot count.
    """
    if knot_counts is None:
        knot_counts = [2, 4, 6, 8, 12]

    x_raw, y_raw = _single_feature_subsample(X, y, feat, max_rows=max_rows)
    x2d = x_raw.reshape(-1, 1)
    X_train_1d, X_test_1d, y_train_1d, y_test_1d = train_test_split(x2d, y_raw, test_size=0.25, random_state=0)

    grid = np.linspace(float(x_raw.min()), float(x_raw.max()), 400).reshape(-1, 1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=X_train_1d.ravel(),
            y=y_train_1d,
            mode="markers",
            name="train",
            marker={"size": 3, "opacity": 0.25, "color": "steelblue"},
        )
    )

    train_mse_list, test_mse_list = [], []
    for k in knot_counts:
        pipe = make_pipeline(
            SplineTransformer(n_knots=k, degree=3, knots="quantile", extrapolation="linear"), LinearRegression()
        )
        pipe.fit(X_train_1d, y_train_1d)
        train_mse_list.append(float(mean_squared_error(y_train_1d, pipe.predict(X_train_1d))))
        test_mse_list.append(float(mean_squared_error(y_test_1d, pipe.predict(X_test_1d))))
        fig.add_trace(go.Scatter(x=grid.ravel(), y=pipe.predict(grid), mode="lines", name=f"NCS {k} knots"))

    fig.update_layout(
        title=f"Natural cubic splines on `{feat}`: varying knot count (§5.2.1)",
        xaxis_title=feat,
        yaxis_title="revenue",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig, {"knot_counts": knot_counts, "train_mse": train_mse_list, "test_mse": test_mse_list}


def spline_knot_bias_variance_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feat: str = "budget",
    *,
    max_knots: int = 20,
    max_rows: int = 3000,
) -> go.Figure:
    """Train vs test MSE as a function of knot count — spline bias-variance tradeoff."""
    x_raw, y_raw = _single_feature_subsample(X, y, feat, max_rows=max_rows)
    x2d = x_raw.reshape(-1, 1)
    X_tr, X_te, y_tr, y_te = train_test_split(x2d, y_raw, test_size=0.25, random_state=0)

    ks = list(range(2, max_knots + 1))
    train_mse, test_mse = [], []
    for k in ks:
        pipe = make_pipeline(SplineTransformer(n_knots=k, degree=3, knots="quantile"), LinearRegression())
        pipe.fit(X_tr, y_tr)
        train_mse.append(float(mean_squared_error(y_tr, pipe.predict(X_tr))))
        test_mse.append(float(mean_squared_error(y_te, pipe.predict(X_te))))

    best_k = ks[int(np.argmin(test_mse))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=train_mse, mode="lines+markers", name="train MSE"))
    fig.add_trace(go.Scatter(x=ks, y=test_mse, mode="lines+markers", name="test MSE"))
    fig.add_vline(x=best_k, line_dash="dash", line_color="grey", annotation_text=f"best k={best_k}")
    fig.update_layout(
        title=f"Spline bias-variance tradeoff on `{feat}` (§5.2)",
        xaxis_title="number of knots",
        yaxis_title="MSE",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


# ---------------------------------------------------------------------------
# Smoothing splines (§5.4)
# ---------------------------------------------------------------------------


def smoothing_spline_lambda_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feat: str = "budget",
    *,
    rel_factors: list[float] | None = None,
    max_rows: int = 2000,
) -> go.Figure:
    """
    Smoothing splines (scipy UnivariateSpline) for various smoothing factors s.

    The smoothing spline minimises:
        sum (y_i - f(x_i))^2  +  lambda * integral [f''(x)]^2 dx

    Fitting is done in log1p-space so the spline stays within the data range on
    heavy-tailed distributions like revenue.  Predictions are back-transformed to
    the original scale for display.  s is expressed as a fraction of n·var(log1p(y))
    so the range is data-scale-invariant.
    """
    if rel_factors is None:
        rel_factors = [1e3, 1e1, 1e-1, 1e-3, 1e-5]

    x_raw, y_raw = _single_feature_subsample(X, y, feat, max_rows=max_rows)

    # log1p transform: stabilises fitting on heavy-tailed economic data
    x_fit = np.log1p(x_raw)
    y_fit = np.log1p(y_raw.astype(float))

    scale = len(y_fit) * float(np.var(y_fit))
    grid_fit = np.linspace(float(x_fit.min()), float(x_fit.max()), 400)
    grid_orig = np.expm1(grid_fit)

    label_names = ["very smooth", "smooth", "medium", "flexible", "near-interpolating"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_raw,
            y=y_raw,
            mode="markers",
            name="data",
            marker={"size": 3, "opacity": 0.25},
        )
    )
    for factor, label in zip(rel_factors, label_names, strict=False):
        sp = UnivariateSpline(x_fit, y_fit, s=scale * factor, k=3)
        preds = np.expm1(sp(grid_fit))
        fig.add_trace(go.Scatter(x=grid_orig, y=preds, mode="lines", name=label))

    fig.update_layout(
        title=f"Smoothing splines on `{feat}` (log-space fit, §5.4)",
        xaxis_title=feat,
        yaxis_title="revenue",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


def smoothing_spline_df_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feat: str = "budget",
    *,
    df_values: list[float] | None = None,
    max_rows: int = 2000,
) -> go.Figure:
    """
    Smoothing splines at specific effective degrees of freedom.

    Fits in log1p-space for stability on heavy-tailed data; back-transforms predictions.
    Scans a relative s-grid (fractions of n·var(log1p(y))) to find the s that yields
    each target df.  df is approximated as n_interior_knots + 3.
    """
    if df_values is None:
        df_values = [4, 6, 9, 15, 25]

    x_raw, y_raw = _single_feature_subsample(X, y, feat, max_rows=max_rows)

    x_fit = np.log1p(x_raw)
    y_fit = np.log1p(y_raw.astype(float))
    n = len(x_fit)

    grid_fit = np.linspace(float(x_fit.min()), float(x_fit.max()), 400)
    grid_orig = np.expm1(grid_fit)

    scale = n * float(np.var(y_fit))
    s_values = np.logspace(np.log10(scale * 1e-5), np.log10(scale * 2e2), 120)

    df_approx = [UnivariateSpline(x_fit, y_fit, s=float(s), k=3).get_knots().shape[0] + 3 for s in s_values]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_raw,
            y=y_raw,
            mode="markers",
            name="data",
            marker={"size": 3, "opacity": 0.25},
        )
    )
    plotted_dfs: set[int] = set()
    for target_df in df_values:
        diffs = [abs(d - target_df) for d in df_approx]
        best_idx = int(np.argmin(diffs))
        actual_df = df_approx[best_idx]
        if actual_df in plotted_dfs:
            continue
        plotted_dfs.add(actual_df)
        sp = UnivariateSpline(x_fit, y_fit, s=float(s_values[best_idx]), k=3)
        preds = np.expm1(sp(grid_fit))
        fig.add_trace(go.Scatter(x=grid_orig, y=preds, mode="lines", name=f"df≈{actual_df}"))

    fig.update_layout(
        title="Smoothing splines: effective degrees of freedom (log-space fit, §5.4.1)",
        xaxis_title=feat,
        yaxis_title="revenue",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


def smoothing_spline_bias_variance_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feat: str = "budget",
    *,
    n_lambdas: int = 40,
    max_rows: int = 2000,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    Bias-variance tradeoff for smoothing splines: train vs test MSE vs log(lambda).

    Uses Ridge regression on a dense B-spline basis to approximate the smoothing spline
    (penalised least squares with many knots).  The ridge penalty plays the role of lambda.
    """
    x_raw, y_raw = _single_feature_subsample(X, y, feat, max_rows=max_rows)
    x2d = x_raw.reshape(-1, 1)
    X_tr, X_te, y_tr, y_te = train_test_split(x2d, y_raw, test_size=0.25, random_state=0)

    n_basis_knots = 30
    basis = SplineTransformer(n_knots=n_basis_knots, degree=3, knots="quantile")
    X_tr_b = basis.fit_transform(X_tr)
    X_te_b = basis.transform(X_te)

    alphas = np.logspace(-2, 8, n_lambdas)
    train_mse, test_mse = [], []
    for alpha in alphas:
        ridge = Ridge(alpha=float(alpha))
        ridge.fit(X_tr_b, y_tr)
        train_mse.append(float(mean_squared_error(y_tr, ridge.predict(X_tr_b))))
        test_mse.append(float(mean_squared_error(y_te, ridge.predict(X_te_b))))

    best_idx = int(np.argmin(test_mse))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.log10(alphas), y=train_mse, mode="lines", name="train MSE"))
    fig.add_trace(go.Scatter(x=np.log10(alphas), y=test_mse, mode="lines", name="test MSE"))
    fig.add_vline(
        x=float(np.log10(alphas[best_idx])),
        line_dash="dash",
        line_color="grey",
        annotation_text=f"best λ=10^{np.log10(alphas[best_idx]):.1f}",
    )
    fig.update_layout(
        title=f"Smoothing spline bias-variance tradeoff on `{feat}` (§5.5.2)",
        xaxis_title="log10(lambda)  [lambda = smoothness penalty]",
        yaxis_title="MSE",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig, {
        "best_alpha": float(alphas[best_idx]),
        "min_test_mse": float(min(test_mse)),
        "min_train_mse": float(min(train_mse)),
    }


def gcv_lambda_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feat: str = "budget",
    *,
    n_lambdas: int = 40,
    max_rows: int = 2000,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    Generalised cross-validation (GCV) score as a function of lambda.

    GCV(lambda) = (1/n) * sum [(y_i - hat{f}_lambda(x_i)) / (1 - S_{ii})]^2

    Approximated here as RSS(lambda) / (n * (1 - df(lambda)/n)^2),
    where df(lambda) = trace(S) is computed from the Ridge hat matrix trace.
    """
    x_raw, y_raw = _single_feature_subsample(X, y, feat, max_rows=max_rows)
    x2d = x_raw.reshape(-1, 1)

    n_basis_knots = 20
    basis = SplineTransformer(n_knots=n_basis_knots, degree=3, knots="quantile")
    Xb = basis.fit_transform(x2d)
    n, _ = Xb.shape

    alphas = np.logspace(-1, 7, n_lambdas)
    gcv_scores, train_mse = [], []
    for alpha in alphas:
        ridge = Ridge(alpha=float(alpha), fit_intercept=True)
        ridge.fit(Xb, y_raw)
        resid = y_raw - ridge.predict(Xb)
        rss = float(np.dot(resid, resid))
        train_mse.append(rss / n)

        # hat-matrix trace via SVD: tr(H) = sum d_i^2 / (d_i^2 + alpha)
        _, sv, _ = np.linalg.svd(Xb, full_matrices=False)
        df_lambda = float(np.sum(sv**2 / (sv**2 + alpha)))
        gcv = rss / (n * (1.0 - df_lambda / n) ** 2)
        gcv_scores.append(gcv)

    best_idx = int(np.argmin(gcv_scores))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.log10(alphas), y=gcv_scores, mode="lines", name="GCV"))
    fig.add_vline(
        x=float(np.log10(alphas[best_idx])),
        line_dash="dash",
        line_color="grey",
        annotation_text=f"best λ=10^{np.log10(alphas[best_idx]):.1f}",
    )
    fig.update_layout(
        title=f"GCV score vs lambda on `{feat}` (§5.5)",
        xaxis_title="log10(lambda)",
        yaxis_title="GCV",
        template="plotly_white",
    )
    return fig, {
        "best_alpha": float(alphas[best_idx]),
        "min_gcv": float(min(gcv_scores)),
    }

"""Shared helpers for ESL Chapter 9 (Additive Models, Trees and Related Methods) notebooks."""

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
from sklearn.preprocessing import SplineTransformer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


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


def load_tmdb_classification_xy(inputs_dir: Path) -> tuple[pd.DataFrame, pd.Series, str]:
    from financial_machine_learning.esl_loaders import load_tmdb_revenue_classification

    return load_tmdb_revenue_classification(inputs_dir)


def _log_features(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str],
    *,
    max_rows: int = 3000,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if len(X) > max_rows:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=max_rows, replace=False)
        X, y = X.iloc[idx], y.iloc[idx]
    cols = []
    for f in feats:
        cols.append(np.log1p(X[f].values.astype(float)))
    x_arr = np.column_stack(cols)
    y_log = np.log1p(np.asarray(y, dtype=float))
    return x_arr, y_log


# ---------------------------------------------------------------------------
# GAM via backfitting (§9.1)
# ---------------------------------------------------------------------------


def _spline_smoother(n_knots: int = 5, degree: int = 3) -> Any:
    return make_pipeline(SplineTransformer(n_knots=n_knots, degree=degree, include_bias=False), LinearRegression())


def _backfitting(
    X: np.ndarray,
    y: np.ndarray,
    n_knots: int = 5,
    max_iter: int = 20,
    tol: float = 1e-4,
) -> tuple[list[Any], np.ndarray, list[float]]:
    """
    Backfitting algorithm for a GAM: y = alpha + sum_j f_j(x_j) (§9.1).

    Each f_j is a cubic spline smoother fit to partial residuals.
    Returns (fitted_smoothers, intercept_per_step, rss_trace).
    """
    n, p = X.shape
    alpha = float(y.mean())
    funcs = [_spline_smoother(n_knots=n_knots) for _ in range(p)]
    f_hats = np.zeros((n, p))
    rss_trace: list[float] = []

    for _ in range(max_iter):
        old_f = f_hats.copy()
        for j in range(p):
            partial_resid = y - alpha - f_hats.sum(axis=1) + f_hats[:, j]
            x_j = X[:, j].reshape(-1, 1)
            funcs[j].fit(x_j, partial_resid)
            f_j = funcs[j].predict(x_j)
            f_j -= f_j.mean()
            f_hats[:, j] = f_j

        pred = alpha + f_hats.sum(axis=1)
        rss = float(mean_squared_error(y, pred))
        rss_trace.append(rss)

        delta = float(np.max(np.abs(f_hats - old_f)))
        if delta < tol:
            break

    return funcs, f_hats, rss_trace


def gam_partial_plots_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    n_knots: int = 5,
    max_rows: int = 2000,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    GAM backfitting partial effect plots (§9.1).

    Shows f_j(x_j) for each feature — the smooth contribution of each predictor
    after adjusting for all others, the key diagnostic in additive models.
    """
    if feats is None:
        feats = ["budget", "popularity", "runtime"]
    feats = [f for f in feats if f in X.columns]

    x_arr, y_log = _log_features(X, y, feats, max_rows=max_rows)
    alpha = float(y_log.mean())
    _, f_hats, rss_trace = _backfitting(x_arr, y_log - alpha, n_knots=n_knots)

    fig = go.Figure()
    colors = ["steelblue", "tomato", "green", "purple", "orange"]
    for j, feat in enumerate(feats):
        sort_idx = np.argsort(x_arr[:, j])
        fig.add_trace(
            go.Scatter(
                x=x_arr[sort_idx, j],
                y=f_hats[sort_idx, j],
                mode="lines",
                name=f"f({feat})",
                line={"color": colors[j % len(colors)]},
            )
        )

    pred = alpha + f_hats.sum(axis=1)
    test_mse = float(mean_squared_error(y_log, pred))

    fig.update_layout(
        title=f"GAM partial effects (backfitting, n_knots={n_knots}) — §9.1",
        xaxis_title="log₁p(feature)",
        yaxis_title="partial effect f_j(x_j)",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig, {
        "final_rss": rss_trace[-1] if rss_trace else float("nan"),
        "n_iter": len(rss_trace),
        "train_mse": test_mse,
    }


def gam_vs_linear_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    n_knots: int = 5,
    max_rows: int = 2000,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    GAM vs linear model cross-validated test MSE comparison (§9.1).

    The GAM with nonlinear spline components should outperform the linear model
    when the true response surface is nonlinear.
    """
    if feats is None:
        feats = ["budget", "popularity", "runtime"]
    feats = [f for f in feats if f in X.columns]

    x_arr, y_log = _log_features(X, y, feats, max_rows=max_rows)

    linear_pipe = make_pipeline(LinearRegression())
    linear_scores = cross_val_score(linear_pipe, x_arr, y_log, cv=5, scoring="neg_mean_squared_error")
    linear_mse = float(-linear_scores.mean())
    linear_std = float(linear_scores.std())

    spline_pipes = [
        make_pipeline(SplineTransformer(n_knots=n_knots, degree=d, include_bias=False), LinearRegression())
        for d in [1, 2, 3]
    ]
    spline_labels = [f"Spline degree={d}" for d in [1, 2, 3]]
    spline_mses = []
    spline_stds = []
    for pipe in spline_pipes:
        scores = cross_val_score(pipe, x_arr, y_log, cv=5, scoring="neg_mean_squared_error")
        spline_mses.append(float(-scores.mean()))
        spline_stds.append(float(scores.std()))

    all_labels = ["Linear", *spline_labels]
    all_mses = [linear_mse, *spline_mses]
    all_stds = [linear_std, *spline_stds]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=all_labels,
            y=all_mses,
            error_y={"type": "data", "array": all_stds, "visible": True},
            marker_color=["grey", "steelblue", "tomato", "green"],
        )
    )
    fig.update_layout(
        title=f"5-fold CV MSE: linear vs additive spline models — §9.1 (features: {feats})",
        xaxis_title="model",
        yaxis_title="CV MSE (log₁p-space)",
        template="plotly_white",
    )
    best = all_labels[int(np.argmin(all_mses))]
    return fig, {"best_model": best, "linear_mse": linear_mse, "best_mse": min(all_mses)}


# ---------------------------------------------------------------------------
# CART decision trees (§9.2)
# ---------------------------------------------------------------------------


def tree_depth_error_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    max_depth_range: int = 12,
    max_rows: int = 2000,
    task: str = "regression",
) -> tuple[go.Figure, dict[str, Any]]:
    """
    CART tree depth vs train/test error (§9.2).

    The characteristic U-shaped test error curve shows that shallow trees underfit
    (high bias) while deep trees overfit (high variance).
    """
    if feats is None:
        feats = ["budget", "popularity", "runtime", "vote_average", "vote_count"]
    feats = [f for f in feats if f in X.columns]

    x_arr, y_log = _log_features(X, y, feats, max_rows=max_rows)
    X_tr, X_te, y_tr, y_te = train_test_split(x_arr, y_log, test_size=0.25, random_state=0)

    depths = list(range(1, max_depth_range + 1))
    train_errs, test_errs = [], []

    for d in depths:
        if task == "regression":
            tree: DecisionTreeRegressor | DecisionTreeClassifier = DecisionTreeRegressor(max_depth=d, random_state=0)
        else:
            tree = DecisionTreeClassifier(max_depth=d, random_state=0)
        tree.fit(X_tr, y_tr)
        train_errs.append(float(mean_squared_error(y_tr, tree.predict(X_tr))))
        test_errs.append(float(mean_squared_error(y_te, tree.predict(X_te))))

    best_d = depths[int(np.argmin(test_errs))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=depths, y=train_errs, mode="lines+markers", name="Train MSE"))
    fig.add_trace(go.Scatter(x=depths, y=test_errs, mode="lines+markers", name="Test MSE"))
    fig.add_vline(x=best_d, line_dash="dash", line_color="grey", annotation_text=f"best depth={best_d}")
    fig.update_layout(
        title=f"CART tree depth vs train/test error — §9.2 (features: {feats[:3]}…)",
        xaxis_title="tree max_depth",
        yaxis_title="MSE (log₁p-space)",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig, {"best_depth": best_d, "best_test_mse": min(test_errs)}


def cost_complexity_pruning_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    max_rows: int = 2000,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    Cost-complexity pruning path (§9.2, minimal cost-complexity pruning).

    sklearn's ``cost_complexity_pruning_path`` traces the Breiman et al. alpha-sequence:
        R_alpha(T) = R(T) + alpha |T_leaves|
    where |T_leaves| is the number of terminal nodes.  As alpha increases, the tree is pruned
    back and test error first decreases (removing noisy splits) then increases (over-pruning).
    """
    if feats is None:
        feats = ["budget", "popularity", "runtime", "vote_average", "vote_count"]
    feats = [f for f in feats if f in X.columns]

    x_arr, y_log = _log_features(X, y, feats, max_rows=max_rows)
    X_tr, X_te, y_tr, y_te = train_test_split(x_arr, y_log, test_size=0.25, random_state=0)

    path = DecisionTreeRegressor(random_state=0).cost_complexity_pruning_path(X_tr, y_tr)
    ccp_alphas = path.ccp_alphas[:-1]  # last alpha gives a single-node tree

    if len(ccp_alphas) > 40:
        step = max(1, len(ccp_alphas) // 40)
        ccp_alphas = ccp_alphas[::step]

    train_errs, test_errs, n_leaves = [], [], []
    for alpha in ccp_alphas:
        tree = DecisionTreeRegressor(ccp_alpha=float(alpha), random_state=0)
        tree.fit(X_tr, y_tr)
        train_errs.append(float(mean_squared_error(y_tr, tree.predict(X_tr))))
        test_errs.append(float(mean_squared_error(y_te, tree.predict(X_te))))
        n_leaves.append(int(tree.get_n_leaves()))

    best_alpha = float(ccp_alphas[int(np.argmin(test_errs))])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=n_leaves, y=train_errs, mode="lines+markers", name="Train MSE"))
    fig.add_trace(go.Scatter(x=n_leaves, y=test_errs, mode="lines+markers", name="Test MSE"))
    fig.update_layout(
        title="Cost-complexity pruning: test MSE vs number of leaves — §9.2",
        xaxis_title="number of terminal nodes |T̃|",
        yaxis_title="MSE (log₁p-space)",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig, {
        "best_alpha": best_alpha,
        "best_n_leaves": n_leaves[int(np.argmin(test_errs))],
        "best_test_mse": min(test_errs),
    }

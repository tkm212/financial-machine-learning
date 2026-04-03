"""Shared setup and modeling helpers for ESL Chapter 2 (supervised learning) notebooks."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def find_project_root(max_up: int = 12) -> Path:
    p = Path.cwd().resolve()
    for _ in range(max_up):
        if (p / "pyproject.toml").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    raise RuntimeError("Could not find project root (pyproject.toml).")


def ensure_package_on_path(root: Path) -> None:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def init_paths() -> tuple[Path, Path, Path]:
    """Return ``(project_root, inputs_dir, outputs_dir)`` and put the package on ``sys.path``."""
    root = find_project_root()
    ensure_package_on_path(root)
    return root, root / "inputs", root / "outputs"


def load_tmdb_xy(inputs_dir: Path) -> tuple[pd.DataFrame, pd.Series, str]:
    from financial_machine_learning.esl_loaders import load_tmdb_revenue_regression

    return load_tmdb_revenue_regression(inputs_dir)


def fit_linear_train_test_mse(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.25,
    random_state: int = 0,
) -> dict[str, Any]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    train_mse = mean_squared_error(y_train, lin.predict(X_train))
    test_mse = mean_squared_error(y_test, lin.predict(X_test))
    return {
        "model": lin,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def subsample_xy(
    X: pd.DataFrame,
    y: pd.Series,
    max_rows: int,
    *,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    if len(X) <= max_rows:
        return X, y
    idx = np.random.default_rng(random_state).choice(len(X), size=max_rows, replace=False)
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


def knn_train_test_mse_figure(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    ks: np.ndarray | None = None,
    max_rows: int = 40_000,
    test_size: float = 0.25,
    random_state: int = 0,
) -> tuple[go.Figure, dict[str, Any]]:
    """Scale once per split, scan $k$, plot train vs test MSE; return figure and summary dict."""
    X_run, y_run = subsample_xy(X, y, max_rows, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X_run, y_run, test_size=test_size, random_state=random_state)

    lin = LinearRegression().fit(X_train, y_train)
    y_te_lin = lin.predict(X_test)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if ks is None:
        ks = np.arange(2, 10)
    train_mse: list[float] = []
    test_mse: list[float] = []
    for k in ks:
        knn = KNeighborsRegressor(n_neighbors=int(k), n_jobs=-1)
        knn.fit(X_train_s, y_train)
        train_mse.append(mean_squared_error(y_train, knn.predict(X_train_s)))
        test_mse.append(mean_squared_error(y_test, knn.predict(X_test_s)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=train_mse, name="train MSE", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=ks, y=test_mse, name="test MSE", mode="lines+markers"))
    fig.update_layout(
        title="$k$-NN: train vs test MSE",
        xaxis_title="k (neighbors)",
        yaxis_title="MSE",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    k_best = int(ks[int(np.argmin(test_mse))])
    summary = {
        "k_best": k_best,
        "min_test_mse": min(test_mse),
        "linear_test_mse": float(mean_squared_error(y_test, y_te_lin)),
        "ks": ks,
        "X_train": X_train,
        "y_train": y_train,
    }
    return fig, summary


def plot_predicted_vs_actual(
    y_true: pd.Series,
    y_pred: np.ndarray,
    *,
    title: str = "Predicted vs actual",
    label: str = "test",
) -> go.Figure:
    """Scatter of predicted vs actual with a perfect-fit diagonal."""
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            name=label,
            marker={"size": 5, "opacity": 0.5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="perfect fit",
            line={"dash": "dash", "color": "black"},
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="actual",
        yaxis_title="predicted",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


def default_single_feature(X: pd.DataFrame) -> str:
    if "budget" in X.columns:
        return "budget"
    if "popularity" in X.columns:
        return "popularity"
    return str(X.columns[0])


def linear_vs_knn_single_feature_figure(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_name: str,
    *,
    feat: str | None = None,
    k_neighbors: int = 15,
    grid_points: int = 200,
) -> tuple[go.Figure, str]:
    col = feat if feat is not None else default_single_feature(X_train)
    x1 = X_train[col].values.ravel()
    order = np.argsort(x1)
    grid = np.linspace(float(np.min(x1)), float(np.max(x1)), grid_points).reshape(-1, 1)

    lin1 = LinearRegression().fit(X_train[[col]], y_train)
    knn1 = make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=k_neighbors, n_jobs=-1),
    ).fit(X_train[[col]], y_train)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x1[order],
            y=y_train.iloc[order],
            mode="markers",
            name="train",
            marker=dict(size=4, opacity=0.35),
        )
    )
    fig.add_trace(go.Scatter(x=grid.ravel(), y=lin1.predict(grid), mode="lines", name="linear"))
    fig.add_trace(go.Scatter(x=grid.ravel(), y=knn1.predict(grid), mode="lines", name=f"k-NN (k={k_neighbors})"))
    fig.update_layout(
        title=f"Response vs {col}",
        xaxis_title=col,
        yaxis_title=target_name,
        template="plotly_white",
    )
    return fig, col

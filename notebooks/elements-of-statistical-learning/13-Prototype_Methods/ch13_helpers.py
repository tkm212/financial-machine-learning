"""Shared helpers for ESL Chapter 13 (Prototype Methods and Nearest-Neighbors) notebooks."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler


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


def _prepare_arrays(
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


# ---------------------------------------------------------------------------
# K-nearest-neighbor: k selection (§13.3)
# ---------------------------------------------------------------------------


def knn_k_selection_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    k_values: list[int] | None = None,
    max_rows: int = 2000,
    n_cv: int = 5,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    Cross-validated accuracy of KNN over a range of k values (§13.3).

    The k-NN classifier assigns the class by majority vote among the k nearest
    neighbours under Euclidean distance:

        C(x) = argmax_g  #{i in N_k(x) : y_i = g}

    where N_k(x) is the set of k indices closest to x.

    Bias-variance trade-off as k varies:
    - k=1: zero training error (each point is its own neighbour), high variance.
    - k=N: constant classifier (global majority vote), high bias.
    - Optimal k balances these; CV on a log-spaced grid is the standard approach (§13.3).

    The error rate of 1-NN is bounded above by twice the Bayes error rate as N -> inf,
    so KNN is always within a factor of 2 of optimal (Cover & Hart, 1967).
    """
    if k_values is None:
        k_values = [1, 3, 5, 7, 10, 15, 20, 30, 50]

    x_arr, y_cls = _prepare_arrays(X, y, feats, max_rows=max_rows)
    scaler = StandardScaler()
    x_arr = scaler.fit_transform(x_arr)

    mean_accs: list[float] = []
    std_accs: list[float] = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        scores = cross_val_score(knn, x_arr, y_cls, cv=n_cv, scoring="accuracy")
        mean_accs.append(float(scores.mean()))
        std_accs.append(float(scores.std()))

    best_idx = int(np.argmax(mean_accs))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=k_values,
            y=mean_accs,
            error_y={"type": "data", "array": std_accs, "visible": True},
            mode="lines+markers",
            name=f"{n_cv}-fold CV accuracy",
            line={"color": "steelblue"},
        )
    )
    fig.add_vline(
        x=k_values[best_idx],
        line_dash="dash",
        line_color="grey",
        annotation_text=f"best k={k_values[best_idx]}",
    )
    fig.update_layout(
        title=f"KNN: k vs {n_cv}-fold CV accuracy — §13.3",
        xaxis_title="k (number of neighbours)",
        yaxis_title="CV accuracy",
        template="plotly_white",
    )
    return fig, {
        "best_k": k_values[best_idx],
        "best_cv_accuracy": mean_accs[best_idx],
    }


# ---------------------------------------------------------------------------
# KNN: metric comparison (§13.3)
# ---------------------------------------------------------------------------


def knn_metric_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    k: int = 5,
    metrics: list[str] | None = None,
    max_rows: int = 2000,
    n_cv: int = 5,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    Compare distance metrics for k-NN (§13.3).

    The choice of distance metric encodes assumptions about the geometry of the
    input space.  ESL §13.3 discusses:

    - **Euclidean** ($L_2$): isotropic; sensitive to feature scales.
    - **Manhattan** ($L_1$): less sensitive to outliers; sum of absolute differences.
    - **Chebyshev** ($L_\\infty$): maximum absolute difference across dimensions.
    - **Cosine**: angle between vectors; scale-invariant, useful for high-dimensional data.

    All variants are evaluated at a fixed k after standard scaling.
    """
    if metrics is None:
        metrics = ["euclidean", "manhattan", "chebyshev", "cosine"]

    x_arr, y_cls = _prepare_arrays(X, y, feats, max_rows=max_rows)
    scaler = StandardScaler()
    x_arr = scaler.fit_transform(x_arr)

    mean_accs: list[float] = []
    std_accs: list[float] = []

    for metric in metrics:
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        scores = cross_val_score(knn, x_arr, y_cls, cv=n_cv, scoring="accuracy")
        mean_accs.append(float(scores.mean()))
        std_accs.append(float(scores.std()))

    best_idx = int(np.argmax(mean_accs))
    colors = ["steelblue", "tomato", "green", "purple"]

    fig = go.Figure(
        go.Bar(
            x=metrics,
            y=mean_accs,
            error_y={"type": "data", "array": std_accs, "visible": True},
            marker_color=colors[: len(metrics)],
            name=f"k={k}, {n_cv}-fold CV accuracy",
        )
    )
    fig.update_layout(
        title=f"KNN metric comparison: {n_cv}-fold CV accuracy (k={k}) — §13.3",
        xaxis_title="distance metric",
        yaxis_title="CV accuracy",
        yaxis={"range": [max(0.0, min(mean_accs) - 0.05), 1.0]},
        template="plotly_white",
    )
    return fig, {
        "best_metric": metrics[best_idx],
        "best_cv_accuracy": mean_accs[best_idx],
    }


# ---------------------------------------------------------------------------
# K-means prototypes (§13.2)
# ---------------------------------------------------------------------------


def kmeans_prototype_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    R_values: list[int] | None = None,
    max_rows: int = 2000,
    n_cv: int = 5,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    K-means prototypes as a pre-processing step for nearest-centroid classification (§13.2).

    The K-means prototype method (§13.2.1) compresses each class $g$ into $R$ prototypes
    by running K-means with $K = R$ on the class-$g$ observations.  New points are then
    classified by nearest-prototype rule:

        C(x) = argmin_g  min_{r in 1..R}  ||x - m_{g,r}||

    As $R$ increases the prototype set approaches the full training set (1-NN limit).
    For $R = 1$ per class this is the **nearest centroid** (Rocchio) classifier.

    The CV accuracy tracks how well the prototype compression retains class structure:
    a flat curve means most information is captured by a few centroids.
    """
    if R_values is None:
        R_values = [1, 2, 3, 5, 8, 10]

    if feats is None:
        feats = ["budget", "popularity", "runtime", "vote_average", "vote_count"]
    feats = [f for f in feats if f in X.columns]

    x_all, y_all = _prepare_arrays(X, y, feats, max_rows=max_rows)
    scaler = StandardScaler()
    x_all = scaler.fit_transform(x_all)

    classes = np.unique(y_all)
    mean_accs: list[float] = []
    std_accs: list[float] = []

    # Manual CV for prototype classifier
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=0)

    for R in R_values:
        fold_accs: list[float] = []
        for train_idx, test_idx in skf.split(x_all, y_all):
            X_tr, X_te = x_all[train_idx], x_all[test_idx]
            y_tr, y_te = y_all[train_idx], y_all[test_idx]

            # Build prototypes per class
            prototypes: list[np.ndarray] = []
            proto_labels: list[int] = []
            for cls in classes:
                cls_mask = y_tr == cls
                cls_X = X_tr[cls_mask]
                n_clusters = min(R, len(cls_X))
                km = KMeans(n_clusters=n_clusters, n_init=5, random_state=0)
                km.fit(cls_X)
                prototypes.append(km.cluster_centers_)
                proto_labels.extend([int(cls)] * n_clusters)

            proto_arr = np.vstack(prototypes)
            proto_y = np.array(proto_labels)

            # Predict by nearest prototype
            diffs = X_te[:, np.newaxis, :] - proto_arr[np.newaxis, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            nearest = np.argmin(dists, axis=1)
            y_pred = proto_y[nearest]
            fold_accs.append(accuracy_score(y_te, y_pred))

        mean_accs.append(float(np.mean(fold_accs)))
        std_accs.append(float(np.std(fold_accs)))

    best_idx = int(np.argmax(mean_accs))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=R_values,
            y=mean_accs,
            error_y={"type": "data", "array": std_accs, "visible": True},
            mode="lines+markers",
            name=f"{n_cv}-fold CV accuracy",
            line={"color": "steelblue"},
        )
    )
    fig.add_vline(
        x=R_values[best_idx],
        line_dash="dash",
        line_color="grey",
        annotation_text=f"best R={R_values[best_idx]}",
    )
    fig.update_layout(
        title=f"K-means prototypes: R per class vs {n_cv}-fold CV accuracy — §13.2",
        xaxis_title="R (prototypes per class)",
        yaxis_title="CV accuracy",
        template="plotly_white",
    )
    return fig, {
        "best_R": R_values[best_idx],
        "best_cv_accuracy": mean_accs[best_idx],
        "R1_accuracy": mean_accs[0],
    }


# ---------------------------------------------------------------------------
# LVQ: learning vector quantization (§13.2.2)
# ---------------------------------------------------------------------------


def lvq_vs_knn_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    k_values: list[int] | None = None,
    R_values: list[int] | None = None,
    max_rows: int = 2000,
    n_cv: int = 5,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    Compare K-means prototypes vs KNN at various sizes (§13.2.2 vs §13.3).

    LVQ (Kohonen 1989) extends K-means prototypes by **moving** prototypes toward
    correctly-classified training points and away from misclassified ones:

        m_{g,r} <- m_{g,r} + epsilon * (x - m_{g,r})   if C(x) = g  (attract)
        m_{g,r} <- m_{g,r} - epsilon * (x - m_{g,r})   if C(x) != g (repel)

    This refines the centroid positions to minimise training error rather than
    within-class variance (as K-means does).  We approximate LVQ using sklearn's
    KNeighborsClassifier on reduced-prototype sets vs full training data to illustrate
    the compression-accuracy trade-off.

    The plot shows CV accuracy for:
    - **Prototype classifier** (K-means, R prototypes per class)
    - **KNN** (k neighbours, full training data)
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 20]
    if R_values is None:
        R_values = [1, 2, 3, 5, 8]

    if feats is None:
        feats = ["budget", "popularity", "runtime", "vote_average", "vote_count"]
    feats = [f for f in feats if f in X.columns]

    x_all, y_all = _prepare_arrays(X, y, feats, max_rows=max_rows)
    scaler = StandardScaler()
    x_all = scaler.fit_transform(x_all)
    classes = np.unique(y_all)

    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=0)

    # --- KNN accuracies ---
    knn_accs: list[float] = []
    for k in k_values:
        scores = cross_val_score(
            KNeighborsClassifier(n_neighbors=k), x_all, y_all, cv=skf, scoring="accuracy"
        )
        knn_accs.append(float(scores.mean()))

    # --- Prototype accuracies ---
    proto_accs: list[float] = []
    for R in R_values:
        fold_accs: list[float] = []
        for train_idx, test_idx in skf.split(x_all, y_all):
            X_tr, X_te = x_all[train_idx], x_all[test_idx]
            y_tr, y_te = y_all[train_idx], y_all[test_idx]
            prototypes: list[np.ndarray] = []
            proto_labels: list[int] = []
            for cls in classes:
                cls_X = X_tr[y_tr == cls]
                n_clusters = min(R, len(cls_X))
                km = KMeans(n_clusters=n_clusters, n_init=5, random_state=0)
                km.fit(cls_X)
                prototypes.append(km.cluster_centers_)
                proto_labels.extend([int(cls)] * n_clusters)
            proto_arr = np.vstack(prototypes)
            proto_y = np.array(proto_labels)
            dists = np.linalg.norm(X_te[:, np.newaxis, :] - proto_arr[np.newaxis, :, :], axis=2)
            y_pred = proto_y[np.argmin(dists, axis=1)]
            fold_accs.append(accuracy_score(y_te, y_pred))
        proto_accs.append(float(np.mean(fold_accs)))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[f"KNN k={k}" for k in k_values],
            y=knn_accs,
            mode="lines+markers",
            name="KNN (full training data)",
            line={"color": "steelblue"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[f"Proto R={r}" for r in R_values],
            y=proto_accs,
            mode="lines+markers",
            name="K-means prototypes",
            line={"color": "tomato", "dash": "dot"},
        )
    )
    fig.update_layout(
        title=f"Prototypes vs KNN: {n_cv}-fold CV accuracy — §13.2-13.3",
        xaxis_title="method / complexity",
        yaxis_title="CV accuracy",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig, {
        "best_knn": max(knn_accs),
        "best_knn_k": k_values[int(np.argmax(knn_accs))],
        "best_proto": max(proto_accs),
        "best_proto_R": R_values[int(np.argmax(proto_accs))],
    }


# ---------------------------------------------------------------------------
# Train/test error vs k (§13.3)
# ---------------------------------------------------------------------------


def knn_train_test_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    k_values: list[int] | None = None,
    test_size: float = 0.25,
    max_rows: int = 2000,
) -> go.Figure:
    """
    KNN train and test error vs k on a held-out split (§13.3).

    This reproduces the classic bias-variance curve from ESL Fig 13.2:
    - Training error increases monotonically with k (more smoothing).
    - Test error has a U-shape: minimum at the optimal k.
    - The gap between train and test error shrinks as k grows (less overfitting).

    The effective number of parameters for KNN is approximately N/k, so increasing
    k is equivalent to reducing model complexity.
    """
    if k_values is None:
        k_values = list(range(1, 51))

    x_arr, y_cls = _prepare_arrays(X, y, feats, max_rows=max_rows)
    scaler = StandardScaler()
    x_arr = scaler.fit_transform(x_arr)
    X_tr, X_te, y_tr, y_te = train_test_split(x_arr, y_cls, test_size=test_size, random_state=0)

    train_errors: list[float] = []
    test_errors: list[float] = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_tr, y_tr)
        train_errors.append(1.0 - accuracy_score(y_tr, knn.predict(X_tr)))
        test_errors.append(1.0 - accuracy_score(y_te, knn.predict(X_te)))

    best_k = k_values[int(np.argmin(test_errors))]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=k_values,
            y=train_errors,
            mode="lines",
            name="Train error",
            line={"color": "steelblue"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=k_values,
            y=test_errors,
            mode="lines",
            name="Test error",
            line={"color": "tomato"},
        )
    )
    fig.add_vline(x=best_k, line_dash="dash", line_color="grey", annotation_text=f"best k={best_k}")
    fig.update_layout(
        title="KNN train vs test error — §13.3",
        xaxis_title="k (number of neighbours)",
        yaxis_title="error rate",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig

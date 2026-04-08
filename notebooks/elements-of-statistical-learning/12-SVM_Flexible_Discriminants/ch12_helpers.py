"""Shared helpers for ESL Chapter 12 (SVM and Flexible Discriminants) notebooks."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC


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
# SVM: cost parameter C (§12.2)
# ---------------------------------------------------------------------------


def svm_cost_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    C_values: list[float] | None = None,
    kernel: str = "rbf",
    max_rows: int = 1500,
    n_cv: int = 5,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    Effect of the cost parameter C on SVM generalisation (§12.2).

    The soft-margin SVM solves:
        min_{w, b, xi}  (1/2)||w||^2 + C * sum_i xi_i
        s.t.  y_i(w^T x_i + b) >= 1 - xi_i,  xi_i >= 0

    C controls the bias-variance trade-off:
    - C -> 0: wide margin, many margin violations (high bias, low variance).
    - C -> inf: hard margin, no violations allowed, fits every training point
      (low bias, high variance — equivalent to 0 training error, potentially overfit).

    ESL §12.2.1 shows the dual formulation:
        max_alpha  sum_i alpha_i - (1/2) sum_{i,j} alpha_i alpha_j y_i y_j x_i^T x_j
        s.t.  0 <= alpha_i <= C,  sum_i alpha_i y_i = 0

    Points with alpha_i > 0 are the **support vectors** — the only observations that
    determine the decision boundary.
    """
    if C_values is None:
        C_values = [0.01, 0.1, 1.0, 10.0, 100.0]

    x_arr, y_cls = _prepare_arrays(X, y, feats, max_rows=max_rows)
    scaler = StandardScaler()
    x_arr = scaler.fit_transform(x_arr)

    mean_accs: list[float] = []
    std_accs: list[float] = []
    n_svs: list[float] = []

    X_tr, _X_te, y_tr, _y_te = train_test_split(x_arr, y_cls, test_size=0.25, random_state=0)

    for C in C_values:
        svc = SVC(C=C, kernel=kernel, random_state=0)
        scores = cross_val_score(svc, x_arr, y_cls, cv=n_cv, scoring="accuracy")
        mean_accs.append(float(scores.mean()))
        std_accs.append(float(scores.std()))
        svc.fit(X_tr, y_tr)
        n_svs.append(float(svc.n_support_.sum()))

    log_C = [float(np.log10(c)) for c in C_values]
    best_idx = int(np.argmax(mean_accs))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=log_C,
            y=mean_accs,
            error_y={"type": "data", "array": std_accs, "visible": True},
            mode="lines+markers",
            name=f"CV accuracy (kernel={kernel})",
            line={"color": "steelblue"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=log_C,
            y=[n / (len(X_tr) * 0.8) for n in n_svs],
            mode="lines+markers",
            name="fraction support vectors",
            line={"color": "tomato", "dash": "dot"},
            yaxis="y2",
        )
    )
    fig.add_vline(
        x=np.log10(C_values[best_idx]),
        line_dash="dash",
        line_color="grey",
        annotation_text=f"best C={C_values[best_idx]}",
    )
    fig.update_layout(
        title=f"SVM cost parameter C vs accuracy (kernel={kernel}) — §12.2",
        xaxis_title="log₁₀(C)",
        yaxis_title="CV accuracy",
        yaxis2={"title": "fraction support vectors", "overlaying": "y", "side": "right", "showgrid": False},
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig, {
        "best_C": C_values[best_idx],
        "best_cv_accuracy": mean_accs[best_idx],
        "kernel": kernel,
    }


# ---------------------------------------------------------------------------
# SVM: kernel comparison (§12.3)
# ---------------------------------------------------------------------------


def svm_kernel_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    kernels: list[str] | None = None,
    C: float = 1.0,
    max_rows: int = 1500,
    n_cv: int = 5,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    Compare SVM kernels by cross-validated accuracy (§12.3).

    The kernel function $K(x, x') = \langle \phi(x), \phi(x') \rangle$ implicitly
    maps inputs to a high-dimensional (possibly infinite) feature space $\mathcal{H}$:

    | Kernel | $K(x, x')$ | Feature space |
    |---|---|---|
    | Linear | $x^\top x'$ | Original $\mathbb{R}^p$ |
    | Polynomial (d=3) | $(1 + x^\top x')^3$ | All degree ≤ 3 monomials |
    | RBF / Gaussian | $\exp(-\gamma \|x - x'\|^2)$ | Infinite-dimensional |
    | Sigmoid | $\tanh(\kappa x^\top x' + c)$ | Neural-network-like |

    Mercer's theorem guarantees that any positive semi-definite kernel corresponds to
    an inner product in some $\mathcal{H}$, so the SVM dual is valid for all such kernels.
    The choice of kernel encodes **prior beliefs** about the structure of the decision boundary.
    """
    if kernels is None:
        kernels = ["linear", "poly", "rbf", "sigmoid"]

    x_arr, y_cls = _prepare_arrays(X, y, feats, max_rows=max_rows)
    scaler = StandardScaler()
    x_arr = scaler.fit_transform(x_arr)

    mean_accs: list[float] = []
    std_accs: list[float] = []

    for kernel in kernels:
        svc = SVC(C=C, kernel=kernel, random_state=0)
        scores = cross_val_score(svc, x_arr, y_cls, cv=n_cv, scoring="accuracy")
        mean_accs.append(float(scores.mean()))
        std_accs.append(float(scores.std()))

    best_idx = int(np.argmax(mean_accs))
    colors = ["steelblue", "tomato", "green", "purple"]

    fig = go.Figure(
        go.Bar(
            x=kernels,
            y=mean_accs,
            error_y={"type": "data", "array": std_accs, "visible": True},
            marker_color=colors[: len(kernels)],
            name=f"{n_cv}-fold CV accuracy (C={C})",
        )
    )
    fig.update_layout(
        title=f"SVM kernel comparison: {n_cv}-fold CV accuracy (C={C}) — §12.3",
        xaxis_title="kernel",
        yaxis_title="CV accuracy",
        yaxis={"range": [max(0.0, min(mean_accs) - 0.05), 1.0]},
        template="plotly_white",
    )
    return fig, {
        "best_kernel": kernels[best_idx],
        "best_cv_accuracy": mean_accs[best_idx],
    }


# ---------------------------------------------------------------------------
# LDA vs FDA via polynomial features (§12.4-12.5)
# ---------------------------------------------------------------------------


def fda_vs_lda_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    poly_degrees: list[int] | None = None,
    max_rows: int = 2000,
    n_cv: int = 5,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    Flexible Discriminant Analysis vs LDA: effect of polynomial feature expansion (§12.5).

    **LDA** (§4.3) finds the linear combination of features that maximally separates classes.
    It assumes within-class Gaussian distributions with equal covariance.

    **FDA** (Hastie, Tibshirani & Buja 1994) replaces the linear predictor with a nonlinear
    regression:  it solves the same discriminant problem but uses **optimal scoring** with
    an arbitrary regression method $\eta(x)$ instead of linear regression.

    We approximate FDA by feeding polynomial-expanded features to LDA:
    - `degree=1`: standard LDA (linear boundary).
    - `degree=2`: quadratic features — LDA on $\{x_j, x_j^2, x_j x_k\}$.
    - `degree=3`: cubic features — captures higher-order interaction effects.

    Higher-degree expansions allow curved decision boundaries while the LDA step
    still performs the optimal linear combination of (now nonlinear) features.
    """
    if poly_degrees is None:
        poly_degrees = [1, 2, 3]

    x_arr, y_cls = _prepare_arrays(X, y, feats, max_rows=max_rows)

    labels = ["LDA (degree=1)"] + [f"FDA (degree={d})" for d in poly_degrees[1:]]
    mean_accs: list[float] = []
    std_accs: list[float] = []

    for degree in poly_degrees:
        pipe = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=degree, include_bias=False),
            StandardScaler(),
            LinearDiscriminantAnalysis(),
        )
        scores = cross_val_score(pipe, x_arr, y_cls, cv=n_cv, scoring="accuracy")
        mean_accs.append(float(scores.mean()))
        std_accs.append(float(scores.std()))

    best_idx = int(np.argmax(mean_accs))
    colors = ["tomato"] + ["steelblue"] * (len(poly_degrees) - 1)

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=mean_accs,
            error_y={"type": "data", "array": std_accs, "visible": True},
            marker_color=colors,
            name=f"{n_cv}-fold CV accuracy (±1 SD)",
        )
    )
    fig.update_layout(
        title=f"FDA vs LDA: {n_cv}-fold CV accuracy by polynomial degree — §12.5",
        xaxis_title="method",
        yaxis_title="CV accuracy",
        yaxis={"range": [max(0.0, min(mean_accs) - 0.05), 1.0]},
        template="plotly_white",
    )
    return fig, {
        "best_model": labels[best_idx],
        "best_cv_accuracy": mean_accs[best_idx],
        "lda_accuracy": mean_accs[0],
    }


# ---------------------------------------------------------------------------
# Penalised Discriminant Analysis (PDA) via shrinkage (§12.6)
# ---------------------------------------------------------------------------


def pda_shrinkage_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    max_rows: int = 2000,
    n_cv: int = 5,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    Penalised Discriminant Analysis: effect of covariance shrinkage (§12.6).

    **PDA** (Hastie, Buja & Tibshirani 1995) adds a smoothness penalty to the LDA criterion:
        ASR(c) = (1/(N-K)) * sum_k sum_{i in C_k} ||c(x_i) - c_k||^2 + lambda * c^T Omega c

    where $\Omega$ is a penalty matrix (e.g. roughness for splines).  With $\lambda = 0$ this
    reduces to FDA/LDA; large $\lambda$ heavily regularises the discriminant directions.

    sklearn's `LinearDiscriminantAnalysis` approximates PDA via the `shrinkage` parameter:
    - `shrinkage=None`: standard MLE covariance estimate.
    - `shrinkage='auto'`: Ledoit-Wolf analytical shrinkage (§12.6).
    - `shrinkage=t` for $t \in (0, 1]$: interpolate between sample and identity covariance:
      $\hat{\Sigma}_{shrunk} = (1 - t)\hat{\Sigma} + t \cdot \text{tr}(\hat{\Sigma})/p \cdot I$

    Regularised covariance prevents near-singular estimates in high dimensions (p >> N).
    """
    x_arr, y_cls = _prepare_arrays(X, y, feats, max_rows=max_rows)
    scaler = StandardScaler()
    x_arr = scaler.fit_transform(x_arr)

    shrinkage_values = [None, "auto", 0.0, 0.1, 0.3, 0.5, 0.8, 1.0]
    labels = ["MLE", "Ledoit-Wolf"] + [f"t={s}" for s in shrinkage_values[2:]]
    mean_accs: list[float] = []
    std_accs: list[float] = []

    for shrinkage in shrinkage_values:
        solver = "lsqr" if shrinkage is not None else "svd"
        lda = LinearDiscriminantAnalysis(shrinkage=shrinkage, solver=solver)
        scores = cross_val_score(lda, x_arr, y_cls, cv=n_cv, scoring="accuracy")
        mean_accs.append(float(scores.mean()))
        std_accs.append(float(scores.std()))

    best_idx = int(np.argmax(mean_accs))

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=mean_accs,
            error_y={"type": "data", "array": std_accs, "visible": True},
            marker_color="steelblue",
            name=f"{n_cv}-fold CV accuracy (±1 SD)",
        )
    )
    fig.add_hline(
        y=mean_accs[0],
        line_dash="dash",
        line_color="tomato",
        annotation_text="MLE baseline",
        annotation_position="right",
    )
    fig.update_layout(
        title=f"PDA covariance shrinkage: {n_cv}-fold CV accuracy — §12.6",
        xaxis_title="shrinkage parameter",
        yaxis_title="CV accuracy",
        yaxis={"range": [max(0.0, min(mean_accs) - 0.05), 1.0]},
        template="plotly_white",
    )
    return fig, {
        "best_shrinkage": labels[best_idx],
        "best_cv_accuracy": mean_accs[best_idx],
        "mle_accuracy": mean_accs[0],
    }


# ---------------------------------------------------------------------------
# SVM vs FDA vs PDA accuracy summary (§12.5-12.6)
# ---------------------------------------------------------------------------


def method_comparison_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    max_rows: int = 1500,
    n_cv: int = 5,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    Accuracy comparison across SVM (linear & RBF), LDA, FDA, and PDA (§12.4-12.6).

    This gives an empirical view of the trade-offs discussed in ESL §12.4-12.6:
    - **LDA**: fast, interpretable, optimal under Gaussian with equal covariance.
    - **FDA**: retains LDA interpretability but allows nonlinear class boundaries.
    - **PDA**: FDA with regularisation — useful when p is large relative to N.
    - **Linear SVM**: related to LDA but maximises margin; robust to non-Gaussian data.
    - **RBF SVM**: fully nonlinear kernel — most flexible, most prone to overfitting without
      careful C and gamma tuning.

    All methods are evaluated by 5-fold CV on the same standardised features.
    """
    x_arr, y_cls = _prepare_arrays(X, y, feats, max_rows=max_rows)
    scaler = StandardScaler()
    x_arr_s = scaler.fit_transform(x_arr)

    methods: dict[str, Any] = {
        "LDA": LinearDiscriminantAnalysis(),
        "FDA (deg=2)": make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(),
            LinearDiscriminantAnalysis(),
        ),
        "PDA (Ledoit-Wolf)": LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr"),
        "SVM (linear)": SVC(C=1.0, kernel="linear", random_state=0),
        "SVM (RBF)": SVC(C=1.0, kernel="rbf", random_state=0),
    }

    names: list[str] = []
    mean_accs: list[float] = []
    std_accs: list[float] = []

    for name, model in methods.items():
        scores = cross_val_score(model, x_arr_s, y_cls, cv=n_cv, scoring="accuracy")
        names.append(name)
        mean_accs.append(float(scores.mean()))
        std_accs.append(float(scores.std()))

    best_idx = int(np.argmax(mean_accs))
    colors = ["tomato", "steelblue", "steelblue", "green", "green"]

    fig = go.Figure(
        go.Bar(
            x=names,
            y=mean_accs,
            error_y={"type": "data", "array": std_accs, "visible": True},
            marker_color=colors,
            name=f"{n_cv}-fold CV accuracy (±1 SD)",
        )
    )
    fig.update_layout(
        title=f"Method comparison: {n_cv}-fold CV accuracy — §12.4-12.6",
        xaxis_title="method",
        yaxis_title="CV accuracy",
        yaxis={"range": [max(0.0, min(mean_accs) - 0.05), 1.0]},
        template="plotly_white",
    )
    return fig, {
        "best_method": names[best_idx],
        "best_cv_accuracy": mean_accs[best_idx],
        "results": dict(zip(names, mean_accs, strict=False)),
    }

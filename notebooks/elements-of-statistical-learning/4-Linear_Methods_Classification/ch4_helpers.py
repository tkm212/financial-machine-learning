"""Shared helpers for ESL Chapter 4 (Linear Methods for Classification) notebooks."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


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
    from financial_machine_learning.esl_loaders import load_tmdb_revenue_classification

    return load_tmdb_revenue_classification(inputs_dir)


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


# ---------------------------------------------------------------------------
# LDA / QDA (§4.3)
# ---------------------------------------------------------------------------


def _meshgrid_from_z(Z: np.ndarray, *, n_grid: int = 200, margin: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    x_min, x_max = float(Z[:, 0].min()) - margin, float(Z[:, 0].max()) + margin
    y_min, y_max = float(Z[:, 1].min()) - margin, float(Z[:, 1].max()) + margin
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_grid), np.linspace(y_min, y_max, n_grid))
    return xx, yy


def lda_2d_boundary_figure(
    X_train_s: np.ndarray,
    y_train: pd.Series,
    *,
    n_grid: int = 200,
) -> tuple[go.Figure, LinearDiscriminantAnalysis, PCA]:
    """LDA decision boundary plotted in the first two principal components."""
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X_train_s)

    lda = LinearDiscriminantAnalysis()
    lda.fit(Z, y_train)

    xx, yy = _meshgrid_from_z(Z, n_grid=n_grid)
    Z_pred = lda.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    colors = {0: "#4393c3", 1: "#d6604d"}
    labels = {0: "Low revenue", 1: "High revenue"}

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=np.linspace(float(xx.min()), float(xx.max()), n_grid),
            y=np.linspace(float(yy.min()), float(yy.max()), n_grid),
            z=Z_pred.astype(float),
            showscale=False,
            colorscale=[[0, "rgba(67,147,195,0.25)"], [1, "rgba(214,96,77,0.25)"]],
            contours={"coloring": "fill"},
            name="boundary",
        )
    )
    y_arr = np.asarray(y_train)
    for cls in [0, 1]:
        mask = y_arr == cls
        fig.add_trace(
            go.Scatter(
                x=Z[mask, 0],
                y=Z[mask, 1],
                mode="markers",
                marker={"size": 4, "color": colors[cls], "opacity": 0.6},
                name=labels[cls],
            )
        )
    fig.update_layout(
        title="LDA decision boundary (PCA 2D projection, §4.3)",
        xaxis_title="PC1",
        yaxis_title="PC2",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig, lda, pca


def lda_vs_qda_logistic_figure(
    X_train_s: np.ndarray,
    X_test_s: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[go.Figure, pd.DataFrame]:
    """Accuracy bar chart: LDA, QDA, Logistic (L2), Logistic (L1)."""
    models: list[tuple[str, Any]] = [
        ("LDA", LinearDiscriminantAnalysis()),
        ("QDA", QuadraticDiscriminantAnalysis()),
        ("Logistic L2", LogisticRegression(C=1.0, l1_ratio=0, solver="saga", max_iter=1000)),
        ("Logistic L1", LogisticRegression(C=1.0, l1_ratio=1, solver="saga", max_iter=1000)),
    ]
    rows = []
    for name, model in models:
        model.fit(X_train_s, y_train)
        rows.append({
            "model": name,
            "train_accuracy": round(float(accuracy_score(y_train, model.predict(X_train_s))), 4),
            "test_accuracy": round(float(accuracy_score(y_test, model.predict(X_test_s))), 4),
        })
    df = pd.DataFrame(rows)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["model"], y=df["train_accuracy"], name="train accuracy"))
    fig.add_trace(go.Bar(x=df["model"], y=df["test_accuracy"], name="test accuracy"))
    fig.update_layout(
        title="LDA vs QDA vs Logistic: accuracy comparison (§4.3 / §4.4)",
        yaxis_title="accuracy",
        barmode="group",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        yaxis={"range": [0, 1]},
    )
    return fig, df


def rda_shrinkage_figure(
    X_train_s: np.ndarray,
    X_test_s: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    n_steps: int = 40,
) -> tuple[go.Figure, dict[str, Any]]:
    """Regularized LDA: test accuracy vs shrinkage parameter (0 = OLS-covariance, 1 = diagonal)."""
    shrinkages = np.linspace(0.0, 1.0, n_steps)
    train_acc, test_acc = [], []
    for s in shrinkages:
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=float(s))
        lda.fit(X_train_s, y_train)
        train_acc.append(float(accuracy_score(y_train, lda.predict(X_train_s))))
        test_acc.append(float(accuracy_score(y_test, lda.predict(X_test_s))))

    best_idx = int(np.argmax(test_acc))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=shrinkages, y=train_acc, mode="lines", name="train accuracy"))
    fig.add_trace(go.Scatter(x=shrinkages, y=test_acc, mode="lines", name="test accuracy"))
    fig.add_vline(
        x=float(shrinkages[best_idx]),
        line_dash="dash",
        line_color="grey",
        annotation_text=f"best={shrinkages[best_idx]:.2f}",
    )
    fig.update_layout(
        title="Regularized LDA: accuracy vs shrinkage (§4.3.1)",
        xaxis_title="shrinkage (0 = full LDA, 1 = diagonal)",
        yaxis_title="accuracy",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig, {
        "best_shrinkage": float(shrinkages[best_idx]),
        "best_test_acc": float(max(test_acc)),
    }


# ---------------------------------------------------------------------------
# Logistic Regression (§4.4)
# ---------------------------------------------------------------------------


def logistic_l1_coef_path_figure(
    X_train_s: np.ndarray,
    y_train: pd.Series,
    feat_names: list[str],
    *,
    n_cs: int = 50,
) -> go.Figure:
    """L1 logistic regression: coefficient paths as C (= 1/lambda) increases."""
    Cs = np.logspace(-3, 3, n_cs)
    coefs = []
    for C in Cs:
        lr = LogisticRegression(C=float(C), l1_ratio=1, solver="saga", max_iter=2000)
        lr.fit(X_train_s, y_train)
        coefs.append(lr.coef_[0].copy())
    coefs_arr = np.array(coefs)

    fig = go.Figure()
    for i, name in enumerate(feat_names):
        fig.add_trace(go.Scatter(x=np.log10(Cs), y=coefs_arr[:, i], mode="lines", name=name))
    fig.update_layout(
        title="L1 Logistic Regression: coefficient paths as penalty relaxes (§4.4.4)",
        xaxis_title="log10(C)  [C = 1/lambda; left = heavy penalty]",
        yaxis_title="coefficient",
        template="plotly_white",
    )
    return fig


def logistic_l1_vs_l2_accuracy_figure(
    X_train_s: np.ndarray,
    X_test_s: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    n_cs: int = 40,
) -> tuple[go.Figure, dict[str, Any]]:
    """Test accuracy vs C for L1 and L2 logistic regression."""
    Cs = np.logspace(-3, 3, n_cs)
    l1_acc, l2_acc = [], []
    for C in Cs:
        l1 = LogisticRegression(C=float(C), l1_ratio=1, solver="saga", max_iter=2000)
        l2 = LogisticRegression(C=float(C), l1_ratio=0, solver="saga", max_iter=2000)
        l1.fit(X_train_s, y_train)
        l2.fit(X_train_s, y_train)
        l1_acc.append(float(accuracy_score(y_test, l1.predict(X_test_s))))
        l2_acc.append(float(accuracy_score(y_test, l2.predict(X_test_s))))

    best_l1_idx = int(np.argmax(l1_acc))
    best_l2_idx = int(np.argmax(l2_acc))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.log10(Cs), y=l1_acc, mode="lines", name="L1 (lasso-logistic)"))
    fig.add_trace(go.Scatter(x=np.log10(Cs), y=l2_acc, mode="lines", name="L2 (ridge-logistic)"))
    fig.update_layout(
        title="Logistic Regression: L1 vs L2 test accuracy (§4.4.4)",
        xaxis_title="log10(C)  [C = 1/lambda]",
        yaxis_title="test accuracy",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig, {
        "best_l1_C": float(Cs[best_l1_idx]),
        "best_l1_acc": float(max(l1_acc)),
        "best_l2_C": float(Cs[best_l2_idx]),
        "best_l2_acc": float(max(l2_acc)),
    }


# ---------------------------------------------------------------------------
# Separating Hyperplanes (§4.5)
# ---------------------------------------------------------------------------


def make_separable_2d(
    n: int = 200,
    *,
    random_state: int = 42,
    margin: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a linearly separable 2D dataset with a clear margin."""
    rng = np.random.default_rng(random_state)
    half = n // 2
    X0 = rng.standard_normal((half, 2)) + np.array([-margin, 0.0])
    X1 = rng.standard_normal((half, 2)) + np.array([+margin, 0.0])
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(half, dtype=int), np.ones(half, dtype=int)])
    return X, y


class PerceptronTracer:
    """Rosenblatt perceptron that records weight vectors at each update."""

    def __init__(self, *, max_iter: int = 200, lr: float = 1.0, random_state: int = 0) -> None:
        self.max_iter = max_iter
        self.lr = lr
        self.random_state = random_state
        self.w_history: list[np.ndarray] = []
        self.misclassified: list[int] = []
        self.w_: np.ndarray = np.array([])
        self.b_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> PerceptronTracer:
        rng = np.random.default_rng(self.random_state)
        _, p = X.shape
        self.w_ = rng.standard_normal(p) * 0.01
        self.b_ = 0.0
        y_pm = np.where(y == 1, 1, -1)

        for _ in range(self.max_iter):
            errors = 0
            for xi, yi in zip(X, y_pm, strict=False):
                pred = np.sign(self.w_ @ xi + self.b_)
                if pred != yi:
                    self.w_ = self.w_ + self.lr * yi * xi
                    self.b_ = self.b_ + self.lr * yi
                    errors += 1
            self.w_history.append(self.w_.copy())
            self.misclassified.append(errors)
            if errors == 0:
                break
        return self


def perceptron_convergence_figure(X: np.ndarray, y: np.ndarray) -> tuple[go.Figure, go.Figure, PerceptronTracer]:
    """Two figures: (1) misclassification count per epoch; (2) final decision boundary."""
    tracer = PerceptronTracer(max_iter=200, lr=1.0)
    tracer.fit(X, y)

    fig_conv = go.Figure()
    fig_conv.add_trace(
        go.Scatter(x=list(range(1, len(tracer.misclassified) + 1)), y=tracer.misclassified, mode="lines+markers")
    )
    fig_conv.update_layout(
        title="Perceptron: misclassifications per epoch (§4.5.1)",
        xaxis_title="epoch",
        yaxis_title="# misclassified",
        template="plotly_white",
    )

    fig_boundary = _scatter_with_boundary(X, y, tracer.w_, tracer.b_, "Perceptron final boundary (§4.5.1)")
    return fig_conv, fig_boundary, tracer


def _scatter_with_boundary(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    title: str,
) -> go.Figure:
    """Scatter plot + linear decision boundary for 2D data."""
    colors = {0: "#4393c3", 1: "#d6604d"}
    labels = {0: "Class 0", 1: "Class 1"}

    fig = go.Figure()
    for cls in [0, 1]:
        mask = y == cls
        fig.add_trace(
            go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode="markers",
                marker={"size": 7, "color": colors[cls], "opacity": 0.7},
                name=labels[cls],
            )
        )

    x_lo, x_hi = float(X[:, 0].min()) - 0.5, float(X[:, 0].max()) + 0.5
    if abs(w[1]) > 1e-10:
        y_lo = -(w[0] * x_lo + b) / w[1]
        y_hi = -(w[0] * x_hi + b) / w[1]
        fig.add_trace(
            go.Scatter(x=[x_lo, x_hi], y=[y_lo, y_hi], mode="lines", name="boundary", line={"color": "black"})
        )

    fig.update_layout(title=title, xaxis_title="x1", yaxis_title="x2", template="plotly_white")
    return fig


def svm_margin_figure(X: np.ndarray, y: np.ndarray) -> go.Figure:
    """LinearSVC optimal separating hyperplane: boundary + margin bands + support vectors."""
    svm = LinearSVC(C=1e4, max_iter=10_000)
    svm.fit(X, y)
    w = svm.coef_[0]
    b = svm.intercept_[0]

    colors = {0: "#4393c3", 1: "#d6604d"}
    labels = {0: "Class 0", 1: "Class 1"}

    fig = go.Figure()
    for cls in [0, 1]:
        mask = y == cls
        fig.add_trace(
            go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode="markers",
                marker={"size": 7, "color": colors[cls], "opacity": 0.7},
                name=labels[cls],
            )
        )

    x_lo, x_hi = float(X[:, 0].min()) - 0.5, float(X[:, 0].max()) + 0.5
    xs = np.array([x_lo, x_hi])
    if abs(w[1]) > 1e-10:
        for offset, dash, name in [(0, "solid", "boundary"), (+1, "dash", "+margin"), (-1, "dash", "-margin")]:
            ys = -(w[0] * xs + b - offset) / w[1]
            fig.add_trace(
                go.Scatter(
                    x=xs.tolist(),
                    y=ys.tolist(),
                    mode="lines",
                    name=name,
                    line={"color": "black", "dash": dash},
                )
            )

    scores = X @ w + b
    margin = 1.0 / np.linalg.norm(w)
    sv_mask = np.abs(scores) <= 1.0 + 1e-3
    if sv_mask.any():
        fig.add_trace(
            go.Scatter(
                x=X[sv_mask, 0],
                y=X[sv_mask, 1],
                mode="markers",
                marker={"size": 12, "color": "rgba(0,0,0,0)", "line": {"color": "black", "width": 2}},
                name="support vectors",
            )
        )

    fig.update_layout(
        title=f"Optimal separating hyperplane (§4.5.2) — margin = {margin:.3f}",
        xaxis_title="x1",
        yaxis_title="x2",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


def perceptron_vs_svm_figure(X: np.ndarray, y: np.ndarray) -> go.Figure:
    """Overlay perceptron and SVM boundaries on the same scatter."""
    tracer = PerceptronTracer(max_iter=200)
    tracer.fit(X, y)

    svm = LinearSVC(C=1e4, max_iter=10_000)
    svm.fit(X, y)
    w_svm = svm.coef_[0]
    b_svm = svm.intercept_[0]

    colors = {0: "#4393c3", 1: "#d6604d"}
    labels = {0: "Class 0", 1: "Class 1"}

    fig = go.Figure()
    for cls in [0, 1]:
        mask = y == cls
        fig.add_trace(
            go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode="markers",
                marker={"size": 7, "color": colors[cls], "opacity": 0.6},
                name=labels[cls],
            )
        )

    x_lo, x_hi = float(X[:, 0].min()) - 0.5, float(X[:, 0].max()) + 0.5
    xs = np.array([x_lo, x_hi])

    w_p, b_p = tracer.w_, tracer.b_
    if abs(w_p[1]) > 1e-10:
        ys = -(w_p[0] * xs + b_p) / w_p[1]
        fig.add_trace(go.Scatter(x=xs.tolist(), y=ys.tolist(), mode="lines", name="Perceptron", line={"dash": "dot"}))

    if abs(w_svm[1]) > 1e-10:
        ys = -(w_svm[0] * xs + b_svm) / w_svm[1]
        fig.add_trace(
            go.Scatter(x=xs.tolist(), y=ys.tolist(), mode="lines", name="SVM (max-margin)", line={"color": "black"})
        )

    fig.update_layout(
        title="Perceptron vs SVM: boundary comparison",
        xaxis_title="x1",
        yaxis_title="x2",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig

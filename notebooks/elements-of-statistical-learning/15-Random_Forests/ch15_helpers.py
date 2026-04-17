"""Shared helpers for ESL Chapter 15 (Random Forests) notebooks."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


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
# OOB error vs number of trees (§15.3.1)
# ---------------------------------------------------------------------------


def rf_oob_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    n_estimators_values: list[int] | None = None,
    max_features_options: list[str | int] | None = None,
    max_rows: int = 2000,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    Random forest OOB error vs number of trees, for multiple ``max_features`` values (§15.3.1).

    **Out-of-bag (OOB) error** is a free internal estimate of generalisation error that
    requires no separate validation set (§15.3.1):

    - Each tree $T_b$ is fit on a bootstrap sample $Z^{*b}$ of size $N$.
    - Observations *not* in $Z^{*b}$ (approximately $N/e \approx 0.368 N$) are **out-of-bag**.
    - The OOB prediction for $x_i$ averages only the trees for which $i$ was OOB.
    - OOB error approximates the leave-one-out CV error without the extra computation.

    As the number of trees grows, OOB error **decreases monotonically** and stabilises —
    unlike boosting, which can overfit with too many trees.  The plot shows the point of
    diminishing returns where additional trees cost compute but buy little accuracy.

    **``max_features``** (denoted $m$ in ESL §15.2) controls how many features are
    considered at each split:
    - $m = p$: **bagging** — every feature is considered at every split.
    - $m = \\sqrt{p}$: ESL's default for classification.
    - $m = p/3$: ESL's default for regression.
    - Smaller $m$ → less correlated trees → lower ensemble variance (§15.4.1).
    """
    if n_estimators_values is None:
        n_estimators_values = [10, 25, 50, 100, 200, 300]
    if max_features_options is None:
        max_features_options = ["sqrt", "log2", 1]

    x_arr, y_cls = _prepare_arrays(X, y, feats, max_rows=max_rows)
    colors = ["steelblue", "tomato", "green", "purple"]

    fig = go.Figure()
    summary: dict[str, Any] = {}

    for i, mf in enumerate(max_features_options):
        oob_errors: list[float] = []
        for n_est in n_estimators_values:
            rf = RandomForestClassifier(
                n_estimators=n_est,
                max_features=mf,
                oob_score=True,
                random_state=0,
                n_jobs=-1,
            )
            rf.fit(x_arr, y_cls)
            oob_errors.append(1.0 - float(rf.oob_score_))

        label = f"max_features={mf}"
        fig.add_trace(
            go.Scatter(
                x=n_estimators_values,
                y=oob_errors,
                mode="lines+markers",
                name=label,
                line={"color": colors[i % len(colors)]},
            )
        )
        summary[label] = {
            "final_oob_error": oob_errors[-1],
            "min_oob_error": min(oob_errors),
        }

    fig.update_layout(
        title="Random forest OOB error vs number of trees — §15.3.1",
        xaxis_title="number of trees B",
        yaxis_title="OOB error rate",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig, summary


# ---------------------------------------------------------------------------
# Variable importance (§15.3.2)
# ---------------------------------------------------------------------------


def rf_variable_importance_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    n_estimators: int = 200,
    max_rows: int = 2000,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    Random forest variable importance (MDI and permutation) (§15.3.2).

    ESL §15.3.2 defines two importance measures:

    **Mean Decrease in Impurity (MDI)** (also called Gini importance):
    For each feature $j$, sum the weighted impurity reductions across all nodes and trees
    where $j$ was used to split:

    $$\\mathcal{I}^2(j) = \\frac{1}{B} \\sum_{b=1}^B \\sum_{t : j_t = j}
      p_t \\cdot i(t)$$

    where $p_t = N_t / N$ and $i(t)$ = impurity decrease at node $t$.

    **Permutation importance** (MDA): for each tree, measure OOB accuracy before and
    after randomly permuting feature $j$ for the OOB observations.  The importance is
    the average drop in accuracy across trees.

    MDI is faster but biased toward high-cardinality features (§15.3.2).
    Permutation importance is less biased but slower.
    """
    if feats is None:
        feats = ["budget", "popularity", "runtime", "vote_average", "vote_count"]
    feats = [f for f in feats if f in X.columns]

    x_arr, y_cls = _prepare_arrays(X, y, feats, max_rows=max_rows)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        oob_score=True,
        random_state=0,
        n_jobs=-1,
    )
    rf.fit(x_arr, y_cls)

    mdi_importances = rf.feature_importances_
    order = np.argsort(mdi_importances)[::-1]
    sorted_feats = [feats[i] for i in order]
    sorted_imp = mdi_importances[order]

    fig = go.Figure(
        go.Bar(
            x=sorted_feats,
            y=list(sorted_imp),
            marker_color="steelblue",
            name="MDI importance",
        )
    )
    fig.update_layout(
        title=f"Random forest variable importance (MDI, B={n_estimators}) — §15.3.2",
        xaxis_title="feature",
        yaxis_title="mean decrease in impurity",
        template="plotly_white",
    )
    return fig, {
        "importances": dict(zip(sorted_feats, sorted_imp.tolist(), strict=False)),
        "oob_accuracy": float(rf.oob_score_),
        "top_feature": sorted_feats[0],
    }


# ---------------------------------------------------------------------------
# max_features: bias-variance trade-off (§15.4.1)
# ---------------------------------------------------------------------------


def rf_max_features_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    n_estimators: int = 200,
    max_features_options: list[str | int] | None = None,
    max_rows: int = 2000,
    n_cv: int = 5,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    Effect of ``max_features`` (m) on random forest bias and variance (§15.4.1).

    ESL Theorem 15.1 (Breiman 2001): the generalisation error of a random forest is
    bounded by:

    $$PE^* \\leq \\bar{\\rho} \\cdot \\frac{1 - s^2}{s^2}$$

    where $\\bar{\\rho}$ is the mean pairwise correlation between trees and $s$ is
    the mean tree strength (accuracy).

    Reducing $m$ (fewer candidate features per split):
    - **Decreases** $\\bar{\\rho}$ — trees see different features, making them less correlated.
    - **Decreases** $s$ — individual trees are weaker (less data-adaptive per node).

    The optimal $m$ minimises the bound by finding the best trade-off.
    ESL's heuristic: $m = \\lfloor \\sqrt{p} \\rfloor$ for classification, $m = \\lfloor p/3 \\rfloor$ for regression.

    We compare OOB accuracy across multiple ``max_features`` settings to illustrate this.
    """
    if max_features_options is None:
        max_features_options = [1, 2, "sqrt", "log2", None]

    x_arr, y_cls = _prepare_arrays(X, y, feats, max_rows=max_rows)

    labels: list[str] = []
    oob_accs: list[float] = []
    cv_accs: list[float] = []
    cv_stds: list[float] = []

    for mf in max_features_options:
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=mf,
            oob_score=True,
            random_state=0,
            n_jobs=-1,
        )
        rf.fit(x_arr, y_cls)
        oob_accs.append(float(rf.oob_score_))

        scores = cross_val_score(
            RandomForestClassifier(n_estimators=n_estimators, max_features=mf, random_state=0, n_jobs=-1),
            x_arr, y_cls, cv=n_cv, scoring="accuracy",
        )
        cv_accs.append(float(scores.mean()))
        cv_stds.append(float(scores.std()))
        label = "all (bagging)" if mf is None else f"m={mf}"
        labels.append(label)

    best_idx = int(np.argmax(cv_accs))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=cv_accs,
            error_y={"type": "data", "array": cv_stds, "visible": True},
            name=f"{n_cv}-fold CV accuracy",
            marker_color="steelblue",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=oob_accs,
            mode="markers",
            name="OOB accuracy",
            marker={"color": "tomato", "size": 10, "symbol": "diamond"},
        )
    )
    fig.update_layout(
        title=f"max_features: CV and OOB accuracy (B={n_estimators}) — §15.4.1",
        xaxis_title="max_features (m)",
        yaxis_title="accuracy",
        yaxis={"range": [max(0.0, min(cv_accs) - 0.05), 1.0]},
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig, {
        "best_max_features": labels[best_idx],
        "best_cv_accuracy": cv_accs[best_idx],
        "results": dict(zip(labels, cv_accs, strict=False)),
    }


# ---------------------------------------------------------------------------
# RF vs single tree vs gradient boosting (§15.3)
# ---------------------------------------------------------------------------


def rf_comparison_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    n_estimators: int = 200,
    max_rows: int = 2000,
    n_cv: int = 5,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    Comparison: single decision tree, bagging, random forest, and gradient boosting (§15.3).

    This contextualises random forests within the ensemble methods covered in ESL:

    | Method | Idea | Bias | Variance |
    |---|---|---|---|
    | **Single tree** | CART grown to full depth | low | high |
    | **Bagging** (§8.7) | Avg. of B full trees, $m = p$ | low | lower |
    | **Random forest** | Avg. of B trees, $m = \\sqrt{p}$ | slightly higher | lowest |
    | **Gradient boosting** (§10.3) | Sequential correction of residuals | lowest | higher |

    Random forests trade a small amount of individual tree accuracy for much lower
    inter-tree correlation, achieving better ensemble variance reduction than bagging.
    Boosting can have lower bias but requires careful regularisation to avoid overfitting.
    """
    x_arr, y_cls = _prepare_arrays(X, y, feats, max_rows=max_rows)

    methods: dict[str, Any] = {
        "Single tree": DecisionTreeClassifier(random_state=0),
        f"Bagging (B={n_estimators})": RandomForestClassifier(
            n_estimators=n_estimators, max_features=None, random_state=0, n_jobs=-1
        ),
        f"RF √p (B={n_estimators})": RandomForestClassifier(
            n_estimators=n_estimators, max_features="sqrt", random_state=0, n_jobs=-1
        ),
        f"GBM (B={n_estimators})": GradientBoostingClassifier(
            n_estimators=n_estimators, learning_rate=0.1, max_depth=3, random_state=0
        ),
    }

    names: list[str] = []
    mean_accs: list[float] = []
    std_accs: list[float] = []

    for name, model in methods.items():
        scores = cross_val_score(model, x_arr, y_cls, cv=n_cv, scoring="accuracy")
        names.append(name)
        mean_accs.append(float(scores.mean()))
        std_accs.append(float(scores.std()))

    best_idx = int(np.argmax(mean_accs))
    colors = ["tomato", "steelblue", "steelblue", "green"]

    fig = go.Figure(
        go.Bar(
            x=names,
            y=mean_accs,
            error_y={"type": "data", "array": std_accs, "visible": True},
            marker_color=colors,
            name=f"{n_cv}-fold CV accuracy",
        )
    )
    fig.update_layout(
        title=f"Ensemble comparison: {n_cv}-fold CV accuracy — §15.3",
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


# ---------------------------------------------------------------------------
# Tree depth effect (§15.3)
# ---------------------------------------------------------------------------


def rf_tree_depth_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    n_estimators: int = 200,
    max_depth_values: list[int | None] | None = None,
    max_rows: int = 2000,
    n_cv: int = 5,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    Effect of individual tree depth on random forest accuracy (§15.3).

    ESL §15.3 recommends growing each tree to full depth (no pruning):
    - Each deep tree has low bias but high variance.
    - The averaging across $B$ de-correlated trees reduces variance without increasing bias.
    - Shallow trees reduce variance less through averaging (higher bias dominates).

    We sweep ``max_depth`` from 1 (decision stumps) to None (full trees) and measure
    OOB accuracy to confirm the full-depth recommendation.
    """
    if max_depth_values is None:
        max_depth_values = [1, 2, 3, 5, 8, None]

    x_arr, y_cls = _prepare_arrays(X, y, feats, max_rows=max_rows)

    labels: list[str] = []
    oob_accs: list[float] = []
    cv_accs: list[float] = []
    cv_stds: list[float] = []

    for depth in max_depth_values:
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features="sqrt",
            max_depth=depth,
            oob_score=True,
            random_state=0,
            n_jobs=-1,
        )
        rf.fit(x_arr, y_cls)
        oob_accs.append(float(rf.oob_score_))

        scores = cross_val_score(
            RandomForestClassifier(
                n_estimators=n_estimators, max_features="sqrt", max_depth=depth, random_state=0, n_jobs=-1
            ),
            x_arr, y_cls, cv=n_cv, scoring="accuracy",
        )
        cv_accs.append(float(scores.mean()))
        cv_stds.append(float(scores.std()))
        labels.append("full" if depth is None else f"depth={depth}")

    best_idx = int(np.argmax(cv_accs))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=cv_accs,
            error_y={"type": "data", "array": cv_stds, "visible": True},
            name=f"{n_cv}-fold CV accuracy",
            marker_color="steelblue",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=oob_accs,
            mode="markers",
            name="OOB accuracy",
            marker={"color": "tomato", "size": 10, "symbol": "diamond"},
        )
    )
    fig.update_layout(
        title=f"Tree depth: CV and OOB accuracy (B={n_estimators}, m=√p) — §15.3",
        xaxis_title="max_depth",
        yaxis_title="accuracy",
        yaxis={"range": [max(0.0, min(cv_accs) - 0.05), 1.0]},
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig, {
        "best_depth": labels[best_idx],
        "best_cv_accuracy": cv_accs[best_idx],
    }

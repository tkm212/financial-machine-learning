"""Shared helpers for ESL Chapter 16 (Ensemble Learning) notebooks."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score


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


def load_tmdb_regression_xy(inputs_dir: Path) -> tuple[pd.DataFrame, pd.Series, str]:
    from financial_machine_learning.esl_loaders import load_tmdb_revenue_regression

    return load_tmdb_revenue_regression(inputs_dir)


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
# Stacking & voting vs base learners (§16.2)
# ---------------------------------------------------------------------------


def ensemble_comparison_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    max_rows: int = 2000,
    n_cv: int = 5,
    random_state: int = 0,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    Compare **stacking** (level-1 meta-learner) and **majority / soft voting** with
    individual base learners (§16.2).

    **Stacking** trains level-0 models on the full data (or on CV folds internally);
    their out-of-fold predictions become features for a **meta-learner** that learns
    how to combine them — strictly more flexible than a fixed average when the
    base errors are not identical.

    **Voting** averages predicted probabilities (`soft` voting) from the same bases
    without learning combination weights.
    """
    x_arr, y_cls = _prepare_arrays(X, y, feats, max_rows=max_rows, random_state=random_state)

    rf = RandomForestClassifier(n_estimators=150, max_features="sqrt", random_state=random_state, n_jobs=1)
    gbm = GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=3, random_state=random_state)
    lr = LogisticRegression(max_iter=2000, random_state=random_state)

    bases: list[tuple[str, Any]] = [
        ("logistic", lr),
        ("random_forest", rf),
        ("gradient_boosting", gbm),
    ]

    voting_soft = VotingClassifier(estimators=bases, voting="soft", n_jobs=1)

    stack = StackingClassifier(
        estimators=bases,
        final_estimator=LogisticRegression(max_iter=2000, random_state=random_state),
        stack_method="predict_proba",
        cv=n_cv,
        n_jobs=1,
    )

    models: dict[str, Any] = {
        "Logistic regression": lr,
        "Random forest": rf,
        "Gradient boosting": gbm,
        "Soft voting (RF+GBM+LR)": voting_soft,
        "Stacking (meta=LR)": stack,
    }

    names: list[str] = []
    mean_accs: list[float] = []
    std_accs: list[float] = []

    for name, model in models.items():
        scores = cross_val_score(model, x_arr, y_cls, cv=n_cv, scoring="accuracy", n_jobs=1)
        names.append(name)
        mean_accs.append(float(scores.mean()))
        std_accs.append(float(scores.std()))

    best_idx = int(np.argmax(mean_accs))
    colors = ["tomato", "steelblue", "steelblue", "green", "purple"]

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
        title=f"Ensemble learning: stacking vs voting vs bases — §16.2 ({n_cv}-fold CV)",
        xaxis_title="method",
        yaxis_title="accuracy",
        yaxis={"range": [max(0.0, min(mean_accs) - 0.05), 1.0]},
        template="plotly_white",
    )
    return fig, {
        "best_method": names[best_idx],
        "best_cv_accuracy": mean_accs[best_idx],
        "results": dict(zip(names, mean_accs, strict=False)),
    }


def meta_learner_sweep_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    max_rows: int = 2000,
    n_cv: int = 5,
    random_state: int = 0,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    Compare **meta-learners** for stacking: ridge-like logistic vs unpenalised logistic (§16.2).

    A small amount of L2 on the meta-learner can stabilise weights when level-0 predictions
    are collinear.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import LinearSVC

    x_arr, y_cls = _prepare_arrays(X, y, feats, max_rows=max_rows, random_state=random_state)

    rf = RandomForestClassifier(n_estimators=120, max_features="sqrt", random_state=random_state, n_jobs=1)
    gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state)
    # LinearSVC + calibration for predict_proba (diverse linear base)
    lsvc = CalibratedClassifierCV(LinearSVC(random_state=random_state, dual="auto"), cv=3)

    bases: list[tuple[str, Any]] = [
        ("rf", rf),
        ("gbm", gbm),
        ("lsvc", lsvc),
    ]

    meta_options: dict[str, Any] = {
        "meta: LogisticRegression (C=1)": LogisticRegression(max_iter=2000, C=1.0, random_state=random_state),
        "meta: LogisticRegression (strong L2, C=0.3)": LogisticRegression(
            max_iter=2000, C=0.3, random_state=random_state
        ),
        "meta: LogisticRegression (weak L2, C=3)": LogisticRegression(max_iter=2000, C=3.0, random_state=random_state),
    }

    labels: list[str] = []
    means: list[float] = []
    stds: list[float] = []

    for label, meta in meta_options.items():
        stack = StackingClassifier(
            estimators=bases,
            final_estimator=meta,
            stack_method="predict_proba",
            cv=n_cv,
            n_jobs=1,
        )
        scores = cross_val_score(stack, x_arr, y_cls, cv=n_cv, scoring="accuracy", n_jobs=1)
        labels.append(label)
        means.append(float(scores.mean()))
        stds.append(float(scores.std()))

    best_idx = int(np.argmax(means))

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=means,
            error_y={"type": "data", "array": stds, "visible": True},
            marker_color="steelblue",
        )
    )
    fig.update_layout(
        title=f"Stacking: meta-learner comparison — §16.2 ({n_cv}-fold CV)",
        xaxis_title="final estimator",
        yaxis_title="accuracy",
        yaxis={"range": [max(0.0, min(means) - 0.05), 1.0]},
        template="plotly_white",
    )
    return fig, {
        "best_meta": labels[best_idx],
        "best_cv_accuracy": means[best_idx],
    }


# ---------------------------------------------------------------------------
# Diversity: correlation of out-of-fold errors (§16.1)
# ---------------------------------------------------------------------------


def base_error_correlation_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    max_rows: int = 2000,
    n_cv: int = 5,
    random_state: int = 0,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    **Diversity** among bases matters: if all models err on the same examples, averaging
    cannot help.  We compute **out-of-fold** predictions for each base learner and form
    binary **mistake indicators** $e_m(i) = \\mathbf{1}[\\hat{y}_m(i) \\neq y_i]$, then show
    the Pearson correlation matrix of $(e_1, \\ldots, e_M)$ across training points (§16.1).

    Lower off-diagonal correlation ⇒ more complementary errors ⇒ more room for stacking /
    learned combinations to beat a fixed vote.
    """
    x_arr, y_cls = _prepare_arrays(X, y, feats, max_rows=max_rows, random_state=random_state)
    y_cls = np.asarray(y_cls, dtype=int)

    bases: dict[str, Any] = {
        "logistic": LogisticRegression(max_iter=2000, random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=120, max_features="sqrt", random_state=random_state, n_jobs=1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state
        ),
    }

    cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=random_state)
    labels = list(bases.keys())
    err_cols: list[np.ndarray] = []

    for _name, est in bases.items():
        pred = cross_val_predict(est, x_arr, y_cls, cv=cv, n_jobs=1)
        err_cols.append((pred != y_cls).astype(np.float64))

    e_mat = np.column_stack(err_cols)
    corr = np.corrcoef(e_mat.T)

    fig = go.Figure(
        go.Heatmap(
            z=corr,
            x=labels,
            y=labels,
            colorscale="RdBu",
            zmid=0.0,
            zmin=-1.0,
            zmax=1.0,
            colorbar={"title": "corr."},
        )
    )
    fig.update_layout(
        title="Pairwise correlation of base learners' mistake indicators (OOF) — §16.1",
        template="plotly_white",
        xaxis_title="base learner",
        yaxis_title="base learner",
    )

    off = corr - np.diag(np.diag(corr))
    mean_off = float(np.sum(np.abs(off)) / (len(labels) * (len(labels) - 1)))

    return fig, {
        "correlation_matrix": corr.tolist(),
        "mean_abs_offdiag": mean_off,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Regression: voting vs stacking (§16.2)
# ---------------------------------------------------------------------------


def _prepare_regression_arrays(
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
        y_sub = np.asarray(y, dtype=float).ravel()[idx]
    else:
        x_sub = X[feats].values.astype(float)
        y_sub = np.asarray(y, dtype=float).ravel()

    x_sub = np.log1p(np.maximum(x_sub, 0))
    y_sub = np.log1p(np.maximum(y_sub, 0))
    return x_sub, y_sub


def regression_stacking_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    max_rows: int = 2000,
    n_cv: int = 5,
    random_state: int = 0,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    **Regression** ensembles: **voting** averages base predictions; **stacking** learns
    coefficients on out-of-fold base predictions (§16.2).  We use ridge, random forest,
    and gradient boosting as level-0 regressors and a **ridge** meta-learner (default in
    `StackingRegressor`).
    """
    x_arr, y_reg = _prepare_regression_arrays(X, y, feats, max_rows=max_rows, random_state=random_state)

    ridge = Ridge(alpha=1.0)
    rf = RandomForestRegressor(n_estimators=120, max_features=0.4, random_state=random_state, n_jobs=1)
    gbr = GradientBoostingRegressor(n_estimators=120, learning_rate=0.08, max_depth=3, random_state=random_state)

    bases: list[tuple[str, Any]] = [
        ("ridge", ridge),
        ("random_forest", rf),
        ("gradient_boosting", gbr),
    ]

    voting = VotingRegressor(estimators=bases, n_jobs=1)
    stack = StackingRegressor(
        estimators=bases,
        final_estimator=Ridge(alpha=1.0),
        cv=n_cv,
        n_jobs=1,
    )

    models: dict[str, Any] = {
        "Ridge": ridge,
        "Random forest": rf,
        "Gradient boosting": gbr,
        "Voting (avg.)": voting,
        "Stacking (meta=Ridge)": stack,
    }

    names: list[str] = []
    r2_vals: list[float] = []
    stds: list[float] = []

    kf = KFold(n_splits=n_cv, shuffle=True, random_state=random_state)
    for name, model in models.items():
        scores = cross_val_score(model, x_arr, y_reg, cv=kf, scoring="r2", n_jobs=1)
        names.append(name)
        r2_vals.append(float(scores.mean()))
        stds.append(float(scores.std()))

    best_idx = int(np.argmax(r2_vals))
    colors = ["tomato", "steelblue", "steelblue", "green", "purple"]

    fig = go.Figure(
        go.Bar(
            x=names,
            y=r2_vals,
            error_y={"type": "data", "array": stds, "visible": True},
            marker_color=colors,
        )
    )
    fig.update_layout(
        title=f"Regression ensembles: voting vs stacking — §16.2 (TMDB log revenue, {n_cv}-fold CV R²)",
        xaxis_title="method",
        yaxis_title="R² (per-fold mean)",
        template="plotly_white",
        yaxis={"range": [min(0.0, min(r2_vals) - 0.05), min(1.0, max(r2_vals) + 0.1)]},
    )
    return fig, {
        "best_method": names[best_idx],
        "best_r2": r2_vals[best_idx],
        "results": dict(zip(names, r2_vals, strict=False)),
    }

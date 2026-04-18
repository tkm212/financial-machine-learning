"""Shared helpers for ESL Chapter 18 (High-Dimensional Problems) notebooks."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


def load_tmdb_revenue_xy(inputs_dir: Path) -> tuple[pd.DataFrame, pd.Series, str]:
    from financial_machine_learning.esl_loaders import load_tmdb_revenue_regression

    return load_tmdb_revenue_regression(inputs_dir)


def simulate_sparse_regression(
    n: int,
    p: int,
    n_nonzero: int,
    *,
    snr: float = 3.0,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate $y = X\\beta + \\varepsilon$ with $n \\ll p$ and **sparse** $\\beta$ (§18.1).

    Only `n_nonzero` coefficients are non-zero; the rest are exact zeros — the regime
    where **lasso** targets subset selection while **ridge** spreads mass across all coordinates.
    """
    rng = np.random.default_rng(random_state)
    x = rng.standard_normal((n, p))
    beta = np.zeros(p)
    idx = rng.choice(p, size=min(n_nonzero, p), replace=False)
    beta[idx] = rng.standard_normal(len(idx))

    sigma_eps = float(np.linalg.norm(x @ beta) / (snr * np.sqrt(n)))
    y = x @ beta + rng.normal(0.0, sigma_eps, size=n)
    return x, y, beta


# ---------------------------------------------------------------------------
# p >> n: ridge vs lasso vs elastic net (§18.2-18.3)
# ---------------------------------------------------------------------------


def highdim_regularization_comparison_figure(
    n: int = 45,
    p: int = 180,
    n_nonzero: int = 12,
    *,
    n_cv: int = 5,
    random_state: int = 0,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    Compare **ridge**, **lasso**, and **elastic net** when $p \\gg n$ (§18.2-18.3).

    Ridge always uses all $p$ features (shrinkage toward zero but not exact zeros).
    Lasso can set many coefficients to **exactly zero**; with only $n$ samples it selects
    at most $n$ active predictors in the Gaussian linear model (under genericity).

    Each pipeline standardises $X$ so penalties apply on a common scale.
    """
    x, y, _beta_true = simulate_sparse_regression(n, p, n_nonzero, random_state=random_state)

    models: dict[str, Any] = {
        f"RidgeCV ({n_cv}-fold)": Pipeline(
            [
                ("scale", StandardScaler()),
                ("reg", RidgeCV(alphas=np.logspace(-3, 4, 60), cv=n_cv)),
            ]
        ),
        f"LassoCV ({n_cv}-fold)": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "reg",
                    LassoCV(
                        n_alphas=100,
                        cv=n_cv,
                        random_state=random_state,
                        max_iter=5000,
                    ),
                ),
            ]
        ),
        f"ElasticNetCV ({n_cv}-fold)": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "reg",
                    ElasticNetCV(
                        l1_ratio=[0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
                        cv=n_cv,
                        random_state=random_state,
                        max_iter=5000,
                    ),
                ),
            ]
        ),
    }

    names: list[str] = []
    r2_vals: list[float] = []
    nonzero_counts: list[int] = []

    for name, pipe in models.items():
        y_hat = cross_val_predict(pipe, x, y, cv=n_cv, n_jobs=1)
        r2_vals.append(float(r2_score(y, y_hat)))

        pipe.fit(x, y)
        coef = pipe.named_steps["reg"].coef_.ravel()
        names.append(name)
        nonzero_counts.append(int(np.sum(np.abs(coef) > 1e-8)))

    best_idx = int(np.argmax(r2_vals))
    colors = ["steelblue", "tomato", "green"]

    fig = go.Figure(
        go.Bar(
            x=names,
            y=r2_vals,
            marker_color=colors,
            text=[f"{v:.3f}" for v in r2_vals],
            textposition="auto",
        )
    )
    fig.update_layout(
        title=f"High-dimensional regression (n={n}, p={p}, {n_nonzero} true nonzeros) — §18.2-18.3",
        yaxis_title="cross-validated R² (predicted)",
        template="plotly_white",
        yaxis={"range": [min(0.0, min(r2_vals) - 0.05), min(1.0, max(r2_vals) + 0.1)]},
    )

    return fig, {
        "best_method": names[best_idx],
        "best_r2": r2_vals[best_idx],
        "results": dict(zip(names, r2_vals, strict=False)),
        "nonzeros": dict(zip(names, nonzero_counts, strict=False)),
        "true_nonzeros": int(n_nonzero),
    }


def lasso_path_sparsity_figure(
    n: int = 40,
    p: int = 120,
    n_nonzero: int = 8,
    *,
    random_state: int = 1,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    Plot the number of **nonzero** lasso coefficients vs penalty strength for a single fit.

    As $\\lambda$ decreases, more coefficients enter; for $p > n$, the fully saturated
    active set has size at most $n$ (§18.4).
    """
    from sklearn.linear_model import lasso_path

    x, y, _beta = simulate_sparse_regression(n, p, n_nonzero, random_state=random_state)
    xs = StandardScaler().fit_transform(x)

    alphas, coefs, _ = lasso_path(xs, y, alphas=None, max_iter=5000)
    # coefs shape (n_features, n_alphas)
    n_nonzeros = [int(np.sum(np.abs(coefs[:, j]) > 1e-8)) for j in range(coefs.shape[1])]

    fig = go.Figure(
        go.Scatter(
            x=np.log10(alphas + 1e-15),
            y=n_nonzeros,
            mode="lines+markers",
            name="|active|",
        )
    )
    fig.add_hline(y=n, line_dash="dash", line_color="gray", annotation_text="n")
    fig.update_layout(
        title=f"Lasso active set size vs log₁₀(λ) — §18.4 (n={n}, p={p})",
        xaxis_title="log₁₀(λ)",
        yaxis_title="number of nonzero coefficients",
        template="plotly_white",
    )
    return fig, {"max_active": max(n_nonzeros), "n": n}


def tmdb_with_noise_features_figure(
    inputs_dir: Path,
    *,
    n_noise: int = 200,
    max_rows: int = 600,
    n_cv: int = 5,
    random_state: int = 0,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    Augment TMDB numeric features with **pure noise** columns so $p \\gg n$ on real data.

    Signal lives only in the original features; noise dimensions let us inspect whether
    regularisation suppresses spurious coefficients.
    """
    x_df, y, _target = load_tmdb_revenue_xy(inputs_dir)
    rng = np.random.default_rng(random_state)

    if len(x_df) > max_rows:
        idx = rng.choice(len(x_df), size=max_rows, replace=False)
        x_df = x_df.iloc[idx]
        y = y.iloc[idx]

    x_num = x_df.select_dtypes(include=[np.number]).fillna(0.0)
    x_base = np.log1p(np.maximum(x_num.values.astype(float), 0.0))
    y_log = np.log1p(np.maximum(np.asarray(y, dtype=float), 0.0))

    noise = rng.standard_normal((len(y_log), n_noise))
    x_aug = np.hstack([x_base, noise])

    p_tot = x_aug.shape[1]
    n_samp = x_aug.shape[0]

    ridge = Pipeline(
        [("scale", StandardScaler()), ("reg", RidgeCV(alphas=np.logspace(-4, 5, 50), cv=n_cv))]
    )
    lasso = Pipeline(
        [
            ("scale", StandardScaler()),
            ("reg", LassoCV(n_alphas=100, cv=n_cv, random_state=random_state, max_iter=8000)),
        ]
    )

    r2_ridge = float(cross_val_score(ridge, x_aug, y_log, cv=n_cv, scoring="r2", n_jobs=1).mean())
    r2_lasso = float(cross_val_score(lasso, x_aug, y_log, cv=n_cv, scoring="r2", n_jobs=1).mean())

    ridge.fit(x_aug, y_log)
    lasso.fit(x_aug, y_log)
    coef_r = np.abs(ridge.named_steps["reg"].coef_.ravel())
    coef_l = np.abs(lasso.named_steps["reg"].coef_.ravel())

    n_signal = x_base.shape[1]
    noise_imp_r = float(coef_r[n_signal:].sum() / (coef_r.sum() + 1e-12))
    noise_imp_l = float(coef_l[n_signal:].sum() / (coef_l.sum() + 1e-12))

    fig = go.Figure(
        go.Bar(
            x=["RidgeCV", "LassoCV"],
            y=[r2_ridge, r2_lasso],
            marker_color=["steelblue", "tomato"],
            text=[f"{r2_ridge:.3f}", f"{r2_lasso:.3f}"],
            textposition="auto",
        )
    )
    fig.update_layout(
        title=f"TMDB log-revenue: p={p_tot} (incl. {n_noise} noise), n={n_samp} — §18",
        yaxis_title=f"mean {n_cv}-fold CV R²",
        template="plotly_white",
    )

    return fig, {
        "r2_ridge": r2_ridge,
        "r2_lasso": r2_lasso,
        "fraction_abs_coef_noise_ridge": noise_imp_r,
        "fraction_abs_coef_noise_lasso": noise_imp_l,
        "p": p_tot,
        "n": n_samp,
    }

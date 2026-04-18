"""Shared helpers for ESL Chapter 17 (Undirected Graphical Models) notebooks."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV
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


def load_tmdb_numeric_features(
    inputs_dir: Path,
    *,
    max_rows: int = 1000,
    random_state: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """
    Numeric TMDB columns (log1p), for a **small-$p$** Gaussian graphical model demo.

    Real movie features are not exactly multivariate normal; the plot is illustrative.
    """
    from financial_machine_learning.esl_loaders import load_tmdb_revenue_regression

    x_df, _y, _ = load_tmdb_revenue_regression(inputs_dir)
    feats = ["budget", "popularity", "runtime", "vote_average", "vote_count"]
    feats = [f for f in feats if f in x_df.columns]
    x_df = x_df[feats].select_dtypes(include=[np.number]).fillna(0.0)

    if len(x_df) > max_rows:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(x_df), size=max_rows, replace=False)
        x_df = x_df.iloc[idx]

    x_raw = np.log1p(np.maximum(x_df.values.astype(float), 0.0))
    return x_raw, feats


def sample_gaussian_precision(
    p: int,
    n: int,
    *,
    n_edges: int = 18,
    edge_scale: float = 0.35,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a **sparse precision matrix** $\\Omega \\succ 0$ by adding random symmetric
    off-diagonal edges, then sample $X \\sim \\mathcal{N}(0, \\Omega^{-1})$.

    Zeros in $\\Omega_{jk}$ encode **conditional independence** $X_j \\perp X_k \\mid X_{\\setminus \\{j,k\\}}$
    in the Gaussian graphical model (§17.3).
    """
    rng = np.random.default_rng(random_state)
    omega = np.eye(p, dtype=float)
    added = 0
    tries = 0
    while added < n_edges and tries < n_edges * 50:
        tries += 1
        i, j = int(rng.integers(0, p)), int(rng.integers(0, p))
        if i == j or omega[i, j] != 0:
            continue
        v = float(rng.normal(0.0, edge_scale))
        omega[i, j] = v
        omega[j, i] = v
        added += 1

    # Ensure positive definite
    w, _ = np.linalg.eigh(omega)
    w_min = float(w.min())
    if w_min < 0.2:
        omega += np.eye(p) * (0.3 - w_min)

    sigma = np.linalg.inv(omega)
    x = rng.multivariate_normal(np.zeros(p), sigma, size=n)
    return x, omega, sigma


def graphical_lasso_demo_figure(
    p: int = 20,
    n: int = 120,
    *,
    cv: int = 5,
    random_state: int = 0,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    **Graphical lasso** (§17.3.1) maximises the $\\ell_1$-penalised Gaussian log-likelihood:

    $$\\log\\det \\Theta - \\mathrm{tr}(S \\Theta) - \\rho \\|\\Theta\\|_1$$

    over positive definite $\\Theta$, where $S$ is the sample covariance.  Zeros in
    $\\hat{\\Theta}$ estimate missing edges in the conditional-independence graph.

    We plot the true precision $\\Omega$, the graphical-lasso estimate $\\hat{\\Theta}$,
    and the CV score as a function of $\\rho$.
    """
    x_raw, omega_true, _sigma = sample_gaussian_precision(p, n, random_state=random_state)
    x = StandardScaler(with_mean=True, with_std=True).fit_transform(x_raw)

    gl = GraphicalLassoCV(cv=cv, n_jobs=1, max_iter=500)
    gl.fit(x)

    theta_hat = np.asarray(gl.precision_, dtype=float)

    # CV curve: sklearn stores negative log-likelihood scores per alpha
    alphas = np.asarray(gl.cv_results_["alphas"], dtype=float)
    scores = np.asarray(gl.cv_results_["mean_test_score"], dtype=float)

    zmin = float(min(omega_true.min(), theta_hat.min()))
    zmax = float(max(omega_true.max(), theta_hat.max()))

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("True precision Ω", "Graphical lasso Θ̂", "CV score vs alpha"),
        horizontal_spacing=0.06,
    )
    fig.add_trace(
        go.Heatmap(z=omega_true, colorscale="RdBu", zmid=0.0, zmin=zmin, zmax=zmax, showscale=True),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(z=theta_hat, colorscale="RdBu", zmid=0.0, zmin=zmin, zmax=zmax, showscale=False),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=alphas, y=scores, mode="lines+markers", name="CV score"),
        row=1,
        col=3,
    )
    fig.add_vline(x=float(gl.alpha_), line_dash="dash", line_color="gray", row=1, col=3)

    fig.update_layout(
        title_text=f"Gaussian graphical model — graphical lasso (p={p}, n={n}) — §17.3.1",
        template="plotly_white",
        height=420,
        showlegend=False,
    )
    fig.update_xaxes(title_text="alpha (penalty)", row=1, col=3)
    fig.update_yaxes(title_text="mean CV score", row=1, col=3)

    off_true = omega_true - np.diag(np.diag(omega_true))
    off_hat = theta_hat - np.diag(np.diag(theta_hat))
    fro_err = float(np.linalg.norm(off_hat - off_true, ord="fro"))

    return fig, {
        "alpha_selected": float(gl.alpha_),
        "frobenius_offdiag_error": fro_err,
        "n_nonzero_offdiag_true": int(np.sum(np.abs(off_true) > 1e-10)),
        "n_nonzero_offdiag_hat": int(np.sum(np.abs(off_hat) > 1e-6)),
    }


def partial_correlation_from_precision(theta: np.ndarray) -> np.ndarray:
    """
    **Partial correlations** from precision (§17.3.2): for $j \\neq k$,

    $$\\rho_{jk \\mid \\setminus \\{j,k\\}} = -\\frac{\\Theta_{jk}}{\\sqrt{\\Theta_{jj} \\Theta_{kk}}}$$
    """
    d = np.sqrt(np.diag(theta))
    outer = np.outer(d, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = -theta / outer
    np.fill_diagonal(r, 0.0)
    return r


def partial_correlation_figure(
    p: int = 12,
    n: int = 200,
    *,
    random_state: int = 1,
) -> tuple[go.Figure, dict[str, Any]]:
    """Visualise partial correlations implied by the estimated precision matrix."""
    x_raw, _omega_true, _ = sample_gaussian_precision(p, n, random_state=random_state)
    x = StandardScaler().fit_transform(x_raw)
    gl = GraphicalLassoCV(cv=5, n_jobs=1, max_iter=500)
    gl.fit(x)
    prec = np.asarray(gl.precision_, dtype=float)
    pc = partial_correlation_from_precision(prec)

    fig = go.Figure(
        go.Heatmap(
            z=pc,
            colorscale="RdBu",
            zmid=0.0,
            zmin=-1.0,
            zmax=1.0,
            colorbar={"title": "partial corr."},
        )
    )
    fig.update_layout(
        title=f"Partial correlation matrix from Θ̂ — §17.3.2 (p={p}, n={n})",
        template="plotly_white",
        xaxis_title="feature",
        yaxis_title="feature",
    )
    return fig, {"alpha": float(gl.alpha_)}


# ---------------------------------------------------------------------------
# Edge stability (bootstrap) — §17.3
# ---------------------------------------------------------------------------


def graphical_lasso_stability_figure(
    p: int = 18,
    n: int = 100,
    *,
    n_bootstrap: int = 45,
    cv: int = 5,
    edge_eps: float = 1e-5,
    random_state: int = 0,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    **Stability selection** (Meinshausen & Bühlmann; see ESL discussion of structure
    uncertainty): refit graphical lasso on **bootstrap** resamples at a fixed penalty
    (chosen once by CV on the full data).  The heatmap shows the **frequency** with which
    each off-diagonal edge $| \\hat{\\Theta}_{jk} | > \\varepsilon$.
    """
    rng = np.random.default_rng(random_state)
    x_raw, _omega, _ = sample_gaussian_precision(p, n, random_state=random_state)
    x = StandardScaler().fit_transform(x_raw)

    glcv = GraphicalLassoCV(cv=cv, n_jobs=1, max_iter=500)
    glcv.fit(x)
    alpha_sel = float(glcv.alpha_)

    accum = np.zeros((p, p), dtype=float)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        gl = GraphicalLasso(alpha=alpha_sel, max_iter=1000)
        gl.fit(xb)
        theta = np.asarray(gl.precision_, dtype=float)
        off = np.abs(theta) > edge_eps
        np.fill_diagonal(off, False)
        accum += off.astype(float)

    freq = accum / float(n_bootstrap)

    fig = go.Figure(
        go.Heatmap(
            z=freq,
            colorscale="Blues",
            zmin=0.0,
            zmax=1.0,
            colorbar={"title": "freq."},
        )
    )
    fig.update_layout(
        title=(f"Edge stability (bootstrap, n={n_bootstrap}) at alpha={alpha_sel:.3f} — §17.3 (p={p})"),
        template="plotly_white",
        xaxis_title="feature",
        yaxis_title="feature",
    )
    off_freq = freq.copy()
    np.fill_diagonal(off_freq, 0.0)
    return fig, {
        "alpha_fixed": alpha_sel,
        "mean_edge_freq": float(off_freq[np.triu_indices(p, k=1)].mean()),
    }


def tmdb_precision_figure(
    inputs_dir: Path,
    *,
    max_rows: int = 900,
    cv: int = 5,
    random_state: int = 0,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    `GraphicalLassoCV` on **standardised** TMDB numeric features (small $p$).  Partial
    correlations and precision are best interpreted when $p \\ll n$ and approximate
    normality holds — here we only illustrate the machinery on real tabular data.
    """
    x_raw, feat_names = load_tmdb_numeric_features(inputs_dir, max_rows=max_rows, random_state=random_state)
    x = StandardScaler().fit_transform(x_raw)
    p = x.shape[1]

    gl = GraphicalLassoCV(cv=cv, n_jobs=1, max_iter=500)
    gl.fit(x)
    theta = np.asarray(gl.precision_, dtype=float)
    pc = partial_correlation_from_precision(theta)

    fig = go.Figure(
        go.Heatmap(
            z=pc,
            x=feat_names,
            y=feat_names,
            colorscale="RdBu",
            zmid=0.0,
            zmin=-1.0,
            zmax=1.0,
            colorbar={"title": "partial corr."},
        )
    )
    fig.update_layout(
        title=f"TMDB: partial correlations from graphical lasso (p={p}, n={len(x)}) — §17.3",
        template="plotly_white",
    )
    return fig, {"alpha": float(gl.alpha_), "n": len(x), "p": p}

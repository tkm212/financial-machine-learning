"""Shared helpers for ESL Chapter 14 (Unsupervised Learning) notebooks."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, PCA
from sklearn.metrics import silhouette_score
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


def load_tmdb_xy(inputs_dir: Path) -> tuple[pd.DataFrame, pd.Series, str]:
    from financial_machine_learning.esl_loaders import load_tmdb_revenue_regression

    return load_tmdb_revenue_regression(inputs_dir)


def _prepare_arrays(
    X: pd.DataFrame,
    feats: list[str] | None = None,
    *,
    max_rows: int = 2000,
    random_state: int = 0,
) -> np.ndarray:
    if feats is None:
        feats = ["budget", "popularity", "runtime", "vote_average", "vote_count"]
    feats = [f for f in feats if f in X.columns]

    if len(X) > max_rows:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=max_rows, replace=False)
        x_sub = X.iloc[idx][feats].values.astype(float)
    else:
        x_sub = X[feats].values.astype(float)

    x_sub = np.log1p(np.maximum(x_sub, 0))
    return x_sub


def _safe_silhouette_score(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    sample_size: int = 500,
    random_state: int = 0,
) -> float:
    """
    Wrap :func:`silhouette_score` when sklearn's preconditions hold.

    sklearn requires ``1 < n_unique_labels < n_samples`` (i.e. at least two clusters and
    not every point in its own cluster).  Otherwise return ``nan`` instead of raising.
    """
    n_samples = X.shape[0]
    n_labels = len(np.unique(labels))
    if n_samples < 3 or not (1 < n_labels < n_samples):
        return float("nan")
    # Random subsampling can yield a single cluster even when full labels are valid;
    # sklearn then raises.  Prefer full score when n is moderate.
    ss: int | None
    if n_samples <= 2500:
        ss = None
    else:
        ss = min(sample_size, n_samples)
    try:
        return float(silhouette_score(X, labels, sample_size=ss, random_state=random_state))
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# K-means: elbow and silhouette (§14.3.6)
# ---------------------------------------------------------------------------


def kmeans_elbow_figure(
    X: pd.DataFrame,
    feats: list[str] | None = None,
    *,
    k_values: list[int] | None = None,
    max_rows: int = 2000,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    K-means inertia (within-cluster sum of squares) and silhouette score vs K (§14.3.6).

    **K-means** iterates two steps until convergence (Lloyd's algorithm):

    1. **Assignment**: assign each point to the nearest centroid:
       $C(i) = \\arg\\min_k \\|x_i - m_k\\|^2$

    2. **Update**: move each centroid to the mean of its assigned points:
       $m_k = \\frac{1}{|C_k|} \\sum_{i \\in C_k} x_i$

    This minimises the **within-cluster sum of squares** (WCSS / inertia):
    $$W(C) = \\sum_{k=1}^{K} \\sum_{i \\in C_k} \\|x_i - m_k\\|^2$$

    **Elbow method**: plot WCSS vs K and look for the "elbow" — the point of
    diminishing returns where adding another cluster buys little reduction in WCSS.

    **Silhouette score** (Rousseeuw 1987): for each point $i$,
    $s_i = (b_i - a_i) / \\max(a_i, b_i)$ where $a_i$ = mean intra-cluster distance,
    $b_i$ = mean nearest-cluster distance.  Range: $[-1, 1]$; higher is better.
    The silhouette often gives a clearer cluster-count signal than the elbow.
    """
    if k_values is None:
        k_values = list(range(2, 11))

    x_arr = _prepare_arrays(X, feats, max_rows=max_rows)
    scaler = StandardScaler()
    x_arr = scaler.fit_transform(x_arr)
    n_samples = x_arr.shape[0]

    inertias: list[float] = []
    silhouettes: list[float] = []

    for k in k_values:
        k_clamped = min(k, n_samples)
        if k_clamped < 2:
            inertias.append(float("nan"))
            silhouettes.append(float("nan"))
            continue
        km = KMeans(n_clusters=k_clamped, n_init=10, random_state=0)
        labels = km.fit_predict(x_arr)
        inertias.append(float(km.inertia_))
        silhouettes.append(_safe_silhouette_score(x_arr, labels, sample_size=500, random_state=0))

    sil_arr = np.asarray(silhouettes, dtype=float)
    if np.all(np.isnan(sil_arr)):
        best_k = max(2, min(k_values[0], n_samples))
    else:
        best_k = k_values[int(np.nanargmax(sil_arr))]
    best_k = int(max(2, min(best_k, n_samples)))

    fig = make_subplots(rows=1, cols=2, subplot_titles=["WCSS (elbow method)", "Silhouette score"])
    fig.add_trace(
        go.Scatter(x=k_values, y=inertias, mode="lines+markers", name="WCSS", line={"color": "steelblue"}),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=k_values, y=silhouettes, mode="lines+markers", name="Silhouette", line={"color": "tomato"}),
        row=1,
        col=2,
    )
    fig.add_vline(x=best_k, line_dash="dash", line_color="grey", row=1, col=2, annotation_text=f"best K={best_k}")
    fig.update_layout(
        title="K-means: elbow and silhouette vs K — §14.3.6",
        template="plotly_white",
        showlegend=False,
    )
    fig.update_xaxes(title_text="K (number of clusters)")
    fig.update_yaxes(title_text="WCSS", row=1, col=1)
    fig.update_yaxes(title_text="silhouette score", row=1, col=2)

    return fig, {
        "best_K_silhouette": best_k,
        "inertias": dict(zip(k_values, inertias, strict=False)),
        "silhouettes": dict(zip(k_values, silhouettes, strict=False)),
    }


# ---------------------------------------------------------------------------
# K-means: cluster sizes and feature centroids (§14.3.6)
# ---------------------------------------------------------------------------


def kmeans_centroid_figure(
    X: pd.DataFrame,
    feats: list[str] | None = None,
    *,
    k: int = 3,
    max_rows: int = 2000,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    Visualise K-means cluster centroids as a heatmap (§14.3.6).

    After fitting with K clusters, the centroid matrix $M \\in \\mathbb{R}^{K \times p}$
    has row $k$ = the mean of all points in cluster $k$.

    Plotting $M$ as a heatmap (features on x-axis, clusters on y-axis) reveals:
    - Which features drive cluster separation.
    - Whether clusters correspond to interpretable groups.
    - The magnitude of centroid differences across clusters.

    Centroid values are shown in the **standardised** feature space so they represent
    the number of standard deviations from the global mean.
    """
    if feats is None:
        feats = ["budget", "popularity", "runtime", "vote_average", "vote_count"]
    feats = [f for f in feats if f in X.columns]

    x_arr = _prepare_arrays(X, feats, max_rows=max_rows)
    scaler = StandardScaler()
    x_arr_s = scaler.fit_transform(x_arr)

    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = km.fit_predict(x_arr_s)
    centroids = km.cluster_centers_  # shape (k, p) in standardised space
    sizes = [int((labels == c).sum()) for c in range(k)]

    fig = go.Figure(
        go.Heatmap(
            z=centroids,
            x=feats,
            y=[f"Cluster {c + 1} (n={sizes[c]})" for c in range(k)],
            colorscale="RdBu_r",
            zmid=0,
            colorbar={"title": "std. deviations"},
        )
    )
    fig.update_layout(
        title=f"K-means centroids heatmap (K={k}, standardised) — §14.3.6",
        xaxis_title="feature",
        yaxis_title="cluster",
        template="plotly_white",
    )
    return fig, {
        "k": k,
        "cluster_sizes": sizes,
        "centroids": centroids,
        "feature_names": feats,
    }


# ---------------------------------------------------------------------------
# Hierarchical clustering: linkage comparison (§14.3.12)
# ---------------------------------------------------------------------------


def hierarchical_linkage_figure(
    X: pd.DataFrame,
    feats: list[str] | None = None,
    *,
    method: str = "ward",
    max_rows: int = 300,
) -> go.Figure:
    """
    Dendrogram for hierarchical agglomerative clustering (§14.3.12).

    **Agglomerative clustering** builds a hierarchy bottom-up:
    1. Start with each point as its own cluster.
    2. At each step, merge the two closest clusters.
    3. Repeat until all points are in one cluster.

    The **linkage** function defines the distance between clusters:

    | Linkage | $d(A, B)$ | Notes |
    |---|---|---|
    | **Single** | $\\min_{i \\in A, j \\in B} d(i, j)$ | Chaining / elongated clusters |
    | **Complete** | $\\max_{i \\in A, j \\in B} d(i, j)$ | Compact, roughly equal-size |
    | **Average** | $\\frac{1}{|A||B|} \\sum_{i,j} d(i,j)$ | Compromise |
    | **Ward** | increase in WCSS from merging | Minimises within-cluster variance |

    Ward linkage tends to produce the most balanced, interpretable dendrograms and
    is the most widely used in practice.

    The **dendrogram** is plotted using scipy's dendrogram function, with leaves
    corresponding to individual data points and branch heights representing the
    distance at which clusters were merged.
    """
    if feats is None:
        feats = ["budget", "popularity", "runtime", "vote_average", "vote_count"]
    feats = [f for f in feats if f in X.columns]

    x_arr = _prepare_arrays(X, feats, max_rows=max_rows)
    scaler = StandardScaler()
    x_arr = scaler.fit_transform(x_arr)

    Z = linkage(x_arr, method=method)
    dg = dendrogram(Z, no_plot=True, truncate_mode="lastp", p=30)

    icoord = np.array(dg["icoord"])
    dcoord = np.array(dg["dcoord"])

    fig = go.Figure()
    for xs, ys in zip(icoord, dcoord):
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line={"color": "steelblue", "width": 1},
                showlegend=False,
            )
        )
    fig.update_layout(
        title=f"Hierarchical clustering dendrogram ({method} linkage) — §14.3.12",
        xaxis={"showticklabels": False, "title": "observations"},
        yaxis={"title": "merge distance"},
        template="plotly_white",
    )
    return fig


def linkage_comparison_figure(
    X: pd.DataFrame,
    feats: list[str] | None = None,
    *,
    k: int = 3,
    methods: list[str] | None = None,
    max_rows: int = 2000,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    Compare silhouette scores across linkage methods for hierarchical clustering (§14.3.12).

    We cut each dendrogram at $K$ clusters and compare the resulting cluster quality
    via silhouette score.  This makes explicit the sensitivity of the partition to the
    choice of linkage function.

    In practice:
    - **Ward** usually gives the best silhouette for compact, isotropic clusters.
    - **Single** linkage suffers from chaining — one long chain drags on silhouette.
    - **Complete** gives tighter bounds but can break truly elongated clusters.
    - **Average** is a robust middle ground.
    """
    from scipy.cluster.hierarchy import fcluster

    if methods is None:
        methods = ["ward", "complete", "average", "single"]
    if feats is None:
        feats = ["budget", "popularity", "runtime", "vote_average", "vote_count"]
    feats = [f for f in feats if f in X.columns]

    x_arr = _prepare_arrays(X, feats, max_rows=max_rows)
    scaler = StandardScaler()
    x_arr = scaler.fit_transform(x_arr)
    n_samples = x_arr.shape[0]
    # silhouette_score requires 1 < n_labels < n_samples; fcluster(..., t=1) yields one label.
    k_eff = int(max(2, min(int(k), max(2, n_samples - 1))))

    silhouettes: list[float] = []
    for method in methods:
        Z = linkage(x_arr, method=method)
        labels = fcluster(Z, t=k_eff, criterion="maxclust")
        silhouettes.append(_safe_silhouette_score(x_arr, labels, sample_size=500, random_state=0))

    sil_arr = np.asarray(silhouettes, dtype=float)
    best_idx = int(np.nanargmax(sil_arr)) if np.any(np.isfinite(sil_arr)) else 0
    colors = ["steelblue", "tomato", "green", "orange"]

    fig = go.Figure(
        go.Bar(
            x=methods,
            y=silhouettes,
            marker_color=colors[: len(methods)],
            name=f"silhouette (K={k_eff})",
        )
    )
    finite = sil_arr[np.isfinite(sil_arr)]
    y_lo = float(np.nanmin(finite)) - 0.05 if finite.size else -0.2
    y_hi = float(np.nanmax(finite)) + 0.05 if finite.size else 1.0
    fig.update_layout(
        title=f"Hierarchical linkage comparison: silhouette score (K={k_eff}) — §14.3.12",
        xaxis_title="linkage method",
        yaxis_title="silhouette score",
        yaxis={"range": [max(-1.0, y_lo), min(1.0, y_hi)]},
        template="plotly_white",
    )
    return fig, {
        "best_method": methods[best_idx],
        "best_silhouette": float(sil_arr[best_idx]) if np.isfinite(sil_arr[best_idx]) else float("nan"),
        "results": dict(zip(methods, silhouettes, strict=False)),
        "k_used": k_eff,
    }


# ---------------------------------------------------------------------------
# PCA: variance explained (§14.5)
# ---------------------------------------------------------------------------


def pca_variance_figure(
    X: pd.DataFrame,
    feats: list[str] | None = None,
    *,
    max_rows: int = 2000,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    PCA scree plot: proportion of variance explained by each principal component (§14.5).

    **Principal Component Analysis** finds the orthogonal directions of maximum variance.
    The $m$-th principal component direction $v_m$ solves:

    $$v_m = \\arg\\max_{\\|v\\|=1,\\, v \\perp v_1,\\ldots,v_{m-1}} \\text{Var}(Xv)$$

    Equivalently, PCA is the eigendecomposition of the sample covariance:
    $S = \\frac{1}{N-1} X^\\top X = V \\Lambda V^\\top$

    where $\\lambda_j = $ variance along the $j$-th principal direction.

    The **scree plot** plots the eigenvalues $\\lambda_j$ (or their cumulative fraction
    $\\sum_{k=1}^j \\lambda_k / \\sum_k \\lambda_k$) vs component index.  An "elbow"
    at component $m$ suggests that $m$ components capture the bulk of the variance.

    **Proportion of variance explained** (PVE):
    $$\\text{PVE}_m = \\frac{\\lambda_m}{\\sum_{j=1}^p \\lambda_j}$$
    """
    if feats is None:
        feats = ["budget", "popularity", "runtime", "vote_average", "vote_count"]
    feats = [f for f in feats if f in X.columns]

    x_arr = _prepare_arrays(X, feats, max_rows=max_rows)
    scaler = StandardScaler()
    x_arr = scaler.fit_transform(x_arr)

    pca = PCA()
    pca.fit(x_arr)
    pve = pca.explained_variance_ratio_
    cumulative = np.cumsum(pve)
    n_comp = len(pve)
    comp_idx = list(range(1, n_comp + 1))

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Proportion of variance explained", "Cumulative PVE"])
    fig.add_trace(
        go.Bar(x=comp_idx, y=list(pve), name="PVE", marker_color="steelblue"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=comp_idx, y=list(cumulative), mode="lines+markers", name="Cumulative PVE", line={"color": "tomato"}
        ),
        row=1,
        col=2,
    )
    fig.add_hline(y=0.90, line_dash="dash", line_color="grey", annotation_text="90%", row=1, col=2)
    fig.update_layout(
        title="PCA scree plot — §14.5",
        template="plotly_white",
        showlegend=False,
    )
    fig.update_xaxes(title_text="principal component")
    fig.update_yaxes(title_text="proportion of variance", row=1, col=1)
    fig.update_yaxes(title_text="cumulative PVE", row=1, col=2)

    n_for_90 = int(np.searchsorted(cumulative, 0.90)) + 1
    return fig, {
        "pve": list(pve),
        "cumulative_pve": list(cumulative),
        "n_components_for_90pct": n_for_90,
    }


# ---------------------------------------------------------------------------
# PCA: biplot — loadings and scores (§14.5)
# ---------------------------------------------------------------------------


def pca_biplot_figure(
    X: pd.DataFrame,
    feats: list[str] | None = None,
    *,
    pc_x: int = 1,
    pc_y: int = 2,
    max_rows: int = 500,
) -> go.Figure:
    r"""
    PCA biplot: scores (observations) and loadings (features) on the first two PCs (§14.5.1).

    A **biplot** superimposes:
    - **Scores** $z_i = X V_{1:2}$: how each observation projects onto the first two PCs.
    - **Loadings** $v_{1:2}$: how each feature contributes to those PCs.

    Observations close together in score space are similar in feature space.
    Features with large loadings in the same direction are positively correlated;
    opposite directions indicate negative correlation.  The angle between loading
    vectors approximates $\cos^{-1}(r)$ where $r$ is the Pearson correlation.
    """
    if feats is None:
        feats = ["budget", "popularity", "runtime", "vote_average", "vote_count"]
    feats = [f for f in feats if f in X.columns]

    x_arr = _prepare_arrays(X, feats, max_rows=max_rows)
    scaler = StandardScaler()
    x_arr = scaler.fit_transform(x_arr)

    pca = PCA(n_components=max(pc_x, pc_y))
    scores = pca.fit_transform(x_arr)
    loadings = pca.components_  # shape (n_components, p)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=scores[:, pc_x - 1],
            y=scores[:, pc_y - 1],
            mode="markers",
            marker={"color": "steelblue", "opacity": 0.5, "size": 5},
            name="observations (scores)",
        )
    )

    scale = float(np.max(np.abs(scores[:, :2]))) * 0.4
    for j, feat in enumerate(feats):
        lx = loadings[pc_x - 1, j] * scale
        ly = loadings[pc_y - 1, j] * scale
        fig.add_annotation(
            x=lx,
            y=ly,
            ax=0,
            ay=0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text=feat,
            showarrow=True,
            arrowhead=2,
            arrowcolor="tomato",
            font={"color": "tomato", "size": 11},
        )

    fig.update_layout(
        title=f"PCA biplot (PC{pc_x} vs PC{pc_y}) — §14.5.1",
        xaxis_title=f"PC{pc_x} ({pca.explained_variance_ratio_[pc_x - 1]:.1%} var)",
        yaxis_title=f"PC{pc_y} ({pca.explained_variance_ratio_[pc_y - 1]:.1%} var)",
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# NMF: reconstruction error vs rank (§14.6)
# ---------------------------------------------------------------------------


def nmf_rank_figure(
    X: pd.DataFrame,
    feats: list[str] | None = None,
    *,
    ranks: list[int] | None = None,
    max_rows: int = 2000,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    NMF reconstruction error vs rank r (§14.6).

    **Non-negative Matrix Factorization** (Lee & Seung 1999) factorises $X \approx WH$
    where $W \\in \\mathbb{R}_{\\geq 0}^{N \times r}$ and $H \\in \\mathbb{R}_{\\geq 0}^{r \times p}$.

    Unlike PCA, the non-negativity constraint enforces **parts-based** representations:
    each observation is a non-negative combination of $r$ non-negative basis vectors.
    This often yields more interpretable components (e.g. topics in text, facial parts).

    The Frobenius reconstruction error $\\|X - WH\\|_F$ decreases as $r$ increases.
    Choosing $r$ involves the same bias-variance trade-off as PCA's component count:
    - Small $r$: under-representation, high reconstruction error.
    - Large $r$: overfitting the non-negative constraint, components less interpretable.

    We use non-negative input features (log1p-transformed counts/budgets) to satisfy
    NMF's non-negativity requirement.
    """
    if ranks is None:
        ranks = list(range(1, 6))
    if feats is None:
        feats = ["budget", "popularity", "runtime", "vote_average", "vote_count"]
    feats = [f for f in feats if f in X.columns]

    x_arr = _prepare_arrays(X, feats, max_rows=max_rows)
    x_arr = np.maximum(x_arr, 0)

    errors: list[float] = []
    for r in ranks:
        nmf = NMF(n_components=r, max_iter=500, random_state=0)
        W = nmf.fit_transform(x_arr)
        H = nmf.components_
        err = float(np.linalg.norm(x_arr - W @ H, "fro"))
        errors.append(err)

    fig = go.Figure(
        go.Scatter(x=ranks, y=errors, mode="lines+markers", line={"color": "steelblue"}, name="Frobenius error")
    )
    fig.update_layout(
        title="NMF reconstruction error vs rank r — §14.6",
        xaxis_title="rank r",
        yaxis_title="||X - WH||_F",
        template="plotly_white",
    )
    return fig, {
        "ranks": ranks,
        "errors": errors,
    }

import marimo

__generated_with = "0.22.3"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cluster Analysis — ESL Ch. 14

    *Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning. §14.3.*

    ## Proximity and dissimilarity (§14.3.1)

    Clustering requires a notion of **dissimilarity** between observations.  Given features
    $x_1, \ldots, x_p$ for each observation, a common choice is squared Euclidean distance:

    $$d(x_i, x_{i'}) = \sum_{j=1}^p (x_{ij} - x_{i'j})^2$$

    The **dissimilarity matrix** $D \in \mathbb{R}^{N \times N}$ with $D_{ii'} = d(x_i, x_{i'})$
    is the input to most clustering algorithms.

    ## K-means clustering (§14.3.6)

    **K-means** partitions $N$ observations into $K$ clusters by minimising the
    **within-cluster point scatter** (a.k.a. inertia):

    $$W(C) = \frac{1}{2} \sum_{k=1}^K \sum_{C(i)=k} \sum_{C(i')=k} \|x_i - x_{i'}\|^2
             = \sum_{k=1}^K N_k \sum_{C(i)=k} \|x_i - \bar{x}_k\|^2$$

    This is NP-hard in general; Lloyd's algorithm finds a local minimum via two
    alternating steps:
    1. **Assign** each point to the nearest centroid.
    2. **Update** each centroid to the mean of its assigned points.

    ## Hierarchical clustering (§14.3.12)

    **Agglomerative** clustering starts with $N$ singleton clusters and successively
    merges the closest pair.  The **linkage function** defines inter-cluster distance:

    | Linkage | Formula | Tendency |
    |---|---|---|
    | Single | $\min_{i,i'} d(x_i, x_{i'})$ | Chaining |
    | Complete | $\max_{i,i'} d(x_i, x_{i'})$ | Compact |
    | Average | $\frac{1}{N_k N_{k'}} \sum_{i,i'} d(x_i, x_{i'})$ | Balanced |
    | Ward | $\Delta W$ from merging | Minimises WCSS |

    The result is a **dendrogram** — a binary tree showing the merge history.
    Any partition into $K$ clusters is obtained by cutting the dendrogram at a
    horizontal threshold.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    import ch14_helpers as helpers

    _root, INPUTS, _outputs = helpers.init_paths()
    return INPUTS, helpers


@app.cell
def _(INPUTS, helpers):
    X, y, target = helpers.load_tmdb_xy(INPUTS)
    print(f"Loaded {len(X):,} rows | target: {target!r}")
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## K-means: elbow method and silhouette score (§14.3.6)

    Two standard criteria for selecting the number of clusters $K$:

    **Elbow method**: plot WCSS vs $K$.  The "elbow" is the last $K$ that gives a large
    marginal reduction in WCSS; beyond it, adding clusters yields diminishing returns.
    The elbow can be ambiguous — use together with silhouette.

    **Silhouette score** (Rousseeuw 1987): for each observation $i$ in cluster $A$, let
    $a_i$ = mean distance within $A$ and $b_i$ = mean distance to the nearest other cluster:

    $$s_i = \frac{b_i - a_i}{\max(a_i, b_i)} \in [-1, 1]$$

    The **mean silhouette** across all $i$ measures global cluster cohesion and separation.
    Values near 1 indicate well-separated clusters; near 0 suggests overlapping clusters.
    """)
    return


@app.cell
def _(X, helpers):
    fig_elbow, elbow_summary = helpers.kmeans_elbow_figure(X, k_values=list(range(2, 10)))
    fig_elbow.show()
    print(f"Best K by silhouette: {elbow_summary['best_K_silhouette']}")
    for k_val, sil in elbow_summary["silhouettes"].items():
        print(f"  K={k_val}: silhouette={sil:.4f}, WCSS={elbow_summary['inertias'][k_val]:.1f}")
    return elbow_summary


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## K-means cluster centroids (§14.3.6)

    With the chosen $K$, we inspect the **centroid matrix** — the mean of each feature
    within each cluster, shown in standardised units.

    - Positive values (red) indicate the cluster has above-average values for that feature.
    - Negative values (blue) indicate below-average.
    - Clusters with distinct centroid patterns are well-separated; similar rows suggest
      those clusters may not be meaningfully different.

    This heatmap provides an interpretable summary of *what each cluster represents*.
    """)
    return


@app.cell
def _(X, elbow_summary, helpers):
    best_k = elbow_summary["best_K_silhouette"]
    fig_centroids, centroid_info = helpers.kmeans_centroid_figure(X, k=best_k)
    fig_centroids.show()
    print(f"K={best_k} cluster sizes: {centroid_info['cluster_sizes']}")
    return best_k


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hierarchical clustering: dendrogram (§14.3.12)

    The **dendrogram** displays the complete merge history.  Each leaf is an observation;
    each internal node represents a merge, with the branch height equal to the dissimilarity
    at which the merge occurred.

    Reading the dendrogram:
    - **Cut height** determines the number of clusters: cut at height $h$ to get clusters
      separated by at least $h$.
    - **Tall branches** indicate well-separated clusters that merged late.
    - **Short branches** near leaves indicate noisy or outlier points.

    We use Ward linkage, which tends to produce compact, balanced dendrograms.
    """)
    return


@app.cell
def _(X, helpers):
    fig_dend = helpers.hierarchical_linkage_figure(X, method="ward", max_rows=200)
    fig_dend.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Linkage method comparison (§14.3.12)

    We cut each dendrogram at $K = 3$ clusters and compare the resulting partitions
    by silhouette score.

    - **Ward** tends to win on compact, isotropic data.
    - **Single linkage** is susceptible to chaining: one long, thin cluster drags down
      the silhouette.
    - **Complete** and **average** are robust alternatives when Ward is too aggressive
      in equalising cluster sizes.

    In practice, run all linkages and use domain knowledge together with quantitative
    criteria to choose the final partition.
    """)
    return


@app.cell
def _(X, best_k, helpers):
    fig_link, link_summary = helpers.linkage_comparison_figure(X, k=best_k)
    fig_link.show()
    print(f"Best linkage: {link_summary['best_method']} | silhouette: {link_summary['best_silhouette']:.4f}")
    for method, sil in link_summary["results"].items():
        print(f"  {method:<10}: {sil:.4f}")
    return


if __name__ == "__main__":
    app.run()

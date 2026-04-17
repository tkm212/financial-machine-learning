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
    # Prototype Methods — ESL Ch. 13

    *Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning. §13.2.*

    ## Prototypes and the nearest-centroid rule (§13.2)

    A **prototype** is a representative point in feature space associated with a class.
    Classification is by **nearest prototype** under some distance metric:

    $$C(x) = \underset{g}{\arg\min} \; \min_{r=1}^{R} \|x - m_{g,r}\|$$

    where $m_{g,r}$ is the $r$-th prototype for class $g$.

    The simplest case $R = 1$ per class gives the **nearest-centroid** (Rocchio) classifier:
    each class is represented by its mean $\bar{x}_g$, and a new point is assigned to the
    class with the closest centroid.

    ## K-means prototypes (§13.2.1)

    For $R > 1$, prototypes are found by running **K-means** separately on each class:

    1. For each class $g$, collect $\{x_i : y_i = g\}$.
    2. Run K-means with $K = R$ to find $R$ centroids $\{m_{g,1}, \ldots, m_{g,R}\}$.
    3. Classify by nearest prototype across all classes.

    As $R$ increases the prototype set becomes richer, approaching the full training set
    in the limit $R = N_g$.  K-means minimises **within-class inertia**, which approximates
    (but does not directly minimise) classification error — see LVQ (§13.2.2) for that.

    ## Learning Vector Quantization (§13.2.2)

    **LVQ** (Kohonen 1989) modifies prototypes to directly reduce classification error.
    Given a training point $x$ and its closest prototype $m^*$:

    - **Correct** classification ($y = C(m^*)$): attract prototype toward $x$:
      $m^* \leftarrow m^* + \varepsilon (x - m^*)$
    - **Wrong** classification ($y \neq C(m^*)$): repel prototype away from $x$:
      $m^* \leftarrow m^* - \varepsilon (x - m^*)$

    The learning rate $\varepsilon$ is typically decreased over iterations.
    LVQ often outperforms K-means prototypes because it directly optimises the decision
    boundary rather than within-class variance.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    import ch13_helpers as helpers

    _root, INPUTS, _outputs = helpers.init_paths()
    return INPUTS, helpers


@app.cell
def _(INPUTS, helpers):
    X, y, target = helpers.load_tmdb_classification_xy(INPUTS)
    print(f"Loaded {len(X):,} rows | target: {target!r} | class balance: {y.mean():.2%} positive")
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## K-means prototypes: accuracy vs R (§13.2.1)

    We sweep the number of prototypes per class $R \in \{1, 2, 3, 5, 8, 10\}$ and measure
    5-fold cross-validated accuracy of the nearest-prototype classifier.

    - $R = 1$: nearest-centroid (Rocchio) — fast and interpretable, but a single mean
      may poorly represent a multimodal class distribution.
    - $R > 1$: K-means subdivides each class into $R$ clusters, capturing multiple modes.
    - Diminishing returns as $R$ grows: beyond a class's natural cluster structure,
      additional prototypes don't improve generalisation.

    The flat or asymptoting curve tells us where the class distributions stop being
    well-described by more centroids.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_proto, proto_summary = helpers.kmeans_prototype_figure(
        X, y, R_values=[1, 2, 3, 5, 8, 10]
    )
    fig_proto.show()
    print(
        f"Best R: {proto_summary['best_R']} | "
        f"CV accuracy: {proto_summary['best_cv_accuracy']:.3%} | "
        f"R=1 (nearest centroid): {proto_summary['R1_accuracy']:.3%}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prototypes vs K-nearest-neighbors (§13.2 vs §13.3)

    Prototype methods offer a **compressed** representation of the training set: instead
    of storing all $N$ training points (as KNN does), we store only $K \cdot R$ prototypes.

    Trade-offs:
    - **Memory**: prototypes scale as $O(K \cdot R)$ vs $O(N)$ for KNN.
    - **Prediction speed**: nearest-prototype search is $O(K \cdot R \cdot p)$ vs $O(N \cdot p)$.
    - **Accuracy**: KNN with a good $k$ often beats a small prototype set, but the gap narrows
      as $R$ increases.

    The plot below shows both methods on the same CV protocol so the accuracy-complexity
    trade-off is directly comparable.
    """)
    return


@app.cell
def _(X, helpers, y):
    fig_cmp, cmp_summary = helpers.lvq_vs_knn_figure(
        X, y, k_values=[1, 3, 5, 10, 20], R_values=[1, 2, 3, 5, 8]
    )
    fig_cmp.show()
    print(
        f"Best KNN: k={cmp_summary['best_knn_k']} accuracy={cmp_summary['best_knn']:.3%} | "
        f"Best prototypes: R={cmp_summary['best_proto_R']} accuracy={cmp_summary['best_proto']:.3%}"
    )
    return


if __name__ == "__main__":
    app.run()

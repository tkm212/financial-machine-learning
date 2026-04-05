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
    # Model Inference: EM Algorithm — ESL Ch. 8

    *Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning. §8.5.*

    The **Expectation-Maximisation (EM) algorithm** is a general approach for maximum likelihood
    estimation when the data have latent (unobserved) structure.  For a Gaussian mixture model:

    $$p(x \mid \theta) = \sum_{k=1}^K \pi_k \, \mathcal{N}(x \mid \mu_k, \sigma_k^2)$$

    the complete-data log-likelihood is intractable because we do not observe which component
    generated each observation.  EM iterates between two steps:

    **E-step** — compute responsibilities (soft assignments):
    $$r_{ik} = \frac{\pi_k \, \mathcal{N}(x_i \mid \mu_k, \sigma_k^2)}{\sum_{j=1}^K \pi_j \, \mathcal{N}(x_i \mid \mu_j, \sigma_j^2)}$$

    **M-step** — update parameters to maximise the expected complete-data log-likelihood:
    $$\mu_k = \frac{\sum_i r_{ik} x_i}{\sum_i r_{ik}}, \qquad
      \sigma_k^2 = \frac{\sum_i r_{ik}(x_i - \mu_k)^2}{\sum_i r_{ik}}, \qquad
      \pi_k = \frac{\sum_i r_{ik}}{N}$$

    EM is guaranteed to increase the observed-data log-likelihood at every iteration, but
    converges only to a local maximum — the result depends on initialisation.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    import ch8_helpers as helpers

    _root, INPUTS, _outputs = helpers.init_paths()
    return INPUTS, helpers


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fitted mixture components (§8.5)

    We generate 500 samples from a known two-component mixture
    $0.4 \cdot \mathcal{N}(-2, 1^2) + 0.6 \cdot \mathcal{N}(2, 1.5^2)$
    and run EM to recover the components.

    The **fitted mixture** (black) closely tracks the **true density** (grey dashed),
    and the individual component curves reveal the inferred means, standard deviations
    and mixing weights.  Note that EM can recover the true parameters even when the
    components overlap substantially.
    """)
    return


@app.cell
def _(helpers):
    fig_em = helpers.em_1d_figure(n_samples=500, K=2, random_state=0)
    fig_em.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Log-likelihood convergence and local maxima (§8.5)

    EM monotonically increases the observed-data log-likelihood
    $\ell(\theta) = \sum_i \log \sum_k \pi_k \mathcal{N}(x_i \mid \mu_k, \sigma_k^2)$
    at each iteration — it is guaranteed to never decrease.

    Running EM from **multiple random initialisations** demonstrates two key properties:
    - Different starts often converge to the **same local maximum** (good news for this simple example).
    - The number of iterations to convergence varies widely — some restarts need only a handful
      of steps while others wander slowly through a flat region before snapping into place.

    For mixture models with more components or in higher dimensions, local maxima become
    a genuine concern; random restarts or deterministic initialisation strategies (e.g. k-means++)
    are used to mitigate this.
    """)
    return


@app.cell
def _(helpers):
    fig_conv = helpers.em_convergence_figure(n_samples=500, K=2, n_restarts=5, random_state=0)
    fig_conv.show()
    return


if __name__ == "__main__":
    app.run()

# Notebooks

All notebooks live under the `notebooks/` directory and are organised into two tracks.

---

## Advances in Financial Machine Learning

Implementations based on **López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.**

| Chapter | Topic | Files |
|---------|-------|-------|
| 2 | Financial Data Structures | `information_bars.py`, `cusum_pca_weights.py` |
| 3 | Labeling | `labeling.py` |
| 4 | Sample Weights | `sample_weights.py` |

Source path: `notebooks/financial-machine-learning/`

### Chapter 2 — Financial Data Structures

Compares time, tick, volume, and dollar bars on real order-book data.
Dollar bars are shown to produce returns that are more stationary and closer
to IID than fixed-time sampling.

The CUSUM filter is applied to the close price of dollar bars to select event
times for labeling, demonstrating how it avoids the clustering issue of
Bollinger-band–style triggers.

### Chapter 3 — Labeling

The triple-barrier method is applied to the events identified by the CUSUM filter.
Each event is labeled +1 (profit take hit first), -1 (stop loss hit first),
or 0 (vertical barrier — maximum holding period elapsed).

### Chapter 4 — Sample Weights

Overlapping triple-barrier labels break the IID assumption. This notebook
computes:

- **Concurrent label count** per bar \( c(t) \)
- **Average uniqueness** \( \bar{u}_i = \frac{1}{T_i} \sum_{t \in [t_{i,0}, t_{i,1}]} \frac{1}{c(t)} \)
- **Time-decay weights** \( w_i = e^{-\text{age}_i / \tau} \)

These weights are passed to `sample_weight` in scikit-learn estimators to
down-weight stale and redundant observations.

---

## Elements of Statistical Learning

Implementations based on **Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning (2nd ed.). Springer.**

Source path: `notebooks/elements-of-statistical-learning/`

| Chapter | Topic | Files |
|---------|-------|-------|
| 2 | Supervised Learning | `least_squares_regression.py`, `k_nearest_neighbors.py` |
| 3 | Linear Methods | `subset_selection.py`, `ridge_regression.py`, `lasso.py`, `pcr_pls.py` |
| 4 | Linear Methods for Classification | `lda.py`, `logistic_regression.py`, `separating_hyperplanes.py` |
| 5 | Basis Expansions | `splines.py`, `smoothing_splines.py` |
| 6 | Kernel Smoothing | `kernel_smoothers.py`, `kernel_density.py` |
| 7 | Model Assessment | `bias_variance.py`, `cross_validation.py` |
| 8 | Model Inference | `em_algorithm.py`, `bagging.py` |
| 9 | Additive Models & Trees | `additive_models.py`, `decision_trees.py` |
| 10 | Boosting | `boosting.py`, `gradient_boosting.py` |
| 13 | Prototype Methods | `nearest_neighbors.py`, `prototype_methods.py` |
| 14 | Unsupervised Learning | `principal_components.py`, `clustering.py` |
| 15 | Random Forests | `random_forests.py` |
| 16 | Ensemble Learning | `ensemble_learning.py` |
| 17 | Undirected Graphical Models | `graphical_models.py` |
| 18 | High-Dimensional Problems | `high_dimensional.py` |

Each chapter folder also contains a `ch{N}_helpers.py` module with dataset
loading and plotting utilities shared across that chapter's notebooks.

### Running a notebook

All notebooks use `uv` for dependency management. From the repo root:

```bash
uv run python notebooks/elements-of-statistical-learning/10-Boosting/boosting.py
```

External datasets (ATP/WTA tennis, TMDB movies) must be downloaded first:

```bash
uv run python scripts/download_atpwta_tennis_data.py
uv run python scripts/download_tmdb_movie_metadata.py
```

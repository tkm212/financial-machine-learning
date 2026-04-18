# financial-machine-learning

[![Release](https://img.shields.io/github/v/release/tkm212bc/financial-machine-learning)](https://github.com/tkm212bc/financial-machine-learning/releases)
[![Build status](https://img.shields.io/github/actions/workflow/status/tkm212bc/financial-machine-learning/main.yml?branch=main)](https://github.com/tkm212bc/financial-machine-learning/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/tkm212bc/financial-machine-learning/branch/main/graph/badge.svg)](https://codecov.io/gh/tkm212bc/financial-machine-learning)
[![Commit activity](https://img.shields.io/github/commit-activity/m/tkm212bc/financial-machine-learning)](https://github.com/tkm212bc/financial-machine-learning/commits/main)
[![License](https://img.shields.io/github/license/tkm212bc/financial-machine-learning)](https://github.com/tkm212bc/financial-machine-learning/blob/main/LICENSE)

Python library and [Marimo](https://marimo.io) notebooks covering two bodies of work:

- **[Advances in Financial Machine Learning](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)** — López de Prado (2018): alternative bar types, CUSUM filtering, triple-barrier labeling, and sample weighting for financial time series.
- **[Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/)** — Hastie, Tibshirani & Friedman (2nd ed.): supervised learning through random forests, implemented across Chapters 2–15.

**[Documentation](https://tkm212bc.github.io/financial-machine-learning/) · [API Reference](https://tkm212bc.github.io/financial-machine-learning/modules/) · [Notebooks](https://tkm212bc.github.io/financial-machine-learning/notebooks/)**

---

## Installation

```bash
uv add financial-machine-learning
```

```bash
pip install financial-machine-learning
```

---

## Library

The `financial_machine_learning` package implements the core pipeline from AFML:

| Module | What it does |
|--------|-------------|
| `bars` | Build time, tick, volume, and dollar bars from raw tick data |
| `filters` | Symmetric CUSUM filter for event-driven sampling (Snippet 2.4) |
| `labeling` | Triple-barrier labeling: profit-take, stop-loss, and vertical barriers |
| `weights` | Concurrent label counts, average uniqueness, and time-decay sample weights |

```python
from financial_machine_learning.bars import dollar_bars
from financial_machine_learning.filters import cusum_filter
from financial_machine_learning.labeling import triple_barrier_labels

bars   = dollar_bars(ticks_df, threshold=1_000_000)
events = cusum_filter(bars["close"], threshold=0.02)
labels = triple_barrier_labels(bars, events, pt=0.02, sl=0.02, num_bars=20)
```

---

## Notebooks

Interactive [Marimo](https://marimo.io) notebooks, runnable locally with `uv`:

```bash
uv run marimo run notebooks/financial-machine-learning/2-Financial_Data_Structures/information_bars.py
```

**Advances in Financial Machine Learning**

| Chapter | Topic |
|---------|-------|
| 2 | Financial Data Structures — bar types, CUSUM filter, PCA weights |
| 3 | Labeling — triple-barrier method |
| 4 | Sample Weights — concurrency, uniqueness, time decay |

**Elements of Statistical Learning**

| Chapters | Topics |
|----------|--------|
| 2–3 | Supervised learning, linear methods |
| 4 | Linear classification (LDA, logistic regression, SVMs) |
| 5–6 | Basis expansions, kernel smoothing |
| 7–8 | Model assessment, bootstrap, bagging |
| 9–10 | Additive models, decision trees, boosting |
| 11–12 | Neural networks, SVMs, flexible discriminants |
| 13 | Prototype methods (K-means, LVQ) and K-nearest-neighbors |
| 14 | Unsupervised learning — clustering, PCA, NMF |
| 15 | Random forests — OOB error, variable importance, tuning |

---

## Development

```bash
git clone https://github.com/tkm212bc/financial-machine-learning.git
cd financial-machine-learning
make install          # create venv + install pre-commit hooks
make check            # lint, type check, dependency audit
make test             # pytest with coverage (requires ≥ 80%)
make docs             # serve docs locally
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

---

## Releasing

1. Bump `version` in `pyproject.toml` and add an entry to `CHANGELOG.md`
2. Commit and push to `main`
3. Create a GitHub release — the `release-main` workflow will automatically deploy the updated docs to GitHub Pages

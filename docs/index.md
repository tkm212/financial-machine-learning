# Financial Machine Learning

[![Release](https://img.shields.io/github/v/release/tkm212bc/financial-machine-learning)](https://github.com/tkm212bc/financial-machine-learning/releases)
[![Build status](https://img.shields.io/github/actions/workflow/status/tkm212bc/financial-machine-learning/main.yml?branch=main)](https://github.com/tkm212bc/financial-machine-learning/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/tkm212bc/financial-machine-learning)](https://github.com/tkm212bc/financial-machine-learning/commits/main)
[![License](https://img.shields.io/github/license/tkm212bc/financial-machine-learning)](https://github.com/tkm212bc/financial-machine-learning/blob/main/LICENSE)

Notebooks and a reusable Python library covering two bodies of work:

- **Advances in Financial Machine Learning** — López de Prado (2018): alternative data structures, event filtering, triple-barrier labeling, and sample weighting for financial time series.
- **Elements of Statistical Learning** — Hastie, Tibshirani & Friedman: supervised learning, linear methods, basis expansions, kernel smoothing, model assessment, additive models, and boosting.

---

## Installation

```bash
uv add financial-machine-learning
```

Or with pip:

```bash
pip install financial-machine-learning
```

To work from source with all development dependencies:

```bash
git clone https://github.com/tkm212bc/financial-machine-learning.git
cd financial-machine-learning
make install
```

---

## Library overview

The `financial_machine_learning` package contains four modules:

| Module | Description |
|--------|-------------|
| `bars` | Alternative bar types: time, tick, volume, and dollar bars |
| `filters` | CUSUM filter for sampling event-driven time series |
| `labeling` | Triple-barrier method for labeling financial observations |
| `weights` | Concurrency, average uniqueness, and time-decay sample weights |

A fifth module, `esl_loaders`, provides dataset loaders used by the ESL notebooks (ATP/WTA tennis and TMDB movie data).

---

## Quick example

```python
from financial_machine_learning.bars import dollar_bars
from financial_machine_learning.filters import cusum_filter
from financial_machine_learning.labeling import triple_barrier_labels

# Build dollar bars from raw tick data
bars = dollar_bars(ticks_df, threshold=1_000_000)

# Identify event times with CUSUM filter
events = cusum_filter(bars["close"], threshold=0.02)

# Label each event with the triple-barrier method
labels = triple_barrier_labels(bars, events, pt=0.02, sl=0.02, num_bars=20)
```

---

## Links

- [GitHub repository](https://github.com/tkm212bc/financial-machine-learning)
- [PyPI package](https://pypi.org/project/financial-machine-learning)
- [API reference](modules.md)
- [Notebooks overview](notebooks.md)

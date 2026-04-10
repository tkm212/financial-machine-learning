"""Financial machine learning utilities.

Public API: bars, filters, labeling, and sample weights from López de Prado (2018).

``esl_loaders`` is intentionally not re-exported here — it requires external
datasets (downloaded via Kaggle) and is only used by the ESL notebooks.
Import it directly: ``from financial_machine_learning.esl_loaders import ...``
"""

from financial_machine_learning.bars import dollar_bars, tick_bars, time_bars, volume_bars
from financial_machine_learning.filters import cusum_filter
from financial_machine_learning.labeling import triple_barrier_labels
from financial_machine_learning.weights import (
    average_uniqueness,
    concurrent_labels_per_bar,
    time_decay_weights,
)

__all__ = [
    "average_uniqueness",
    "concurrent_labels_per_bar",
    "cusum_filter",
    "dollar_bars",
    "tick_bars",
    "time_bars",
    "time_decay_weights",
    "triple_barrier_labels",
    "volume_bars",
]

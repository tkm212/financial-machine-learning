# API Reference

## Financial Machine Learning

### Bars

Alternative sampling methods for financial data (López de Prado, Ch. 2).
Dollar and volume bars sample by economic information flow rather than clock time,
yielding returns that are closer to IID than time-sampled data.

::: financial_machine_learning.bars

---

### Filters

Event filters used to down-sample financial time series before labeling (López de Prado, Ch. 2).

::: financial_machine_learning.filters

---

### Labeling

Triple-barrier labeling assigns a direction (+1, -1, 0) to each sampled event
based on which of three barriers — profit take, stop loss, or vertical — is hit first (López de Prado, Ch. 3).

::: financial_machine_learning.labeling

---

### Weights

Sample weights that account for the overlap between labels, correcting the IID
assumption broken by the triple-barrier method (López de Prado, Ch. 4).

::: financial_machine_learning.weights

---

## ESL Loaders

Dataset loaders for the Elements of Statistical Learning notebooks.
Provides cleaned `(X, y, target_name)` tuples ready for use with scikit-learn estimators.

::: financial_machine_learning.esl_loaders

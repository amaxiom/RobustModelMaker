# RobustModelMaker

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.3.1-green.svg)](CHANGELOG.md)

**A reproducible model-building pipeline for small-to-medium scientific datasets.**

RobustModelMaker (ROBUST) combines bootstrap stability selection with leakage-safe nested cross-validation to identify a stable, minimal feature subset and produce honest performance estimates. It is designed for scientific datasets where reproducibility, interpretability, and honest generalisation estimates matter as much as raw predictive performance.

---

## Why RobustModelMaker?

Standard machine learning pipelines applied to scientific data suffer from two problems that ROBUST addresses directly:

**Optimistic performance estimates.** When feature selection, hyperparameter tuning, and model evaluation share the same data, the reported score reflects the data used for model building, not future data. ROBUST uses a strict nested cross-validation design in which each of those steps is performed entirely on the training partition of each fold. The test partition is used only to evaluate the final fold model, never to inform any modelling decision.

**Unstable feature selection.** Single-run feature selection methods produce a feature set that can change substantially with small changes in the data. ROBUST runs bootstrap stability selection: features are ranked by how consistently they are selected across hundreds of random subsamples of the training data. Only features that exceed a stability threshold (selected in at least 70% of bootstrap runs by default) are retained.

The result is a model built on a smaller, more reproducible feature set whose estimated performance is trustworthy.

---

## Key capabilities

| Capability | Detail |
|---|---|
| Task types | Binary classification, multiclass classification, regression |
| Algorithms | 9 built-in: `eln`, `rdg`, `las`, `log`, `svm`, `rf`, `xgb`, `mlp`, `lin` |
| Feature selection | Bootstrap stability selection with configurable threshold and bootstrap count |
| Performance estimation | Nested CV (outer + inner), repeated nested CV, grouped CV |
| Preprocessing | Median imputation + optional standard scaling, fitted inside each fold |
| Missing data | NaN-tolerant by default; optional data-driven missingness filter |
| Cutoff determination | Bootstrap specificity-targeted threshold for binary classification |
| Probability calibration | Platt scaling (sigmoid) or isotonic regression |
| Post-hoc analysis | Permutation importance, SHAP-ready export, feature stability plots |
| External validation | One-call evaluation on a held-out set with full metric suite |
| Reproducibility | Fully deterministic given a fixed random seed, verified by test suite |
| Save/load | JSON metadata, CSV tables, and pickle of the fitted result |

---

## Installation

ROBUST is a single-file library with no build step. Copy `RobustModelMaker.py` into your project and import it:

```python
import sys
sys.path.append("/path/to/RobustModelMaker")
from RobustModelMaker import RobustModelMaker
```

**Required:** Python >= 3.9, numpy, pandas, scikit-learn, scipy

**Optional:** xgboost (for `alg="xgb"`)

---

## Quick start

```python
import pandas as pd
from RobustModelMaker import RobustModelMaker

X = pd.read_csv("features.csv")
y = pd.read_csv("labels.csv").squeeze()

maker = RobustModelMaker(
    alg="eln",           # elastic net: interpretable and fast
    task_type="binary",  # "binary", "multiclass", or "regression"
    outer_cv=5,
    inner_cv=5,
    n_bootstrap=100,
    stability_threshold=0.7,
    random_state=42,
).fit(X, y)

result = maker.result_
print(f"Selected {len(result.selected_features)} of {len(result.feature_names)} features")
print(f"Nested CV AUC: {result.nested_cv_result.mean_score:.4f} "
      f"+/- {result.nested_cv_result.std_score:.4f}")

# Predict on new data (preprocessing and feature selection applied automatically)
predictions = maker.predict(X_new)
probabilities = maker.predict_proba(X_new)
```

The functional API is also available:

```python
from RobustModelMaker import run_pipeline
result = run_pipeline(X, y, alg="eln", task_type="binary",
                      outer_cv=5, inner_cv=5, random_state=42)
```

---

## Algorithms

| Code | Model | Task types | Notes |
|---|---|---|---|
| `eln` | Elastic net | all | Fastest; coefficient-based importance; auto-scales features |
| `rdg` | Ridge regression / L2 logistic | all | Stable; good default for many scientific datasets |
| `las` | Lasso / L1 logistic | all | Sparse coefficients; strong feature selector |
| `log` | L2 logistic regression | classification | Reliable baseline for binary and multiclass |
| `svm` | Linear SVM | all | Effective in high-dimensional spaces |
| `rf` | Random forest | all | Non-linear; no scaling needed; class_weight balanced |
| `xgb` | XGBoost | all | Highest raw performance; requires xgboost package |
| `mlp` | Multi-layer perceptron | all | Neural baseline; slower on small datasets |
| `lin` | Linear regression (OLS) | regression only | Interpretable; no regularisation |

---

## Selected results from the benchmark suite

Three real scientific datasets are used to evaluate ROBUST against a full-feature nested-CV baseline using the same algorithm and fold structure. All three benchmarks use Random Forest (`rf`) for both ROBUST and the baseline, isolating the effect of bootstrap stability selection from any algorithm differences. The benchmark uses BenchMake archetypal splits to ensure train and test sets are representative rather than randomly sampled.

| Dataset | Task | n_train x p | ROBUST feats | Reduction | BL score | ROBUST score | p | Outcome |
|---|---|---|---|---|---|---|---|---|
| SECOM Manufacturing | binary | 1253 x 590 | 301 | 49.0% | 0.6814 AUC | 0.6835 AUC | 0.770 | preserved |
| Urban Land Cover | multiclass | 540 x 147 | 66 | 55.1% | 0.9827 AUC | 0.9849 AUC | 0.432 | preserved |
| Graphene Oxide Bulk | regression | 1293 x 309 | 150 | 51.5% | 0.0266 RMSE | 0.0343 RMSE | 0.193 | preserved |

Classification metrics are AUC-ROC (binary) and weighted OVR AUC (multiclass), higher is better. Regression metric is RMSE in eV, lower is better. The p-value column is from the paired Wilcoxon signed-rank test on per-fold scores. Across all three tasks ROBUST roughly halves the feature count with no statistically significant change in performance, yielding score-per-feature efficiency gains of 1.97x (SECOM), 2.23x (Urban Land Cover), and 2.66x (Graphene Oxide).

**Benchmark configuration:** `outer_cv=10`, `inner_cv=5`, `n_bootstrap=25`, `stability_threshold=0.6`, `n_iter=10`, `random_state=42`. These differ from the production defaults (`n_bootstrap=100`, `stability_threshold=0.7`, `n_iter=100`) because the full benchmark runs ~4 hours of wall-clock time as configured; production defaults would multiply that several-fold.

**Outcome key:** `preserved` is the primary success criterion: the stability-selected feature subset achieves statistically equivalent performance to the full-feature baseline (paired Wilcoxon, p >= 0.05) while using a fraction of the features. The selected features are robust across bootstrap resamples of the training data, not optimal for any single model fit; a small non-significant performance difference from the baseline is the expected and intended outcome. The other two outcomes the benchmark can return are `sig. better *` (unexpected improvement) and `sig. worse *` (significant loss).

**Regression scores** are reported as RMSE (lower is better). Internally, ROBUST stores negative RMSE following sklearn convention so that all metrics can be maximised; the benchmark console report and README table always display positive RMSE for readability.

**Note on split methodology:** All benchmarks use [BenchMake](https://github.com/amaxiom/benchmake) archetypal splits, which are adversarial by design. BenchMake selects maximally representative train/test partitions that keep the two sets apart in feature space, producing more conservative (lower) scores than conventional random splits would on the same datasets. This is intentional: the benchmark is a worst-case assessment. Scores you observe when running ROBUST on your own data with the default random splits will typically be higher. The ROBUST vs. full-feature baseline comparison within each benchmark is internally consistent because both models use the same split.

Exact scores depend on the random seed and runtime environment. Run `python benchmarks/benchmark_suite.py` or open `benchmarks/Benchmark_Suite.ipynb` for a full console report, including a 25+ test statistical battery for each dataset comparing ROBUST and baseline per-fold scores.

---

## Repository structure

```
RobustModelMaker/
├── RobustModelMaker.py              Single-file library (all you need to use ROBUST)
├── requirements.txt                 Minimum dependency versions
├── LICENSE                          MIT
├── CHANGELOG.md                     Version history
│
├── tests/
│   ├── unit_test_suite.py           96+ unit tests covering all algorithms, task types,
│   │                                edge cases, and API contracts
│   ├── performance_test_suite.py    Runtime and memory budget tests
│   ├── reproducibility_test_suite.py  30 determinism tests (same seed -> same result)
│   └── Test_Suite.ipynb             Interactive test runner notebook
│
├── benchmarks/
│   ├── benchmark_suite.py           Three real-dataset benchmarks with full statistical
│   │                                comparison battery
│   ├── Benchmark_Suite.ipynb        Interactive benchmark runner
│   └── Graphene_Oxide_Bulk.csv      CSIRO benchmark dataset (local; not downloaded)
│
├── docs/
│   ├── USER_GUIDE.md                All parameters, methods, and usage patterns
│   ├── IMPLEMENTATION_GUIDE.md      Internal design, tuning, and extension guide
│   └── INTERPRETATION_GUIDE.md      How to read and report results correctly
│
```

---

## Running the tests

```bash
# Unit tests (all algorithms, all task types, edge cases)
pytest tests/unit_test_suite.py -v

# Performance and memory budget tests
RUN_PERFORMANCE=1 pytest tests/performance_test_suite.py -v -s

# Reproducibility tests (determinism verification)
pytest tests/reproducibility_test_suite.py -v

# Benchmarks (requires network for SECOM and Urban Land Cover datasets)
pytest benchmarks/benchmark_suite.py -v -s

# Full console benchmark report
python benchmarks/benchmark_suite.py
```

---

## Documentation

| Guide | Contents |
|---|---|
| [User Guide](docs/USER_GUIDE.md) | Parameters, methods, prediction, validation, SHAP, saving |
| [Implementation Guide](docs/IMPLEMENTATION_GUIDE.md) | Internal design, tuning for speed and rigor, algorithm details, extending ROBUST |
| [Interpretation Guide](docs/INTERPRETATION_GUIDE.md) | Reading results correctly, statistical tests, what to report in a paper |

---

## Example: comparing to a full-feature baseline

```python
from RobustModelMaker import RobustModelMaker
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ROBUST: stability-selected feature subset
maker = RobustModelMaker(alg="rdg", task_type="binary",
                          outer_cv=5, n_bootstrap=100, random_state=42).fit(X, y)
print(f"ROBUST: {len(maker.result_.selected_features)} features, "
      f"AUC {maker.result_.nested_cv_result.mean_score:.4f}")

# Access out-of-fold predictions for calibration or downstream analysis
oof_proba = maker.result_.nested_cv_result.outer_predictions
oof_true  = maker.result_.nested_cv_result.outer_true_labels
```

---

## Example: grouped cross-validation

Use `groups` when samples from the same experimental unit (patient, batch, synthesis run) appear more than once, to prevent data leakage across groups:

```python
maker = RobustModelMaker(
    alg="eln", task_type="regression",
    outer_cv=5, n_bootstrap=100, random_state=42,
).fit(X, y, groups=sample_ids)
```

---

## Example: permutation importance after fitting

```python
pi = maker.permutation_importance(X_val, y_val, n_repeats=20, random_state=42, n_jobs=1)
print(pi.summary().head(10))
#    feature  importance_mean  importance_std
# 0  feature_23          0.189           0.031
# 1  feature_7           0.142           0.028
# ...
```

---

## Citing this work

If you use RobustModelMaker in your research, please cite:

```
Barnard, A. S. (2026). RobustModelMaker: A reproducible stability-selection pipeline
for scientific machine learning (v0.3). GitHub: https://github.com/amaxiom/RobustModelMaker
```

---

## Author

**Prof Amanda S Barnard**
GitHub: [amaxiom](https://github.com/amaxiom)

RobustModelMaker is developed and maintained as a tool for rigorous, reproducible machine learning in scientific research.

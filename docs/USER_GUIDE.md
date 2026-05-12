# RobustModelMaker User Guide

RobustModelMaker (ROBUST) is a reproducible model-building pipeline for small-to-medium scientific datasets. It combines bootstrap stability selection with nested cross-validation to produce feature-reduced models that generalise reliably, with honest performance estimates that do not leak information from the test set into model-building decisions.

---

## Contents

1. [Installation](#1-installation)
2. [Quick start](#2-quick-start)
3. [Task types](#3-task-types)
4. [Algorithms](#4-algorithms)
5. [Full parameter reference](#5-full-parameter-reference)
6. [The result object](#6-the-result-object)
7. [Prediction and probability](#7-prediction-and-probability)
8. [External validation](#8-external-validation)
9. [Permutation importance](#9-permutation-importance)
10. [SHAP integration](#10-shap-integration)
11. [Saving and loading results](#11-saving-and-loading-results) — class API, automatic save, output files, print report, reload
12. [Grouped cross-validation](#12-grouped-cross-validation)
13. [Probability calibration](#13-probability-calibration)
14. [Working with missing values](#14-working-with-missing-values)
15. [Using the functional API](#15-using-the-functional-api)

---

## 1. Installation

ROBUST is a single-file library. Copy `RobustModelMaker.py` into your project and import it:

```python
import sys
sys.path.append("/path/to/RobustModelMaker")
from RobustModelMaker import RobustModelMaker, run_pipeline
```

**Required packages:** `numpy`, `pandas`, `scikit-learn`, `scipy`

**Optional:** `xgboost` (for `alg="xgb"`)

---

## 2. Quick start

```python
import pandas as pd
from RobustModelMaker import RobustModelMaker

X = pd.read_csv("features.csv")
y = pd.read_csv("labels.csv").squeeze()

maker = RobustModelMaker(
    alg="eln",          # elastic net: fast and interpretable
    task_type="binary", # "binary", "multiclass", or "regression"
    outer_cv=5,
    inner_cv=5,
    n_bootstrap=50,
    random_state=42,
)
maker.fit(X, y)

# See what was selected and how well it performed
result = maker.result_
print(f"Selected {len(result.selected_features)} features")
print(f"Nested CV AUC: {result.nested_cv_result.mean_score:.4f} "
      f"+/- {result.nested_cv_result.std_score:.4f}")

# Predict on new data
predictions = maker.predict(X_new)
```

The equivalent functional call:

```python
from RobustModelMaker import run_pipeline

result = run_pipeline(X, y, alg="eln", task_type="binary",
                      outer_cv=5, inner_cv=5, n_bootstrap=50, random_state=42)
```

---

## 3. Task types

Set `task_type` to one of:

| Value | When to use | Scoring metric |
|---|---|---|
| `"binary"` | Two-class outcome (0/1, True/False, case/control) | ROC-AUC |
| `"multiclass"` | Three or more classes | Weighted OVR ROC-AUC |
| `"regression"` | Continuous numeric target | Negative RMSE |
| `"auto"` | Let ROBUST infer from `y` (see note below) | as above |

**Auto-detection rules:** if `y` is float-typed and has more than 20 unique values (or > 20% of samples), ROBUST infers regression; 2 unique values gives binary; 3 to 20 gives multiclass.

For scientific use it is best practice to set `task_type` explicitly rather than relying on auto-detection.

**Labels:** Any hashable type is supported for classification (strings, integers, booleans). ROBUST encodes them internally and decodes predictions back to the original label space, so `predict()` always returns values in the same format as `y`.

---

## 4. Algorithms

| Code | Algorithm | Supported tasks | Notes |
|---|---|---|---|
| `"eln"` | Elastic Net | all | Fastest; coefficient-based feature importance; feature scaling applied automatically |
| `"rdg"` | Ridge (L2 logistic / ridge regression) | all | Stable baseline; good default for many scientific datasets |
| `"las"` | Lasso (L1 logistic / lasso regression) | all | Sparse coefficients; strong built-in feature selector |
| `"log"` | L2 logistic regression | classification only | Reliable baseline for binary and multiclass problems |
| `"svm"` | Linear SVM | all | Effective in high-dimensional feature spaces |
| `"rf"` | Random Forest | all | No scaling needed; handles non-linear relationships; `class_weight="balanced"` |
| `"xgb"` | XGBoost | all | Highest raw performance; requires `pip install xgboost`; slowest |
| `"mlp"` | Multi-layer perceptron | all | Neural baseline; slower on small datasets |
| `"lin"` | Ordinary least squares | regression only | Fully interpretable; no regularisation |

**Note on task restrictions:** `"log"` raises an error if used with `task_type="regression"`. `"lin"` raises an error if used with any classification task type.

**Choosing an algorithm:**

- Start with `"eln"` for interpretability and speed, especially when `n_features >> n_samples`.
- Use `"rdg"` or `"las"` for similar interpretability with different regularisation penalties.
- Use `"rf"` when relationships are non-linear and you have enough samples for tree methods.
- Use `"xgb"` for maximum predictive performance when runtime is not a concern.
- Use `"lin"` for regression when you want the simplest possible model with no shrinkage.

The algorithm governs both the stability selection phase (bootstrap subsampling) and the final nested CV phase (hyperparameter tuning), so feature selection and model assessment use the same estimator family.

---

## 5. Full parameter reference

### `RobustModelMaker` constructor / `run_pipeline` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `alg` | str | `"eln"` | Algorithm: one of `"eln"`, `"rdg"`, `"las"`, `"log"`, `"svm"`, `"rf"`, `"xgb"`, `"mlp"`, `"lin"` |
| `task_type` | str | `"auto"` | Task: `"binary"`, `"multiclass"`, `"regression"`, `"auto"` |
| `outer_cv` | int | `10` | Number of outer CV folds (for performance estimation) |
| `inner_cv` | int | `10` | Number of inner CV folds (for hyperparameter search) |
| `repeated_outer_cv` | int | `1` | Repeat outer CV this many times and average; increases runtime linearly |
| `n_iter` | int | `100` | Hyperparameter search iterations per inner fold (RandomizedSearchCV) |
| `stability_threshold` | float | `0.7` | Selection frequency required for a feature to be retained (0 to 1) |
| `n_bootstrap` | int | `100` | Bootstrap resamples for stability selection |
| `cutoff_n_bootstrap` | int | `1000` | Bootstrap resamples for binary classification cutoff determination |
| `spec` | float | `0.98` | Target specificity for binary cutoff determination |
| `random_state` | int | `42` | Seed for all random operations |
| `preprocess` | str | `"auto"` | Preprocessing: `"auto"` (scale only for algorithms that need it), `"standard"` (always scale), `"none"` |
| `calibration` | str | `"none"` | Probability calibration: `"none"`, `"sigmoid"`, `"isotonic"` |
| `groups` | array-like | `None` | Group labels for grouped CV (prevents data leakage across groups) |
| `X_validation` | DataFrame/array | `None` | External validation set features (evaluated after fitting) |
| `y_validation` | array-like | `None` | External validation set labels |
| `n_jobs` | int | `-1` | Parallelism (-1 uses all available cores) |
| `verbose` | bool | `True` | Print progress during fitting |
| `preserve_nans` | bool | `True` | If `False`, drop high-missingness rows and columns before processing |
| `save_results` | bool | `False` | If `True`, automatically save all outputs after fitting |
| `output_dir` | str | `"robust_model_results"` | Directory for saved outputs (used when `save_results=True`) |
| `output_prefix` | str | `"robust_model"` | Filename prefix for all saved files (used when `save_results=True`) |

### `.fit()` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `X` | DataFrame or array | required | Feature matrix |
| `y` | array-like | required | Target vector |
| `groups` | array-like | `None` | Group labels (overrides constructor `groups` if both are provided) |
| `X_validation` | DataFrame or array | `None` | External validation features (overrides constructor value) |
| `y_validation` | array-like | `None` | External validation labels (overrides constructor value) |
| `feature_names` | list of str | `None` | Feature names to use when `X` is a numpy array without column names |

### `stability_threshold` guidance

| Value | Meaning | When to use |
|---|---|---|
| 0.5 | Selected in more than half of bootstrap samples | Lenient; more features retained; benchmark suite default |
| 0.6 | 60% bootstrap stability | Balanced starting point |
| 0.7 | 70% stability (default) | Standard scientific usage |
| 0.8 | High stability | When interpretability and parsimony matter most |

A lower threshold retains more features and typically achieves scores closer to the full-feature baseline. A higher threshold produces a smaller, more reproducible feature set that may trade a small amount of performance for stability.

---

## 6. The result object

After fitting, the full result is available as `maker.result_` (a `PipelineResult` dataclass) or as the return value of `run_pipeline()`.

### Key attributes

```python
result = maker.result_

# Selected feature names (as they appeared in your DataFrame columns)
result.selected_features          # np.ndarray of feature name strings

# All feature names (input order)
result.feature_names              # np.ndarray

# Algorithm and task type
result.algorithm                  # e.g. "eln"
result.task_type                  # e.g. "binary"

# Nested CV performance
result.nested_cv_result.mean_score       # float: mean outer-fold score
result.nested_cv_result.std_score        # float: std across outer folds
result.nested_cv_result.outer_scores     # np.ndarray: one score per outer fold
result.nested_cv_result.outer_predictions  # out-of-fold predictions (full dataset)
result.nested_cv_result.outer_true_labels  # original labels, same order

# Shortcut properties on PipelineResult (same values, fewer keystrokes)
result.mean_score                        # same as result.nested_cv_result.mean_score
result.std_score                         # same as result.nested_cv_result.std_score

# Stability selection frequencies (one value per feature, 0.0 to 1.0)
result.stability_result.selection_frequencies   # np.ndarray
result.stability_result.summary()               # pd.DataFrame with columns:
                                                #   feature, selection_frequency, selected

# Binary classification cutoff (None for multiclass / regression)
result.cutoff_result               # CutoffResult or None
result.cutoff_result.cutoff_median # float: recommended decision threshold
result.cutoff_result.summary()     # human-readable cutoff string

# External validation (None unless X_validation was provided)
result.validation_result
result.validation_result.metrics      # dict of metric names to floats
result.validation_result.confusion    # confusion matrix (classification only)

# For multiclass
result.class_names                 # np.ndarray of class name strings
result.label_mapping               # dict: original_label -> integer code
```

### Full text summary

```python
print(result.summary())
```

Prints a formatted block covering task, algorithm, selected features, nested CV scores, cutoff, and external validation if available.

### Results tables (for export or inspection)

```python
tables = result.results_tables()
# Returns a dict of pd.DataFrames. Keys always present:
#   "overview"                           -- task, algorithm, n_features, score
#   "selected_features"                  -- selected feature names
#   "stability_selection"                -- per-feature selection frequency
#   "feature_stability_cv"               -- per-fold feature selection table
#   "nested_cv_scores"                   -- per-fold scores
#   "nested_cv_predictions"              -- out-of-fold predictions
#
# Present for binary classification only:
#   "cutoff_distribution"                -- bootstrap cutoff values
#
# Present when external validation was run:
#   "external_validation"                -- summary metrics
#   "external_validation_metrics"        -- full metric dict as a table
#   "external_validation_predictions"    -- predicted vs actual for each sample
#   "external_validation_confusion_matrix" -- confusion matrix (classification)

# Save all tables as CSVs alongside JSON and pickle
result.save_results(output_dir="results/", prefix="my_model")
```

**Note:** the `save_results()` method on `PipelineResult` takes a `prefix` parameter (not `output_prefix`). See section 11 for the full save API.

---

## 7. Prediction and probability

Once fitted, `maker` (or `result`) can predict directly:

```python
# Class labels (returns pd.Series for DataFrame input)
predictions = maker.predict(X_new)

# Binary: use a specific cutoff instead of the auto-determined one
predictions = maker.predict(X_new, cutoff=0.6)

# Probabilities
# Binary: returns pd.Series of positive-class probability
proba = maker.predict_proba(X_new)

# Multiclass: returns pd.DataFrame, one column per class
proba_df = maker.predict_proba(X_new)

# Regression: predict_proba raises AttributeError (use predict)
values = maker.predict(X_new)
```

**Important:** `predict()` and `predict_proba()` automatically apply the same preprocessing (imputation, scaling) and feature selection that was learned during fitting. You do not need to preprocess `X_new` yourself.

---

## 8. External validation

Pass a held-out set at fit time:

```python
maker = RobustModelMaker(alg="eln", task_type="binary", random_state=42)
maker.fit(X_train, y_train, X_validation=X_val, y_validation=y_val)

val = maker.result_.validation_result
print(val.summary())    # pd.DataFrame of metrics
print(val.metrics)      # dict: auc, accuracy, sensitivity, specificity, etc.
```

Or evaluate after fitting:

```python
val = maker.result_.evaluate_verification(X_val, y_val)
print(val.metrics)
```

**Binary classification metrics:** `auc`, `accuracy`, `balanced_accuracy`, `sensitivity`, `specificity`, `tn`, `fp`, `fn`, `tp`, `cutoff`

**Multiclass classification metrics:** `auc_ovr_weighted`, `accuracy`, `balanced_accuracy`, `f1_weighted`, `macro_f1`

**Regression metrics:** `r2`, `rmse`, `mae`

---

## 9. Permutation importance

After fitting, compute permutation importance on any dataset (typically a held-out set):

```python
pi = maker.permutation_importance(X_val, y_val, n_repeats=20, random_state=42, n_jobs=1)

# Access raw arrays
pi.importances_mean     # np.ndarray, one value per selected feature
pi.importances_std      # np.ndarray
pi.feature_names        # np.ndarray, aligned with importances_mean

# Get a sorted DataFrame
summary = pi.summary()
# Columns: feature, importance_mean, importance_std
print(summary.head(10))

# Or pass as_frame=True directly
df = maker.permutation_importance(X_val, y_val, as_frame=True)
```

Permutation importance is computed on the selected features only (post-selection), not the full feature space.

---

## 10. SHAP integration

Export the model and processed feature matrix in SHAP-ready format:

```python
shap_data = maker.result_.export_shap_ready(X)

model = shap_data["model"]         # fitted sklearn estimator
X_df  = shap_data["X"]             # processed, selected features as pd.DataFrame
names = shap_data["feature_names"] # np.ndarray

import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_df)
shap.summary_plot(shap_values, X_df)
```

For linear models (`alg="eln"`), use `shap.LinearExplainer`. For XGBoost, use `shap.TreeExplainer`.

---

## 11. Saving and loading results

### Saving via the class

```python
# Save all outputs after fitting
maker.save_results(output_dir="results/", output_prefix="my_model")
# maker.save() is an alias for maker.save_results()
maker.save(output_dir="results/", output_prefix="my_model")
```

### Saving automatically at fit time

Pass `save_results=True` to the constructor to save immediately after `.fit()` completes:

```python
maker = RobustModelMaker(
    alg="eln", task_type="binary",
    save_results=True,
    output_dir="results/",
    output_prefix="my_model",
    random_state=42,
)
maker.fit(X, y)
# All outputs written automatically; no separate save call needed
```

### What is written

```
results/
    my_model_metadata.json                     -- parameters and summary metrics
    my_model_pipeline_result.pkl               -- full PipelineResult object (pickle)
    my_model_overview.csv
    my_model_selected_features.csv
    my_model_stability_selection.csv
    my_model_feature_stability_cv.csv
    my_model_nested_cv_scores.csv
    my_model_nested_cv_predictions.csv
    my_model_cutoff_distribution.csv           -- binary classification only
    my_model_external_validation_metrics.csv   -- if validation set provided
    my_model_external_validation_predictions.csv
    my_model_external_validation_confusion_matrix.csv
    my_model_summary.txt                       -- formatted text report
```

After saving, `maker.result_.results_dir` is set to the output directory path.

### Saving via the result object directly

```python
result = maker.result_
result.save_results(output_dir="results/", prefix="my_model")
# Note: parameter is "prefix" (not "output_prefix") on PipelineResult
```

### Printing a report

```python
# Print a formatted summary to the console
maker.print_results()
maker.print_results(top_n=10)   # show top 10 selected features

# Equivalent standalone function
from RobustModelMaker import print_pipeline_results
print_pipeline_results(result, top_n=20)
```

### Reloading from pickle

```python
import pickle
with open("results/my_model_pipeline_result.pkl", "rb") as f:
    result = pickle.load(f)

print(result.mean_score)
predictions = result.predict(X_new)
```

---

## 12. Grouped cross-validation

Use `groups` to prevent data leakage when samples from the same subject, batch, or experimental unit appear in multiple rows:

```python
# groups has one entry per sample identifying which group it belongs to
groups = df["patient_id"].values

maker = RobustModelMaker(alg="rf", task_type="binary", outer_cv=5, random_state=42)
maker.fit(X, y, groups=groups)
```

When `groups` is provided, ROBUST uses `GroupKFold` for both outer and inner folds, ensuring all rows from a given group are always in the same fold. This is critical for longitudinal data, repeated-measures designs, or any dataset with non-independent observations.

**Note:** Grouped CV is deterministic (no shuffling), so `repeated_outer_cv > 1` has no effect and is automatically set to 1.

---

## 13. Probability calibration

For binary and multiclass tasks, predicted probabilities can be calibrated using Platt scaling (sigmoid) or isotonic regression:

```python
maker = RobustModelMaker(
    alg="rf",
    task_type="binary",
    calibration="sigmoid",   # or "isotonic"
    random_state=42,
)
maker.fit(X, y)
```

Calibration is applied after hyperparameter selection in each fold and to the final model. It has no effect on class-label predictions, only on the probability values returned by `predict_proba()`.

**When to calibrate:** Random forests and XGBoost classifiers often produce poorly calibrated probabilities (overconfident or underconfident). If downstream decisions depend on the actual probability values (e.g. expected-value calculations, threshold selection), calibration is recommended.

---

## 14. Working with missing values

ROBUST handles NaN values in `X` automatically by default:

- `preserve_nans=True` (default): NaN values are passed through to the preprocessing pipeline, which uses median imputation inside each CV fold. The imputer is always fitted on training data only, with no leakage.
- `preserve_nans=False`: ROBUST first applies a data-driven missingness filter that drops columns and rows whose missing fraction exceeds optimised thresholds, then proceeds with median imputation. Use this when very sparse features or heavily missing rows would otherwise dominate the analysis.

```python
# Let ROBUST decide which rows/columns to drop based on missingness
maker = RobustModelMaker(alg="eln", task_type="regression",
                         preserve_nans=False, random_state=42)
maker.fit(X, y)

# Inspect what was dropped
d = maker.result_.nan_dropping_result
print(f"Original: {d['original_n_samples']} rows x {d['original_n_features']} features")
print(f"Retained: {d['retained_n_samples']} rows x {d['retained_n_features']} features")
```

The benchmark on the SECOM semiconductor dataset (1567 x 590 features, extensive real NaN values) confirms that ROBUST works reliably out of the box with `preserve_nans=True`.

---

## 15. Using the functional API

All operations are also available as standalone functions:

```python
from RobustModelMaker import (
    stability_selection,
    nested_cross_validation,
    determine_cutoff,
    get_algorithm_config,
    set_global_seed,
)

# Standalone stability selection on preprocessed data
stab = stability_selection(X_proc, y, feature_names=names,
                            alg="eln", n_bootstrap=100, threshold=0.7)
print(stab.summary())

# Standalone nested CV (runs stability selection inside each fold)
nested = nested_cross_validation(X, y, feature_names=names,
                                  alg="eln", outer_cv=5, inner_cv=5)
print(nested.summary())

# Determine a binary cutoff from out-of-fold scores
cutoff = determine_cutoff(y_true, oof_scores, target_specificity=0.98)
print(cutoff.summary())

# Get a configured estimator and its hyperparameter search space
model, param_dist = get_algorithm_config("rf", "binary", random_state=42)
```

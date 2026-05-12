# RobustModelMaker Implementation Guide

This guide describes how ROBUST works internally, how its components fit together, and how to tune each stage to suit your dataset and runtime budget. It is intended for users who want to go beyond defaults and for developers who need to understand or extend the code.

---

## Contents

1. [Pipeline overview](#1-pipeline-overview)
2. [Stage 1: Input validation and preprocessing](#2-stage-1-input-validation-and-preprocessing)
3. [Stage 2: Bootstrap stability selection](#3-stage-2-bootstrap-stability-selection)
4. [Stage 3: Nested cross-validation](#4-stage-3-nested-cross-validation)
5. [Stage 4: Final model and cutoff](#5-stage-4-final-model-and-cutoff)
6. [Reproducibility by design](#6-reproducibility-by-design)
7. [Parallelism and runtime](#7-parallelism-and-runtime)
8. [Memory usage](#8-memory-usage)
9. [Tuning for speed](#9-tuning-for-speed)
10. [Tuning for rigor](#10-tuning-for-rigor)
11. [Algorithm internals](#11-algorithm-internals)
12. [Handling class imbalance](#12-handling-class-imbalance)
13. [Understanding the scoring metrics](#13-understanding-the-scoring-metrics)
14. [Extending ROBUST](#14-extending-robust)

---

## 1. Pipeline overview

ROBUST runs five distinct phases:

```
Input X, y
    |
    v
[1] Validation + Preprocessing
    |
    v
[2] Nested CV with per-fold stability selection
    |   For each outer fold:
    |     - Preprocess training fold (impute + scale)
    |     - Stability selection: bootstrap n_bootstrap times,
    |       record which features are selected above threshold
    |     - Inner CV: RandomizedSearchCV on selected features only
    |     - Score on outer-fold test set (no preprocessing of test fold
    |       from training information)
    |
    v
[3] Final stability selection on all training data
    |
    v
[4] Final model fit on selected features (full training data)
    |
    v
[5] Cutoff determination (binary only): bootstrap controls
```

The critical design property is that **the test fold is never seen during preprocessing, feature selection, or hyperparameter tuning** within any fold. Each of those operations happens entirely on the training partition of that fold. This prevents optimistic bias and ensures that the nested CV score is an honest estimate of generalisation performance.

---

## 2. Stage 1: Input validation and preprocessing

### Validation checks

ROBUST validates inputs before running anything. Errors are raised immediately and clearly:

- `X` must be a 2D numpy array or pandas DataFrame with at least 4 samples and at least 1 feature.
- Infinite values in `X` raise an error (NaNs are allowed; infinities are not).
- All-missing feature columns raise an error.
- Duplicate feature names raise an error.
- For classification: each class must appear in at least `min(outer_cv, inner_cv)` samples.
- Group labels must have the same length as `y` and contain at least 2 distinct groups.

### Preprocessing pipeline

Preprocessing is a two-step sklearn `Pipeline`:

1. **Median imputation** (`SimpleImputer(strategy="median")`): replaces NaNs with the column median, computed from training data only.
2. **Standard scaling** (`StandardScaler()`): applied only when `preprocess="standard"` or `preprocess="auto"` with `alg="eln"`. Tree methods (rf, xgb) do not require scaling.

The preprocessor is fitted independently inside each outer CV fold on the training partition. The final preprocessor is refitted on all training data after CV is complete. This means that imputation values and scaling statistics are different across folds and across the final model fit, all determined only from the data available at that stage.

### Missingness strategy (`preserve_nans=False`)

When `preserve_nans=False`, ROBUST first runs `_smart_drop_nans`:

1. For each combination of column-missingness threshold (0.1 to 0.9) and row-missingness threshold (0.1 to 0.9), it scores the retained region: `density * sqrt(retained_row_fraction) * sqrt(retained_col_fraction)`. This balances data completeness against sample retention.
2. The thresholds achieving the best score are selected.
3. Columns and rows exceeding their respective thresholds are dropped.
4. Remaining NaNs are handled by the per-fold median imputer.

Use `preserve_nans=False` when a substantial fraction of features or samples is very sparse. On the Graphene Oxide benchmark (1617 x 462, real NaN values), this approach retains a consistent feature set for all folds and prevents dimension mismatches during nested CV.

---

## 3. Stage 2: Bootstrap stability selection

### What it does

For each of `n_bootstrap` iterations:
1. Draw a stratified subsample of `sample_fraction * n_samples` rows (stratified for classification, random for regression).
2. Fit a fixed-configuration version of the chosen algorithm on the subsample (no hyperparameter search at this stage).
3. Extract feature importances: `|coef_|` for elastic net (max over classes for multiclass), `feature_importances_` for tree methods.
4. Mark as selected any feature with importance above the median of all importances for that bootstrap run.
5. Accumulate selection counts.

After all bootstraps, divide counts by `n_bootstrap` to get selection frequencies in [0, 1]. Features with frequency >= `stability_threshold` are included in the final feature set.

### Fixed algorithm configurations for stability selection

During stability selection, no hyperparameter search is performed. Each algorithm uses a fixed configuration that balances speed with informativeness:

| Algorithm | Fixed config |
|---|---|
| `eln` | `alpha=0.01, l1_ratio=0.7` |
| `rf` | `n_estimators=80, max_depth=10` |
| `xgb` | `n_estimators=80, max_depth=6, learning_rate=0.05` |

These values were chosen to give useful feature rankings without overfitting. The same configuration is used in every bootstrap run, so the only source of variation is the subsample.

### Placement inside nested CV

Stability selection runs inside each outer fold, on the preprocessed training data for that fold. This means:

- Feature selection frequencies are computed from the fold's training data only.
- The test partition never influences which features are selected.
- Different folds may select slightly different feature sets; this is expected and its extent can be inspected via `nested_cv_result.selected_features_per_fold` and `nested_cv_result.feature_stability`.

### `n_bootstrap` and `stability_threshold` interactions

| `n_bootstrap` | `stability_threshold` | Effect |
|---|---|---|
| 15 (fast/benchmark default) | 0.5 | Coarser frequencies (multiples of 1/15); threshold at 50% means selected in 8+ runs |
| 50 | 0.6 | Reasonable approximation for exploratory work |
| 100 (default) | 0.7 | Standard; selected in 70+ out of 100 bootstrap samples |
| 200 | 0.7 | Smoother frequencies; useful when n_features is large |

Fewer than 50 bootstraps can make the frequencies noisy enough that small random fluctuations determine inclusion. For publication-quality results, use `n_bootstrap >= 100`.

---

## 4. Stage 3: Nested cross-validation

### Structure

ROBUST implements a true nested design:

```
Outer fold split (outer_cv folds, repeated repeated_outer_cv times)
  |
  +-- Training data (outer fold) -->
  |       1. Preprocess (fit on training data only)
  |       2. Stability selection (bootstrap on training data)
  |       3. Inner fold split (inner_cv folds) on selected features
  |              RandomizedSearchCV: n_iter hyperparameter configs
  |              scored by inner-fold cross-validation
  |       4. Refit best config on full training partition
  |
  +-- Test data (outer fold) -->
          5. Preprocess using training-fitted preprocessor
          6. Select features using training-determined set
          7. Score (AUC / weighted OVR AUC / neg-RMSE)
          8. Store out-of-fold predictions
```

No information from the test fold touches steps 1-4. This is stricter than a simple train/test split: the hyperparameter search itself is cross-validated on the training data only.

### `outer_cv` and `inner_cv` guidance

| Dataset size | Recommended outer_cv | Recommended inner_cv |
|---|---|---|
| < 100 samples | 10 (leave-many-out) | 5 |
| 100 to 500 | 5 to 10 | 5 |
| 500 to 2000 | 5 | 3 to 5 |
| > 2000 | 5 | 3 |

More outer folds gives a more reliable performance estimate at the cost of more stability-selection and inner-CV runs. The total number of model fits scales as `outer_cv * inner_cv * n_iter * repeated_outer_cv`.

### Repeated nested CV

Setting `repeated_outer_cv > 1` repeats the entire outer CV with a different random seed per repeat. Outer fold scores and out-of-fold predictions are averaged across repeats. This reduces the variance of the performance estimate for small datasets:

```python
maker = RobustModelMaker(alg="eln", task_type="binary",
                          outer_cv=5, repeated_outer_cv=3, random_state=42)
```

Each repeat uses seed `random_state + repeat * 1000`, ensuring different fold splits while remaining fully reproducible.

### Per-fold feature stability table

```python
fs = maker.result_.nested_cv_result.feature_stability
# Columns: feature, mean_frequency, std_frequency, selected_in_n_folds
print(fs.head(10))
```

A feature with `selected_in_n_folds == outer_cv * repeated_outer_cv` was selected in every fold of every repeat, indicating very high stability. High `std_frequency` suggests the feature's relevance is data-partition-dependent.

---

## 5. Stage 4: Final model and cutoff

After nested CV is complete (and the performance estimate is fully established), ROBUST fits the **final model** on all training data:

1. Fit the final preprocessor on all training data.
2. Run stability selection on all training data.
3. Select features at `stability_threshold`.
4. Fit a final `RandomizedSearchCV` over all training data with inner CV to find the best hyperparameters.
5. Refit the best estimator on all training data.

The final model is stored in `result_.robust_model`. It is the model used for `predict()` and `predict_proba()`.

### Binary classification cutoff determination

For `task_type="binary"`, ROBUST determines a probability cutoff by bootstrapping the control-class (negative class) out-of-fold predictions:

1. Collect the `outer_predictions` for all samples with true label 0 (controls).
2. For each of `cutoff_n_bootstrap` bootstrap resamples of the control scores, find the `spec`-th quantile (default `spec=0.98`, meaning the 98th percentile of control scores).
3. The final cutoff is the median of the bootstrap cutoff distribution. A 95% CI is also reported.

This cutoff ensures that approximately `spec * 100`% of controls score below it (i.e. target specificity), while maximising sensitivity. The cutoff is stored in `result_.cutoff_result.cutoff_median`.

For a different cutoff strategy, use:

```python
from RobustModelMaker import determine_cutoff

# Use a different target specificity
cutoff = determine_cutoff(y_true, oof_scores, target_specificity=0.95)

# Apply manually
predictions = maker.predict(X_new, cutoff=cutoff.cutoff_median)
```

---

## 6. Reproducibility by design

ROBUST is designed to be fully deterministic given the same `random_state`:

- All random operations use explicit seeds derived from `random_state` via offsets (e.g. `random_state + fold_idx`, `random_state + 10000 + bootstrap_idx`).
- `set_global_seed()` sets `numpy.random.seed` and `PYTHONHASHSEED`.
- Each bootstrap run uses `random_state + 10000 + b` so bootstrap sequences are independent of fold sequences.
- `RandomizedSearchCV` receives `random_state=fold_seed`, not a global RNG state.

The reproducibility test suite (`tests/reproducibility_test_suite.py`) verifies this with 30 tests covering:
- Identical selected features across two runs with the same seed.
- Identical fold scores (to machine precision).
- Identical stability frequencies.
- Different results with a different seed.
- Deterministic predictions on new data.

**Caveats:**
- Parallelism (`n_jobs != 1`) can introduce ordering non-determinism with some sklearn versions on some platforms. For exact reproducibility, use `n_jobs=1`.
- Floating-point accumulation order may differ across CPU architectures for tree methods.

---

## 7. Parallelism and runtime

ROBUST passes `n_jobs` to `RandomizedSearchCV` and to `stability_selection` (which passes it to the model). The total number of model fits is:

```
fits = outer_cv * repeated_outer_cv * (n_bootstrap + n_iter * inner_cv + 1)
     + n_bootstrap   (final stability selection)
     + n_iter * inner_cv  (final hyperparameter search)
     + 1  (final model)
```

For the benchmark defaults (`outer_cv=5, inner_cv=2, n_bootstrap=15, n_iter=8, repeated_outer_cv=1`):

```
5 * (15 + 8*2 + 1) + 15 + 8*2 + 1 = 5 * 32 + 32 = 192 fits
```

For production defaults (`outer_cv=10, inner_cv=10, n_bootstrap=100, n_iter=100`):

```
10 * (100 + 100*10 + 1) + 100 + 100*10 + 1 = 10 * 1101 + 1101 = 12,111 fits
```

Each "fit" involves training on a subsample of the data. Runtime is roughly linear in `n_samples * n_features` for linear models and super-linear for tree methods.

---

## 8. Memory usage

Peak memory occurs during `RandomizedSearchCV` with `n_jobs > 1`, when each worker holds a copy of the training fold. Approximate memory per worker:

```
bytes_per_worker ≈ n_samples * n_selected_features * 8 bytes (float64)
```

For large feature spaces, use `n_jobs=1` to avoid duplication or set `n_jobs` to a smaller value (e.g. 2 or 4).

The `preserve_nans=False` mode can significantly reduce memory by dropping sparse columns and rows before the pipeline runs.

---

## 9. Tuning for speed

A fast configuration for exploration or CI:

```python
FAST_KWARGS = dict(
    outer_cv=3,
    inner_cv=2,
    n_bootstrap=15,
    n_iter=8,
    stability_threshold=0.5,
    cutoff_n_bootstrap=100,
    n_jobs=1,
    verbose=False,
    random_state=42,
)
maker = RobustModelMaker(alg="eln", task_type="binary", **FAST_KWARGS)
```

The performance test suite (`tests/performance_test_suite.py`) uses a similar configuration and verifies that binary/multiclass/regression runs complete within a per-sample budget of 0.08 seconds (plus a fixed overhead), enforced on 120-sample synthetic datasets.

**Trade-offs when reducing parameters:**

| Reduction | What is lost |
|---|---|
| Fewer `n_bootstrap` | Coarser selection frequencies; more variance in which features are selected |
| Fewer `n_iter` | Coarser hyperparameter search; potentially worse final model |
| Fewer `inner_cv` | Noisier inner-fold score estimates; less reliable hyperparameter ranking |
| Fewer `outer_cv` | Wider confidence intervals on the performance estimate |
| Lower `n_jobs` | Longer wall-clock time, but lower memory and more reproducible |

---

## 10. Tuning for rigor

A rigorous configuration for final results:

```python
RIGOROUS_KWARGS = dict(
    outer_cv=10,
    inner_cv=10,
    n_bootstrap=200,
    n_iter=200,
    stability_threshold=0.7,
    repeated_outer_cv=3,
    cutoff_n_bootstrap=2000,
    n_jobs=-1,
    random_state=42,
)
```

For very small datasets (< 100 samples), consider:

```python
# Leave-one-out outer CV with repeated splits to reduce estimate variance
maker = RobustModelMaker(
    alg="eln", task_type="binary",
    outer_cv=10,          # use all available folds
    repeated_outer_cv=5,  # repeat 5 times to stabilise the estimate
    n_bootstrap=200,
    random_state=42,
)
```

---

## 11. Algorithm internals

### Elastic net (`eln`)

**Stability selection:** uses `LogisticRegression(penalty="elasticnet", l1_ratio=0.7, C=1.0)` for classification; `ElasticNet(alpha=0.01, l1_ratio=0.7)` for regression. The L1 component produces sparse coefficients, so the above-median importance threshold selects roughly the top half of features per bootstrap run.

**Nested CV search space:**
- Classification: `C ~ LogUniform(1e-4, 1e2)`, `l1_ratio ~ Uniform(0, 1)`
- Regression: `alpha ~ LogUniform(1e-4, 1e2)`, `l1_ratio ~ Uniform(0, 1)`

**Preprocessing:** always scale (standard normalisation). Required because regularisation strength is not scale-invariant.

**Feature importance:** `|coef_|` for binary; `max(|coef_|, axis=0)` across classes for multiclass.

### Ridge (`rdg`)

**Stability selection:** `LogisticRegression(penalty="l2", C=1.0)` for classification; `Ridge(alpha=1.0)` for regression. Uses `|coef_|` for importance ranking.

**Nested CV search space:** `C ~ LogUniform(1e-4, 1e2)` (classification) or `alpha ~ LogUniform(1e-4, 1e2)` (regression).

**Preprocessing:** always scale. L2 penalty is not scale-invariant.

### Lasso (`las`)

**Stability selection:** `LogisticRegression(penalty="l1", C=0.1, solver="liblinear")` for classification; `Lasso(alpha=0.01)` for regression. Strong L1 sparsity makes this a natural feature selector.

**Nested CV search space:** `C ~ LogUniform(1e-4, 1e2)` (classification) or `alpha ~ LogUniform(1e-4, 1e2)` (regression).

**Preprocessing:** always scale.

### Logistic regression (`log`) — classification only

**Stability selection:** `LogisticRegression(penalty="l2", C=1.0)`. Raises an error if used with `task_type="regression"`.

**Nested CV search space:** `C ~ LogUniform(1e-4, 1e2)`.

**Preprocessing:** always scale.

### Linear SVM (`svm`)

**Stability selection:** `LinearSVC(C=0.1)` for classification; `LinearSVR(C=0.1)` for regression. Uses `|coef_|` for importance.

**Nested CV search space:** `C ~ LogUniform(1e-4, 1e2)`.

**Preprocessing:** always scale. SVM margin is distance-based and requires comparable feature magnitudes.

### Random forest (`rf`)

**Stability selection:** `n_estimators=80, max_depth=10` with `class_weight="balanced_subsample"` for classification. Uses `feature_importances_` (mean decrease in impurity).

**Nested CV search space:** `n_estimators ~ Randint(100, 500)`, `max_depth ~ Randint(2, 20)`, `min_samples_split`, `min_samples_leaf`, `max_features`.

**Preprocessing:** median imputation only (no scaling needed for trees).

### XGBoost (`xgb`)

**Stability selection:** `n_estimators=80, max_depth=6, learning_rate=0.05`. Uses `feature_importances_` (weight-based gain).

**Nested CV search space:** `n_estimators`, `max_depth`, `learning_rate ~ LogUniform(0.01, 0.3)`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`.

**Preprocessing:** median imputation only.

### Multi-layer perceptron (`mlp`)

**Stability selection:** `MLPClassifier` or `MLPRegressor` with a single hidden layer of 100 units and `max_iter=500`. Uses permutation importance internally (no native `coef_` or `feature_importances_`).

**Nested CV search space:** `hidden_layer_sizes`, `alpha ~ LogUniform(1e-5, 1e-1)`, `learning_rate_init ~ LogUniform(1e-4, 1e-2)`.

**Preprocessing:** always scale. Neural networks are sensitive to feature magnitude.

### Ordinary least squares (`lin`) — regression only

**Stability selection:** `LinearRegression()`. Raises an error if used with classification task types. Uses `|coef_|` for importance.

**Nested CV search space:** no hyperparameters; a single deterministic fit per inner fold.

**Preprocessing:** always scale.

---

## 12. Handling class imbalance

ROBUST addresses class imbalance at two levels:

1. **Stratified splitting:** `StratifiedKFold` is used for all classification tasks, ensuring class proportions are preserved in each fold.

2. **Algorithm-level balancing:**
   - Elastic net (`eln`), Ridge (`rdg`), Lasso (`las`), Logistic (`log`), SVM (`svm`), MLP (`mlp`): `class_weight` is not set by default. For severe imbalance these models rely on stratification and the AUC metric.
   - Random forest (`rf`): `class_weight="balanced_subsample"` is always set, which weights samples inversely proportional to class frequency within each bootstrap sample of the tree.
   - XGBoost (`xgb`): no automatic weighting, but the hyperparameter search covers `reg_alpha` and `reg_lambda` which help regularise minority-class patterns.

3. **Stability subsampling:** `_stratified_or_random_subsample()` draws stratified subsamples for classification, preserving class proportions in each bootstrap run.

For extreme imbalance (< 5% minority class, as in the SECOM benchmark with ~7% failure rate), the benchmark results show that the combination of stratified splitting and `class_weight="balanced_subsample"` is sufficient for AUC-based evaluation. If recall or F1 on the minority class is the target metric, consider post-hoc threshold selection using `determine_cutoff()` with a lower `target_specificity`.

---

## 13. Understanding the scoring metrics

| Task | Metric | Score range | Best value | Notes |
|---|---|---|---|---|
| Binary | ROC-AUC | 0 to 1 | 1.0 | 0.5 = random; reported as positive float |
| Multiclass | Weighted OVR ROC-AUC | 0 to 1 | 1.0 | Weighted by class frequency |
| Regression | Negative RMSE | -inf to 0 | 0.0 | Negated so sklearn maximisation applies |

**Regression scores are always negative** (they are negated RMSE, not RMSE itself). A score of -1.23 means the RMSE is 1.23 in the target's units. A less negative score (e.g. -0.8) is better than a more negative one (e.g. -1.5). When comparing ROBUST to a baseline, a positive delta means ROBUST had a smaller RMSE.

The `mean_score` and `std_score` attributes use this sign convention throughout. The `floor_score` parameter in the benchmark suite is set accordingly (e.g. `floor_score=-8.0` for the Graphene Oxide dataset means ROBUST must achieve a mean neg-RMSE better than -8.0, i.e. RMSE < 8.0 eV).

---

## 14. Extending ROBUST

### Adding a new algorithm

1. Add the algorithm code to the `Algorithm` Literal type alias.
2. Add a branch to `get_algorithm_config()` returning `(estimator, param_distributions)`.
3. Add a fixed-configuration branch to `stability_selection()` (the `if alg == ...` block that sets model parameters before bootstrapping).
4. Add a test to `tests/unit_test_suite.py` using the `CLASSIFICATION_ALGS` / `REGRESSION_ALGS` lists.

### Using a custom estimator outside the pipeline

```python
from RobustModelMaker import stability_selection, nested_cross_validation

# Preprocess first (or let ROBUST handle it inside nested_cross_validation)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pre = Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler())])
X_proc = pre.fit_transform(X_train)

# Run stability selection
stab = stability_selection(X_proc, y_train, feature_names=feature_names,
                            alg="eln", n_bootstrap=100, threshold=0.7)
selected = stab.selected_features
print(f"Selected: {selected}")
```

### Custom scoring metric

The scoring metric is determined by `_default_scoring(task_type)`. To use a different metric, pass the sklearn scorer string as a `scoring` argument to the underlying `nested_cross_validation` call or override `_default_scoring` directly. For `permutation_importance`, pass `scoring=` explicitly:

```python
pi = maker.permutation_importance(X_val, y_val,
                                   scoring="balanced_accuracy", n_repeats=20)
```

# RobustModelMaker Interpretation Guide

This guide explains what ROBUST produces, how to read each output, and how to draw valid scientific conclusions from the results. It covers performance estimates, feature selection outputs, cutoff determination, and the statistical comparison framework used in the benchmark suite.

---

## Contents

1. [The central question ROBUST answers](#1-the-central-question-robust-answers)
2. [Nested CV performance estimates](#2-nested-cv-performance-estimates)
3. [Per-fold scores: what variance means](#3-per-fold-scores-what-variance-means)
4. [Feature selection frequencies](#4-feature-selection-frequencies)
5. [Selected features and the final model](#5-selected-features-and-the-final-model)
6. [The binary classification cutoff](#6-the-binary-classification-cutoff)
7. [External validation results](#7-external-validation-results)
8. [Comparing ROBUST to a full-feature baseline](#8-comparing-robust-to-a-full-feature-baseline)
9. [Interpreting the statistical test battery](#9-interpreting-the-statistical-test-battery)
10. [Benchmark evidence](#10-benchmark-evidence)
11. [What to report in a paper](#11-what-to-report-in-a-paper)
12. [Common misinterpretations](#12-common-misinterpretations)

---

## 1. The central question ROBUST answers

ROBUST addresses a specific problem common in scientific machine learning: you have a moderately sized dataset with many candidate features, and you want to know:

1. How well can a model predict the outcome using only a stable, reproducible subset of features?
2. Which features are robustly predictive across different subsamples of the data?
3. How much performance is preserved (or gained) by reducing the feature set?

The goal is not to maximise raw predictive performance. It is to find the smallest feature set that preserves generalisation performance, and to quantify that performance honestly using a design that does not leak information from the test set.

---

## 2. Nested CV performance estimates

### What the score means

The `nested_cv_result.mean_score` is the average of `outer_cv` (times `repeated_outer_cv`) out-of-fold scores. Each score was computed on data that was never used for preprocessing, feature selection, or hyperparameter tuning in that fold.

This is the correct estimate to report as the model's expected generalisation performance on new data drawn from the same distribution.

```python
result = maker.result_
print(f"Mean AUC: {result.nested_cv_result.mean_score:.4f}")
print(f"Std  AUC: {result.nested_cv_result.std_score:.4f}")
```

**What "nested" means:** A simple cross-validated AUC (outer CV only) is optimistically biased because hyperparameter selection uses the outer-fold test set. ROBUST avoids this by running a separate inner CV for hyperparameter selection within each outer fold, so the outer-fold test set is used only for evaluation.

### Regression scores (negative RMSE)

Regression scores are reported as **negative RMSE**: a score of `-1.23` means the root-mean-squared error is 1.23 in target units. A less negative score (closer to zero) is better. When comparing two models, a positive delta means the model with the less negative score has a lower RMSE.

```python
rmse = abs(result.nested_cv_result.mean_score)
print(f"Mean RMSE: {rmse:.3f} {target_units}")
```

### Score standard deviation

The standard deviation across folds reflects two things:

1. **Natural variation** in how hard different test partitions are to predict (e.g. one fold might contain more hard-to-classify boundary cases).
2. **Data scarcity**: with few samples, each fold contains very little test data, making fold-level scores noisy.

A large `std_score` relative to `mean_score` is expected on small datasets. It does not mean the model is unreliable in the same way that a wide confidence interval does not mean a parameter is unknown.

---

## 3. Per-fold scores: what variance means

```python
print(result.nested_cv_result.outer_scores)
# e.g. [0.762, 0.701, 0.798, 0.744, 0.719]
```

Inspect per-fold scores to check:

- **Consistency:** all folds scoring near the mean indicates a well-generalising model and a reasonably balanced dataset.
- **Outlier folds:** one fold with a very different score (high or low) may indicate a class-imbalanced test partition, or that a particular data region is harder to predict. Investigate what is different about that fold's data.
- **Trend:** if scores are noticeably lower in later folds, this can indicate temporal autocorrelation (the splits are not truly independent). Use `groups=` if your data has batch or time structure.

The out-of-fold predictions cover every training sample exactly once (or `repeated_outer_cv` times with averaging for repeated CV), making them suitable for calibration assessment:

```python
oof_pred = result.nested_cv_result.outer_predictions  # probabilities or continuous values
oof_true = result.nested_cv_result.outer_true_labels
```

---

## 4. Feature selection frequencies

The stability selection frequency for a feature is the proportion of bootstrap runs in which that feature was selected (importance above the median) when trained on a random subsample of the training data.

```python
stab_df = result.stability_result.summary()
# Columns: feature, selection_frequency, selected
print(stab_df.head(20))
```

### Interpreting the frequency values

| Frequency | Meaning |
|---|---|
| 1.00 | Selected in every single bootstrap run on every fold. Extremely robust signal. |
| 0.80 to 0.99 | High stability. Reliably selected across different data subsamples. |
| 0.70 (threshold default) | Meets the selection threshold. Considered stable. |
| 0.50 to 0.69 | Marginal stability. Selected in roughly half of runs. Treat with caution. |
| < 0.50 | Not robustly selected. May still have some predictive value but is unstable. |

### What selection frequency is not

Selection frequency is not the same as effect size or variable importance. A feature with frequency 1.0 is robustly selected, but may have a small effect on the outcome. A feature with frequency 0.4 may have a strong effect in the right subpopulation but be unstable across different training samples.

Similarly, a feature not selected by ROBUST is not necessarily irrelevant: it may be correlated with a selected feature (multicollinearity reduces both), or it may only be predictive in interaction with another feature.

### Visualising stability

```python
ax = result.stability_result.plot_feature_stability(top_n=30)
# Horizontal bar chart; vertical dashed line marks the threshold
```

### Feature stability across folds

The per-fold stability table shows whether the same features are selected across all outer CV folds:

```python
fs = result.nested_cv_result.feature_stability
# Columns: feature, mean_frequency, std_frequency, selected_in_n_folds
print(fs[fs["selected_in_n_folds"] == maker.outer_cv].head())
# Features selected in every outer fold
```

A feature with `selected_in_n_folds == outer_cv` was considered stable in every fold. A feature with `selected_in_n_folds == 1` was selected in only one fold; this may be a fold-specific artefact.

---

## 5. Selected features and the final model

The final feature set is determined by running stability selection on all training data (after nested CV is complete and the performance estimate is established):

```python
print(result.selected_features)
# np.ndarray of feature name strings
```

**This is the feature set used by `predict()`.** It is the result of a single, full-data stability selection run, not an average across folds. For well-behaved datasets, it should closely resemble the features that were consistently selected across all outer folds.

### When the final set differs from fold-level sets

If some features appear in `selected_features` but rarely in `nested_cv_result.feature_stability`, or vice versa, this indicates the feature's relevance is sensitive to the particular training sample. This is useful information: it means the feature's contribution is not robust to small changes in the dataset. Treat such features with extra caution in downstream analysis.

### Accessing the final fitted model directly

```python
model = result.robust_model   # fitted sklearn estimator (operates on selected features)
pre   = result.preprocessor   # fitted preprocessing pipeline

# Coefficients (eln)
if hasattr(model, "coef_"):
    coefs = pd.Series(model.coef_.ravel(), index=result.selected_features)
    print(coefs.sort_values(key=abs, ascending=False))
```

---

## 6. The binary classification cutoff

### What the cutoff is

ROBUST determines a probability threshold by bootstrapping the out-of-fold scores of the **negative (control) class**:

1. Take all out-of-fold predicted probabilities for control samples.
2. For each of `cutoff_n_bootstrap` bootstrap resamples, compute the `spec`-th quantile of control scores (default 98th percentile).
3. Report the median of the bootstrap cutoff distribution.

This means: if you classify a new sample as positive when its predicted probability exceeds the cutoff, you expect approximately 98% of true negatives to score below the threshold (98% specificity by design).

```python
cutoff = result.cutoff_result
print(f"Cutoff:              {cutoff.cutoff_median:.4f}")
print(f"95% CI:              [{cutoff.cutoff_ci_lower:.4f}, {cutoff.cutoff_ci_upper:.4f}]")
print(f"Target specificity:  {cutoff.target_specificity:.1%}")
print(f"Achieved specificity:{cutoff.achieved_specificity:.1%}  (on training OOF predictions)")
print(f"Achieved sensitivity:{cutoff.achieved_sensitivity:.1%}  (on training OOF predictions)")
```

### Important limitations

- The achieved specificity and sensitivity are computed on the out-of-fold predictions, which are an honest estimate of generalisation but are still training data. Report them as internal validation estimates.
- If the cutoff 95% CI is wide, the threshold is uncertain. Use external validation to confirm the cutoff holds on new data.
- The cutoff is appropriate for the specific `spec` target. Adjust `spec` if clinical or scientific requirements differ:

```python
# 95% specificity target instead of 98%
maker = RobustModelMaker(alg="eln", task_type="binary", spec=0.95, random_state=42)
```

---

## 7. External validation results

External validation provides an independent performance estimate on data not used at any stage of model building:

```python
val = result.validation_result   # or result.evaluate_verification(X_val, y_val)
print(val.metrics)
```

### Interpreting validation vs nested CV scores

| Scenario | Likely meaning |
|---|---|
| Validation AUC ≈ nested CV mean AUC | The nested CV estimate was accurate; good generalisation |
| Validation AUC > nested CV mean AUC | Slight positive fluctuation or the external set is somewhat easier |
| Validation AUC < nested CV mean AUC by a small amount (< 2 std) | Expected random variation; model generalises adequately |
| Validation AUC < nested CV mean AUC by > 2 std | Potential distribution shift, temporal drift, or label noise in one set |

A validation score substantially higher than the nested CV estimate should prompt investigation: the validation set may not be truly independent (shared preprocessing, same time period, correlated samples).

---

## 8. Comparing ROBUST to a full-feature baseline

The benchmark suite runs ROBUST alongside a full-feature nested-CV baseline using the same algorithm, fold structure, and scoring metric. This answers: how much performance is traded for the feature reduction?

### The outcome classification

The benchmark reports one of three outcomes based on the paired statistical test (Wilcoxon signed-rank preferred; paired t-test if Wilcoxon is unavailable):

| Outcome | Meaning |
|---|---|
| `preserved` | Score difference is not statistically significant (p >= 0.05). ROBUST achieves comparable performance with fewer features. This is the target result. |
| `improved *` | ROBUST score is significantly higher (p < 0.05, delta > 0). ROBUST outperforms the full-feature model, likely because feature reduction acts as regularisation. |
| `degraded *` | ROBUST score is significantly lower (p < 0.05, delta < 0). Feature reduction caused a measurable performance loss. |

### Why `preserved` is a success

The purpose of ROBUST is not to beat the baseline but to match it with fewer features. A `preserved` outcome means the feature subset is sufficient to capture the signal, and the model built on it will be more interpretable, more stable, and less prone to overfitting on new data from the same distribution.

### The efficiency metric

The score-per-feature ratio in the scenario report quantifies how efficiently ROBUST uses information:

```
efficiency_ratio = (|ROBUST_score| / n_selected_features) / (|BL_score| / n_total_features)
```

A ratio of 10x means ROBUST achieves the same score per feature with one-tenth the features, or equivalently, each selected feature carries ten times more predictive signal than the average feature in the full set. This is meaningful when features have acquisition costs (e.g. clinical assays, sensor channels) or interpretability constraints.

### Why a fixed score-delta threshold is misleading

Early versions of ROBUST used a 0.001 score threshold to declare a "winner". This is problematic for two reasons:

1. A delta of 0.002 is within the noise of a 5-fold CV estimate. With 5 folds, the standard error of the mean is `std / sqrt(5)`, which for typical std values of 0.05 to 0.10 gives a standard error of 0.02 to 0.04. A threshold of 0.001 is far below the noise floor.
2. The threshold is arbitrary and does not adapt to the difficulty of the problem or the scale of the metric (a delta of 0.001 is meaningless for regression RMSE, which may be in the thousands).

The statistically-grounded outcome label addresses both issues.

---

## 9. Interpreting the statistical test battery

The benchmark suite runs 25+ statistical tests comparing ROBUST and baseline per-fold scores. This section explains the most important ones.

### Descriptive statistics

```
ROBUST mean +/- std    per-fold mean and standard deviation for ROBUST
BL     mean +/- std    same for the full-feature baseline
ROBUST median [IQR]    robust location and spread estimates
```

Compare mean vs median: if they differ substantially, the fold-score distribution is skewed, and the median is the more reliable central tendency estimate.

### Paired tests (most important)

**Wilcoxon signed-rank (preferred):** a non-parametric test for the location of the difference distribution. Tests whether the paired differences (ROBUST score minus BL score in each fold) are symmetric around zero. Does not assume normality. Preferred for the small sample sizes typical of CV (5 to 10 folds).

**Paired t-test:** parametric equivalent. More powerful when normality holds but sensitive to skew. With 5 folds, normality cannot be meaningfully assessed (the Shapiro-Wilk test will almost never reject with n=5).

**Sign test:** counts how many folds ROBUST scored higher. Reports whether this count is significantly greater or lesser than chance (binomial test). Robust to outlier folds. `ROBUST wins k/n non-tied folds` tells you the raw fold counts.

**Interpretation:** for the `preserved` outcome to be reliable, **all three tests should be non-significant** (p >= 0.05). If one is significant and others are not, the result is ambiguous and more data (more folds or more samples) would be needed.

### Effect sizes (independent of significance)

**Cohen's d:** standardised mean difference. Values < 0.2 are negligible, 0.2 to 0.5 small, 0.5 to 0.8 medium, > 0.8 large. A significant p-value with a negligible Cohen's d means the difference is statistically detectable but practically unimportant.

**Common language effect size P(ROBUST > BL):** the probability that a randomly chosen ROBUST fold score exceeds a randomly chosen BL fold score. P = 0.5 means the methods are indistinguishable; P = 0.7 means ROBUST wins 70% of random comparisons.

**Rank-biserial correlation r:** non-parametric effect size from the Mann-Whitney U test. r = 0 (no difference) to r = 1 (ROBUST always higher). For a `preserved` result, |r| should be close to 0.

### Bootstrap confidence interval for the mean difference

```
Bootstrap delta-mean (ROBUST - BL), obs
  95% bootstrap CI for delta-mean    [lo, hi]
```

This non-parametric CI for the mean score difference is the most direct summary of practical significance. A CI that includes zero is consistent with no meaningful difference. A CI of [-0.05, +0.03] means the data are consistent with ROBUST being up to 5% worse or 3% better than the baseline.

### Normality and variance tests

These are informative rather than decision-making:

- **Shapiro-Wilk / Anderson-Darling:** test whether fold scores follow a normal distribution. With 5 folds, the tests have very low power. Non-rejection does not confirm normality.
- **Levene's / Bartlett's tests, Variance ratio:** ROBUST may have lower fold-to-fold variance than the baseline (variance ratio < 1). This is a useful secondary outcome: even if mean performance is similar, a more stable model (lower variance) is preferable in practice.

---

## 10. Benchmark evidence

The benchmark suite (`benchmarks/benchmark_suite.py`) evaluates ROBUST on three real scientific datasets:

### SECOM Semiconductor Manufacturing

- 1567 samples, 590 sensor features, binary pass/fail, ~7% failure rate, extensive NaN values
- Algorithm: Ridge regression (logistic), task: binary classification
- Floor score (min acceptable AUC): 0.60
- Expected outcome: `preserved` (feature reduction with no significant AUC loss)
- This benchmark tests ROBUST under severe class imbalance and high missingness.

### Urban Land Cover

- 675 samples, 147 spectral/texture features, 9-class aerial imagery, no NaN values
- Algorithm: Ridge logistic regression, task: multiclass classification
- Floor score (min acceptable weighted OVR AUC): 0.75
- Expected outcome: `preserved`
- This benchmark tests multiclass discrimination on a moderately sized, well-structured dataset.

### Graphene Oxide Bulk

- 1617 samples, 462 structural chemistry descriptors, regression target: Formation_energy (eV), real NaN values
- Algorithm: Lasso regression, task: regression
- Floor score (minimum acceptable neg-RMSE): -8.0 (RMSE < 8 eV)
- Expected outcome: `preserved` or `improved`
- This benchmark tests regression under high feature dimensionality and domain-specific sparse descriptors.

### Reading the benchmark output

The scenario report prints:

1. **Feature selection comparison:** how many features ROBUST selected vs the full-feature baseline, and the score delta with p-value and outcome label.
2. **Stability-selected features:** the top-15 features by bootstrap frequency. These are the features ROBUST considers robustly informative.
3. **Per-fold scores:** ROBUST score and BL score for each outer fold, with per-fold delta.
4. **Statistical test battery:** the full battery described in Section 9.
5. **Cross-scenario summary table:** all three datasets in one aligned table.

---

## 11. What to report in a paper

When reporting ROBUST results in a scientific paper, include the following:

### Methods section

```
Feature selection and model assessment were performed using RobustModelMaker v0.3
(https://github.com/your_repo). Bootstrap stability selection [Meinshausen & Buhlmann, 2010]
with n_bootstrap=100 bootstrap resamples and a selection threshold of 0.7 was used to
identify a stable feature subset. Model performance was estimated using nested cross-validation
(outer_cv=10, inner_cv=10) with [algorithm] and n_iter=100 hyperparameter search iterations
per fold. All preprocessing (median imputation, standard scaling) was performed inside each
fold on training data only. The random seed was fixed to 42 for full reproducibility.
```

### Results section

Report:
- Number of features selected out of total (e.g. "47 of 590 features, 92% reduction")
- Mean nested CV score +/- std (e.g. "AUC 0.724 +/- 0.031")
- Comparison to full-feature baseline: score delta and statistical outcome (e.g. "comparable to the full-feature baseline, delta = -0.012, Wilcoxon p = 0.41")
- Cutoff and achieved sensitivity/specificity at that cutoff (binary classification only)
- External validation score if available

### What not to over-claim

- Do not report the nested CV score as the model's "accuracy on new data". It is an estimate of expected performance on future data from the same distribution. Performance on a genuinely different population may differ.
- Do not interpret non-significant paired tests as proof of equivalence. They mean the data are consistent with no difference; a larger study might reveal a small difference.
- Do not report selection frequencies as effect sizes or biomarker confidence. A frequency of 0.95 means the feature is a stable predictor in this dataset and model class; it does not quantify the feature's biological importance.

---

## 12. Common misinterpretations

### "The model selected feature X, so X is the most important predictor"

ROBUST selects features based on stability across bootstrap samples, not on raw importance magnitude. A feature with frequency 1.0 may have a smaller effect size than one with frequency 0.6 that is sometimes swamped by correlated features. Use permutation importance or SHAP values (on the selected features) to rank by effect magnitude after selection.

### "Features not selected are irrelevant"

Non-selected features may be correlated with selected ones (and thus redundant rather than uninformative), or may only be predictive in interaction with other features (which stability selection does not capture). Non-selection is evidence of instability or redundancy, not evidence of irrelevance.

### "The nested CV score is the model's performance on these data"

The nested CV score is an out-of-fold estimate. The final model is fit on all training data, so its in-sample performance will be higher. The nested CV score is an estimate of performance on future data. Do not compare it directly to in-sample scores from other methods.

### "A higher stability threshold always gives a better model"

A higher threshold produces fewer, more consistently selected features. If the threshold is too high for your dataset size or feature signal strength, it may exclude genuinely predictive features, reducing performance. The right threshold is dataset-dependent. If `preserved` results are obtained at 0.5 and 0.7, the higher threshold is preferable for parsimony. If results are `degraded` at 0.7 but `preserved` at 0.5, the lower threshold is the working point.

### "p >= 0.05 means the methods are identical"

Non-significance means the data are insufficient to distinguish the methods at alpha = 0.05. With 5 outer folds, the paired tests have very limited power to detect differences smaller than about 0.5 standard deviations. Use the bootstrap confidence interval for the mean difference to understand the range of practically plausible differences.

### "The efficiency gain ratio is a measure of information compression"

The score-per-feature ratio compares the ratio of performance to feature count. It is a useful practical metric when features have acquisition costs, but it conflates two different quantities (performance scale and feature count) in a way that depends on the scoring metric's scale. Treat it as a summary heuristic, not a rigorous information-theoretic measure.

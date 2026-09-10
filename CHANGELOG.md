# Changelog

All notable changes to RobustModelMaker are documented here.

---

## v0.3.3 (2026-09-11)

Maintenance and documentation release. No change to the public API, and no change to
any numerical result. Every pre-existing test still passes unchanged: 102 unit, 30
reproducibility and 28 performance tests.

### Removed

- 262 lines of unreachable code in `RobustModelMaker.py`. The file rebinds several names
  near the end of the module (the "v0.3 compatibility and reporting/performance patches"
  section), so earlier definitions of the same names were shadowed and could never run.
  Removed: the v0.2 bodies of `get_algorithm_config` and `stability_selection`, the
  class-body `RobustModelMaker.__init__` and `.fit`, `PipelineResult.permutation_importance`,
  the first `print_pipeline_results`, and `_stratified_or_random_subsample` (reachable only
  from the shadowed `stability_selection`). Each binding was confirmed at runtime via
  `__code__.co_firstlineno` before removal. The first `run_pipeline` was kept, because
  `_ROBUST_ORIGINAL_RUN_PIPELINE` captures it before the wrapper shadows the name.

### Added

- `tests/edge_case_test_suite.py`: 75 tests covering input-validation failures, defensive
  fallbacks, verbose output paths, and the private helpers that only fire on awkward data.
- `pytest.ini`: registers `*_test_suite.py` as a discovery pattern. Without it, `pytest`
  and `pytest tests/` silently collected nothing, because the suite filenames match
  neither of pytest's defaults (`test_*.py`, `*_test.py`).
- Statement coverage of `RobustModelMaker.py` is now 100 per cent (835 statements),
  measured across the unit, edge-case and reproducibility suites.

### Fixed

- `tools/threshold_optimizer.py` could not find the library when it was installed from
  PyPI. `_import_robust_module()` tried the file-path candidates and then
  `import RobustModelMaker`, but the installed distribution exposes the module as
  `robustmodelmaker`, so the import failed with a misleading "Cannot locate
  RobustModelMaker.py" error. A lowercase package fallback was added.

### Documentation

Corrections found by auditing every claim, parameter, default and code example in the
README and the three guides against the code. Examples that raised on copy-paste:

- User Guide: the reload example opened `{prefix}_pipeline_result.pkl`; the file written
  is `{prefix}_result.pkl`.
- User Guide and PyPI README: `save_results(output_prefix=...)` raised `TypeError`. Both
  `save()` and `save_results()` take `prefix`; only the constructor and `run_pipeline()`
  take `output_prefix`. The distinction is now stated explicitly.
- User Guide: `groups`, `X_validation` and `y_validation` were listed as constructor
  parameters. They belong to `.fit()` and `run_pipeline()` only.
- User Guide: the `.fit()` parameter table listed `feature_names` last when it is the
  third positional parameter, so `fit(X, y, groups)` silently bound groups to
  `feature_names`. The table is now in signature order, with a worked warning.
- Interpretation Guide: `stability_result.plot_feature_stability()` raised
  `AttributeError`; the method is on `PipelineResult` and `RobustModelMaker`.
- Implementation Guide: described a `scoring` argument to `nested_cross_validation`,
  which has no such parameter.
- User Guide: the SHAP example used `TreeExplainer` on the guide's own elastic-net
  example. It now selects the explainer from the `"algorithm"` key.

Factual corrections:

- `rf` uses `class_weight="balanced_subsample"`, not `"balanced"` (README, PyPI README,
  User Guide).
- `rf` search space is `Randint(20, 100)` and `Randint(2, 12)`, not `(100, 500)` and
  `(2, 20)`.
- `xgb` does not search `reg_alpha` or `reg_lambda`.
- `rdg` classification searches `C ~ LogUniform(1e-3, 1e2)`, not `1e-4`.
- `svm` regression also searches `epsilon`.
- Multiclass linear importance is `mean(|coef_|, axis=0)`, not `max`.
- `rdg`, `las`, `log` and `svm` classification all set `class_weight="balanced"`; only
  `eln` and `mlp` leave it unset.
- `StratifiedKFold` is replaced by `GroupKFold` for every task type when `groups` is given.
- `preprocess="auto"` scales for `eln` only; `preprocess="none"` still imputes.
- The class-count validation rule is `max(2, min(outer_cv, inner_cv))`, and is skipped
  under grouped CV.
- `determine_cutoff` does not maximise sensitivity; it takes a quantile of control scores
  and reports the sensitivity achieved there.
- The benchmark configuration quoted in the README and both guides was wrong in four of
  five values. It is `outer_cv=10, inner_cv=10, n_bootstrap=100, n_iter=100,
  cutoff_n_bootstrap=500`, with a base `stability_threshold` of 0.75 and per-dataset
  overrides of 0.60 (SECOM) and 0.80 (Urban Land Cover).
- The performance suite uses flat 90-second and 750 MB guards on 90 to 96 sample
  datasets, not a per-sample budget on 120 samples, and asserts only under
  `ROBUST_PERF_STRICT=1`.
- `selected_in_n_folds` has a maximum of `outer_cv * repeated_outer_cv`, not `outer_cv`.
- The benchmark emits `preserved`, `sig. better *` and `sig. worse *`; there is no
  `degraded` outcome.
- "Save/load" was listed as a capability. There is no load API; use `pickle.load` on the
  saved `_result.pkl`.

Newly documented behaviour that was previously silent:

- Calibration changes class-label predictions, not only probabilities, because `predict()`
  thresholds or argmaxes the calibrated scores.
- Calibration is silently skipped under grouped CV, with a `UserWarning`.
- `stability_selection()` returns the top three features by frequency when nothing clears
  `stability_threshold`, so a raised threshold never yields an empty feature set.
- `stability_selection()` builds its own preprocessor with `preprocess="auto"` hardcoded,
  so a user's `preprocess="standard"` does not reach the resamples.
- The draws in stability selection are subsamples taken **without** replacement, not
  bootstrap resamples. Only `determine_cutoff` draws with replacement. The Methods
  boilerplate in the Interpretation Guide has been corrected accordingly.
- `set_global_seed` sets `PYTHONHASHSEED` only when absent, and that has no effect on the
  running interpreter.
- Resample seed streams overlap between neighbouring outer folds. Determinism is
  unaffected; independence was overstated.

Also added: an "Other methods worth knowing" section to the User Guide covering
`plot_feature_stability`, the maker-level reporting helpers, the `mean_auc`/`std_auc`
aliases, `PermutationImportanceResult.importances`, and `stability_selection`'s
`sample_fraction`. Installation now documents `pip install robustmodelmaker` alongside
the single-file copy, and lists `matplotlib` and `shap` as optional. The repository
structure and test-running sections of the README were refreshed. The citation is now
the arXiv BibTeX entry.

---

## v0.3.2 (2026-05-20)

Documentation-only release. No library code changes; `RobustModelMaker.py` is byte-identical to v0.3.1.

### Changed

- README, pypi_staging README, and Interpretation Guide refreshed with current benchmark results from `benchmarks/Benchmark_Suite.ipynb`: SECOM 301/590 features (49.0% reduction, AUC 0.6835 vs 0.6814, paired Wilcoxon p=0.770, preserved); Urban Land Cover 66/147 features (55.1% reduction, AUC 0.9849 vs 0.9827, p=0.432, preserved); Graphene Oxide Bulk 150/309 features (51.5% reduction, RMSE 0.0343 vs 0.0266 eV, p=0.193, preserved). Previous tables overstated reduction percentages (e.g. ~92% for SECOM) and used incorrect train sizes and feature counts.
- Interpretation Guide outcome labels corrected from `improved *` / `degraded *` to `sig. better *` / `sig. worse *` to match the actual benchmark console output.
- Implementation Guide Graphene Oxide reference updated from 1617 x 462 to 1617 x 309 (post-cleanup column count, after dropping all-NaN and constant columns).
- Interpretation Guide section 10 expanded with a cross-scenario summary table and per-dataset observed-result paragraphs (deltas, p-values, supporting statistics).

---

## v0.3.1 (2026-05-18)

### Fixed

- `CutoffResult.bootstrap_cutoffs` property added as an alias for `cutoff_distribution`; previously accessing this attribute raised `AttributeError`
- `Algorithm` type literal corrected to include all nine supported algorithms (`"eln"`, `"rdg"`, `"las"`, `"log"`, `"svm"`, `"rf"`, `"xgb"`, `"mlp"`, `"lin"`); the previous definition listed only three
- Removed unreachable `y.ndim != 1` guard from `_prepare_y` (always `False` after `ravel()`)
- Removed unused `_ROBUST_ORIGINAL_INIT` module-level variable

### Changed

- All three benchmark datasets (SECOM, Urban Land Cover, Graphene Oxide) now use Random Forest (`rf`) for both the ROBUST run and the full-feature baseline, isolating the effect of bootstrap stability selection from algorithm differences
- Documentation corrections in Implementation Guide: `las` solver and default `C`; `svm` estimator class (`SVC` with `kernel="linear"`, not `LinearSVC`); `mlp` importance method (first-layer weight magnitudes, not permutation importance); preprocessing notes for `rdg`, `las`, `log`, `svm`, `mlp`, and `lin` under default `preprocess="auto"`

---

## v0.3 (2026)

### Added

- Binary classification support with ROC-AUC scoring and bootstrap specificity-targeted cutoff determination (`determine_cutoff`)
- Multiclass classification support with weighted one-vs-rest ROC-AUC scoring
- Regression support with negative RMSE scoring
- External validation: pass `X_validation` and `y_validation` at fit time, or call `evaluate_verification()` post-fit
- Probability calibration: Platt scaling (`calibration="sigmoid"`) and isotonic regression (`calibration="isotonic"`)
- Permutation importance: `permutation_importance()` on any dataset, returns `PermutationImportanceResult` with `.summary()` DataFrame
- SHAP-ready export: `export_shap_ready()` returns the fitted model and processed selected-feature matrix
- Grouped cross-validation: pass `groups=` to enforce group integrity across folds (GroupKFold)
- Repeated nested CV: `repeated_outer_cv > 1` repeats the outer CV with different seeds and averages predictions
- Feature stability plot: `plot_feature_stability(top_n=30)` returns a matplotlib axis
- Results tables: `results_tables()` returns a dict of DataFrames suitable for export or inspection
- Save/load: `save_results()` writes JSON metadata, CSV tables, and a pickle of the full result
- Extended algorithm support: `rdg` (Ridge/L2-logistic), `las` (Lasso/L1-logistic), `log` (logistic), `svm` (LinearSVM), `mlp` (MLP), `lin` (OLS) added alongside existing `eln`, `rf`, `xgb`
- `RobustModelMaker` class with scikit-learn-style `.fit()` / `.predict()` / `.predict_proba()` API
- `task_type="auto"` inference from target variable characteristics
- `preserve_nans=False` mode with data-driven missingness threshold optimisation (`_smart_drop_nans`)
- Per-fold feature stability table in `NestedCVResult.feature_stability`
- `set_global_seed()` utility for environment-level determinism
- Full test suite: 96+ unit tests, performance budget tests, 30 reproducibility tests
- Benchmark suite: three real scientific datasets (SECOM, Urban Land Cover, Graphene Oxide) with 25-test statistical battery and BenchMake archetypal splits

### Changed

- Public entry points from v0.2 (`run_pipeline`, `RobustModelMaker`) retained with backward-compatible signatures
- Preprocessing is now a `sklearn.pipeline.Pipeline` (imputer + optional scaler) fitted strictly inside each fold
- Feature importance extraction unified across algorithms: `|coef_|` for linear models, `feature_importances_` for trees, permutation fallback for models without native importance
- `NestedCVResult` extended with `selected_features_per_fold`, `feature_stability`, `repeats`, `task_type`
- `PipelineResult` extended with `algorithm`, `task_type`, `class_names`, `label_mapping`, `calibration`, `validation_result`, `nan_dropping_col_mask`

---

## v0.2

- Initial public release
- Binary classification with elastic net stability selection and nested CV
- Bootstrap cutoff determination at target specificity
- Basic save/load functionality

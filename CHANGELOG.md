# Changelog

All notable changes to RobustModelMaker are documented here.

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

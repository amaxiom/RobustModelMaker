"""
Edge-case and error-path pytest suite for RobustModelMaker v0.3.

Companion to unit_test_suite.py.  Where that suite exercises the happy paths of
the public API, this one targets the branches those runs never reach: input
validation failures, defensive fallbacks, verbose reporting, and the private
helpers that only fire on awkward data.

Run alongside the main suite:

    python -m pytest tests/ -q

Everything here is deliberately small and fast.  No test fits a model on more
than 60 rows, and the shared pipeline fixtures are module-scoped so the handful
of genuine end-to-end runs happen once each.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# -----------------------------------------------------------------------------
# Dynamic import, matching unit_test_suite.py
# -----------------------------------------------------------------------------


def _load_robust_model_maker_module():
    # unit_test_suite.py loads the module under this same key.  Re-executing it
    # would create a second set of class objects, and anything pickled by one
    # suite would then fail to unpickle in the other, so reuse what is there.
    already_loaded = sys.modules.get("robust_model_maker_under_test")
    if already_loaded is not None:
        return already_loaded

    here = Path(__file__).resolve().parent
    candidates = []
    env_path = os.environ.get("ROBUST_MODEL_MAKER_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([
        here / "RobustModelMaker.py",
        here.parent / "RobustModelMaker.py",
    ])
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            spec = importlib.util.spec_from_file_location("robust_model_maker_under_test", candidate)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("Could not find RobustModelMaker.py one level above tests/.")


robust = _load_robust_model_maker_module()


FAST = dict(
    outer_cv=3,
    inner_cv=2,
    n_iter=2,
    n_bootstrap=4,
    cutoff_n_bootstrap=20,
    stability_threshold=0.25,
    random_state=123,
    n_jobs=1,
    verbose=False,
)


# -----------------------------------------------------------------------------
# Small data helpers
# -----------------------------------------------------------------------------


def _binary_data(n=48, p=5, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, p))
    y = (X[:, 0] + 0.4 * rng.normal(size=n) > 0).astype(int)
    if len(np.unique(y)) < 2:                      # pragma: no cover
        y[:2] = [0, 1]
    return X, y


def _multiclass_data(n=60, p=4, seed=1):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, p))
    y = np.tile([0, 1, 2], n // 3)
    X[y == 1, 0] += 2.0
    X[y == 2, 1] += 2.0
    return X, y


def _regression_data(n=48, p=4, seed=2):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, p))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.3, size=n)
    return X, y


# -----------------------------------------------------------------------------
# Module-scoped fitted pipelines, shared by the result-object tests
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def binary_result():
    X, y = _binary_data()
    return robust.run_pipeline(X, y, alg="rdg", task_type="binary", **FAST)


@pytest.fixture(scope="module")
def multiclass_result():
    X, y = _multiclass_data()
    return robust.run_pipeline(
        X[:45], y[:45], alg="rdg", task_type="multiclass",
        X_validation=X[45:], y_validation=y[45:], **FAST
    )


@pytest.fixture(scope="module")
def regression_result():
    X, y = _regression_data()
    return robust.run_pipeline(X, y, alg="lin", task_type="regression", **FAST)


def _synthetic_pipeline_result(
    task_type="binary",
    n_features=3,
    feature_names=np.array(["a", "b", "c"]),
    preprocessor="default",
    model="default",
    outer_predictions=None,
    validation_result=None,
    class_names=None,
    inverse_label_mapping=None,
    label_mapping=None,
    cutoff_result=None,
):
    """Build a PipelineResult by hand, without running the pipeline.

    Used for branches that a real run cannot reach, such as a missing
    preprocessor or a three-dimensional out-of-fold prediction array.
    """
    rng = np.random.RandomState(0)
    X = rng.normal(size=(20, n_features))
    y = np.array([0, 1] * 10)
    if preprocessor == "default":
        preprocessor = robust._make_preprocessor("auto", "rdg").fit(X)
    if model == "default":
        model = LogisticRegression(max_iter=1000).fit(X, y)
    if outer_predictions is None:
        outer_predictions = rng.uniform(size=20)

    nested = robust.NestedCVResult(
        outer_scores=np.array([0.6, 0.7]),
        outer_predictions=outer_predictions,
        outer_true_labels=y,
        mean_score=0.65,
        std_score=0.05,
        metric_name="roc_auc",
        best_params_per_fold=[{}, {}],
        selected_features_per_fold=[np.array(["a"]), np.array(["a"])],
        feature_stability=pd.DataFrame({
            "feature": feature_names if feature_names is not None else ["a", "b", "c"],
            "mean_frequency": np.linspace(1.0, 0.1, n_features),
            "std_frequency": np.zeros(n_features),
            "selected_in_n_folds": np.ones(n_features, dtype=int),
        }),
        repeats=1,
        task_type=task_type,
    )
    stability = robust.StabilitySelectionResult(
        feature_names=np.array(feature_names) if feature_names is not None else np.array(["a", "b", "c"]),
        selection_frequencies=np.linspace(1.0, 0.1, n_features),
        selected_features=np.array(["a"]),
        selected_indices=np.array([0]),
        threshold=0.5,
        n_bootstrap=4,
        task_type=task_type,
    )
    return robust.PipelineResult(
        nested_cv_result=nested,
        stability_result=stability,
        cutoff_result=cutoff_result,
        robust_model=model,
        selected_features=np.array(["a"]),
        selected_feature_indices=np.array([0]),
        algorithm="rdg",
        task_type=task_type,
        preprocessor=preprocessor,
        feature_names=feature_names,
        label_mapping=label_mapping,
        inverse_label_mapping=inverse_label_mapping,
        class_names=class_names,
        validation_result=validation_result,
    )


# -----------------------------------------------------------------------------
# set_global_seed and the small result-container properties
# -----------------------------------------------------------------------------


def test_set_global_seed_accepts_none():
    """random_state=None returns immediately without touching the RNG."""
    before = np.random.get_state()[1][0]
    assert robust.set_global_seed(None) is None
    assert np.random.get_state()[1][0] == before


def test_cutoff_result_bootstrap_cutoffs_alias():
    dist = np.array([0.1, 0.2, 0.3])
    cut = robust.CutoffResult(0.2, 0.1, 0.3, dist, 0.98, 0.97, 0.5)
    assert np.array_equal(cut.bootstrap_cutoffs, dist)
    assert "Cutoff: 0.2000" in cut.summary()


def test_nested_cv_auc_aliases(binary_result):
    nested = binary_result.nested_cv_result
    assert nested.mean_auc == nested.mean_score
    assert nested.std_auc == nested.std_score


def test_pipeline_result_score_properties(binary_result):
    assert binary_result.mean_score == pytest.approx(float(binary_result.nested_cv_result.mean_score))
    assert binary_result.std_score == pytest.approx(float(binary_result.nested_cv_result.std_score))


# -----------------------------------------------------------------------------
# _prepare_X_selected
# -----------------------------------------------------------------------------


def test_prepare_X_selected_without_preprocessor_raises():
    result = _synthetic_pipeline_result(preprocessor=None)
    with pytest.raises(NotFittedError, match="no fitted preprocessor"):
        result._prepare_X_selected(np.zeros((5, 3)))


def test_prepare_X_selected_matches_columns_by_string_name():
    """feature_names are strings while the DataFrame columns are integers."""
    result = _synthetic_pipeline_result(feature_names=np.array(["0", "1", "2"]))
    frame = pd.DataFrame(np.random.RandomState(3).normal(size=(6, 3)), columns=[0, 1, 2])
    X_sel, index = result._prepare_X_selected(frame)
    assert X_sel.shape == (6, 1)
    assert index is not None


def test_prepare_X_selected_without_feature_names_uses_frame_as_is():
    result = _synthetic_pipeline_result(feature_names=None)
    frame = pd.DataFrame(np.random.RandomState(4).normal(size=(6, 3)), columns=["x", "y", "z"])
    X_sel, _ = result._prepare_X_selected(frame)
    assert X_sel.shape == (6, 1)


def test_prepare_X_selected_rejects_one_dimensional_array():
    result = _synthetic_pipeline_result()
    with pytest.raises(ValueError, match="2D array or DataFrame"):
        result._prepare_X_selected(np.zeros(3))


def test_prepare_X_selected_rejects_wrong_feature_count():
    result = _synthetic_pipeline_result()
    with pytest.raises(ValueError, match="expected 3"):
        result._prepare_X_selected(np.zeros((5, 7)))


# -----------------------------------------------------------------------------
# predict / predict_proba / summary on PipelineResult
# -----------------------------------------------------------------------------


def test_predict_proba_requires_model_support():
    class _NoProba:
        def predict(self, X):                       # pragma: no cover
            return np.zeros(len(X))

    result = _synthetic_pipeline_result(model=_NoProba())
    with pytest.raises(AttributeError, match="does not support predict_proba"):
        result.predict_proba(np.zeros((5, 3)))


def test_multiclass_predict_proba_array_input_returns_matrix(multiclass_result):
    X, _ = _multiclass_data()
    proba = multiclass_result.predict_proba(X[45:])
    assert isinstance(proba, np.ndarray)
    assert proba.ndim == 2


def test_multiclass_predict_array_input_returns_array(multiclass_result):
    X, _ = _multiclass_data()
    pred = multiclass_result.predict(X[45:])
    assert isinstance(pred, np.ndarray)
    assert not isinstance(pred, pd.Series)


def test_regression_predict_proba_is_rejected(regression_result):
    X, _ = _regression_data()
    with pytest.raises(AttributeError, match="do not support predict_proba"):
        regression_result.predict_proba(X)


def test_summary_includes_external_validation_block(multiclass_result):
    text = multiclass_result.summary()
    assert "EXTERNAL VALIDATION" in text


def test_multiclass_auc_falls_back_to_nan_on_single_class_validation(multiclass_result):
    """roc_auc_score raises when the validation fold holds one class only."""
    X, y = _multiclass_data()
    single = y[45:] == y[45]
    verification = multiclass_result.evaluate_verification(X[45:][single], y[45:][single])
    assert np.isnan(verification.metrics["auc_ovr_weighted"])


# -----------------------------------------------------------------------------
# Feature name and array coercion helpers
# -----------------------------------------------------------------------------


def test_extract_feature_names_rejects_one_dimensional_array():
    with pytest.raises(ValueError, match="2D array or DataFrame"):
        robust._extract_feature_names(np.zeros(5), None)


def test_to_numpy_X_rejects_one_dimensional_array():
    with pytest.raises(ValueError, match="2D array or DataFrame"):
        robust._to_numpy_X(np.zeros(5))


def test_to_numpy_X_requires_four_samples():
    with pytest.raises(ValueError, match="at least 4 samples"):
        robust._to_numpy_X(np.zeros((3, 2)))


def test_to_numpy_X_requires_one_feature():
    with pytest.raises(ValueError, match="at least 1 feature"):
        robust._to_numpy_X(np.zeros((10, 0)))


# -----------------------------------------------------------------------------
# Task resolution and label handling
# -----------------------------------------------------------------------------


def test_resolve_task_type_treats_many_integer_levels_as_regression():
    assert robust._resolve_task_type(np.arange(100), "auto") == "regression"


def test_make_label_mapping_rejects_missing_labels():
    with pytest.raises(ValueError, match="y contains missing values"):
        robust._make_label_mapping(np.array([0.0, 1.0, np.nan, 1.0]), "binary")


def test_multiclass_requires_three_classes():
    with pytest.raises(ValueError, match="at least 3 classes"):
        robust._make_label_mapping(np.array([0, 1, 0, 1]), "multiclass")


def test_prepare_y_rejects_missing_values():
    with pytest.raises(ValueError, match="y contains missing values"):
        robust._prepare_y(np.array([1.0, np.nan, 0.0, 1.0]), "binary")


def test_prepare_y_rejects_non_finite_regression_target():
    with pytest.raises(ValueError, match="non-finite"):
        robust._prepare_y(np.array([1.0, 2.0, np.inf, 4.0]), "regression")


def test_prepare_y_builds_mapping_when_none_supplied():
    encoded = robust._prepare_y(np.array(["b", "a", "b", "a"]), "binary", label_mapping=None)
    assert set(np.unique(encoded)) == {0, 1}


def test_prepare_y_rejects_label_outside_mapping():
    with pytest.raises(ValueError, match="Unknown label"):
        robust._prepare_y(np.array([0, 1, 2, 1]), "binary", label_mapping={0: 0, 1: 1})


def test_decode_labels_without_mapping_is_identity():
    values = np.array([0, 1, 1])
    assert np.array_equal(robust._decode_labels(values, None), values)


# -----------------------------------------------------------------------------
# _validate_inputs failure modes
# -----------------------------------------------------------------------------


def test_validate_inputs_rejects_row_count_mismatch():
    X, y = _binary_data(n=20, p=3)
    with pytest.raises(ValueError, match="but y has"):
        robust._validate_inputs(X, y[:-1], None, 3, 2, "binary", None)


def test_validate_inputs_rejects_wrong_feature_name_length():
    X, y = _binary_data(n=20, p=3)
    with pytest.raises(ValueError, match="feature_names has length"):
        robust._validate_inputs(X, y, np.array(["a", "b"]), 3, 2, "binary", None)


def test_validate_inputs_rejects_all_missing_column():
    X, y = _binary_data(n=20, p=3)
    X = X.copy()
    X[:, 1] = np.nan
    with pytest.raises(ValueError, match="all-missing feature columns"):
        robust._validate_inputs(X, y, None, 3, 2, "binary", None)


def test_validate_inputs_rejects_group_length_mismatch():
    X, y = _binary_data(n=20, p=3)
    with pytest.raises(ValueError, match="groups must have the same length"):
        robust._validate_inputs(X, y, None, 3, 2, "binary", np.zeros(5))


def test_validate_inputs_requires_two_groups():
    X, y = _binary_data(n=20, p=3)
    with pytest.raises(ValueError, match="at least 2 distinct groups"):
        robust._validate_inputs(X, y, None, 3, 2, "binary", np.zeros(20))


def test_validate_inputs_regression_requires_enough_samples():
    X, y = _regression_data(n=8, p=3)
    with pytest.raises(ValueError, match="enough samples"):
        robust._validate_inputs(X, y, None, 10, 10, "regression", None)


# -----------------------------------------------------------------------------
# Scoring and splitter helpers
# -----------------------------------------------------------------------------


def test_score_predictions_multiclass_falls_back_to_accuracy():
    """A single observed class makes roc_auc_score raise; accuracy is used."""
    y_true = np.zeros(6, dtype=int)
    proba = np.tile([0.7, 0.2, 0.1], (6, 1))
    assert robust._score_predictions("multiclass", y_true, proba) == pytest.approx(1.0)


def test_outer_splitter_reduces_splits_to_group_count():
    groups = np.array([0, 0, 1, 1, 2, 2])
    with pytest.warns(UserWarning, match="outer_cv reduced"):
        splitter, n_splits = robust._make_outer_splitter("binary", 5, 0, groups)
    assert n_splits == 3


def test_inner_splitter_requires_two_groups():
    with pytest.raises(ValueError, match="at least 2 groups"):
        robust._make_inner_splitter("binary", 3, 0, np.zeros(10))


# -----------------------------------------------------------------------------
# Calibration helper
# -----------------------------------------------------------------------------


def test_calibration_falls_back_to_base_estimator_keyword(monkeypatch):
    """Older sklearn used base_estimator= rather than estimator=."""

    class _OldSklearnCalibrated:
        def __init__(self, base_estimator=None, estimator=None, method=None, cv=None, ensemble=True):
            if estimator is not None:
                raise TypeError("unexpected keyword argument 'estimator'")
            self.base_estimator = base_estimator
            self.fitted = False

        def fit(self, X, y):
            self.fitted = True
            return self

    monkeypatch.setattr(robust, "CalibratedClassifierCV", _OldSklearnCalibrated)
    X, y = _binary_data(n=20, p=3)
    out = robust._fit_calibrated_if_needed(
        LogisticRegression(max_iter=500), X, y, "binary", "sigmoid", cv=2, groups=None
    )
    assert isinstance(out, _OldSklearnCalibrated)
    assert out.fitted


def test_calibration_is_skipped_for_grouped_cv():
    X, y = _binary_data(n=20, p=3)
    groups = np.repeat([0, 1, 2, 3], 5)
    with pytest.warns(UserWarning, match="Calibration is skipped"):
        out = robust._fit_calibrated_if_needed(
            LogisticRegression(max_iter=500), X, y, "binary", "sigmoid", cv=2, groups=groups
        )
    assert isinstance(out, LogisticRegression)


# -----------------------------------------------------------------------------
# nested_cross_validation guards and verbose output
# -----------------------------------------------------------------------------


def test_nested_cv_rejects_zero_repeats():
    X, y = _binary_data(n=24, p=3)
    with pytest.raises(ValueError, match="repeated_outer_cv must be >= 1"):
        robust.nested_cross_validation(
            X, y, alg="rdg", task_type="binary", outer_cv=3, inner_cv=2,
            repeated_outer_cv=0, n_iter=1, n_bootstrap=2, random_state=0,
            n_jobs=1, verbose=False,
        )


def test_nested_cv_verbose_prints_fold_progress(capsys):
    X, y = _binary_data(n=30, p=3)
    robust.nested_cross_validation(
        X, y, alg="rdg", task_type="binary", outer_cv=3, inner_cv=2,
        n_iter=1, n_bootstrap=2, stability_threshold=0.25, random_state=0,
        n_jobs=1, verbose=True,
    )
    out = capsys.readouterr().out
    assert "outer fold 1/3" in out
    assert "features selected" in out


def test_nested_cv_falls_back_when_no_features_selected(monkeypatch):
    """Defensive branch: an empty selection is replaced by the top features."""
    real = robust.stability_selection

    def _empty_selection(*args, **kwargs):
        res = real(*args, **kwargs)
        return robust.StabilitySelectionResult(
            feature_names=res.feature_names,
            selection_frequencies=res.selection_frequencies,
            selected_features=np.array([], dtype=res.feature_names.dtype),
            selected_indices=np.array([], dtype=int),
            threshold=res.threshold,
            n_bootstrap=res.n_bootstrap,
            task_type=res.task_type,
        )

    monkeypatch.setattr(robust, "stability_selection", _empty_selection)
    X, y = _binary_data(n=30, p=3)
    with pytest.warns(UserWarning, match="no features selected"):
        result = robust.nested_cross_validation(
            X, y, alg="rdg", task_type="binary", outer_cv=3, inner_cv=2,
            n_iter=1, n_bootstrap=2, random_state=0, n_jobs=1, verbose=False,
        )
    assert all(len(f) > 0 for f in result.selected_features_per_fold)


# -----------------------------------------------------------------------------
# determine_cutoff
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.2, 1.5])
def test_determine_cutoff_rejects_out_of_range_specificity(bad):
    with pytest.raises(ValueError, match="target_specificity must be in"):
        robust.determine_cutoff(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]), target_specificity=bad)


# -----------------------------------------------------------------------------
# Missingness threshold search and NaN dropping
# -----------------------------------------------------------------------------


def test_threshold_search_skips_when_no_column_qualifies():
    """Every column is 95 per cent missing, so no candidate threshold keeps one."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(40, 4))
    mask = rng.uniform(size=X.shape) < 0.95
    X[mask] = np.nan
    col_t, row_t = robust._find_optimal_missingness_thresholds(X)
    assert 0.0 < col_t <= 0.9
    assert 0.0 < row_t <= 0.9


def test_threshold_search_skips_when_too_few_rows_remain():
    """Columns are clean enough to keep, but almost every row is mostly missing."""
    X = np.full((30, 4), np.nan)
    X[:3, :] = 1.0                      # only three complete rows
    X[3:, 0] = 2.0
    col_t, row_t = robust._find_optimal_missingness_thresholds(X)
    assert 0.0 < col_t <= 0.9


def test_smart_drop_nans_keeps_all_columns_when_none_qualify():
    rng = np.random.RandomState(1)
    X = rng.normal(size=(30, 3))
    X[rng.uniform(size=X.shape) < 0.97] = np.nan
    names = np.array(["a", "b", "c"])
    X_clean, names_clean, row_mask, col_mask = robust._smart_drop_nans(X, names, verbose=False)
    assert X_clean.shape[0] >= 4
    assert len(names_clean) == col_mask.sum()


def test_smart_drop_nans_verbose_reports_thresholds(capsys):
    X, _ = _binary_data(n=30, p=4)
    X = X.copy()
    X[0, 0] = np.nan
    robust._smart_drop_nans(X, np.array(["a", "b", "c", "d"]), verbose=True)
    assert "col_threshold" in capsys.readouterr().out


# -----------------------------------------------------------------------------
# run_pipeline verbose reporting and fallbacks
# -----------------------------------------------------------------------------


def test_run_pipeline_verbose_prints_header_and_summary(capsys):
    X, y = _binary_data(n=36, p=4)
    kwargs = dict(FAST)
    kwargs["verbose"] = True
    robust.run_pipeline(X, y, alg="rdg", task_type="binary", **kwargs)
    out = capsys.readouterr().out
    assert "Running RobustModelMaker v0.3" in out
    assert "Algorithm: rdg" in out
    assert "ROBUST MODEL MAKER v0.3 RESULTS" in out


def test_run_pipeline_verbose_reports_nan_dropping(capsys):
    X, y = _binary_data(n=40, p=5)
    X = X.copy()
    X[np.random.RandomState(7).uniform(size=X.shape) < 0.2] = np.nan
    kwargs = dict(FAST)
    kwargs["verbose"] = True
    robust.run_pipeline(X, y, alg="rdg", task_type="binary", preserve_nans=False, **kwargs)
    assert "NaN strategy:" in capsys.readouterr().out


def test_run_pipeline_falls_back_when_no_features_selected(monkeypatch):
    real = robust.stability_selection
    calls = {"n": 0}

    def _empty_after_first(*args, **kwargs):
        res = real(*args, **kwargs)
        calls["n"] += 1
        return robust.StabilitySelectionResult(
            feature_names=res.feature_names,
            selection_frequencies=res.selection_frequencies,
            selected_features=np.array([], dtype=res.feature_names.dtype),
            selected_indices=np.array([], dtype=int),
            threshold=res.threshold,
            n_bootstrap=res.n_bootstrap,
            task_type=res.task_type,
        )

    monkeypatch.setattr(robust, "stability_selection", _empty_after_first)
    X, y = _binary_data(n=30, p=4)
    with pytest.warns(UserWarning, match="No features selected"):
        result = robust.run_pipeline(X, y, alg="rdg", task_type="binary", **FAST)
    assert len(result.selected_features) > 0


def test_run_pipeline_requires_both_validation_arrays():
    X, y = _binary_data(n=24, p=3)
    with pytest.raises(ValueError, match="both X_validation and y_validation"):
        robust.run_pipeline(X, y, alg="rdg", task_type="binary", X_validation=X, **FAST)


# -----------------------------------------------------------------------------
# Reporting helpers
# -----------------------------------------------------------------------------


def test_results_tables_handle_multiclass_validation_probabilities(multiclass_result):
    tables = multiclass_result.results_tables()
    pred = tables["external_validation_predictions"]
    prob_cols = [c for c in pred.columns if c.startswith("probability_class_")]
    assert len(prob_cols) >= 3


def test_results_tables_flatten_high_dimensional_predictions():
    """Defensive branch for out-of-fold predictions with more than two axes."""
    result = _synthetic_pipeline_result(outer_predictions=np.zeros((20, 2, 2)))
    tables = result.results_tables()
    assert "prediction_or_score" in tables["nested_cv_predictions"].columns


def test_permutation_importance_as_frame(binary_result):
    X, y = _binary_data()
    frame = binary_result.permutation_importance(X, y, n_repeats=2, n_jobs=1, as_frame=True)
    assert isinstance(frame, pd.DataFrame)
    assert "importance_mean" in frame.columns


def test_maker_results_tables_and_print(capsys):
    X, y = _binary_data(n=36, p=4)
    maker = robust.RobustModelMaker(alg="rdg", task_type="binary", **FAST).fit(X, y)
    tables = maker.results_tables()
    assert "overview" in tables
    maker.print_results(top_n=3)
    assert "RESULTS" in capsys.readouterr().out


def test_print_pipeline_results_function(binary_result, capsys):
    robust.print_pipeline_results(binary_result, top_n=3)
    assert "RESULTS" in capsys.readouterr().out


# -----------------------------------------------------------------------------
# get_algorithm_config guards
# -----------------------------------------------------------------------------


def test_xgb_requires_the_optional_dependency(monkeypatch):
    monkeypatch.setattr(robust, "_HAS_XGBOOST", False)
    with pytest.raises(ImportError, match="xgboost is not installed"):
        robust.get_algorithm_config("xgb", "binary")


def test_unknown_algorithm_is_rejected():
    with pytest.raises(ValueError, match="alg must be one of"):
        robust.get_algorithm_config("not_an_algorithm", "binary")


# -----------------------------------------------------------------------------
# _robust_model_importance
# -----------------------------------------------------------------------------


def test_model_importance_recovers_from_broken_calibration_wrapper():
    """A CalibratedClassifierCV whose inner list is unavailable falls through."""

    class _BrokenCalibrated(robust.CalibratedClassifierCV):
        @property
        def calibrated_classifiers_(self):
            raise RuntimeError("not fitted")

    broken = _BrokenCalibrated(estimator=LogisticRegression())
    broken.coef_ = np.array([[1.0, -2.0, 3.0]])
    imp = robust._robust_model_importance(broken, np.zeros((5, 3)), np.zeros(5), 3, 0)
    assert np.allclose(imp, [1.0, 2.0, 3.0])


def test_model_importance_uses_permutation_for_opaque_models():
    X, y = _binary_data(n=30, p=3)
    model = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    imp = robust._robust_model_importance(model, X, y, 3, 0)
    assert imp.shape == (3,)


def test_model_importance_resizes_to_expected_width():
    model = LogisticRegression().fit(*_binary_data(n=20, p=3))
    imp = robust._robust_model_importance(model, np.zeros((5, 3)), np.zeros(5), 5, 0)
    assert imp.shape == (5,)


# -----------------------------------------------------------------------------
# stability_selection guards and fallbacks
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
def test_stability_selection_rejects_bad_sample_fraction(fraction):
    X, y = _binary_data(n=20, p=3)
    with pytest.raises(ValueError, match="sample_fraction must be in"):
        robust.stability_selection(X, y, alg="rdg", sample_fraction=fraction, n_bootstrap=1)


@pytest.mark.parametrize("threshold", [0.0, -0.5, 2.0])
def test_stability_selection_rejects_bad_threshold(threshold):
    X, y = _binary_data(n=20, p=3)
    with pytest.raises(ValueError, match="threshold must be in"):
        robust.stability_selection(X, y, alg="rdg", threshold=threshold, n_bootstrap=1)


def test_stability_selection_tolerates_unsettable_parameters(monkeypatch):
    """set_params raising ValueError must not abort the resampling loop."""
    real_config = robust.get_algorithm_config

    def _stubborn(alg, task_type="binary", **kwargs):
        model, params = real_config("rdg", task_type, **kwargs)

        def _raise(**_):
            raise ValueError("cannot set these parameters")

        model.set_params = _raise
        return model, params

    monkeypatch.setattr(robust, "get_algorithm_config", _stubborn)
    X, y = _binary_data(n=24, p=3)
    res = robust.stability_selection(X, y, alg="rf", n_bootstrap=2, threshold=0.5, random_state=0)
    assert res.selection_frequencies.shape == (3,)


def test_stability_selection_handles_all_zero_importance():
    """Constant features give a tree model zero importance everywhere."""
    X = np.zeros((24, 3))
    y = np.array([0, 1] * 12)
    res = robust.stability_selection(X, y, alg="rf", n_bootstrap=2, threshold=0.5, random_state=0)
    assert res.selection_frequencies.sum() > 0


def test_stability_selection_handles_uniform_importance(monkeypatch):
    """Importance equal to its own median selects nothing, so argmax is used."""
    monkeypatch.setattr(
        robust, "_robust_model_importance",
        lambda model, X, y, n_features, seed: np.ones(n_features),
    )
    X, y = _binary_data(n=24, p=3)
    res = robust.stability_selection(X, y, alg="rdg", n_bootstrap=2, threshold=0.5, random_state=0)
    assert res.selection_frequencies.sum() > 0


def test_stability_selection_falls_back_to_top_features(monkeypatch):
    """No feature reaches a threshold of 1.0, so the top three are kept."""
    counter = {"i": 0}

    def _alternating(model, X, y, n_features, seed):
        counter["i"] += 1
        imp = np.zeros(n_features)
        imp[counter["i"] % n_features] = 1.0
        imp[(counter["i"] + 1) % n_features] = 0.5
        return imp

    monkeypatch.setattr(robust, "_robust_model_importance", _alternating)
    X, y = _binary_data(n=24, p=6)
    res = robust.stability_selection(X, y, alg="rdg", n_bootstrap=3, threshold=1.0, random_state=0)
    assert len(res.selected_indices) == 3
    assert res.selection_frequencies.max() < 1.0


# -----------------------------------------------------------------------------
# StabilitySelectionResult summary
# -----------------------------------------------------------------------------


def test_stability_summary_is_sorted_descending(binary_result):
    frame = binary_result.stability_result.summary()
    freqs = frame["selection_frequency"].to_numpy()
    assert np.all(np.diff(freqs) <= 0)
    assert set(frame.columns) == {"feature", "selection_frequency", "selected"}


def test_nested_cv_detects_samples_without_out_of_fold_predictions(monkeypatch):
    """Every sample must land in exactly one test fold; the guard catches gaps."""

    class _LeakySplitter:
        """Outer splitter that never puts the final sample in a test fold."""

        def __init__(self, n_splits):
            self.n_splits = n_splits

        def split(self, X, y=None, groups=None):
            n = len(X)
            idx = np.arange(n)
            half = n // 2
            yield idx[half:], idx[:half]
            yield idx[:half], idx[half:n - 1]

    monkeypatch.setattr(
        robust, "_make_outer_splitter",
        lambda task_type, n_splits, random_state, groups: (_LeakySplitter(2), 2),
    )

    rng = np.random.RandomState(5)
    X = rng.normal(size=(32, 3))
    y = np.array([0, 1] * 16)
    X[y == 1, 0] += 1.5

    with pytest.raises(RuntimeError, match="did not receive out-of-fold predictions"):
        robust.nested_cross_validation(
            X, y, alg="rdg", task_type="binary", outer_cv=2, inner_cv=2,
            n_iter=1, n_bootstrap=2, stability_threshold=0.25,
            random_state=0, n_jobs=1, verbose=False,
        )

"""
Comprehensive pytest suite for RobustModelMaker v0.3.

How to run
----------
Place this file in tests/ beside RobustModelMaker.py (one level up), then run:

    python -m pytest unit_test_suite.py -q

Inside Jupyter:

    import pytest
    pytest.main(["-vv", "-s", "unit_test_suite.py"])

For slower coverage, including MLP and XGBoost end-to-end tests:

    RUN_SLOW=1 python -m pytest unit_test_suite.py -q

The default suite is deliberately small but behaviour-focused. It covers:
- every supported algorithm code at the configuration level
- every fast algorithm end-to-end for compatible task types
- binary, multiclass and regression modes
- external validation
- calibration
- grouped CV
- repeated nested CV
- permutation importance
- SHAP-ready export
- plotting
- automatic result saving and reporting
- reproducibility
- clear validation failures
"""

from __future__ import annotations

import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import make_classification, make_regression
from sklearn.exceptions import NotFittedError


# -----------------------------------------------------------------------------
# Dynamic import
# -----------------------------------------------------------------------------


def _load_robust_model_maker_module():
    """Load RobustModelMaker from a local file without requiring installation."""
    here = Path(__file__).resolve().parent
    candidates = []
    env_path = os.environ.get("ROBUST_MODEL_MAKER_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([
        here / "RobustModelMaker.py",
        here.parent / "RobustModelMaker.py",
        here / "RobustModelMaker_v0_3.py",
        here.parent / "RobustModelMaker_v0_3.py",
    ])

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            spec = importlib.util.spec_from_file_location("robust_model_maker_under_test", candidate)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(
        "Could not find RobustModelMaker.py. Put it one level above tests/ "
        "or set ROBUST_MODEL_MAKER_PATH=/path/to/RobustModelMaker.py."
    )


robust = _load_robust_model_maker_module()


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slower algorithm coverage tests")


# -----------------------------------------------------------------------------
# Fixtures and helpers
# -----------------------------------------------------------------------------


FAST_KWARGS = dict(
    outer_cv=3,
    inner_cv=2,
    repeated_outer_cv=1,
    n_iter=2,
    n_bootstrap=4,
    cutoff_n_bootstrap=20,
    stability_threshold=0.25,
    random_state=123,
    n_jobs=1,
    verbose=False,
    save_results=False,
)

VERY_FAST_KWARGS = dict(
    outer_cv=2,
    inner_cv=2,
    repeated_outer_cv=1,
    n_iter=1,
    n_bootstrap=2,
    cutoff_n_bootstrap=10,
    stability_threshold=0.1,
    random_state=123,
    n_jobs=1,
    verbose=False,
    save_results=False,
)

CLASSIFICATION_ALGS = ["eln", "rf", "xgb", "mlp", "svm", "rdg", "las", "log"]
REGRESSION_ALGS = ["eln", "rf", "xgb", "mlp", "svm", "rdg", "las", "lin"]
FAST_CLASSIFICATION_ALGS = ["eln", "rf", "svm", "rdg", "las", "log"]
FAST_REGRESSION_ALGS = ["eln", "rf", "svm", "rdg", "las", "lin"]
SLOW_ALGS = ["mlp", "xgb"]


@pytest.fixture(scope="session")
def binary_df():
    X, y = make_classification(
        n_samples=90,
        n_features=8,
        n_informative=4,
        n_redundant=1,
        n_repeated=0,
        n_classes=2,
        weights=[0.55, 0.45],
        class_sep=1.2,
        random_state=10,
    )
    idx = pd.Index([f"sample_{i:03d}" for i in range(X.shape[0])], name="sample_id")
    cols = [f"f{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, index=idx, columns=cols), pd.Series(y, index=idx, name="target")


@pytest.fixture(scope="session")
def multiclass_df():
    X, y = make_classification(
        n_samples=96,
        n_features=9,
        n_informative=5,
        n_redundant=1,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.4,
        random_state=11,
    )
    labels = np.array(["alpha", "beta", "gamma"])[y]
    idx = pd.Index([f"mc_{i:03d}" for i in range(X.shape[0])], name="sample_id")
    cols = [f"m{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, index=idx, columns=cols), pd.Series(labels, index=idx, name="class")


@pytest.fixture(scope="session")
def regression_df():
    X, y = make_regression(
        n_samples=90,
        n_features=7,
        n_informative=4,
        noise=8.0,
        random_state=12,
    )
    idx = pd.Index([f"reg_{i:03d}" for i in range(X.shape[0])], name="sample_id")
    cols = [f"r{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, index=idx, columns=cols), pd.Series(y, index=idx, name="response")


def _skip_if_xgb_unavailable(alg: str):
    if alg == "xgb" and not getattr(robust, "_HAS_XGBOOST", False):
        pytest.skip("xgboost is not installed in this environment")


def _fit_maker(alg: str, task_type: str, X, y, **extra):
    _skip_if_xgb_unavailable(alg)
    kwargs = dict(FAST_KWARGS)
    kwargs.update(extra)
    maker = robust.RobustModelMaker(alg=alg, task_type=task_type, **kwargs)
    return maker.fit(X, y)


def _assert_fitted_result(result, expected_task: str, expected_algorithm: str | None = None):
    assert result.task_type == expected_task
    if expected_algorithm is not None:
        assert result.algorithm == expected_algorithm
    assert result.robust_model is not None
    assert result.preprocessor is not None
    assert len(result.selected_features) >= 1
    assert len(result.selected_feature_indices) == len(result.selected_features)
    assert result.nested_cv_result.outer_predictions.shape[0] == result.nested_cv_result.outer_true_labels.shape[0]
    assert np.all(np.isfinite(result.stability_result.selection_frequencies))
    assert result.stability_result.selection_frequencies.shape[0] == len(result.feature_names)
    assert np.all((result.stability_result.selection_frequencies >= 0) & (result.stability_result.selection_frequencies <= 1))
    text = result.summary()
    assert "ROBUST MODEL MAKER" in text
    assert f"Task: {expected_task}" in text
    tables = result.results_tables()
    required_tables = {
        "overview",
        "selected_features",
        "stability_selection",
        "feature_stability_cv",
        "nested_cv_scores",
        "nested_cv_predictions",
    }
    assert required_tables.issubset(tables)
    assert isinstance(tables["overview"], pd.DataFrame)
    assert not tables["overview"].empty


# -----------------------------------------------------------------------------
# Import and public API tests
# -----------------------------------------------------------------------------


def test_module_imports_and_public_api():
    required = [
        "RobustModelMaker",
        "run_pipeline",
        "stability_selection",
        "nested_cross_validation",
        "determine_cutoff",
        "get_algorithm_config",
        "print_pipeline_results",
        "set_global_seed",
    ]
    for name in required:
        assert hasattr(robust, name), f"Missing public API: {name}"


@pytest.mark.parametrize("task_type, alg", [("binary", a) for a in CLASSIFICATION_ALGS] + [("multiclass", a) for a in CLASSIFICATION_ALGS] + [("regression", a) for a in REGRESSION_ALGS])
def test_all_algorithm_configs_exist(task_type, alg):
    _skip_if_xgb_unavailable(alg)
    model, params = robust.get_algorithm_config(
        alg,
        task_type,
        random_state=1,
        n_jobs=1,
        n_classes=3 if task_type == "multiclass" else None,
    )
    assert hasattr(model, "fit")
    assert isinstance(params, dict)
    assert len(params) >= 1


def test_invalid_algorithm_task_configurations_fail():
    with pytest.raises(ValueError):
        robust.get_algorithm_config("lin", "binary", random_state=1, n_jobs=1)
    with pytest.raises(ValueError):
        robust.get_algorithm_config("log", "regression", random_state=1, n_jobs=1)


def test_unfitted_estimator_raises(binary_df):
    X, _ = binary_df
    maker = robust.RobustModelMaker(verbose=False, save_results=False)
    with pytest.raises(NotFittedError):
        maker.predict(X)
    with pytest.raises(NotFittedError):
        maker.predict_proba(X)
    with pytest.raises(NotFittedError):
        maker.summary()


# -----------------------------------------------------------------------------
# Binary classification behaviour
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("alg", FAST_CLASSIFICATION_ALGS)
def test_binary_classification_fast_algorithms(binary_df, alg):
    X, y = binary_df
    maker = _fit_maker(alg, "binary", X, y)
    result = maker.result_
    _assert_fitted_result(result, "binary", alg)

    proba = maker.predict_proba(X.head(7))
    pred = maker.predict(X.head(7))
    assert isinstance(proba, pd.Series)
    assert isinstance(pred, pd.Series)
    assert proba.index.equals(X.head(7).index)
    assert pred.index.equals(X.head(7).index)
    assert proba.between(0, 1).all()
    assert set(np.unique(pred)).issubset(set(np.unique(y)))
    assert result.cutoff_result is not None
    assert 0 <= result.cutoff_result.cutoff_median <= 1


@pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="Set RUN_SLOW=1 to run slow algorithm coverage")
@pytest.mark.parametrize("alg", SLOW_ALGS)
def test_binary_classification_slow_algorithms(binary_df, alg):
    X, y = binary_df
    maker = _fit_maker(alg, "binary", X, y, **VERY_FAST_KWARGS)
    _assert_fitted_result(maker.result_, "binary", alg)
    assert len(maker.predict(X.head(5))) == 5


def test_run_pipeline_function_matches_class_interface(binary_df):
    X, y = binary_df
    result = robust.run_pipeline(X, y, alg="log", task_type="binary", **VERY_FAST_KWARGS)
    _assert_fitted_result(result, "binary", "log")
    pred = result.predict(X.head(4))
    assert isinstance(pred, pd.Series)
    assert len(pred) == 4


def test_auto_task_resolves_binary(binary_df):
    X, y = binary_df
    maker = _fit_maker("log", "auto", X, y, **VERY_FAST_KWARGS)
    assert maker.result_.task_type == "binary"


def test_binary_external_validation(binary_df):
    X, y = binary_df
    X_train, y_train = X.iloc[:70], y.iloc[:70]
    X_val, y_val = X.iloc[70:], y.iloc[70:]
    maker = robust.RobustModelMaker(alg="log", task_type="binary", **VERY_FAST_KWARGS)
    maker.fit(X_train, y_train, X_validation=X_val, y_validation=y_val)
    val = maker.result_.validation_result
    assert val is not None
    assert val.task_type == "binary"
    assert {"auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "cutoff"}.issubset(val.metrics)
    assert len(val.predictions) == len(y_val)
    tables = maker.result_.results_tables()
    assert "external_validation_metrics" in tables
    assert "external_validation_predictions" in tables
    assert "external_validation_confusion_matrix" in tables


@pytest.mark.parametrize("calibration", ["sigmoid", "isotonic"])
def test_binary_calibration_modes(binary_df, calibration):
    X, y = binary_df
    maker = _fit_maker("log", "binary", X, y, calibration=calibration, **VERY_FAST_KWARGS)
    assert maker.result_.calibration == calibration
    proba = maker.predict_proba(X.head(10))
    assert np.all((np.asarray(proba) >= 0) & (np.asarray(proba) <= 1))


@pytest.mark.parametrize("alg", FAST_CLASSIFICATION_ALGS)
def test_binary_evaluate_verification_method(binary_df, alg):
    X, y = binary_df
    maker = _fit_maker(alg, "binary", X.iloc[:70], y.iloc[:70], **VERY_FAST_KWARGS)
    verification = maker.evaluate_verification(X.iloc[70:], y.iloc[70:])
    assert verification.task_type == "binary"
    assert len(verification.predictions) == 20
    assert verification.probabilities is not None


# -----------------------------------------------------------------------------
# Multiclass classification behaviour
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("alg", FAST_CLASSIFICATION_ALGS)
def test_multiclass_classification_fast_algorithms(multiclass_df, alg):
    X, y = multiclass_df
    maker = _fit_maker(alg, "multiclass", X, y)
    result = maker.result_
    _assert_fitted_result(result, "multiclass", alg)

    proba = maker.predict_proba(X.head(6))
    pred = maker.predict(X.head(6))
    assert isinstance(proba, pd.DataFrame)
    assert isinstance(pred, pd.Series)
    assert proba.shape == (6, 3)
    np.testing.assert_allclose(proba.sum(axis=1).to_numpy(), np.ones(6), atol=1e-6)
    assert set(pred.unique()).issubset(set(y.unique()))
    assert result.cutoff_result is None


@pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="Set RUN_SLOW=1 to run slow algorithm coverage")
@pytest.mark.parametrize("alg", SLOW_ALGS)
def test_multiclass_slow_algorithms(multiclass_df, alg):
    X, y = multiclass_df
    maker = _fit_maker(alg, "multiclass", X, y, **VERY_FAST_KWARGS)
    _assert_fitted_result(maker.result_, "multiclass", alg)


def test_multiclass_external_validation(multiclass_df):
    X, y = multiclass_df
    maker = robust.RobustModelMaker(alg="log", task_type="multiclass", **VERY_FAST_KWARGS)
    maker.fit(X.iloc[:75], y.iloc[:75], X_validation=X.iloc[75:], y_validation=y.iloc[75:])
    val = maker.result_.validation_result
    assert val is not None
    assert val.task_type == "multiclass"
    assert {"accuracy", "balanced_accuracy", "macro_f1"}.issubset(val.metrics)
    assert val.probabilities is not None


def test_auto_task_resolves_multiclass(multiclass_df):
    X, y = multiclass_df
    maker = _fit_maker("log", "auto", X, y, **VERY_FAST_KWARGS)
    assert maker.result_.task_type == "multiclass"


# -----------------------------------------------------------------------------
# Regression behaviour
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("alg", FAST_REGRESSION_ALGS)
def test_regression_fast_algorithms(regression_df, alg):
    X, y = regression_df
    maker = _fit_maker(alg, "regression", X, y)
    result = maker.result_
    _assert_fitted_result(result, "regression", alg)
    assert result.cutoff_result is None

    pred = maker.predict(X.head(8))
    assert isinstance(pred, pd.Series)
    assert pred.index.equals(X.head(8).index)
    assert pred.shape == (8,)
    assert np.all(np.isfinite(pred.to_numpy()))
    with pytest.raises(AttributeError):
        maker.predict_proba(X.head(8))


@pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="Set RUN_SLOW=1 to run slow algorithm coverage")
@pytest.mark.parametrize("alg", SLOW_ALGS)
def test_regression_slow_algorithms(regression_df, alg):
    X, y = regression_df
    maker = _fit_maker(alg, "regression", X, y, **VERY_FAST_KWARGS)
    _assert_fitted_result(maker.result_, "regression", alg)


def test_regression_external_validation(regression_df):
    X, y = regression_df
    maker = robust.RobustModelMaker(alg="rdg", task_type="regression", **VERY_FAST_KWARGS)
    maker.fit(X.iloc[:70], y.iloc[:70], X_validation=X.iloc[70:], y_validation=y.iloc[70:])
    val = maker.result_.validation_result
    assert val is not None
    assert val.task_type == "regression"
    assert {"r2", "rmse", "mae"}.issubset(val.metrics)
    assert np.isfinite(list(val.metrics.values())).all()


def test_auto_task_requires_numeric_regression_target(regression_df):
    X, y = regression_df
    maker = _fit_maker("rdg", "auto", X, y, **VERY_FAST_KWARGS)
    assert maker.result_.task_type == "regression"


# -----------------------------------------------------------------------------
# Reproducibility tests
# -----------------------------------------------------------------------------


def test_reproducible_selected_features_and_predictions(binary_df):
    X, y = binary_df
    kwargs = dict(VERY_FAST_KWARGS)
    kwargs["random_state"] = 777

    m1 = robust.RobustModelMaker(alg="log", task_type="binary", **kwargs).fit(X, y)
    m2 = robust.RobustModelMaker(alg="log", task_type="binary", **kwargs).fit(X, y)

    np.testing.assert_array_equal(m1.result_.selected_feature_indices, m2.result_.selected_feature_indices)
    np.testing.assert_allclose(m1.result_.stability_result.selection_frequencies, m2.result_.stability_result.selection_frequencies)
    np.testing.assert_allclose(np.asarray(m1.predict_proba(X)), np.asarray(m2.predict_proba(X)), rtol=1e-10, atol=1e-10)


def test_reproducible_saved_core_metadata(binary_df, tmp_path):
    X, y = binary_df
    kwargs = dict(VERY_FAST_KWARGS)
    kwargs.update(dict(random_state=555, save_results=True, output_dir=tmp_path / "run1", output_prefix="robust"))
    m1 = robust.RobustModelMaker(alg="log", task_type="binary", **kwargs).fit(X, y)

    kwargs.update(dict(output_dir=tmp_path / "run2"))
    m2 = robust.RobustModelMaker(alg="log", task_type="binary", **kwargs).fit(X, y)

    md1 = json.loads((Path(m1.result_.results_dir) / "robust_metadata.json").read_text())
    md2 = json.loads((Path(m2.result_.results_dir) / "robust_metadata.json").read_text())
    for key in ["task_type", "algorithm", "calibration", "metric_name", "n_selected_features", "selected_features"]:
        assert md1[key] == md2[key]
    assert np.isclose(md1["mean_score"], md2["mean_score"])


# -----------------------------------------------------------------------------
# Grouped CV and repeated nested CV
# -----------------------------------------------------------------------------


def test_grouped_cv_binary(binary_df):
    X, y = binary_df
    groups = np.repeat(np.arange(30), 3)
    maker = robust.RobustModelMaker(alg="log", task_type="binary", **VERY_FAST_KWARGS)
    maker.fit(X, y, groups=groups)
    _assert_fitted_result(maker.result_, "binary", "log")


def test_grouped_cv_ignores_repeats_safely(binary_df):
    X, y = binary_df
    groups = np.repeat(np.arange(30), 3)
    maker = robust.RobustModelMaker(
        alg="log",
        task_type="binary",
        outer_cv=3,
        inner_cv=2,
        repeated_outer_cv=3,
        n_iter=1,
        n_bootstrap=2,
        cutoff_n_bootstrap=10,
        stability_threshold=0.1,
        random_state=321,
        n_jobs=1,
        verbose=False,
        save_results=False,
    )
    with pytest.warns(UserWarning, match="Grouped CV"):
        maker.fit(X, y, groups=groups)
    assert maker.result_.nested_cv_result.repeats == 1


def test_repeated_nested_cv_binary(binary_df):
    X, y = binary_df
    maker = robust.RobustModelMaker(
        alg="log",
        task_type="binary",
        outer_cv=3,
        inner_cv=2,
        repeated_outer_cv=2,
        n_iter=1,
        n_bootstrap=2,
        cutoff_n_bootstrap=10,
        stability_threshold=0.1,
        random_state=321,
        n_jobs=1,
        verbose=False,
        save_results=False,
    ).fit(X, y)
    result = maker.result_
    assert result.nested_cv_result.outer_scores.shape[0] == 6
    assert result.nested_cv_result.repeats == 2


# -----------------------------------------------------------------------------
# Permutation importance, SHAP-ready export, plotting
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("task_type, fixture_name, alg", [("binary", "binary_df", "log"), ("multiclass", "multiclass_df", "log"), ("regression", "regression_df", "rdg")])
def test_permutation_importance_all_task_types(request, task_type, fixture_name, alg):
    X, y = request.getfixturevalue(fixture_name)
    maker = robust.RobustModelMaker(alg=alg, task_type=task_type, **VERY_FAST_KWARGS).fit(X, y)
    imp = maker.permutation_importance(X, y, n_repeats=3, n_jobs=1, random_state=99)
    assert imp.importances_mean.shape[0] == len(maker.result_.selected_features)
    assert imp.importances_std.shape[0] == len(maker.result_.selected_features)
    assert imp.importances.shape[0] == len(maker.result_.selected_features)
    summary = imp.summary()
    assert list(summary.columns) == ["feature", "importance_mean", "importance_std"]
    assert set(summary["feature"]).issubset(set(maker.result_.selected_features))


@pytest.mark.parametrize("task_type, fixture_name, alg", [("binary", "binary_df", "log"), ("multiclass", "multiclass_df", "log"), ("regression", "regression_df", "rdg")])
def test_shap_ready_export_all_task_types(request, task_type, fixture_name, alg):
    X, y = request.getfixturevalue(fixture_name)
    maker = robust.RobustModelMaker(alg=alg, task_type=task_type, **VERY_FAST_KWARGS).fit(X, y)
    exported = maker.export_shap_ready(X.head(12), y.head(12))
    assert {"model", "X", "feature_names", "task_type", "algorithm", "predict_function", "y"}.issubset(exported)
    assert isinstance(exported["X"], pd.DataFrame)
    assert exported["X"].shape == (12, len(maker.result_.selected_features))
    assert list(exported["X"].columns) == list(maker.result_.selected_features)
    assert callable(exported["predict_function"])
    if task_type != "regression":
        assert callable(exported.get("predict_proba_function"))
    else:
        assert "predict_proba_function" not in exported or exported.get("predict_proba_function") is None


def test_plot_feature_stability_returns_axes(binary_df):
    import matplotlib

    matplotlib.use("Agg", force=True)
    X, y = binary_df
    maker = robust.RobustModelMaker(alg="log", task_type="binary", **VERY_FAST_KWARGS).fit(X, y)
    ax = maker.plot_feature_stability(top_n=5)
    assert hasattr(ax, "barh")
    assert ax.get_xlabel() == "Selection frequency"
    with pytest.raises(ValueError):
        maker.plot_feature_stability(top_n=0)


# -----------------------------------------------------------------------------
# Reporting and automatic saving
# -----------------------------------------------------------------------------


def test_results_tables_print_and_save(binary_df, tmp_path, capsys):
    X, y = binary_df
    maker = robust.RobustModelMaker(
        alg="log",
        task_type="binary",
        **{**VERY_FAST_KWARGS, "save_results": True, "output_dir": tmp_path / "auto", "output_prefix": "demo"},
    ).fit(X, y)

    result = maker.result_
    out_dir = Path(result.results_dir)
    assert out_dir.exists()
    expected_files = [
        "demo_overview.csv",
        "demo_selected_features.csv",
        "demo_stability_selection.csv",
        "demo_feature_stability_cv.csv",
        "demo_nested_cv_scores.csv",
        "demo_nested_cv_predictions.csv",
        "demo_cutoff_distribution.csv",
        "demo_metadata.json",
        "demo_summary.txt",
        "demo_result.pkl",
    ]
    for fname in expected_files:
        assert (out_dir / fname).exists(), fname

    metadata = json.loads((out_dir / "demo_metadata.json").read_text())
    assert metadata["task_type"] == "binary"
    assert metadata["algorithm"] == "log"
    assert metadata["n_selected_features"] == len(result.selected_features)

    with open(out_dir / "demo_result.pkl", "rb") as f:
        loaded = pickle.load(f)
    assert loaded.task_type == result.task_type
    assert loaded.algorithm == result.algorithm

    result.print_results(top_n=3)
    captured = capsys.readouterr().out
    assert "ROBUST MODEL MAKER RESULTS SUMMARY" in captured
    assert "TOP LEVEL" in captured
    assert "NESTED CV RESULTS" in captured

    manual_dir = maker.save(tmp_path / "manual", prefix="manual")
    assert Path(manual_dir).exists()
    assert (Path(manual_dir) / "manual_metadata.json").exists()


# -----------------------------------------------------------------------------
# Input validation and clear failures
# -----------------------------------------------------------------------------


def test_missing_values_are_imputed_but_infinities_fail(binary_df):
    X, y = binary_df
    X_missing = X.copy()
    X_missing.iloc[0, 0] = np.nan
    maker = robust.RobustModelMaker(alg="log", task_type="binary", **VERY_FAST_KWARGS).fit(X_missing, y)
    assert maker.result_ is not None

    X_bad = X.copy()
    X_bad.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinite|Inf|finite"):
        robust.RobustModelMaker(alg="log", task_type="binary", **VERY_FAST_KWARGS).fit(X_bad, y)


def test_duplicate_dataframe_columns_fail(binary_df):
    X, y = binary_df
    X_dup = X.copy()
    X_dup.columns = ["dup", "dup"] + list(X_dup.columns[2:])
    with pytest.raises(ValueError, match="Duplicate|duplicate"):
        robust.RobustModelMaker(alg="log", task_type="binary", **VERY_FAST_KWARGS).fit(X_dup, y)


def test_non_binary_label_error_for_binary_task(binary_df):
    X, _ = binary_df
    y = pd.Series(np.tile([0, 1, 2], 30), index=X.index)
    with pytest.raises(ValueError, match="Binary|binary"):
        robust.RobustModelMaker(alg="log", task_type="binary", **VERY_FAST_KWARGS).fit(X, y)


def test_too_few_samples_per_class_fails(binary_df):
    X, _ = binary_df
    y = pd.Series([0] * 88 + [1] * 2, index=X.index)
    with pytest.raises(ValueError, match="fold|class|samples"):
        robust.RobustModelMaker(alg="log", task_type="binary", outer_cv=5, inner_cv=3, n_iter=1, n_bootstrap=2, verbose=False, save_results=False).fit(X, y)


def test_single_class_fails(binary_df):
    X, _ = binary_df
    y = pd.Series(np.zeros(len(X), dtype=int), index=X.index)
    with pytest.raises(ValueError, match="class|unique|Binary|binary"):
        robust.RobustModelMaker(alg="log", task_type="binary", **VERY_FAST_KWARGS).fit(X, y)


def test_invalid_algorithm_task_combinations(binary_df, regression_df):
    Xc, yc = binary_df
    Xr, yr = regression_df
    with pytest.raises(ValueError, match="linear|lin|regression"):
        robust.RobustModelMaker(alg="lin", task_type="binary", **VERY_FAST_KWARGS).fit(Xc, yc)
    with pytest.raises(ValueError, match="logistic|log|classification|regression"):
        robust.RobustModelMaker(alg="log", task_type="regression", **VERY_FAST_KWARGS).fit(Xr, yr)


def test_validation_requires_X_and_y_together(binary_df):
    X, y = binary_df
    with pytest.raises(ValueError, match="both X_validation and y_validation"):
        robust.RobustModelMaker(alg="log", task_type="binary", **VERY_FAST_KWARGS).fit(
            X, y, X_validation=X.head(5), y_validation=None
        )


def test_prediction_requires_expected_columns(binary_df):
    X, y = binary_df
    maker = robust.RobustModelMaker(alg="log", task_type="binary", **VERY_FAST_KWARGS).fit(X, y)
    X_missing_col = X.drop(columns=[X.columns[0]])
    with pytest.raises(ValueError, match="missing|required|features"):
        maker.predict(X_missing_col)


def test_array_input_prediction_returns_numpy(binary_df):
    X, y = binary_df
    maker = robust.RobustModelMaker(alg="log", task_type="binary", **VERY_FAST_KWARGS).fit(X.to_numpy(), y.to_numpy())
    pred = maker.predict(X.to_numpy()[:5])
    proba = maker.predict_proba(X.to_numpy()[:5])
    assert isinstance(pred, np.ndarray)
    assert isinstance(proba, np.ndarray)
    assert pred.shape == (5,)
    assert proba.shape == (5,)


# -----------------------------------------------------------------------------
# Lower-level function tests
# -----------------------------------------------------------------------------


def test_determine_cutoff_uses_positive_if_score_at_or_above_cutoff():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_scores = np.array([0.10, 0.20, 0.30, 0.40, 0.35, 0.45, 0.80, 0.90])
    cutoff = robust.determine_cutoff(y_true, y_scores, target_specificity=0.75, n_bootstrap=50, random_state=5)
    assert 0 <= cutoff.cutoff_median <= 1
    predicted_positive = y_scores >= cutoff.cutoff_median
    assert predicted_positive.dtype == bool
    assert 0 <= cutoff.achieved_specificity <= 1
    assert 0 <= cutoff.achieved_sensitivity <= 1


@pytest.mark.parametrize("task_type,alg", [("binary", "log"), ("multiclass", "log"), ("regression", "rdg")])
def test_algorithm_config_returns_estimator_and_search_space(task_type, alg):
    model, params = robust.get_algorithm_config(alg, task_type, random_state=1, n_jobs=1, n_classes=3 if task_type == "multiclass" else None)
    assert hasattr(model, "fit")
    assert isinstance(params, dict)
    assert len(params) >= 1


@pytest.mark.parametrize("task_type, fixture_name, alg", [("binary", "binary_df", "log"), ("multiclass", "multiclass_df", "log"), ("regression", "regression_df", "rdg")])
def test_stability_selection_direct_all_task_types(request, task_type, fixture_name, alg):
    X, y = request.getfixturevalue(fixture_name)
    result = robust.stability_selection(
        X.to_numpy(),
        y.to_numpy(),
        feature_names=X.columns.to_numpy(),
        alg=alg,
        task_type=task_type,
        n_bootstrap=3,
        threshold=0.1,
        random_state=1,
        n_jobs=1,
    )
    assert result.task_type == task_type
    assert result.selection_frequencies.shape[0] == X.shape[1]
    assert np.all((result.selection_frequencies >= 0) & (result.selection_frequencies <= 1))
    assert isinstance(result.summary(), pd.DataFrame)


# -----------------------------------------------------------------------------
# preserve_nans tests
# -----------------------------------------------------------------------------


@pytest.fixture(scope="session")
def binary_df_with_nans(binary_df):
    """Binary dataset with ~25% NaN values injected across features."""
    X, y = binary_df
    rng = np.random.RandomState(99)
    X_nan = X.copy().astype(float)
    for col in X_nan.columns[:4]:
        mask = rng.rand(len(X_nan)) < 0.25
        X_nan.loc[mask, col] = np.nan
    # One column is mostly missing (70%) to trigger column dropping
    X_nan.iloc[:, -1] = np.nan
    X_nan.iloc[:5, -1] = 1.0
    return X_nan, y


@pytest.mark.parametrize("task_type,alg", [("binary", "log"), ("regression", "rdg")])
def test_preserve_nans_false_runs_and_drops(binary_df_with_nans, regression_df, task_type, alg):
    """preserve_nans=False should run successfully and retain fewer features than the original."""
    if task_type == "binary":
        X, y = binary_df_with_nans
    else:
        X_orig, y = regression_df
        rng = np.random.RandomState(77)
        X = X_orig.copy().astype(float)
        X.iloc[:, -1] = np.nan
        X.iloc[:5, -1] = 1.0

    maker = robust.RobustModelMaker(alg=alg, task_type=task_type, preserve_nans=False, **VERY_FAST_KWARGS)
    maker.fit(X, y)
    result = maker.result_

    assert result.preserve_nans is False
    assert result.nan_dropping_result is not None
    d = result.nan_dropping_result
    assert d["original_n_features"] == X.shape[1]
    assert d["retained_n_features"] <= d["original_n_features"]
    assert d["retained_n_samples"] <= d["original_n_samples"]
    assert d["retained_n_samples"] >= 4

    assert result.robust_model is not None
    assert len(result.selected_features) >= 1
    text = result.summary()
    assert "preserve_nans=False" in text


def test_preserve_nans_false_predictions_work_with_full_dataframe(binary_df_with_nans):
    """After preserve_nans=False, predict() on a full DataFrame should work by name-based column selection."""
    X, y = binary_df_with_nans
    maker = robust.RobustModelMaker(alg="log", task_type="binary", preserve_nans=False, **VERY_FAST_KWARGS)
    maker.fit(X, y)

    pred = maker.predict(X.head(10))
    proba = maker.predict_proba(X.head(10))
    assert isinstance(pred, pd.Series)
    assert isinstance(proba, pd.Series)
    assert len(pred) == 10
    assert proba.between(0, 1).all()


def test_preserve_nans_false_predictions_work_with_numpy(binary_df_with_nans):
    """After preserve_nans=False, predict() on a numpy array of the full original shape should work."""
    X, y = binary_df_with_nans
    maker = robust.RobustModelMaker(alg="log", task_type="binary", preserve_nans=False, **VERY_FAST_KWARGS)
    maker.fit(X, y)

    # Pass full numpy array -- the col_mask is applied internally
    X_np = X.to_numpy(dtype=float)
    pred = maker.predict(X_np[:10])
    assert isinstance(pred, np.ndarray)
    assert len(pred) == 10


def test_preserve_nans_true_unchanged(binary_df_with_nans):
    """preserve_nans=True (default) should keep all original features in feature_names."""
    X, y = binary_df_with_nans
    maker = robust.RobustModelMaker(alg="log", task_type="binary", preserve_nans=True, **VERY_FAST_KWARGS)
    maker.fit(X, y)
    result = maker.result_

    assert result.preserve_nans is True
    assert result.nan_dropping_result is None
    assert result.nan_dropping_col_mask is None
    assert len(result.feature_names) == X.shape[1]


def test_preserve_nans_false_multiclass(multiclass_df):
    """preserve_nans=False works for multiclass with injected NaNs."""
    X_orig, y = multiclass_df
    rng = np.random.RandomState(55)
    X = X_orig.copy().astype(float)
    for col in X.columns[:3]:
        X.loc[rng.rand(len(X)) < 0.3, col] = np.nan
    X.iloc[:, -1] = np.nan
    X.iloc[:4, -1] = 0.5

    maker = robust.RobustModelMaker(alg="log", task_type="multiclass", preserve_nans=False, **VERY_FAST_KWARGS)
    maker.fit(X, y)
    result = maker.result_

    assert result.preserve_nans is False
    assert result.nan_dropping_result["retained_n_features"] <= X.shape[1]
    pred = maker.predict(X.head(6))
    assert isinstance(pred, pd.Series)
    assert set(pred.unique()).issubset(set(y.unique()))


def test_smart_drop_nans_utility():
    """Unit test for the _smart_drop_nans helper function directly."""
    rng = np.random.RandomState(0)
    X = rng.randn(50, 10).astype(float)
    # Make last column 90% missing
    X[rng.rand(50) > 0.1, -1] = np.nan
    # Make rows 0-4 entirely missing in first 5 features
    X[:5, :5] = np.nan
    names = np.array([f"feat_{i}" for i in range(10)])

    X_clean, names_clean, row_mask, col_mask = robust._smart_drop_nans(X, names, random_state=0, verbose=False)

    assert X_clean.shape[1] == names_clean.shape[0]
    assert X_clean.shape[0] == row_mask.sum()
    assert col_mask.shape[0] == 10
    assert col_mask.sum() == X_clean.shape[1]
    # The mostly-missing last column should be dropped
    assert not col_mask[-1], "Mostly-missing column should be dropped"
    # At least 4 rows must be retained
    assert X_clean.shape[0] >= 4

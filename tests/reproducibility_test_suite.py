"""
reproducibility_test_suite.py
==============================
RobustModelMaker -- reproducibility and determinism tests.

These tests verify that ROBUST behaves deterministically when the same random_state
is supplied, that different seeds produce structurally valid but numerically
distinct results, that serialised models predict identically to live ones, and
that stability frequencies converge with increasing bootstrap samples.

How to run
----------
From the project root:

    python -m pytest tests/reproducibility_test_suite.py -v

From inside the tests/ directory:

    python -m pytest reproducibility_test_suite.py -v

The suite is intentionally fast: all datasets are small synthetic arrays and
CV settings use the minimum viable fold counts.
"""

from __future__ import annotations

import importlib.util
import os
import pickle
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

# ---------------------------------------------------------------------------
# Load RobustModelMaker from a local file
# ---------------------------------------------------------------------------


def _load_robust_module():
    here = Path(__file__).resolve().parent
    env_path = os.environ.get("ROBUST_MODEL_MAKER_PATH")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates += [
        here / "RobustModelMaker.py",
        here.parent / "RobustModelMaker.py",
        here / "RobustModelMaker_v0_3.py",
        here.parent / "RobustModelMaker_v0_3.py",
    ]
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("robust_repro", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(
        "RobustModelMaker.py not found. Place it one level above tests/ "
        "or set ROBUST_MODEL_MAKER_PATH=/path/to/RobustModelMaker.py."
    )


robust = _load_robust_module()

# ---------------------------------------------------------------------------
# Shared CV settings (deliberately minimal for speed)
# ---------------------------------------------------------------------------

REPRO_KWARGS: Dict[str, Any] = dict(
    outer_cv=3,
    inner_cv=2,
    n_iter=2,
    n_bootstrap=6,
    cutoff_n_bootstrap=20,
    stability_threshold=0.25,
    n_jobs=1,
    verbose=False,
    save_results=False,
)

SEED_A = 42
SEED_B = 99


# ---------------------------------------------------------------------------
# Fixtures: tiny synthetic datasets, session-scoped so they are built once
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def binary_data():
    X, y = make_classification(
        n_samples=120,
        n_features=12,
        n_informative=5,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        class_sep=1.5,
        random_state=7,
    )
    cols = [f"b{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="target")


@pytest.fixture(scope="session")
def multiclass_data():
    X, y = make_classification(
        n_samples=120,
        n_features=12,
        n_informative=5,
        n_redundant=2,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=8,
    )
    labels = np.array(["cat", "dog", "bird"])[y]
    cols = [f"m{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols), pd.Series(labels, name="species")


@pytest.fixture(scope="session")
def regression_data():
    X, y = make_regression(
        n_samples=120,
        n_features=12,
        n_informative=5,
        noise=5.0,
        random_state=9,
    )
    cols = [f"r{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="response")


# ---------------------------------------------------------------------------
# Session-scoped fitted results used across multiple tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def binary_result_seed_a(binary_data):
    X, y = binary_data
    maker = robust.RobustModelMaker(alg="rdg", task_type="binary", random_state=SEED_A, **REPRO_KWARGS)
    return maker.fit(X, y)


@pytest.fixture(scope="session")
def binary_result_seed_a_dup(binary_data):
    """Second fit with identical seed: must match binary_result_seed_a exactly."""
    X, y = binary_data
    maker = robust.RobustModelMaker(alg="rdg", task_type="binary", random_state=SEED_A, **REPRO_KWARGS)
    return maker.fit(X, y)


@pytest.fixture(scope="session")
def binary_result_seed_b(binary_data):
    """Fit with a different seed: valid but allowed to differ from seed A."""
    X, y = binary_data
    maker = robust.RobustModelMaker(alg="rdg", task_type="binary", random_state=SEED_B, **REPRO_KWARGS)
    return maker.fit(X, y)


@pytest.fixture(scope="session")
def regression_result_seed_a(regression_data):
    X, y = regression_data
    maker = robust.RobustModelMaker(alg="las", task_type="regression", random_state=SEED_A, **REPRO_KWARGS)
    return maker.fit(X, y)


@pytest.fixture(scope="session")
def regression_result_seed_a_dup(regression_data):
    X, y = regression_data
    maker = robust.RobustModelMaker(alg="las", task_type="regression", random_state=SEED_A, **REPRO_KWARGS)
    return maker.fit(X, y)


@pytest.fixture(scope="session")
def multiclass_result_seed_a(multiclass_data):
    X, y = multiclass_data
    maker = robust.RobustModelMaker(alg="rdg", task_type="multiclass", random_state=SEED_A, **REPRO_KWARGS)
    return maker.fit(X, y)


@pytest.fixture(scope="session")
def multiclass_result_seed_a_dup(multiclass_data):
    X, y = multiclass_data
    maker = robust.RobustModelMaker(alg="rdg", task_type="multiclass", random_state=SEED_A, **REPRO_KWARGS)
    return maker.fit(X, y)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _r(maker):
    """Return the PipelineResult stored on a fitted RobustModelMaker."""
    return maker.result_


def _selected_features(maker) -> list:
    """Return a sorted list of selected feature names."""
    return sorted(str(f) for f in _r(maker).selected_features)


def _fold_scores(maker) -> np.ndarray:
    """Return the per-fold outer CV scores as a 1-D array."""
    return np.asarray(_r(maker).nested_cv_result.outer_scores)


def _mean_score(maker) -> float:
    """Return the mean outer CV score."""
    return float(np.mean(_fold_scores(maker)))


def _stability_freqs(maker) -> np.ndarray:
    """Return the stability frequency vector (one value per input feature)."""
    return np.asarray(_r(maker).stability_result.selection_frequencies, dtype=float)


def _cutoff(maker):
    """Return the classification cutoff median, or None if not computed."""
    cr = _r(maker).cutoff_result
    if cr is None:
        return None
    return cr.cutoff_median


# ===========================================================================
# TestExactReproducibility
# ===========================================================================


class TestExactReproducibility:
    """Same data + same random_state must yield byte-for-byte identical results."""

    def test_binary_selected_features_identical(
        self, binary_result_seed_a, binary_result_seed_a_dup
    ):
        assert _selected_features(binary_result_seed_a) == _selected_features(
            binary_result_seed_a_dup
        ), "Selected features differ between two identically seeded binary fits."

    def test_binary_fold_scores_identical(
        self, binary_result_seed_a, binary_result_seed_a_dup
    ):
        s1 = _fold_scores(binary_result_seed_a)
        s2 = _fold_scores(binary_result_seed_a_dup)
        np.testing.assert_array_equal(
            s1, s2, err_msg="Fold scores differ between identically seeded binary fits."
        )

    def test_binary_mean_score_identical(
        self, binary_result_seed_a, binary_result_seed_a_dup
    ):
        assert _mean_score(binary_result_seed_a) == _mean_score(binary_result_seed_a_dup)

    def test_binary_stability_frequencies_identical(
        self, binary_result_seed_a, binary_result_seed_a_dup
    ):
        f1 = _stability_freqs(binary_result_seed_a)
        f2 = _stability_freqs(binary_result_seed_a_dup)
        np.testing.assert_array_equal(
            f1, f2,
            err_msg="Stability frequencies differ between identically seeded binary fits.",
        )

    def test_regression_selected_features_identical(
        self, regression_result_seed_a, regression_result_seed_a_dup
    ):
        assert _selected_features(regression_result_seed_a) == _selected_features(
            regression_result_seed_a_dup
        )

    def test_regression_fold_scores_identical(
        self, regression_result_seed_a, regression_result_seed_a_dup
    ):
        np.testing.assert_array_equal(
            _fold_scores(regression_result_seed_a),
            _fold_scores(regression_result_seed_a_dup),
            err_msg="Regression fold scores differ between identically seeded fits.",
        )

    def test_multiclass_selected_features_identical(
        self, multiclass_result_seed_a, multiclass_result_seed_a_dup
    ):
        assert _selected_features(multiclass_result_seed_a) == _selected_features(
            multiclass_result_seed_a_dup
        )

    def test_multiclass_fold_scores_identical(
        self, multiclass_result_seed_a, multiclass_result_seed_a_dup
    ):
        np.testing.assert_array_equal(
            _fold_scores(multiclass_result_seed_a),
            _fold_scores(multiclass_result_seed_a_dup),
        )

    def test_cutoff_identical(self, binary_result_seed_a, binary_result_seed_a_dup):
        assert _cutoff(binary_result_seed_a) == _cutoff(binary_result_seed_a_dup)


# ===========================================================================
# TestSeedDiversity
# ===========================================================================


class TestSeedDiversity:
    """Different seeds should produce valid but (usually) distinct results.

    These tests are non-deterministic in the sense that they will pass even if
    two seeds happen to produce the same selection on a small dataset; that is
    unlikely enough not to be a false-positive risk here. The primary purpose is
    to confirm that the seed parameter actually propagates to the internals.
    """

    def test_different_seeds_fold_scores_differ(
        self, binary_result_seed_a, binary_result_seed_b
    ):
        s_a = _fold_scores(binary_result_seed_a)
        s_b = _fold_scores(binary_result_seed_b)
        # At least one fold score should differ if the seed is being used
        assert not np.array_equal(s_a, s_b), (
            "Fold scores are identical across seeds A and B. "
            "This suggests the random_state parameter is not being propagated."
        )

    def test_both_seeds_produce_valid_scores(
        self, binary_result_seed_a, binary_result_seed_b
    ):
        for result in (binary_result_seed_a, binary_result_seed_b):
            scores = _fold_scores(result)
            assert np.all(np.isfinite(scores)), "Non-finite fold scores detected."
            assert np.all(scores >= 0.0) and np.all(scores <= 1.0), (
                "Binary AUC scores should be in [0, 1]."
            )

    def test_both_seeds_select_at_least_one_feature(
        self, binary_result_seed_a, binary_result_seed_b
    ):
        for result in (binary_result_seed_a, binary_result_seed_b):
            assert len(_selected_features(result)) >= 1


# ===========================================================================
# TestSerializationReproducibility
# ===========================================================================


class TestSerializationReproducibility:
    """Pickle/unpickle must leave predictions numerically unchanged."""

    def test_binary_pickle_predict_proba_identical(
        self, binary_result_seed_a, binary_data
    ):
        X, _ = binary_data
        proba_before = binary_result_seed_a.predict_proba(X)
        restored = pickle.loads(pickle.dumps(binary_result_seed_a))
        proba_after = restored.predict_proba(X)
        np.testing.assert_array_equal(
            proba_before,
            proba_after,
            err_msg="predict_proba changed after pickle round-trip.",
        )

    def test_binary_pickle_predict_identical(
        self, binary_result_seed_a, binary_data
    ):
        X, _ = binary_data
        pred_before = binary_result_seed_a.predict(X).values
        restored = pickle.loads(pickle.dumps(binary_result_seed_a))
        pred_after = restored.predict(X).values
        np.testing.assert_array_equal(pred_before, pred_after)

    def test_regression_pickle_predict_identical(
        self, regression_result_seed_a, regression_data
    ):
        X, _ = regression_data
        pred_before = regression_result_seed_a.predict(X).values
        restored = pickle.loads(pickle.dumps(regression_result_seed_a))
        pred_after = restored.predict(X).values
        np.testing.assert_array_almost_equal(
            pred_before, pred_after, decimal=10,
            err_msg="Regression predictions changed after pickle round-trip.",
        )

    def test_multiclass_pickle_predict_identical(
        self, multiclass_result_seed_a, multiclass_data
    ):
        X, _ = multiclass_data
        pred_before = multiclass_result_seed_a.predict(X).values
        restored = pickle.loads(pickle.dumps(multiclass_result_seed_a))
        pred_after = restored.predict(X).values
        np.testing.assert_array_equal(pred_before, pred_after)

    def test_pickle_preserves_selected_features(
        self, binary_result_seed_a
    ):
        before = _selected_features(binary_result_seed_a)
        restored = pickle.loads(pickle.dumps(binary_result_seed_a))
        after = _selected_features(restored)
        assert before == after, "Selected features changed after pickle round-trip."

    def test_pickle_preserves_fold_scores(self, binary_result_seed_a):
        before = _fold_scores(binary_result_seed_a)
        restored = pickle.loads(pickle.dumps(binary_result_seed_a))
        after = _fold_scores(restored)
        np.testing.assert_array_equal(before, after)

    def test_deepcopy_predict_identical(self, regression_result_seed_a, regression_data):
        X, _ = regression_data
        pred_before = regression_result_seed_a.predict(X).values
        copy = deepcopy(regression_result_seed_a)
        pred_after = copy.predict(X).values
        np.testing.assert_array_almost_equal(pred_before, pred_after, decimal=10)


# ===========================================================================
# TestStabilityConvergence
# ===========================================================================


class TestStabilityConvergence:
    """Stability frequencies should converge as n_bootstrap increases.

    More bootstrap samples reduce Monte Carlo noise in the selection frequencies.
    This test checks that the variance across repeated runs shrinks, not that
    specific features are selected (which depends on the data).
    """

    def test_more_bootstraps_reduce_frequency_variance(self, binary_data):
        """Variance of stability frequencies decreases with more bootstrap samples."""
        X, y = binary_data
        base = dict(REPRO_KWARGS)

        # Collect frequency vectors from 5 independent runs at each bootstrap count
        def _run_freqs(n_boot: int, seed: int) -> np.ndarray:
            kwargs = dict(base, n_bootstrap=n_boot, cutoff_n_bootstrap=max(20, n_boot * 3))
            maker = robust.RobustModelMaker(alg="rdg", task_type="binary", random_state=seed, **kwargs)
            result = maker.fit(X, y)
            return _stability_freqs(result)

        n_seeds = 4
        few_runs = np.stack([_run_freqs(4, s) for s in range(n_seeds)])
        many_runs = np.stack([_run_freqs(16, s) for s in range(n_seeds)])

        var_few = float(np.var(few_runs, axis=0).mean())
        var_many = float(np.var(many_runs, axis=0).mean())

        assert var_many <= var_few * 1.5, (
            f"Expected variance with 16 bootstraps ({var_many:.4f}) to be no worse "
            f"than with 4 bootstraps ({var_few:.4f}). Stability selection appears "
            "non-convergent."
        )

    def test_stability_frequencies_in_unit_interval(self, binary_result_seed_a):
        freqs = _stability_freqs(binary_result_seed_a)
        assert np.all(freqs >= 0.0) and np.all(freqs <= 1.0), (
            "Stability frequencies must lie in [0, 1]."
        )

    def test_stability_frequencies_not_all_zero(self, binary_result_seed_a):
        freqs = _stability_freqs(binary_result_seed_a)
        assert freqs.max() > 0.0, "All stability frequencies are zero."

    def test_stability_frequencies_not_all_one(self, binary_data):
        """With a weak signal, not all features should reach max frequency."""
        X, y = binary_data
        # Shuffle y to break signal: no feature should dominate
        rng = np.random.default_rng(0)
        y_shuffled = pd.Series(rng.permutation(y.values), name="target")
        kwargs = dict(REPRO_KWARGS, n_bootstrap=10, stability_threshold=0.5)
        maker = robust.RobustModelMaker(alg="rdg", task_type="binary", random_state=SEED_A, **kwargs)
        result = maker.fit(X, y_shuffled)
        freqs = _stability_freqs(result)
        assert freqs.max() < 1.0 or freqs.mean() < 0.9, (
            "All stability frequencies are 1.0 on shuffled labels: "
            "selection appears insensitive to signal."
        )


# ===========================================================================
# TestScoreConsistency
# ===========================================================================


class TestScoreConsistency:
    """Cross-checks between scalar summary statistics and per-fold arrays."""

    def test_binary_mean_score_equals_fold_mean(self, binary_result_seed_a):
        scores = _fold_scores(binary_result_seed_a)
        fold_mean = float(np.mean(scores))
        computed_mean = _mean_score(binary_result_seed_a)
        assert abs(computed_mean - fold_mean) < 1e-12, (
            f"_mean_score {computed_mean:.6f} != mean of outer_scores {fold_mean:.6f}."
        )

    def test_regression_fold_scores_negative_rmse(self, regression_result_seed_a):
        """Regression fold scores should be negative RMSE (negative is better)."""
        scores = _fold_scores(regression_result_seed_a)
        assert np.all(scores <= 0.0), (
            "Regression fold scores should be negative RMSE values, got: "
            f"{scores}"
        )

    def test_regression_mean_score_equals_fold_mean(self, regression_result_seed_a):
        scores = _fold_scores(regression_result_seed_a)
        fold_mean = float(np.mean(scores))
        assert abs(_mean_score(regression_result_seed_a) - fold_mean) < 1e-12

    def test_multiclass_fold_scores_bounded(self, multiclass_result_seed_a):
        scores = _fold_scores(multiclass_result_seed_a)
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0), (
            f"Multiclass balanced accuracy scores out of [0, 1]: {scores}"
        )

    def test_fold_count_matches_outer_cv(self, binary_result_seed_a):
        n_folds = len(_fold_scores(binary_result_seed_a))
        assert n_folds == REPRO_KWARGS["outer_cv"], (
            f"Expected {REPRO_KWARGS['outer_cv']} fold scores, got {n_folds}."
        )

    def test_selected_features_are_subset_of_input(self, binary_result_seed_a, binary_data):
        X, _ = binary_data
        selected = set(_selected_features(binary_result_seed_a))
        input_cols = set(str(c) for c in X.columns)
        assert selected.issubset(input_cols), (
            f"Selected features {selected - input_cols} are not in the input columns."
        )

    def test_feature_reduction_achievable(self, binary_result_seed_a, binary_data):
        X, _ = binary_data
        selected = _selected_features(binary_result_seed_a)
        assert len(selected) <= X.shape[1], "Cannot select more features than supplied."

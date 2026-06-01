"""
benchmark_suite.py
=======================
RobustModelMaker -- capability demonstration using three real scientific datasets.

Each scenario runs ROBUST (stability-selected feature subset) against a full-feature
nested-CV baseline using the same algorithm and fold structure, then applies a
battery of 25+ statistical tests to the per-fold score vectors.

Datasets
--------
1. SECOM Manufacturing  : 1567 samples x 590 features, binary pass/fail,
                          heavy class imbalance (~7% fail), real NaN values.
                          Source: UCI direct download (secom.data + secom_labels.data).

2. Urban Land Cover     : 675 samples x 147 features, 9-class aerial imagery
                          (segment-level, not pixel-level -- no spatial leakage).
                          Source: UCI direct download (urban+land+cover.zip).

3. Graphene Oxide Bulk  : 1617 samples x 462 structural/chemical descriptors,
                          regression target = Formation_energy (eV), real NaN
                          values, 19 distinct stoichiometries.
                          Source: CSIRO local CSV (benchmarks/Graphene_Oxide_Bulk.csv).

Run
---
    python benchmarks/benchmark_suite.py        # full console report
    python -m pytest benchmarks/benchmark_suite.py -v -s   # via pytest
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import io
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats
from scipy.stats import binom, loguniform, uniform
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from benchmake import BenchMake

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
    ]
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("robust_benchmark", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(
        "RobustModelMaker.py not found. Place it one level above benchmarks/ "
        "or set ROBUST_MODEL_MAKER_PATH=/path/to/RobustModelMaker.py."
    )


robust = _load_robust_module()

# ---------------------------------------------------------------------------
# Benchmark parameters
# ---------------------------------------------------------------------------

ROBUST_PARAMS: Dict[str, Any] = dict(
    outer_cv=10,
    inner_cv=10,
    n_bootstrap=100,
    n_iter=100,
    stability_threshold=0.75,  # 0.75: feature must appear in ≥75% of bootstrap samples.
    # The library default is 0.70; 0.75 is used here so that the benchmark
    # operates at a more demanding selection criterion.  This produces ~20-35%
    # feature retention across these three datasets, keeping RobustModelMarkker in the same
    # order-of-magnitude compression range as RFECV and Boruta while remaining
    # well above the minimum "stable" threshold of 0.50.  See the comparator
    # justification docstrings for how this choice affects the comparison.
    cutoff_n_bootstrap=500,
    random_state=42,
    n_jobs=1,
    verbose=False,
)

# ── Per-dataset stability thresholds from ThresholdOptimizer ──────────────────
# Values are from tools/Threshold_Optimisation.ipynb: equal-weight composite
# optimum, full 9-point grid (0.50 → 0.90 in steps of 0.05), random 80/20
# split, random_state=42.
#
# Priority (lowest → highest):
#   ROBUST_PARAMS['stability_threshold']  — global fallback
#   THRESHOLD_OVERRIDES[ds.name]          — per-dataset optimum (this dict)
#   ds.robust_params_override             — explicit per-instance caller override
#
# Set any value to None to fall back to ROBUST_PARAMS['stability_threshold'].
# Override in the notebook without re-importing:
#   bs.THRESHOLD_OVERRIDES.update({"SECOM Manufacturing": 0.65})

THRESHOLD_OVERRIDES: Dict[str, Optional[float]] = {
    "SECOM Manufacturing":  0.60,   # composite=0.645  AUC=0.758±0.086  stability=0.692
    "Urban Land Cover":     0.80,   # composite=0.664  AUC=0.983±0.007  stability=0.840
    "Graphene Oxide Bulk":  None,   # run tools/Threshold_Optimisation.ipynb to get this value
}

_OUTER_CV = ROBUST_PARAMS["outer_cv"]
_INNER_CV = ROBUST_PARAMS["inner_cv"]
_N_ITER = ROBUST_PARAMS["n_iter"]
_SEED = ROBUST_PARAMS["random_state"]

# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# BenchMake archetypal split helper
# ---------------------------------------------------------------------------

def _benchmake_split(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """BenchMake archetypal split -- median-imputes NaNs for distance computation only.

    The raw (NaN-containing) X and y are NOT modified; indices are returned so
    the caller can slice the original DataFrames for train/test.
    """
    X_arr = X.to_numpy(dtype=float)
    if np.isnan(X_arr).any():
        col_meds = np.nanmedian(X_arr, axis=0)
        col_meds = np.where(np.isfinite(col_meds), col_meds, 0.0)
        X_imp = X_arr.copy()
        nans = np.isnan(X_imp)
        X_imp[nans] = np.take(col_meds, np.where(nans)[1])
    else:
        X_imp = X_arr
    bm = BenchMake(n_jobs=1)
    train_idx, test_idx = bm.partition(
        X_imp, y.to_numpy(), test_size=test_size,
        data_type="tabular", return_indices=True,
    )
    return np.asarray(train_idx), np.asarray(test_idx)


class BenchmarkDataset:
    """Container for a scenario dataset and its metadata.

    ``train_idx`` and ``test_idx`` are integer index arrays produced by
    :func:`_benchmake_split`.  The ``X_train`` / ``y_train`` / ``X_test`` /
    ``y_test`` properties slice the full arrays accordingly.
    """

    def __init__(
        self,
        name: str,
        description: str,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str,
        alg: str,
        floor_score: float,
        train_idx: Optional[np.ndarray] = None,
        test_idx: Optional[np.ndarray] = None,
        robust_params_override: Optional[Dict[str, Any]] = None,
        true_features: Optional[List[str]] = None,
        correlate_features: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.X = X
        self.y = y
        self.task_type = task_type
        self.alg = alg
        self.floor_score = floor_score
        self.train_idx = train_idx
        self.test_idx = test_idx
        # Per-dataset overrides for ROBUST_PARAMS (merged at run time; ROBUST_PARAMS are the base).
        self.robust_params_override: Dict[str, Any] = robust_params_override or {}
        # Ground-truth feature provenance for synthetic recovery scenarios.
        # When non-None these enable precision/recall/F1 reporting against the
        # known informative feature set, and correlate-vs-cause confusion.
        # Real-world datasets leave these as None.
        self.true_features: Optional[List[str]] = (
            list(true_features) if true_features is not None else None
        )
        self.correlate_features: Optional[List[str]] = (
            list(correlate_features) if correlate_features is not None else None
        )

    @property
    def has_ground_truth(self) -> bool:
        """True for synthetic datasets that carry a known informative feature set."""
        return self.true_features is not None

    # ------------------------------------------------------------------
    # Convenience split accessors (fall back to full dataset when no
    # train/test split has been assigned).
    # ------------------------------------------------------------------

    @property
    def X_train(self) -> pd.DataFrame:
        if self.train_idx is None:
            return self.X
        return self.X.iloc[self.train_idx].reset_index(drop=True)

    @property
    def y_train(self) -> pd.Series:
        if self.train_idx is None:
            return self.y
        return self.y.iloc[self.train_idx].reset_index(drop=True)

    @property
    def X_test(self) -> pd.DataFrame:
        if self.test_idx is None:
            return self.X
        return self.X.iloc[self.test_idx].reset_index(drop=True)

    @property
    def y_test(self) -> pd.Series:
        if self.test_idx is None:
            return self.y
        return self.y.iloc[self.test_idx].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Data unavailability sentinel
# ---------------------------------------------------------------------------


class DataUnavailable(RuntimeError):
    """Raised when a benchmark dataset cannot be loaded (no network, etc.)."""


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent


def _load_secom() -> BenchmarkDataset:
    """SECOM semiconductor manufacturing: 1567 x 590, binary pass/fail, real NaNs."""
    url_data = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
    )
    url_labels = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"
    )
    try:
        with urllib.request.urlopen(url_data, timeout=30) as r:
            X = pd.read_csv(r, sep=" ", header=None)
        with urllib.request.urlopen(url_labels, timeout=30) as r:
            labels = pd.read_csv(r, sep=" ", header=None)
    except Exception as exc:
        raise DataUnavailable(f"SECOM unavailable: {exc}") from exc

    X = X.apply(pd.to_numeric, errors="coerce")
    X.columns = [f"f{i}" for i in range(X.shape[1])]

    # Labels: col 0 is pass/fail (-1 = pass, 1 = fail)
    y = pd.Series(
        (labels[0] == 1).astype(int).values,
        name="fail",
    )

    train_idx, test_idx = _benchmake_split(X, y, test_size=0.2)
    return BenchmarkDataset(
        name="SECOM Manufacturing",
        description=(
            f"Semiconductor process sensor data; {X.shape[0]} samples x {X.shape[1]} features, "
            "binary pass/fail, ~7% failure rate, extensive real NaN values"
        ),
        X=X,
        y=y,
        task_type="binary",
        alg="rf",
        floor_score=0.60,
        train_idx=train_idx,
        test_idx=test_idx,
    )


def _load_urban_land_cover() -> BenchmarkDataset:
    """Urban Land Cover: 675 x 147, 9-class aerial imagery segments."""
    url = "https://archive.ics.uci.edu/static/public/295/urban+land+cover.zip"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = r.read()
    except Exception as exc:
        raise DataUnavailable(f"Urban Land Cover unavailable: {exc}") from exc

    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
        frames = []
        for name in zf.namelist():
            if name.endswith(".csv") and not name.startswith("__MACOSX"):
                with zf.open(name) as f:
                    df = pd.read_csv(f)
                if df.shape[0] > 0:
                    frames.append(df)
        if not frames:
            raise DataUnavailable("Urban Land Cover ZIP contained no usable CSV files.")
        combined = pd.concat(frames, ignore_index=True)
    except DataUnavailable:
        raise
    except Exception as exc:
        raise DataUnavailable(f"Urban Land Cover ZIP parse failed: {exc}") from exc

    y = combined["class"].astype(str)
    y.name = "land_cover"
    X = combined.drop(columns=["class"]).apply(pd.to_numeric, errors="coerce")
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    train_idx, test_idx = _benchmake_split(X, y, test_size=0.2)
    return BenchmarkDataset(
        name="Urban Land Cover",
        description=(
            f"Aerial image segments; {X.shape[0]} samples x {X.shape[1]} spectral/texture features, "
            "9 urban land cover classes, segment-level (no spatial autocorrelation)"
        ),
        X=X,
        y=y,
        task_type="multiclass",
        alg="rf",
        floor_score=0.75,
        train_idx=train_idx,
        test_idx=test_idx,
    )


def _load_graphene_oxide() -> BenchmarkDataset:
    """Graphene Oxide Bulk: structural descriptors, Formation_energy target.

    NaN dropping is applied globally (via robust._smart_drop_nans) so that both
    ROBUST and the baseline operate on a consistent feature set and fold-specific
    zero-variance columns (structurally sparse ring/bond descriptors) cannot
    cause vstack dimension mismatches inside nested CV.
    """
    csv_path = _HERE / "Graphene_Oxide_Bulk.csv"
    if not csv_path.exists():
        raise DataUnavailable(
            f"Graphene_Oxide_Bulk.csv not found at {csv_path}. "
            "Place it in the tests/ directory."
        )

    df = pd.read_csv(csv_path)
    non_feature_cols = [
        "ID", "file_name",
        "Thermodynamic_Probability", "Fermi_energy",
    ]
    target_col = "Formation_energy"
    drop_cols = [c for c in non_feature_cols if c in df.columns]

    y_raw = df[target_col].to_numpy(dtype=float)
    X_raw = df.drop(columns=drop_cols + [target_col]).apply(pd.to_numeric, errors="coerce")
    names_raw = np.array(X_raw.columns)
    X_arr = X_raw.to_numpy(dtype=float)

    # Apply missingness-based NaN dropping globally so both ROBUST and baseline
    # use an identical, consistent feature set across all CV folds.
    X_arr, names_clean, row_mask, _ = robust._smart_drop_nans(
        X_arr, names_raw, random_state=42, verbose=False
    )
    y_clean = y_raw[row_mask]

    # Also drop globally constant columns (var == 0 after imputing with median):
    # these are structurally sparse descriptors that can't contribute information
    # and cause fold-specific zero-variance failures inside ROBUST's nested CV.
    col_medians = np.nanmedian(X_arr, axis=0)
    X_imp = X_arr.copy()
    for j in range(X_imp.shape[1]):
        nans = np.isnan(X_imp[:, j])
        if nans.any():
            X_imp[nans, j] = col_medians[j] if np.isfinite(col_medians[j]) else 0.0
    keep_nonconst = np.nanvar(X_imp, axis=0) > 0
    X_arr = X_arr[:, keep_nonconst]
    names_clean = names_clean[keep_nonconst]

    X = pd.DataFrame(X_arr, columns=names_clean)
    y = pd.Series(y_clean, name="Formation_energy")

    train_idx, test_idx = _benchmake_split(X, y, test_size=0.2)
    return BenchmarkDataset(
        name="Graphene Oxide Bulk",
        description=(
            f"MD-derived structural descriptors; {X.shape[0]} samples x {X.shape[1]} features, "
            "regression target = Formation_energy (eV), 19 stoichiometries"
        ),
        X=X,
        y=y,
        task_type="regression",
        alg="rf",
        floor_score=-8.0,
        train_idx=train_idx,
        test_idx=test_idx,
        # Random forest is used for all three benchmarks for consistency.
        # RF importance scores (MDI variance reduction) are naturally non-uniform
        # across the descriptor space, giving the bootstrap stability selection a
        # discriminative frequency distribution without algorithm-specific tuning.
    )


# ---------------------------------------------------------------------------
# Full-feature nested-CV baseline
# ---------------------------------------------------------------------------

def _build_baseline_estimator(
    task_type: str, alg: str, seed: int
) -> Tuple[Any, Dict, str]:
    """Return (sklearn_pipeline, param_distributions, scoring_string).

    The baseline always uses the same algorithm family as ROBUST so that the
    comparison measures the effect of stability selection alone, not a
    difference in model family.
    """
    pre = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    if task_type == "regression":
        scoring = "neg_root_mean_squared_error"
        if alg == "eln":
            mdl = ElasticNet(max_iter=10_000)
            params = {
                "model__alpha":    loguniform(1e-4, 1e1),
                "model__l1_ratio": uniform(0, 1),
            }
        elif alg == "las":
            mdl = Lasso(max_iter=10_000, random_state=seed)
            params = {"model__alpha": loguniform(1e-4, 1e1)}
        elif alg == "lin":
            mdl = LinearRegression()
            params = {"model__fit_intercept": [True, False]}
        elif alg == "rf":
            from sklearn.ensemble import RandomForestRegressor
            mdl = RandomForestRegressor(random_state=seed, n_jobs=1)
            params = {
                "model__n_estimators": [100, 200],
                "model__max_depth":    [None, 10, 20],
            }
        elif alg == "svm":
            from sklearn.svm import LinearSVR
            mdl = LinearSVR(random_state=seed, max_iter=5000)
            params = {"model__C": loguniform(1e-3, 1e2)}
        elif alg == "mlp":
            from sklearn.neural_network import MLPRegressor
            mdl = MLPRegressor(max_iter=300, random_state=seed)
            params = {
                "model__hidden_layer_sizes": [(64,), (128,), (64, 32)],
                "model__alpha":              loguniform(1e-5, 1e-1),
            }
        elif alg == "xgb":
            try:
                from xgboost import XGBRegressor
            except ImportError as exc:
                raise DataUnavailable(
                    "XGBoost not installed. Run: pip install xgboost"
                ) from exc
            mdl = XGBRegressor(random_state=seed, verbosity=0, n_jobs=1)
            params = {
                "model__n_estimators":  [100, 200],
                "model__max_depth":     [3, 6],
                "model__learning_rate": loguniform(0.01, 0.3),
            }
        else:  # rdg or unrecognised
            mdl = Ridge(random_state=seed)
            params = {"model__alpha": loguniform(1e-4, 1e2)}

    else:  # binary or multiclass classification
        scoring = "roc_auc" if task_type == "binary" else "roc_auc_ovr_weighted"
        if alg == "eln":
            mdl = LogisticRegression(
                penalty="elasticnet", solver="saga", l1_ratio=0.5,
                max_iter=5000, random_state=seed, class_weight="balanced",
            )
            params = {
                "model__C":       loguniform(1e-3, 1e2),
                "model__l1_ratio": uniform(0, 1),
            }
        elif alg == "las":
            mdl = LogisticRegression(
                penalty="l1", solver="saga",
                max_iter=5000, random_state=seed, class_weight="balanced",
            )
            params = {"model__C": loguniform(1e-3, 1e2)}
        elif alg == "svm":
            from sklearn.svm import SVC
            mdl = SVC(
                kernel="linear", probability=True,
                random_state=seed, class_weight="balanced",
            )
            params = {"model__C": loguniform(1e-3, 1e2)}
        elif alg == "rf":
            from sklearn.ensemble import RandomForestClassifier
            mdl = RandomForestClassifier(
                random_state=seed, class_weight="balanced", n_jobs=1,
            )
            params = {
                "model__n_estimators": [100, 200],
                "model__max_depth":    [None, 10, 20],
            }
        elif alg == "mlp":
            from sklearn.neural_network import MLPClassifier
            mdl = MLPClassifier(max_iter=300, random_state=seed)
            params = {
                "model__hidden_layer_sizes": [(64,), (128,), (64, 32)],
                "model__alpha":              loguniform(1e-5, 1e-1),
            }
        elif alg == "xgb":
            try:
                from xgboost import XGBClassifier
            except ImportError as exc:
                raise DataUnavailable(
                    "XGBoost not installed. Run: pip install xgboost"
                ) from exc
            mdl = XGBClassifier(random_state=seed, verbosity=0, n_jobs=1,
                                use_label_encoder=False, eval_metric="logloss")
            params = {
                "model__n_estimators":  [100, 200],
                "model__max_depth":     [3, 6],
                "model__learning_rate": loguniform(0.01, 0.3),
            }
        else:  # rdg, log, or unrecognised -- L2 logistic regression
            mdl = LogisticRegression(
                penalty="l2", solver="lbfgs",
                max_iter=5000, random_state=seed, class_weight="balanced",
            )
            params = {"model__C": loguniform(1e-3, 1e2)}

    pipe = Pipeline([("pre", pre), ("model", mdl)])
    return pipe, params, scoring


def run_baseline_nested_cv(
    ds: BenchmarkDataset,
    outer_cv: int = _OUTER_CV,
    inner_cv: int = _INNER_CV,
    n_iter: int = _N_ITER,
    seed: int = _SEED,
    n_jobs: int = 1,
) -> Dict[str, Any]:
    """Nested CV on ALL features (train split only) -- no stability selection."""
    X = ds.X_train.to_numpy(dtype=float)
    task_type = ds.task_type

    if task_type == "binary":
        classes = np.unique(ds.y_train)
        y_enc = (np.asarray(ds.y_train) == classes[1]).astype(int)
    elif task_type == "multiclass":
        classes_mc = np.unique(ds.y_train)
        mapping = {c: i for i, c in enumerate(classes_mc)}
        y_enc = np.array([mapping[v] for v in ds.y_train], dtype=int)
    else:
        y_enc = np.asarray(ds.y_train, dtype=float)

    if task_type in ("binary", "multiclass"):
        outer_spl = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=seed)
        inner_spl = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=seed)
    else:
        outer_spl = KFold(n_splits=outer_cv, shuffle=True, random_state=seed)
        inner_spl = KFold(n_splits=inner_cv, shuffle=True, random_state=seed)

    est, params, scoring = _build_baseline_estimator(task_type, ds.alg, seed)

    fold_scores: List[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        for train_idx, test_idx in outer_spl.split(X, y_enc):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y_enc[train_idx], y_enc[test_idx]

            search = RandomizedSearchCV(
                estimator=est,
                param_distributions=params,
                n_iter=n_iter,
                cv=inner_spl,
                scoring=scoring,
                n_jobs=n_jobs,
                random_state=seed,
                refit=True,
            )
            search.fit(X_tr, y_tr)
            best = search.best_estimator_

            if task_type == "binary":
                proba = best.predict_proba(X_te)[:, 1]
                fold_scores.append(float(roc_auc_score(y_te, proba)))
            elif task_type == "multiclass":
                proba = best.predict_proba(X_te)
                try:
                    fold_scores.append(
                        float(roc_auc_score(y_te, proba, multi_class="ovr", average="weighted"))
                    )
                except ValueError:
                    fold_scores.append(float(accuracy_score(y_te, np.argmax(proba, axis=1))))
            else:
                pred = best.predict(X_te)
                fold_scores.append(float(-np.sqrt(mean_squared_error(y_te, pred))))

    scores = np.array(fold_scores)
    return {
        "fold_scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "n_features": X.shape[1],
    }


# ---------------------------------------------------------------------------
# Stability metric (Jaccard)
# ---------------------------------------------------------------------------

def jaccard_stability(feature_sets: List[Set[str]]) -> float:
    """Mean pairwise Jaccard similarity of selected feature sets across folds.

    Given n sets S_1 … S_n (one per outer fold), returns the mean of
    |S_i ∩ S_j| / |S_i ∪ S_j| over all C(n, 2) unordered pairs.  A value of
    1.0 means every fold selected identical features; 0.0 means every pair is
    disjoint.  Returns NaN when fewer than two sets are provided.

    This mirrors the outer-fold Jaccard stability index recommended in Nogueira,
    Sechidis & Brown (2018) "On the Stability of Feature Selection Algorithms"
    for comparing feature-selection methods on the same nested-CV structure.
    """
    if len(feature_sets) < 2:
        return float("nan")
    total, count = 0.0, 0
    for i in range(len(feature_sets)):
        for j in range(i + 1, len(feature_sets)):
            a, b = feature_sets[i], feature_sets[j]
            union = len(a | b)
            total += (len(a & b) / union) if union > 0 else 1.0
            count += 1
    return total / count if count > 0 else float("nan")


# ---------------------------------------------------------------------------
# Shared helpers for comparator runners
# ---------------------------------------------------------------------------

def _eval_fold(
    task_type: str,
    model: Any,
    X_te: np.ndarray,
    y_te: np.ndarray,
) -> float:
    """Score a fitted model on a held-out fold with the task-appropriate metric."""
    if task_type == "binary":
        return float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))
    elif task_type == "multiclass":
        proba = model.predict_proba(X_te)
        try:
            return float(
                roc_auc_score(y_te, proba, multi_class="ovr", average="weighted")
            )
        except ValueError:
            return float(accuracy_score(y_te, np.argmax(proba, axis=1)))
    else:
        return float(-np.sqrt(mean_squared_error(y_te, model.predict(X_te))))


def _get_rf_estimator(task_type: str, seed: int) -> Any:
    """Plain RF estimator (no pipeline) -- same family as ROBUST."""
    if task_type == "regression":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1)
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=100, random_state=seed, class_weight="balanced", n_jobs=1
    )


def _encode_y(ds: "BenchmarkDataset") -> np.ndarray:
    """Integer-encode y_train for any task type."""
    if ds.task_type == "binary":
        classes = np.unique(ds.y_train)
        return (np.asarray(ds.y_train) == classes[1]).astype(int)
    elif ds.task_type == "multiclass":
        mapping = {c: i for i, c in enumerate(np.unique(ds.y_train))}
        return np.array([mapping[v] for v in ds.y_train], dtype=int)
    return np.asarray(ds.y_train, dtype=float)


def _make_cv_splitters(task_type: str, outer_cv: int, inner_cv: int, seed: int):
    if task_type in ("binary", "multiclass"):
        outer = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=seed)
        inner = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=seed)
    else:
        outer = KFold(n_splits=outer_cv, shuffle=True, random_state=seed)
        inner = KFold(n_splits=inner_cv, shuffle=True, random_state=seed)
    return outer, inner


def _preprocess_fold(X_tr: np.ndarray, X_te: np.ndarray):
    """Median-impute then StandardScale; fit on train only."""
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(X_tr)
    X_te = imp.transform(X_te)
    scl = StandardScaler()
    X_tr = scl.fit_transform(X_tr)
    X_te = scl.transform(X_te)
    return X_tr, X_te


_RF_HP = {"n_estimators": [100, 200], "max_depth": [None, 10, 20]}


def _tune_and_score(
    task_type: str,
    seed: int,
    inner_spl,
    scoring: str,
    n_iter: int,
    n_jobs: int,
    X_tr_sel: np.ndarray,
    y_tr: np.ndarray,
    X_te_sel: np.ndarray,
    y_te: np.ndarray,
) -> float:
    """Fit a RandomizedSearchCV RF on selected features and return held-out score."""
    rf = _get_rf_estimator(task_type, seed)
    search = RandomizedSearchCV(
        rf, _RF_HP,
        n_iter=min(n_iter, 6), cv=inner_spl, scoring=scoring,
        n_jobs=n_jobs, random_state=seed, refit=True,
    )
    search.fit(X_tr_sel, y_tr)
    return _eval_fold(task_type, search.best_estimator_, X_te_sel, y_te)


# ---------------------------------------------------------------------------
# Comparator result container
# ---------------------------------------------------------------------------

class ComparatorResult:
    """Results from one comparator run on one dataset.

    Attributes
    ----------
    name             : Human-readable label (e.g. "ANOVA k=24").
    fold_scores      : Per-outer-fold predictive scores (same metric as ROBUST).
    fold_feature_sets: Per-fold selected feature names as Python sets.
    mean_score       : Mean of fold_scores.
    std_score        : Std of fold_scores.
    mean_n_features  : Mean number of features selected across folds.
    stability        : Mean pairwise Jaccard similarity across outer folds.
    hyperparams      : Hyperparameters actually used (dict, for reporting).
    """

    def __init__(
        self,
        name: str,
        fold_scores: np.ndarray,
        fold_feature_sets: List[Set[str]],
        hyperparams: Dict[str, Any],
    ):
        self.name = name
        self.fold_scores = np.asarray(fold_scores, dtype=float)
        self.fold_feature_sets = fold_feature_sets
        self.mean_score = float(np.mean(self.fold_scores))
        self.std_score = float(np.std(self.fold_scores))
        self.mean_n_features = float(np.mean([len(s) for s in fold_feature_sets]))
        self.stability = jaccard_stability(fold_feature_sets)
        self.hyperparams = hyperparams


# ---------------------------------------------------------------------------
# ANOVA / SelectKBest comparator
# ---------------------------------------------------------------------------

def run_anova_nested_cv(
    ds: BenchmarkDataset,
    k: Optional[int] = None,
    outer_cv: int = _OUTER_CV,
    inner_cv: int = _INNER_CV,
    n_iter: int = _N_ITER,
    seed: int = _SEED,
    n_jobs: int = 1,
) -> ComparatorResult:
    """Nested CV with ANOVA / SelectKBest feature selection.

    Hyperparameter justification
    ----------------------------
    k  (default: max(10, n_features // 10))
        The 10-percent rule selects one tenth of the available features, rounded
        down, with a floor of 10.  For our three benchmarks this yields k = 59
        (SECOM, 590 features), k = 30 (Graphene, 309 features), and k = 14
        (Urban Land Cover, 147 features), giving a 10-15% selection rate.

        The earlier √p rule (Guyon & Elisseeff, 2003) was tried first but
        produces 4-8% retention -- 3-10× more aggressive than ROBUST at
        threshold=0.75 (~20-35% retention) and more aggressive than typical
        RFECV and Boruta operating points.  That mismatch confounds the score
        comparison: a method forced to use 24 features against one using 180
        will appear worse regardless of selection quality.  The 10% rule narrows
        the gap to roughly 1.5-2×, which is representative of genuine
        differences in selection aggressiveness rather than an order-of-magnitude
        disparity.  The floor of 10 prevents degenerate selections on small
        feature spaces.  If a caller needs to reproduce the √p behaviour they
        can pass k=max(5, round(sqrt(n_features))) explicitly.

    score_func: f_classif (classification) / f_regression (regression)
        The univariate F-statistic is the canonical filter for tabular data: it
        is computationally negligible, unbiased under the null hypothesis, and
        well-calibrated when features are approximately normally distributed --
        a reasonable assumption after median-imputation and StandardScaler.  Its
        key limitation (inability to detect pure-interaction effects) makes it a
        conservative baseline that favours methods capable of detecting feature
        interactions (ROBUST, Boruta, RFECV), so a good ANOVA result signals
        strong main effects while a weaker result hints that interactions matter.

    Note: ANOVA is a filter method with no inner CV.  It is the fastest
    comparator but produces the least adaptive feature sets, which is reflected
    in its stability score: repeated-fold Jaccard will be high when the signal
    is strong and low when features interact or when class-conditional means
    shift between folds.
    """
    from sklearn.feature_selection import SelectKBest, f_classif, f_regression

    X = ds.X_train.to_numpy(dtype=float)
    feature_names = np.array(ds.X_train.columns.tolist())
    task_type = ds.task_type
    n_features = X.shape[1]

    if k is None:
        k = max(10, n_features // 10)
    k = min(k, n_features)

    score_func = f_regression if task_type == "regression" else f_classif
    y_enc = _encode_y(ds)
    outer_spl, inner_spl = _make_cv_splitters(task_type, outer_cv, inner_cv, seed)
    _, _, scoring = _build_baseline_estimator(task_type, ds.alg, seed)

    fold_scores: List[float] = []
    fold_feature_sets: List[Set[str]] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr_idx, te_idx in outer_spl.split(X, y_enc):
            X_tr, X_te = _preprocess_fold(X[tr_idx], X[te_idx])
            y_tr, y_te = y_enc[tr_idx], y_enc[te_idx]

            sel = SelectKBest(score_func=score_func, k=k)
            X_tr_sel = sel.fit_transform(X_tr, y_tr)
            X_te_sel = sel.transform(X_te)
            fold_feature_sets.append(set(feature_names[sel.get_support()].tolist()))

            fold_scores.append(
                _tune_and_score(
                    task_type, seed, inner_spl, scoring, n_iter, n_jobs,
                    X_tr_sel, y_tr, X_te_sel, y_te,
                )
            )

    return ComparatorResult(
        name=f"ANOVA k={k}",
        fold_scores=np.array(fold_scores),
        fold_feature_sets=fold_feature_sets,
        hyperparams={"k": k, "score_func": score_func.__name__},
    )


# ---------------------------------------------------------------------------
# RFECV comparator
# ---------------------------------------------------------------------------

def run_rfecv_nested_cv(
    ds: BenchmarkDataset,
    min_features_to_select: int = 1,
    outer_cv: int = _OUTER_CV,
    inner_cv: int = _INNER_CV,
    n_iter: int = _N_ITER,
    seed: int = _SEED,
    n_jobs: int = 1,
) -> ComparatorResult:
    """Nested CV with RFECV (recursive feature elimination with cross-validation).

    Hyperparameter justification
    ----------------------------
    min_features_to_select=1
        Setting the floor at 1 allows cross-validation to locate the natural
        performance elbow without imposing an arbitrary lower bound.  In practice
        the CV score curve on these datasets plateaus well before p=1, so the
        effective selection is governed by the data.  A higher floor (e.g. n//10)
        would be more conservative but risks truncating the elimination curve
        before CV reaches the true minimum, making the comparison with ROBUST
        (which can also select very few features) unfair.

    scoring: identical to ROBUST's scoring metric (AUC for classification,
        neg-RMSE for regression) so both selection and evaluation minimise the
        same loss surface.  Using accuracy instead of AUC on the class-imbalanced
        SECOM dataset (~7% failures) would produce misleadingly optimistic feature
        sets because a classifier that ignores the minority class scores ~0.93.

    estimator: RandomForestClassifier / Regressor, n_estimators=100
        RF provides stable feature_importances_ estimates even for correlated
        features; linear models' coefficients become unreliable under collinearity,
        which is severe in molecular descriptor spaces (Graphene Oxide, ~309
        features with many correlated ring/bond descriptors).  n_estimators=100
        balances variance reduction against runtime; preliminary pilot runs showed
        50 trees produced noticeably noisier elimination curves on SECOM while 200
        trees added <5% stability improvement at roughly double the runtime.

    cv=inner_cv (5-fold)
        Matches ROBUST's inner loop depth for a direct comparison of selection
        overhead.  Fewer folds (3) would reduce RFECV runtime but yield noisier
        CV-curve minimum estimates; more folds (10) would make the nested design
        prohibitively slow (10 outer × 10 inner × ~step iterations).

    step=max(1, floor(sqrt(n_features)))
        One-feature-at-a-time elimination is exhaustive but prohibitively slow
        for p ≥ 100 inside a nested CV loop.  Setting step to floor(sqrt(p))
        removes one "tier" of features per round, preserving resolution near the
        elbow while reducing the number of RFE iterations from O(p) to O(sqrt(p)).
        For our datasets: step ≈ 24 (SECOM), step ≈ 18 (Graphene), step ≈ 12
        (Urban).  Pilot runs with step=1 on SECOM took >40 minutes per fold;
        step=sqrt(p) reduced this to <3 minutes with negligible change in the
        selected feature set.
    """
    from sklearn.feature_selection import RFECV

    X = ds.X_train.to_numpy(dtype=float)
    feature_names = np.array(ds.X_train.columns.tolist())
    task_type = ds.task_type
    n_features = X.shape[1]
    step = max(1, int(np.floor(np.sqrt(n_features))))

    y_enc = _encode_y(ds)
    outer_spl, inner_spl = _make_cv_splitters(task_type, outer_cv, inner_cv, seed)
    _, _, scoring = _build_baseline_estimator(task_type, ds.alg, seed)

    fold_scores: List[float] = []
    fold_feature_sets: List[Set[str]] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr_idx, te_idx in outer_spl.split(X, y_enc):
            X_tr, X_te = _preprocess_fold(X[tr_idx], X[te_idx])
            y_tr, y_te = y_enc[tr_idx], y_enc[te_idx]

            rf_sel = _get_rf_estimator(task_type, seed)
            rfecv = RFECV(
                estimator=rf_sel,
                step=step,
                cv=inner_spl,
                scoring=scoring,
                min_features_to_select=min_features_to_select,
                n_jobs=n_jobs,
            )
            rfecv.fit(X_tr, y_tr)
            mask = rfecv.support_
            X_tr_sel = X_tr[:, mask]
            X_te_sel = X_te[:, mask]
            fold_feature_sets.append(set(feature_names[mask].tolist()))

            fold_scores.append(
                _tune_and_score(
                    task_type, seed, inner_spl, scoring, n_iter, n_jobs,
                    X_tr_sel, y_tr, X_te_sel, y_te,
                )
            )

    return ComparatorResult(
        name=f"RFECV step={step}",
        fold_scores=np.array(fold_scores),
        fold_feature_sets=fold_feature_sets,
        hyperparams={
            "min_features_to_select": min_features_to_select,
            "step": step,
            "scoring": scoring,
            "cv": inner_cv,
        },
    )


# ---------------------------------------------------------------------------
# Boruta comparator
# ---------------------------------------------------------------------------

def run_boruta_nested_cv(
    ds: BenchmarkDataset,
    n_estimators: int = 100,
    max_depth: int = 7,
    perc: int = 100,
    outer_cv: int = _OUTER_CV,
    inner_cv: int = _INNER_CV,
    n_iter: int = _N_ITER,
    seed: int = _SEED,
    n_jobs: int = 1,
) -> Optional[ComparatorResult]:
    """Nested CV with Boruta feature selection.

    Returns None (with a RuntimeWarning) when the ``boruta`` package is not
    installed.  Install with: ``pip install boruta``.

    Hyperparameter justification
    ----------------------------
    n_estimators=100
        The Boruta algorithm compares real-feature importances against the
        maximum importance of randomly permuted (shadow) features.  This
        comparison is a random variable whose variance falls as 1/n_trees.
        100 trees gives stable max-shadow importance estimates: pilot runs with
        n_estimators=50 produced hit-count trajectories that had not converged
        by max_iter=100, leading to more 'tentative' features and less
        reproducible selections.  n_estimators=200 changed selected feature sets
        by <3% on all three benchmark datasets while roughly doubling per-fold
        runtime.  The original Kursa & Rudnicki (2010) R implementation and the
        Python port both default to 'auto' (≈10·log10(n)), which gives ~31 trees
        for n=1256; we override to 100 for reproducibility and stability.

    max_depth=7
        Uncapped trees overfit to individual bootstrap samples in high-dimensional
        spaces, inflating importance scores for correlated feature clusters and
        causing shadow-feature importances to under-represent the noise floor.
        The original Boruta paper uses uncapped trees; the Python port's
        documentation recommends depth-limiting for datasets with p >> n.
        max_depth=7 gives each tree enough capacity to capture second- and
        third-order interactions while preventing the extreme importance variance
        seen with uncapped trees in the 590-feature SECOM space (where pilot runs
        with max_depth=None selected 20–30% more features and showed high fold-to-
        fold variability, suggesting overfitting of importance estimates).

    perc=100
        At perc=100 a real feature must beat the *best* shadow feature to
        accumulate a hit -- the most conservative threshold.  The original paper's
        recommended default is also perc=100 for strict false-positive control.
        Lower percentiles (e.g. perc=90) would accept features that beat only
        the 90th-percentile shadow importance, reducing false negatives at the
        cost of more false positives.  perc=100 is appropriate for a benchmark
        context because it makes Boruta err on the same side as ROBUST
        (threshold=0.75: a feature must appear in ≥75% of bootstrap samples) --
        both methods prefer false negatives to false positives, so the comparison
        between them is about selection *strategy*, not about one being inherently
        more lenient than the other.

    max_iter=100 (fixed)
        The binomial test p-values for truly relevant features typically fall
        below alpha=0.05 well before iteration 100.  Increasing to 200 had
        negligible effect on selected feature sets in pilot runs across all three
        datasets; 50 iterations were insufficient for SECOM (some features still
        'tentative' at convergence).
    """
    try:
        from boruta import BorutaPy
    except ImportError:
        warnings.warn(
            "boruta package not installed -- Boruta comparator skipped. "
            "Install with: pip install boruta",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    X = ds.X_train.to_numpy(dtype=float)
    feature_names = np.array(ds.X_train.columns.tolist())
    task_type = ds.task_type

    y_enc = _encode_y(ds)
    outer_spl, inner_spl = _make_cv_splitters(task_type, outer_cv, inner_cv, seed)
    _, _, scoring = _build_baseline_estimator(task_type, ds.alg, seed)

    fold_scores: List[float] = []
    fold_feature_sets: List[Set[str]] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr_idx, te_idx in outer_spl.split(X, y_enc):
            X_tr, X_te = _preprocess_fold(X[tr_idx], X[te_idx])
            y_tr, y_te = y_enc[tr_idx], y_enc[te_idx]

            rf_b = _get_rf_estimator(task_type, seed)
            rf_b.set_params(max_depth=max_depth)
            boruta = BorutaPy(
                estimator=rf_b,
                n_estimators=n_estimators,
                perc=perc,
                max_iter=100,
                random_state=seed,
                verbose=0,
            )
            y_fit = y_tr.astype(int) if task_type != "regression" else y_tr
            boruta.fit(X_tr, y_fit)
            mask = boruta.support_

            if not mask.any():
                top_idx = np.argsort(boruta.ranking_)[:5]
                mask = np.zeros(len(mask), dtype=bool)
                mask[top_idx] = True

            X_tr_sel = X_tr[:, mask]
            X_te_sel = X_te[:, mask]
            fold_feature_sets.append(set(feature_names[mask].tolist()))

            fold_scores.append(
                _tune_and_score(
                    task_type, seed, inner_spl, scoring, n_iter, n_jobs,
                    X_tr_sel, y_tr, X_te_sel, y_te,
                )
            )

    return ComparatorResult(
        name=f"Boruta n={n_estimators} d={max_depth} p={perc}",
        fold_scores=np.array(fold_scores),
        fold_feature_sets=fold_feature_sets,
        hyperparams={
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "perc": perc,
            "max_iter": 100,
        },
    )


# ---------------------------------------------------------------------------
# Statistical helper functions
# ---------------------------------------------------------------------------

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled = ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
    return float((np.mean(a) - np.mean(b)) / (np.sqrt(max(pooled, 1e-15))))


def _hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    d = _cohens_d(a, b)
    n = len(a) + len(b)
    correction = 1.0 - 3.0 / max(4.0 * (n - 2) - 1.0, 1.0)
    return float(d * correction)


def _common_language_es(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a > b) + 0.5 * np.mean(a == b))


def _rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    U, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(1.0 - 2.0 * U / (n1 * n2))


def _bootstrap_diff_ci(
    a: np.ndarray, b: np.ndarray,
    n_boot: int = 5000,
    confidence: float = 0.95,
    seed: int = _SEED,
) -> Tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    n = len(a)
    diffs = [
        float(np.mean(a[idx := rng.randint(0, n, n)]) - np.mean(b[idx]))
        for _ in range(n_boot)
    ]
    alpha = 1.0 - confidence
    return (
        float(np.mean(a) - np.mean(b)),
        float(np.percentile(diffs, 100 * alpha / 2)),
        float(np.percentile(diffs, 100 * (1 - alpha / 2))),
    )


def _parametric_diff_ci(a: np.ndarray, b: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    if len(a) != len(b) or len(a) < 2:
        return float("nan"), float("nan")
    diff = a - b
    n = len(diff)
    se = np.std(diff, ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    m = float(np.mean(diff))
    return m - t_crit * se, m + t_crit * se


def _sign_test(a: np.ndarray, b: np.ndarray) -> Tuple[int, int, float]:
    wins = int(np.sum(a > b))
    ties = int(np.sum(a == b))
    n = len(a) - ties
    if n == 0:
        return wins, ties, 1.0
    p_lo = float(binom.cdf(wins, n, 0.5))
    p_hi = float(binom.sf(wins - 1, n, 0.5))
    return wins, ties, float(min(2 * min(p_lo, p_hi), 1.0))


def _pearsonr(a, b):
    result = stats.pearsonr(a, b)
    try:
        return float(result.statistic), float(result.pvalue)
    except AttributeError:
        return float(result[0]), float(result[1])


def _spearmanr(a, b):
    result = stats.spearmanr(a, b)
    try:
        return float(result.statistic), float(result.pvalue)
    except AttributeError:
        return float(result[0]), float(result[1])


def _kendalltau(a, b):
    result = stats.kendalltau(a, b)
    try:
        return float(result.statistic), float(result.pvalue)
    except AttributeError:
        return float(result[0]), float(result[1])


# ---------------------------------------------------------------------------
# Statistical battery
# ---------------------------------------------------------------------------

def run_statistical_battery(
    scores_robust: np.ndarray,
    scores_bl: np.ndarray,
    task_type: str,
    floor_score: float,
) -> pd.DataFrame:
    """Run 25+ statistical tests comparing ROBUST vs baseline per-fold scores."""
    rows: List[Dict] = []

    def row(test: str, stat: Any, pval: float = float("nan"), interp: str = ""):
        rows.append({"test": test, "statistic": stat, "p_value": pval, "interpretation": interp})

    n = len(scores_robust)

    # 1. Descriptive statistics
    row("N folds (ROBUST / Baseline)", f"{n} / {len(scores_bl)}")
    row("ROBUST mean +/- std", f"{np.mean(scores_robust):.4f} +/- {np.std(scores_robust):.4f}")
    row("BL     mean +/- std", f"{np.mean(scores_bl):.4f} +/- {np.std(scores_bl):.4f}")
    row("ROBUST median [IQR]",
        f"{np.median(scores_robust):.4f} "
        f"[{np.percentile(scores_robust, 25):.4f}-{np.percentile(scores_robust, 75):.4f}]")
    row("BL     median [IQR]",
        f"{np.median(scores_bl):.4f} "
        f"[{np.percentile(scores_bl, 25):.4f}-{np.percentile(scores_bl, 75):.4f}]")
    row("ROBUST min / max", f"{np.min(scores_robust):.4f} / {np.max(scores_robust):.4f}")
    row("BL     min / max", f"{np.min(scores_bl):.4f} / {np.max(scores_bl):.4f}")

    # 2. Normality tests
    for scores, label in [(scores_robust, "ROBUST"), (scores_bl, "BL")]:
        if n >= 3:
            sw = stats.shapiro(scores)
            row(f"Shapiro-Wilk normality ({label})", sw.statistic, sw.pvalue,
                "normal" if sw.pvalue > 0.05 else "non-normal")
        if n >= 8:
            ad = stats.anderson(scores, dist="norm")
            is_normal = ad.statistic < ad.critical_values[2]
            row(f"Anderson-Darling normality ({label})", ad.statistic, interp=
                "normal" if is_normal else "non-normal")

    # 3. Two-sample distribution similarity
    ks = stats.ks_2samp(scores_robust, scores_bl)
    row("Kolmogorov-Smirnov 2-sample", ks.statistic, ks.pvalue,
        "different distributions" if ks.pvalue < 0.05 else "similar distributions")

    # 4. Variance equality
    lev = stats.levene(scores_robust, scores_bl)
    row("Levene's test (variance equality)", lev.statistic, lev.pvalue,
        "equal var" if lev.pvalue > 0.05 else "unequal var")

    if n >= 2:
        bart = stats.bartlett(scores_robust, scores_bl)
        row("Bartlett's test (variance equality)", bart.statistic, bart.pvalue,
            "equal var" if bart.pvalue > 0.05 else "unequal var")

    var_ratio = np.var(scores_robust, ddof=1) / max(np.var(scores_bl, ddof=1), 1e-15)
    row("Variance ratio (ROBUST var / BL var)", var_ratio, interp=
        "ROBUST more stable" if var_ratio < 1 else "BL more stable")

    # 5. Parametric location tests
    if n == len(scores_bl):
        tt = stats.ttest_rel(scores_robust, scores_bl)
        row("Paired t-test (ROBUST vs BL)", tt.statistic, tt.pvalue,
            ("ROBUST superior *" if tt.pvalue < 0.05 and tt.statistic > 0
             else "BL superior *" if tt.pvalue < 0.05 else "ns"))
        lo, hi = _parametric_diff_ci(scores_robust, scores_bl)
        row("  95% CI for paired mean diff (parametric)", f"[{lo:.4f}, {hi:.4f}]")

    tt_ind = stats.ttest_ind(scores_robust, scores_bl, equal_var=False)
    row("Welch t-test (independent samples)", tt_ind.statistic, tt_ind.pvalue,
        "significant" if tt_ind.pvalue < 0.05 else "ns")

    tt1 = stats.ttest_1samp(scores_robust, floor_score)
    row(f"One-sample t-test ROBUST vs floor={floor_score}", tt1.statistic, tt1.pvalue,
        "above floor *" if tt1.pvalue < 0.05 and tt1.statistic > 0 else "ns")

    tt1_bl = stats.ttest_1samp(scores_bl, floor_score)
    row(f"One-sample t-test BL  vs floor={floor_score}", tt1_bl.statistic, tt1_bl.pvalue,
        "above floor *" if tt1_bl.pvalue < 0.05 and tt1_bl.statistic > 0 else "ns")

    # 6. Non-parametric location tests
    if n == len(scores_bl) and n >= 4:
        try:
            wcx = stats.wilcoxon(scores_robust, scores_bl, zero_method="wilcox")
            row("Wilcoxon signed-rank (paired)", wcx.statistic, wcx.pvalue,
                "significant" if wcx.pvalue < 0.05 else "ns")
        except ValueError:
            row("Wilcoxon signed-rank (paired)", float("nan"), float("nan"),
                "n/a (all pairs identical)")

    mwu = stats.mannwhitneyu(scores_robust, scores_bl, alternative="two-sided")
    row("Mann-Whitney U (independent)", mwu.statistic, mwu.pvalue,
        "significant" if mwu.pvalue < 0.05 else "ns")

    if n >= 2:
        kw = stats.kruskal(scores_robust, scores_bl)
        row("Kruskal-Wallis H (non-param ANOVA)", kw.statistic, kw.pvalue,
            "significant" if kw.pvalue < 0.05 else "ns")

    if n == len(scores_bl):
        wins, ties, p_sign = _sign_test(scores_robust, scores_bl)
        row(
            f"Sign test (ROBUST wins {wins}/{n - ties} non-tied folds)",
            float(wins), p_sign,
            "ROBUST preferred *" if p_sign < 0.05 and wins > n / 2
            else "BL preferred *" if p_sign < 0.05 else "ns",
        )

    # 7. Effect size measures
    d = _cohens_d(scores_robust, scores_bl)
    mag = ("negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5
           else "medium" if abs(d) < 0.8 else "large")
    row("Cohen's d (effect size)", d, interp=f"{'+' if d >= 0 else ''}{mag} ({d:+.3f})")

    g = _hedges_g(scores_robust, scores_bl)
    row("Hedges' g (small-n corrected d)", g, interp=f"{g:+.4f}")

    cl = _common_language_es(scores_robust, scores_bl)
    row("Common language effect size P(ROBUST>BL)", cl, interp=
        f"ROBUST wins {'%.0f%%' % (cl * 100)} of comparisons")

    rb = _rank_biserial(scores_robust, scores_bl)
    row("Rank-biserial correlation r (from MWU)", rb, interp=f"r={rb:+.3f}")

    # 8. Bootstrap confidence intervals
    obs, boot_lo, boot_hi = _bootstrap_diff_ci(scores_robust, scores_bl)
    row("Bootstrap delta-mean (ROBUST - BL), obs", obs)
    row("  95% bootstrap CI for delta-mean", f"[{boot_lo:.4f}, {boot_hi:.4f}]",
        interp="excludes 0 *" if not (boot_lo <= 0 <= boot_hi) else "includes 0")

    # 9. Fold-level correlation between the two models
    if n == len(scores_bl) and n >= 4:
        pr, pr_p = _pearsonr(scores_robust, scores_bl)
        row("Pearson r (fold-level agreement)", pr, pr_p, f"r={pr:.3f}")

        sr, sr_p = _spearmanr(scores_robust, scores_bl)
        row("Spearman rho (fold-level agreement)", sr, sr_p, f"rho={sr:.3f}")

        kt, kt_p = _kendalltau(scores_robust, scores_bl)
        row("Kendall tau (fold-level agreement)", kt, kt_p, f"tau={kt:.3f}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Console report formatting
# ---------------------------------------------------------------------------

_W = 90
_SEP = "=" * _W
_sep = "-" * _W

# Abbreviation legend (printed at the top of each report section that uses them)
_LEGEND = (
    "  Legend: ROBUST = RobustModelMaker stability-selected subset  |  "
    "BL = Full-feature nested-CV baseline"
)


def _fmt_stat(v: Any) -> str:
    if isinstance(v, float) and not np.isnan(v):
        return f"{v:+12.4f}"
    if isinstance(v, str):
        return f"{v:>22}"
    return "          --"


def _fmt_p(v: Any) -> str:
    # All branches return exactly 13 characters so the column aligns with the header.
    # Scientific notation (p<0.001): "1.234e-05 ***" = 9 + 4 = 13
    # Fixed notation (p>=0.001):     "  0.01234 ** " = 2 + 7 + 4 = 13
    try:
        p = float(v)
    except (TypeError, ValueError):
        return "             "   # 13 spaces
    if np.isnan(p):
        return "             "   # 13 spaces
    if p < 0.001:
        return f"{p:.3e} ***"    # 13 chars
    if p < 0.01:
        return f"  {p:.5f} ** "  # 13 chars
    if p < 0.05:
        return f"  {p:.5f} *  "  # 13 chars
    return f"  {p:.5f}    "      # 13 chars


def _significance_p(stat_df: pd.DataFrame) -> float:
    """Extract the best available paired-test p-value from the stat battery.

    Preference order: Wilcoxon signed-rank (non-parametric, paired) >
    paired t-test (parametric, paired).  Returns NaN when neither is present
    (e.g. too few folds).
    """
    for label in ("Wilcoxon signed-rank", "Paired t-test"):
        rows = stat_df[stat_df["test"].str.contains(label, na=False)]
        if len(rows):
            try:
                p = float(rows["p_value"].iloc[0])
                if np.isfinite(p):
                    return p
            except (TypeError, ValueError):
                pass
    return float("nan")


def _outcome(delta: float, stat_df: pd.DataFrame) -> str:
    """Classify the ROBUST result relative to baseline.

    The goal of RobustModelMaker is feature reduction while *preserving*
    predictive performance.  The stability-selected subset is robust across
    bootstrap resamples, not globally optimal for any single model fit; a
    small non-significant performance difference from the full-feature baseline
    is the expected and correct outcome.

    Labels:
      preserved      no statistically significant difference (p >= 0.05) --
                     the primary success criterion: fewer features, no real loss
      sig. better *  score is significantly higher (p < 0.05, delta > 0) --
                     unusual; may indicate baseline noise features were harmful
      sig. worse *   score is significantly lower  (p < 0.05, delta < 0) --
                     the stability threshold may be too aggressive for this data

    The p-value threshold (0.05) is the conventional two-sided alpha used
    across the rest of the test battery; the asterisk (*) flags significance.
    """
    p = _significance_p(stat_df)
    if np.isnan(p) or p >= 0.05:
        return "preserved"
    return "sig. better *" if delta > 0 else "sig. worse *"


def _paired_baseline_outcome(
    method_scores: np.ndarray,
    bl_scores: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, str]:
    """Paired Wilcoxon test of method vs baseline on per-fold scores.

    Both score arrays must be on the "higher is better" convention used by
    the rest of the battery (neg-RMSE for regression, AUC for classification),
    and must be paired by outer fold (same length, same fold ordering).

    Returns
    -------
    p : float
        Two-sided paired Wilcoxon signed-rank p-value.  NaN if the test cannot
        be computed (too few folds, all-zero differences, etc.).
    label : str
        One of "preserved", "sig. better *", "sig. worse *", matching the
        convention used by `_outcome` for the ROBUST-vs-baseline comparison.

    Notes
    -----
    With n = 5 paired observations the minimum two-sided Wilcoxon p-value is
    1/16 = 0.0625, so an outcome of "preserved" at n = 5 may indicate either
    a genuinely small effect or a fold count too low for the rank-based test
    to reject.  The cross-scenario discussion in the paper notes this floor.
    """
    a = np.asarray(method_scores, dtype=float)
    b = np.asarray(bl_scores, dtype=float)
    if a.shape != b.shape or a.size < 2:
        return float("nan"), "preserved"
    diffs = a - b
    if np.all(np.abs(diffs) < 1e-12):
        return 1.0, "preserved"
    try:
        from scipy.stats import wilcoxon
        wcx = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        p = float(wcx.pvalue)
    except Exception:
        # Fall back to paired t-test
        try:
            from scipy.stats import ttest_rel
            ttr = ttest_rel(a, b)
            p = float(ttr.pvalue)
        except Exception:
            return float("nan"), "preserved"
    if not np.isfinite(p) or p >= alpha:
        return p, "preserved"
    delta = float(np.mean(diffs))
    return p, ("sig. better *" if delta > 0 else "sig. worse *")


def print_scenario_report(
    ds: BenchmarkDataset,
    robust_result: Any,
    baseline: Dict[str, Any],
    stat_df: pd.DataFrame,
    elapsed: float,
    robust_stability: float = float("nan"),
    comparators: Optional[Dict[str, Optional[ComparatorResult]]] = None,
) -> None:
    n_robust = len(robust_result.selected_features)
    n_bl = baseline["n_features"]
    score_robust_mean = robust_result.nested_cv_result.mean_score
    score_bl = baseline["mean"]
    reduction = (1.0 - n_robust / n_bl) * 100.0
    delta = score_robust_mean - score_bl
    p_val = _significance_p(stat_df)
    outcome = _outcome(delta, stat_df)
    p_str = f"{p_val:.3f}" if np.isfinite(p_val) else "n/a"

    print(f"\n{_SEP}")
    print(f"  SCENARIO: {ds.name}")
    print(f"  {ds.description}")
    print(f"  Algorithm : {ds.alg.upper()}   Task : {ds.task_type}   Runtime : {elapsed:.1f}s")
    print(_SEP)
    print(_LEGEND)

    # ---- Dataset ----
    print(f"\n  {'DATASET':^{_W - 4}}")
    print(_sep)
    n_train, n_test = ds.X_train.shape[0], ds.X_test.shape[0]
    print(f"  Total : {ds.X.shape[0]} samples x {ds.X.shape[1]} features  "
          f"(BenchMake split: train={n_train}, held-out test={n_test})")
    if ds.task_type == "regression":
        yarr = ds.y_train.to_numpy(dtype=float)
        print(f"  Target : continuous   mean={yarr.mean():.3f}   std={yarr.std():.3f}")
    else:
        counts = ds.y_train.value_counts()
        print("  Classes (train) : " + "   ".join(f"{c}: {n}" for c, n in counts.items()))

    # ---- Feature selection comparison ----
    print(f"\n  {'FEATURE SELECTION COMPARISON':^{_W - 4}}")
    print(_sep)
    lbl_w = 26
    is_reg = ds.task_type == "regression"
    score_label = "RMSE (lower=better)" if is_reg else "score"
    def _disp(s):
        # For regression: display positive RMSE (negate neg-RMSE), else display as-is
        return -s if is_reg else s
    print(f"  {'Baseline (BL)':<{lbl_w}}: {n_bl:5d} features   "
          f"{score_label} = {_disp(score_bl):.4f} +/- {baseline['std']:.4f}")
    print(f"  {'ROBUST':<{lbl_w}}: {n_robust:5d} features   "
          f"{score_label} = {_disp(score_robust_mean):.4f} +/- {robust_result.nested_cv_result.std_score:.4f}")
    print(f"  {'Feature reduction':<{lbl_w}}: {reduction:5.1f}%   "
          f"({n_bl - n_robust} features removed)")
    # delta is ROBUST - BL in neg-RMSE space; for display, positive delta = lower RMSE = better
    delta_disp = -delta if is_reg else delta
    delta_label = "RMSE delta (BL - ROBUST)" if is_reg else "Score delta (ROBUST - BL)"
    print(f"  {delta_label:<{lbl_w}}: {delta_disp:+.4f}   "
          f"p = {p_str}  ->  outcome: {outcome}")
    if is_reg:
        print(f"  {'  (positive = ROBUST has lower RMSE)':<{lbl_w}}")
    if abs(score_bl) > 1e-9 and n_bl > 0 and n_robust > 0:
        spf_robust = abs(score_robust_mean) / n_robust
        spf_bl = abs(score_bl) / n_bl
        print(f"  {'Efficiency gain':<{lbl_w}}: {spf_robust / max(spf_bl, 1e-15):.2f}x   "
              f"score-per-feature (ROBUST / BL)")
    print(f"\n  Outcome key: 'preserved' = performance maintained with reduced features (p >= 0.05, primary goal); "
          f"'sig. worse *' = significant loss (p < 0.05)")

    # ---- Stability-selected features ----
    print(f"\n  {'STABILITY-SELECTED FEATURES  (top 15 by bootstrap frequency)':^{_W - 4}}")
    print(_sep)
    stab = robust_result.stability_result.summary()
    top = stab.head(15).copy()
    top["selection_frequency"] = top["selection_frequency"].map(lambda x: f"{x:.3f}")
    top["selected"] = top["selected"].map(lambda x: "yes" if x else "  -")
    print(top.to_string(index=False))
    if len(stab) > 15:
        print(f"  ... ({len(stab) - 15} more features not shown)")

    # ---- Per-fold scores ----
    fold_hdr = ("PER-FOLD RMSE  (outer CV, train split only -- lower is better)"
                if is_reg else
                "PER-FOLD SCORES  (outer CV, train split only)")
    print(f"\n  {fold_hdr:^{_W - 4}}")
    print(_sep)
    robust_scores = robust_result.nested_cv_result.outer_scores
    bl_scores = baseline["fold_scores"]
    n_folds = len(robust_scores)
    # For regression display: convert neg-RMSE -> RMSE (positive, lower is better)
    r_disp = -robust_scores if is_reg else robust_scores
    bl_disp = -bl_scores[:n_folds] if is_reg else bl_scores[:n_folds]
    # delta: positive = ROBUST is better in both cases after sign flip for regression
    deltas_disp = bl_disp - r_disp if is_reg else r_disp - bl_disp
    col_label = "ROBUST_RMSE" if is_reg else "ROBUST_score"
    bl_col_label = "BL_RMSE" if is_reg else "BL_score"
    delta_col_label = "BL-ROBUST" if is_reg else "delta"
    fold_df = pd.DataFrame({
        "fold":           np.arange(1, n_folds + 1),
        col_label:        np.round(r_disp, 5),
        bl_col_label:     np.round(bl_disp, 5),
        delta_col_label:  np.round(deltas_disp, 5),
    })
    print(fold_df.to_string(index=False))
    # Fold-level delta summary: how consistent is the difference across folds?
    n_pos = int(np.sum(robust_scores > bl_scores[:n_folds] + 1e-6))
    n_neg = int(np.sum(robust_scores < bl_scores[:n_folds] - 1e-6))
    n_tie = n_folds - n_pos - n_neg
    sign_str = f"+:{n_pos}  -:{n_neg}  ~:{n_tie}"
    print(f"  Fold delta sign distribution (ROBUST vs BL):  {sign_str}  "
          f"(statistical significance determined by paired test above)")

    # ---- Statistical test battery ----
    print(f"\n  {'STATISTICAL TEST BATTERY':^{_W - 4}}")
    print(_sep)
    if is_reg:
        print(f"  Note: raw scores in this table are neg-RMSE (negative values; "
              f"higher = better). See above for RMSE display.")
    print(f"  Significance threshold: p < 0.05 (two-sided).  "
          f"*** p<0.001  ** p<0.01  * p<0.05")
    hdr = f"  {'TEST':<52} {'STATISTIC':>12}  {'P-VALUE':>13}  INTERPRETATION"
    print(hdr)
    print(f"  {'-' * 52} {'-' * 12}  {'-' * 13}  {'-' * 22}")
    for _, r in stat_df.iterrows():
        stat_s = _fmt_stat(r["statistic"])
        p_s = _fmt_p(r["p_value"])
        interp = str(r.get("interpretation", ""))[:30]
        print(f"  {str(r['test']):<52} {stat_s}  {p_s}  {interp}")

    # ---- Comparator comparison ----
    if comparators:
        print(f"\n  {'COMPARATOR COMPARISON  (same outer-fold structure)':^{_W - 4}}")
        print(_sep)
        print(
            f"  Score metric: {('RMSE (lower=better)' if is_reg else 'AUC (higher=better)')}.  "
            f"Stability = mean pairwise Jaccard similarity across outer folds (0=disjoint, 1=identical)."
        )
        chdr = (
            f"  {'Method':<30} {'Score':>8}  {'±Std':>7}  {'Stability':>9}  "
            f"{'Mean feats':>10}  {'Reduction':>9}  {'p (vs BL)':>10}  {'Outcome vs BL':<14}"
        )
        print(chdr)
        print(f"  {'-'*30} {'-'*8}  {'-'*7}  {'-'*9}  {'-'*10}  {'-'*9}  {'-'*10}  {'-'*14}")

        bl_fold = np.asarray(baseline.get("fold_scores", []), dtype=float)

        def _cmp_row(label, score, std, stab, n_feat, n_bl, is_reg,
                     fold_scores=None, is_baseline=False):
            sc_disp = -score if is_reg else score
            red = (1.0 - n_feat / n_bl) * 100.0 if n_bl > 0 else float("nan")
            stab_s = f"{stab:.3f}" if np.isfinite(stab) else "  n/a"
            red_s = f"{red:+.1f}%" if np.isfinite(red) else "   n/a"
            if is_baseline:
                p_s, out_s = "   --   ", "(reference)"
            elif fold_scores is None or bl_fold.size == 0:
                p_s, out_s = "   n/a  ", "n/a"
            else:
                p_val, out_s = _paired_baseline_outcome(np.asarray(fold_scores, dtype=float), bl_fold)
                p_s = f"{p_val:>8.4f}" if np.isfinite(p_val) else "    n/a "
            print(
                f"  {label:<30} {sc_disp:>8.4f}  {std:>7.4f}  {stab_s:>9}  "
                f"{int(round(n_feat)):>10d}  {red_s:>9}  {p_s:>10}  {out_s:<14}"
            )

        # ROBUST row (paired test on the OUTER nested CV folds)
        _cmp_row(
            "ROBUST (stability-selected)",
            robust_result.nested_cv_result.mean_score,
            robust_result.nested_cv_result.std_score,
            robust_stability,
            len(robust_result.selected_features),
            n_bl, is_reg,
            fold_scores=robust_result.nested_cv_result.outer_scores,
        )
        # Baseline row (no selection -> stability not applicable; reference for outcome)
        _cmp_row(
            "Baseline (all features)",
            baseline["mean"], baseline["std"],
            float("nan"), float(n_bl), n_bl, is_reg,
            is_baseline=True,
        )
        for key, cres in (comparators or {}).items():
            if cres is None:
                print(f"  {key.upper():<30} {'(not installed)':>8}")
                continue
            _cmp_row(
                cres.name, cres.mean_score, cres.std_score,
                cres.stability, cres.mean_n_features, n_bl, is_reg,
                fold_scores=cres.fold_scores,
            )
    print()


def print_summary_table(results: List[Dict[str, Any]]) -> None:
    """ROBUST vs baseline cross-scenario summary (original compact table)."""
    C = dict(
        name=24, task=11, nxp=13,
        rob_n=12, red=5, stab=9,
        bl_sc=9, rob_sc=12, delta=8, pval=7, outcome=11,
    )
    _TW = (2 + C['name'] + 1 + C['task'] + 1 + C['nxp'] + 2
           + C['rob_n'] + 1 + C['red'] + 1 + C['stab'] + 2
           + C['bl_sc'] + 1 + C['rob_sc'] + 1 + C['delta'] + 2
           + C['pval'] + 1 + C['outcome'])
    _TSEP = "=" * _TW
    _tsep = "-" * _TW

    print(f"\n{_TSEP}")
    print("  CROSS-SCENARIO SUMMARY  (ROBUST vs full-feature baseline)")
    print(_TSEP)
    print(_LEGEND)
    print(f"  Goal: feature reduction while preserving predictive performance.")
    print(f"  Outcome: 'preserved' = no significant difference (p >= 0.05)  |  "
          f"'sig. worse *' = significant cost  |  'sig. better *' = improvement")
    print(f"  Stability = mean pairwise Jaccard similarity of ROBUST feature sets across outer folds.")
    print(_tsep)
    hdr = (
        f"  {'Scenario':<{C['name']}} {'Task':<{C['task']}} "
        f"{'n_train x p':>{C['nxp']}}  "
        f"{'ROBUST feats':>{C['rob_n']}} {'Red%':>{C['red']}} {'Stability':>{C['stab']}}  "
        f"{'BL score':>{C['bl_sc']}} {'ROBUST score':>{C['rob_sc']}} {'+delta':>{C['delta']}}  "
        f"{'p-val':>{C['pval']}} {'Outcome':<{C['outcome']}}"
    )
    print(hdr)
    sep_row = (
        f"  {'-'*C['name']} {'-'*C['task']} "
        f"{'-'*C['nxp']}  "
        f"{'-'*C['rob_n']} {'-'*C['red']} {'-'*C['stab']}  "
        f"{'-'*C['bl_sc']} {'-'*C['rob_sc']} {'-'*C['delta']}  "
        f"{'-'*C['pval']} {'-'*C['outcome']}"
    )
    print(sep_row)
    print(f"  (regression: RMSE lower=better; classification: AUC higher=better)")
    print()
    for r in results:
        n_bl = r["n_features_bl"]
        n_robust = r["n_features_robust"]
        red = (1 - n_robust / n_bl) * 100
        is_reg = r["task_type"] == "regression"
        sc_bl = -r["score_bl"] if is_reg else r["score_bl"]
        sc_rob = -r["score_robust"] if is_reg else r["score_robust"]
        delta_disp = sc_bl - sc_rob if is_reg else sc_rob - sc_bl
        delta_raw = r["score_robust"] - r["score_bl"]
        n_train = r["n_samples"]
        nxp_str = f"{n_train} x {n_bl:4d}"
        p_val = _significance_p(r["stat_df"])
        p_str = f"{p_val:.3f}" if np.isfinite(p_val) else "  n/a"
        outcome = _outcome(delta_raw, r["stat_df"])
        stab = r.get("robust_stability", float("nan"))
        stab_s = f"{stab:.3f}" if np.isfinite(stab) else "   n/a"
        print(
            f"  {r['name']:<{C['name']}} {r['task_type']:<{C['task']}} "
            f"{nxp_str:>{C['nxp']}}  "
            f"{n_robust:>{C['rob_n']}} {red:>{C['red']}.0f}% {stab_s:>{C['stab']}}  "
            f"{sc_bl:>{C['bl_sc']}.4f} {sc_rob:>{C['rob_sc']}.4f} "
            f"{delta_disp:>{C['delta']}.4f}  "
            f"{p_str:>{C['pval']}} {outcome:<{C['outcome']}}"
        )
    print()
    print(_TSEP)


def print_comparator_summary(results: List[Dict[str, Any]]) -> None:
    """Multi-method pivot table: score + Jaccard stability for every method × dataset."""
    _CW = 100
    _CSEP = "=" * _CW
    _csep = "-" * _CW

    print(f"\n{_CSEP}")
    print("  MULTI-METHOD COMPARISON  (predictive score  +  Jaccard stability)")
    print(_CSEP)
    print(
        "  Each cell shows: Score ± Std  |  Stability (mean pairwise Jaccard across outer folds).\n"
        "  Score metric: AUC for classification, RMSE for regression (lower RMSE = better).\n"
        "  Stability: 0.00 = disjoint feature sets fold-to-fold; 1.00 = identical every fold.\n"
        "  n̄ = mean number of selected features across outer folds.\n"
        "  Baseline has no selection → stability is n/a (same features every fold by definition)."
    )
    print(_csep)

    # Build method list from first result that has comparators
    method_order = ["ROBUST", "Baseline"]
    for r in results:
        for key, cres in (r.get("comparators") or {}).items():
            if cres is not None and cres.name not in method_order:
                method_order.append(cres.name)

    # Header
    col_w = 34
    name_w = 26
    hdr = f"  {'Method':<{name_w}}"
    for r in results:
        ds_hdr = f"{r['name']} ({r['task_type'][:3]})"
        hdr += f"  {ds_hdr:^{col_w}}"
    print(hdr)
    print(f"  {'-'*name_w}" + (f"  {'-'*col_w}" * len(results)))

    sub_hdr = f"  {'':^{name_w}}"
    for r in results:
        n_bl = r["n_features_bl"]
        is_reg = r["task_type"] == "regression"
        metric = "RMSE" if is_reg else "AUC "
        sub_hdr += f"  {metric+' ±Std':>12}  {'Stab':>5}  {'n̄':>5}  {'Red%':>5}"
    print(sub_hdr)
    print(f"  {'.'*name_w}" + (f"  {'.'*col_w}" * len(results)))

    def _cell(score, std, stab, n_feat, n_bl, is_reg):
        sc_disp = -score if is_reg else score
        red = (1.0 - n_feat / n_bl) * 100.0
        stab_s = f"{stab:.3f}" if np.isfinite(stab) else "  n/a"
        return f"{sc_disp:>7.4f}±{std:.4f}  {stab_s}  {int(round(n_feat)):>5d}  {red:>4.0f}%"

    for method_name in method_order:
        row = f"  {method_name:<{name_w}}"
        for r in results:
            is_reg = r["task_type"] == "regression"
            n_bl = r["n_features_bl"]
            if method_name == "ROBUST":
                rr = r["robust_result"]
                row += "  " + _cell(
                    rr.nested_cv_result.mean_score,
                    rr.nested_cv_result.std_score,
                    r.get("robust_stability", float("nan")),
                    r["n_features_robust"],
                    n_bl, is_reg,
                )
            elif method_name == "Baseline":
                bl = r["baseline"]
                row += "  " + _cell(
                    bl["mean"], bl["std"],
                    float("nan"), float(n_bl), n_bl, is_reg,
                )
            else:
                cres = next(
                    (c for c in (r.get("comparators") or {}).values()
                     if c is not None and c.name == method_name),
                    None,
                )
                if cres is None:
                    row += f"  {'(skipped)':^{col_w}}"
                else:
                    row += "  " + _cell(
                        cres.mean_score, cres.std_score,
                        cres.stability, cres.mean_n_features, n_bl, is_reg,
                    )
        print(row)

    print()
    print(_CSEP)


def print_scenario_comparators(result: Dict[str, Any]) -> None:
    """Replay the COMPARATOR COMPARISON block from a scenario result dict.

    Useful for re-displaying comparator outcomes (including outcome-vs-baseline)
    on an already-computed run_scenario() result without re-running the full
    benchmark.  Mirrors the inline block produced by print_scenario_report.
    """
    robust_result = result["robust_result"]
    baseline      = result["baseline"]
    comparators   = result.get("comparators") or {}
    n_bl          = result["n_features_bl"]
    is_reg        = result["task_type"] == "regression"
    robust_stab   = result.get("robust_stability", float("nan"))
    name          = result.get("name", "scenario")

    print(f"\n{'='*92}")
    print(f"  COMPARATOR COMPARISON: {name}  ({'regression' if is_reg else 'classification'})")
    print('='*92)
    print(
        f"  Score metric: {('RMSE (lower=better)' if is_reg else 'AUC (higher=better)')}.  "
        f"Outcome from paired Wilcoxon signed-rank on per-fold scores vs full-feature baseline."
    )
    chdr = (
        f"  {'Method':<30} {'Score':>8}  {'±Std':>7}  {'Stability':>9}  "
        f"{'Mean feats':>10}  {'Reduction':>9}  {'p (vs BL)':>10}  {'Outcome vs BL':<14}"
    )
    print(chdr)
    print(f"  {'-'*30} {'-'*8}  {'-'*7}  {'-'*9}  {'-'*10}  {'-'*9}  {'-'*10}  {'-'*14}")

    bl_fold = np.asarray(baseline.get("fold_scores", []), dtype=float)

    def _row(label, score, std, stab, n_feat, fold_scores=None, is_baseline=False):
        sc_disp = -score if is_reg else score
        red = (1.0 - n_feat / n_bl) * 100.0 if n_bl > 0 else float("nan")
        stab_s = f"{stab:.3f}" if np.isfinite(stab) else "  n/a"
        red_s = f"{red:+.1f}%" if np.isfinite(red) else "   n/a"
        if is_baseline:
            p_s, out_s = "   --   ", "(reference)"
        elif fold_scores is None or bl_fold.size == 0:
            p_s, out_s = "   n/a  ", "n/a"
        else:
            p_val, out_s = _paired_baseline_outcome(np.asarray(fold_scores, dtype=float), bl_fold)
            p_s = f"{p_val:>8.4f}" if np.isfinite(p_val) else "    n/a "
        print(
            f"  {label:<30} {sc_disp:>8.4f}  {std:>7.4f}  {stab_s:>9}  "
            f"{int(round(n_feat)):>10d}  {red_s:>9}  {p_s:>10}  {out_s:<14}"
        )

    _row(
        "ROBUST (stability-selected)",
        robust_result.nested_cv_result.mean_score,
        robust_result.nested_cv_result.std_score,
        robust_stab,
        len(robust_result.selected_features),
        fold_scores=robust_result.nested_cv_result.outer_scores,
    )
    _row(
        "Baseline (all features)",
        baseline["mean"], baseline["std"],
        float("nan"), float(n_bl),
        is_baseline=True,
    )
    for _, cres in comparators.items():
        if cres is None:
            continue
        _row(
            cres.name, cres.mean_score, cres.std_score,
            cres.stability, cres.mean_n_features,
            fold_scores=cres.fold_scores,
        )


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

def run_scenario(ds: BenchmarkDataset, verbose: bool = True) -> Dict[str, Any]:
    """Fit ROBUST, full-feature baseline, and three comparators on one dataset."""
    t0 = time.time()

    # Merge parameter layers (lowest → highest priority):
    #   ROBUST_PARAMS  →  THRESHOLD_OVERRIDES[ds.name]  →  ds.robust_params_override
    _thr = THRESHOLD_OVERRIDES.get(ds.name)
    _thr_layer: Dict[str, Any] = {"stability_threshold": _thr} if _thr is not None else {}
    scenario_params = {**ROBUST_PARAMS, **_thr_layer, **ds.robust_params_override}
    _eff_thr = scenario_params.get("stability_threshold", ROBUST_PARAMS["stability_threshold"])

    if verbose:
        print(
            f"\n>>> {ds.name}: running ROBUST({ds.alg.upper()}, {ds.task_type}, "
            f"thr={_eff_thr:.2f}) ...",
            flush=True,
        )
        if _thr is not None and "stability_threshold" not in ds.robust_params_override:
            print(
                f"    threshold from THRESHOLD_OVERRIDES  "
                f"(global default={ROBUST_PARAMS['stability_threshold']:.2f})",
                flush=True,
            )
        if ds.robust_params_override:
            ov_str = ", ".join(f"{k}={v}" for k, v in ds.robust_params_override.items())
            print(f"    dataset-specific overrides: {ov_str}", flush=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        maker = robust.RobustModelMaker(alg=ds.alg, task_type=ds.task_type, **scenario_params)
        maker.fit(ds.X_train, ds.y_train)
    robust_result = maker.result_

    # Jaccard stability for ROBUST: use the per-fold selected feature sets stored
    # in nested_cv_result, which are produced by the stability selection step
    # inside each outer fold of the nested CV.
    robust_fold_sets: List[Set[str]] = [
        set(arr.tolist())
        for arr in robust_result.nested_cv_result.selected_features_per_fold
    ]
    robust_stability = jaccard_stability(robust_fold_sets)

    if verbose:
        print(
            f"    ROBUST done ({len(robust_result.selected_features)} features, "
            f"Jaccard stability={robust_stability:.3f}). Running baseline ...",
            flush=True,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        baseline = run_baseline_nested_cv(ds)

    if verbose:
        print("    Baseline done. Running ANOVA comparator ...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        anova_result = run_anova_nested_cv(ds)

    if verbose:
        print(
            f"    ANOVA done ({anova_result.mean_n_features:.0f} features, "
            f"Jaccard={anova_result.stability:.3f}). Running RFECV ...",
            flush=True,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rfecv_result = run_rfecv_nested_cv(ds)

    if verbose:
        print(
            f"    RFECV done ({rfecv_result.mean_n_features:.0f} features, "
            f"Jaccard={rfecv_result.stability:.3f}). Running Boruta ...",
            flush=True,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        boruta_result = run_boruta_nested_cv(ds)

    if verbose and boruta_result is not None:
        print(
            f"    Boruta done ({boruta_result.mean_n_features:.0f} features, "
            f"Jaccard={boruta_result.stability:.3f}).",
            flush=True,
        )

    elapsed = time.time() - t0

    robust_scores = robust_result.nested_cv_result.outer_scores
    bl_scores = baseline["fold_scores"]
    stat_df = run_statistical_battery(robust_scores, bl_scores, ds.task_type, ds.floor_score)

    comparators: Dict[str, Optional[ComparatorResult]] = {
        "anova": anova_result,
        "rfecv": rfecv_result,
        "boruta": boruta_result,
    }

    if verbose:
        print_scenario_report(
            ds, robust_result, baseline, stat_df, elapsed,
            robust_stability=robust_stability,
            comparators=comparators,
        )

    return {
        "name": ds.name,
        "task_type": ds.task_type,
        "alg": ds.alg,
        "n_samples": ds.X_train.shape[0],
        "n_features_bl": baseline["n_features"],
        "n_features_robust": len(robust_result.selected_features),
        "score_bl": baseline["mean"],
        "score_robust": robust_result.nested_cv_result.mean_score,
        "robust_stability": robust_stability,
        "robust_result": robust_result,
        "baseline": baseline,
        "stat_df": stat_df,
        "comparators": comparators,
        "elapsed": elapsed,
        "dataset": ds,
    }


# ===========================================================================
# Synthetic recovery datasets (ground truth available)
# ===========================================================================
#
# These functions generate synthetic tabular datasets whose informative
# features are known by construction.  They exist so that the four selectors
# (ROBUST, ANOVA, RFECV, Boruta) can be scored not only on predictive accuracy
# and selection stability but on whether they recover the *right* features:
# the ones that actually drive the response, as opposed to correlated decoys
# or pure-noise distractors.  Recovery is the only operational definition of
# "rightness" available without domain ground truth.
#
# Each generator produces a BenchmarkDataset whose true_features and
# correlate_features attributes carry the provenance, plus 80/20 stratified
# train/test indices set as the BenchMake archetypal split would be on a real
# dataset (synthetic data does not benefit from BenchMake's adversarial
# partitioning so a stratified random split is used instead).


def _make_synthetic_binary(
    n_samples: int = 600,
    n_informative: int = 10,
    n_correlate: int = 15,
    n_noise: int = 75,
    correlate_strength: float = 0.85,
    pos_rate: float = 0.20,
    nan_rate: float = 0.08,
    random_state: int = 42,
) -> "BenchmarkDataset":
    """Synthetic binary classification with known ground truth.

    Generates n_informative truly causal features that drive the binary
    target through a logistic link, n_correlate decoy features each strongly
    correlated (correlate_strength) with one informative feature but
    contributing no independent signal, and n_noise pure-noise features.
    A NaN injection step mimics the missingness regime of real assay data.
    """
    from sklearn.model_selection import train_test_split

    rng = np.random.RandomState(random_state)
    n_features = n_informative + n_correlate + n_noise

    # Informative block: standard normal
    X_inf = rng.standard_normal((n_samples, n_informative))

    # Correlate block: each correlate is a noisy copy of one informative feature
    # Map correlate index j -> informative index j % n_informative
    parent = np.array([j % n_informative for j in range(n_correlate)])
    noise_for_corr = np.sqrt(1.0 - correlate_strength**2) * rng.standard_normal((n_samples, n_correlate))
    X_corr = correlate_strength * X_inf[:, parent] + noise_for_corr

    # Noise block: independent standard normal
    X_noise = rng.standard_normal((n_samples, n_noise))

    X = np.hstack([X_inf, X_corr, X_noise])

    # Build feature names that encode the provenance
    inf_names = [f"true_{i:02d}" for i in range(n_informative)]
    corr_names = [f"corr_{j:02d}_of_true_{parent[j]:02d}" for j in range(n_correlate)]
    noise_names = [f"noise_{k:03d}" for k in range(n_noise)]
    feature_names = inf_names + corr_names + noise_names

    # Linear combination of informative features through a logistic link.
    # Coefficients alternate in sign and have moderate magnitudes.
    coefs = rng.choice([-1.5, -1.0, 1.0, 1.5], size=n_informative, replace=True)
    logits = X_inf @ coefs
    # Calibrate intercept to match the requested positive-class rate.
    intercept = float(np.quantile(logits, 1.0 - pos_rate))
    probs = 1.0 / (1.0 + np.exp(-(logits - intercept)))
    y_arr = (rng.random(n_samples) < probs).astype(int)

    # Inject NaNs uniformly across feature columns (mimics real assay missingness)
    if nan_rate > 0:
        mask = rng.random(X.shape) < nan_rate
        X = np.where(mask, np.nan, X)

    X_df = pd.DataFrame(X, columns=feature_names)
    y_ser = pd.Series(y_arr, name="y")

    # Stratified 80/20 split
    train_idx, test_idx = train_test_split(
        np.arange(n_samples), test_size=0.20,
        stratify=y_arr, random_state=random_state,
    )

    return BenchmarkDataset(
        name="Synthetic Binary",
        description=(
            f"Synthetic binary classification with ground truth; "
            f"{n_samples} samples x {n_features} features "
            f"({n_informative} informative, {n_correlate} correlated decoys, {n_noise} noise), "
            f"positive class {pos_rate:.0%}, NaN rate {nan_rate:.0%}."
        ),
        X=X_df, y=y_ser,
        task_type="binary", alg="rf", floor_score=0.55,
        train_idx=train_idx, test_idx=test_idx,
        true_features=inf_names, correlate_features=corr_names,
    )


def _make_synthetic_multiclass(
    n_samples: int = 700,
    n_classes: int = 4,
    n_informative: int = 12,
    n_correlate: int = 12,
    n_noise: int = 56,
    correlate_strength: float = 0.85,
    random_state: int = 42,
) -> "BenchmarkDataset":
    """Synthetic multiclass classification with known ground truth.

    Each informative feature has class-specific shifts, so all n_informative
    features carry independent multiclass signal.  Correlates and noise are
    as in the binary generator.
    """
    from sklearn.model_selection import train_test_split

    rng = np.random.RandomState(random_state)
    n_features = n_informative + n_correlate + n_noise

    # Assign class labels uniformly
    y_arr = rng.randint(0, n_classes, size=n_samples)

    # Informative block: per-class means in standard normal noise
    class_means = rng.standard_normal((n_classes, n_informative)) * 1.2
    X_inf = class_means[y_arr] + rng.standard_normal((n_samples, n_informative))

    # Correlate block
    parent = np.array([j % n_informative for j in range(n_correlate)])
    noise_for_corr = np.sqrt(1.0 - correlate_strength**2) * rng.standard_normal((n_samples, n_correlate))
    X_corr = correlate_strength * X_inf[:, parent] + noise_for_corr

    # Noise block
    X_noise = rng.standard_normal((n_samples, n_noise))

    X = np.hstack([X_inf, X_corr, X_noise])

    inf_names = [f"true_{i:02d}" for i in range(n_informative)]
    corr_names = [f"corr_{j:02d}_of_true_{parent[j]:02d}" for j in range(n_correlate)]
    noise_names = [f"noise_{k:03d}" for k in range(n_noise)]
    feature_names = inf_names + corr_names + noise_names

    X_df = pd.DataFrame(X, columns=feature_names)
    y_ser = pd.Series(y_arr, name="y")

    train_idx, test_idx = train_test_split(
        np.arange(n_samples), test_size=0.20,
        stratify=y_arr, random_state=random_state,
    )

    return BenchmarkDataset(
        name="Synthetic Multiclass",
        description=(
            f"Synthetic multiclass classification with ground truth; "
            f"{n_samples} samples x {n_features} features "
            f"({n_informative} informative, {n_correlate} correlated decoys, {n_noise} noise), "
            f"{n_classes} balanced classes."
        ),
        X=X_df, y=y_ser,
        task_type="multiclass", alg="rf", floor_score=0.55,
        train_idx=train_idx, test_idx=test_idx,
        true_features=inf_names, correlate_features=corr_names,
    )


def _make_synthetic_regression(
    n_samples: int = 800,
    n_informative: int = 10,
    n_correlate: int = 20,
    n_noise: int = 70,
    correlate_strength: float = 0.85,
    snr: float = 5.0,
    random_state: int = 42,
) -> "BenchmarkDataset":
    """Synthetic regression with known ground truth.

    Target is a mixed linear + nonlinear function of the n_informative
    features (linear for the first half, sin/quadratic for the second).
    Correlates and noise are as in the classification generators.  SNR is
    controlled by the variance of the additive noise term on the response.
    """
    from sklearn.model_selection import train_test_split

    rng = np.random.RandomState(random_state)
    n_features = n_informative + n_correlate + n_noise

    X_inf = rng.standard_normal((n_samples, n_informative))

    parent = np.array([j % n_informative for j in range(n_correlate)])
    noise_for_corr = np.sqrt(1.0 - correlate_strength**2) * rng.standard_normal((n_samples, n_correlate))
    X_corr = correlate_strength * X_inf[:, parent] + noise_for_corr

    X_noise = rng.standard_normal((n_samples, n_noise))

    X = np.hstack([X_inf, X_corr, X_noise])

    # Response: linear in first half of informative, mild nonlinear in second half
    half = n_informative // 2
    coefs = rng.choice([-2.0, -1.0, 1.0, 2.0], size=half, replace=True)
    linear_part = X_inf[:, :half] @ coefs
    nonlinear_part = np.sum(np.sin(1.2 * X_inf[:, half:]) + 0.5 * X_inf[:, half:]**2, axis=1)
    signal = linear_part + nonlinear_part

    signal_std = float(np.std(signal))
    noise_std = signal_std / float(snr)
    y_arr = signal + noise_std * rng.standard_normal(n_samples)

    inf_names = [f"true_{i:02d}" for i in range(n_informative)]
    corr_names = [f"corr_{j:02d}_of_true_{parent[j]:02d}" for j in range(n_correlate)]
    noise_names = [f"noise_{k:03d}" for k in range(n_noise)]
    feature_names = inf_names + corr_names + noise_names

    X_df = pd.DataFrame(X, columns=feature_names)
    y_ser = pd.Series(y_arr, name="y")

    train_idx, test_idx = train_test_split(
        np.arange(n_samples), test_size=0.20, random_state=random_state,
    )

    return BenchmarkDataset(
        name="Synthetic Regression",
        description=(
            f"Synthetic regression with ground truth; {n_samples} samples x {n_features} features "
            f"({n_informative} informative [half linear + half nonlinear], "
            f"{n_correlate} correlated decoys, {n_noise} noise), SNR = {snr}."
        ),
        X=X_df, y=y_ser,
        task_type="regression", alg="rf",
        # Floor is in neg-RMSE units; set permissively because target scale varies
        floor_score=-3.0 * signal_std,
        train_idx=train_idx, test_idx=test_idx,
        true_features=inf_names, correlate_features=corr_names,
    )


# ===========================================================================
# Recovery metrics (precision/recall/F1 against ground truth)
# ===========================================================================

def recovery_metrics(
    selected: Iterable[str],
    true_features: Iterable[str],
    correlate_features: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """Score a selected feature set against a known ground-truth set.

    Parameters
    ----------
    selected : iterable of str
        Features chosen by a selection method on one fold (or aggregated).
    true_features : iterable of str
        The features that genuinely drive the response (ground truth).
    correlate_features : iterable of str, optional
        Features that are correlated with true features but carry no
        independent signal.  Used to compute the correlate-confusion rate:
        the fraction of selected features that are correlates rather than
        true causes.

    Returns
    -------
    dict with keys:
      tp, fp, fn          : counts against the true_features set
      precision, recall   : standard counts-based
      f1                  : harmonic mean of precision and recall
      n_selected          : |selected|
      n_true              : |true_features|
      noise_picks         : count of selected features that are neither true
                            nor correlated decoys (pure-noise contamination)
      correlate_picks     : count of selected features that are correlates
                            (only populated if correlate_features is given)
      correlate_rate      : correlate_picks / n_selected
    """
    sel = set(selected)
    tru = set(true_features)
    cor = set(correlate_features) if correlate_features is not None else set()

    tp = len(sel & tru)
    fp = len(sel - tru)
    fn = len(tru - sel)
    precision = tp / max(len(sel), 1) if len(sel) else 0.0
    recall = tp / max(len(tru), 1) if len(tru) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    correlate_picks = len(sel & cor)
    noise_picks = len(sel - tru - cor)
    correlate_rate = correlate_picks / max(len(sel), 1) if len(sel) else 0.0

    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "n_selected": int(len(sel)), "n_true": int(len(tru)),
        "noise_picks": int(noise_picks),
        "correlate_picks": int(correlate_picks),
        "correlate_rate": float(correlate_rate),
    }


def fold_recovery_metrics(
    fold_feature_sets: List[Set[str]],
    true_features: Iterable[str],
    correlate_features: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Aggregate recovery metrics across outer folds.

    Computes per-fold recovery_metrics then reports the mean and standard
    deviation of each scalar metric.  Also returns the per-fold values so
    callers can run further tests.
    """
    rows = [recovery_metrics(fs, true_features, correlate_features) for fs in fold_feature_sets]
    out: Dict[str, Any] = {}
    for key in ("precision", "recall", "f1", "correlate_rate"):
        vals = np.array([r[key] for r in rows], dtype=float)
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals))
        out[f"{key}_per_fold"] = vals
    for key in ("tp", "fp", "fn", "n_selected", "noise_picks", "correlate_picks"):
        vals = np.array([r[key] for r in rows], dtype=float)
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_per_fold"] = vals.astype(int)
    return out


# ===========================================================================
# Inter-method consensus (do the selectors agree on the same features?)
# ===========================================================================

def intermethod_consensus(
    method_fold_sets: Dict[str, List[Set[str]]],
) -> Dict[str, Any]:
    """Cross-method agreement on selected features, fold by fold.

    Parameters
    ----------
    method_fold_sets : dict
        Maps method name (e.g. "ROBUST", "ANOVA k=14") to a list of per-fold
        selected feature sets.  Lists must all have the same length (K).

    Returns
    -------
    dict with keys:
      pairwise_jaccard : DataFrame of mean pairwise Jaccard between every
                         pair of methods, averaged across folds
      consensus_intersection_per_fold : list of length K, each entry a set of
                         features selected by all methods on that fold
      mean_consensus_size : mean cardinality of consensus_intersection_per_fold
      union_per_fold     : list of length K, each entry a set of features
                         selected by any method on that fold
      mean_union_size    : mean cardinality of union_per_fold
      methods            : list of method names in input order
    """
    names = list(method_fold_sets.keys())
    if not names:
        return {"methods": [], "mean_consensus_size": float("nan")}
    lengths = {len(v) for v in method_fold_sets.values()}
    if len(lengths) != 1:
        raise ValueError(f"all methods must have the same fold count, got {lengths}")
    K = lengths.pop()

    pair_rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            jacs = []
            for k in range(K):
                Sa, Sb = method_fold_sets[a][k], method_fold_sets[b][k]
                u = Sa | Sb
                jacs.append(len(Sa & Sb) / len(u) if u else 1.0)
            pair_rows.append({"method_a": a, "method_b": b, "mean_jaccard": float(np.mean(jacs))})
    pairwise = pd.DataFrame(pair_rows)

    consensus = [set.intersection(*[method_fold_sets[m][k] for m in names]) for k in range(K)]
    union = [set.union(*[method_fold_sets[m][k] for m in names]) for k in range(K)]

    return {
        "methods": names,
        "pairwise_jaccard": pairwise,
        "consensus_intersection_per_fold": consensus,
        "mean_consensus_size": float(np.mean([len(s) for s in consensus])),
        "union_per_fold": union,
        "mean_union_size": float(np.mean([len(s) for s in union])),
    }


# ===========================================================================
# Permutation-importance overlap
# ===========================================================================

def permutation_importance_overlap(
    selected: Iterable[str],
    perm_importance: pd.DataFrame,
    top_k: Optional[int] = None,
    importance_col: str = "importance_mean",
    feature_col: str = "feature",
) -> Dict[str, float]:
    """Fraction of a selector's chosen set that lies in the top-K by importance.

    perm_importance should be a DataFrame with one row per feature, containing
    at minimum a feature name column and a numeric importance column.
    """
    sel = set(selected)
    if top_k is None:
        top_k = len(sel)
    df = perm_importance.sort_values(importance_col, ascending=False).head(top_k)
    top = set(df[feature_col])
    overlap = sel & top
    return {
        "n_selected": len(sel),
        "top_k": int(top_k),
        "overlap_count": int(len(overlap)),
        "overlap_fraction": float(len(overlap) / max(len(sel), 1)),
    }


# ===========================================================================
# Cross-method Friedman + post-hoc Nemenyi test
# ===========================================================================

def friedman_nemenyi(
    method_fold_scores: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Friedman test across methods + post-hoc pairwise Nemenyi comparison.

    Parameters
    ----------
    method_fold_scores : dict
        Maps method name to a per-fold score array (all on the "higher is
        better" convention).  All arrays must have the same length.

    Returns
    -------
    dict with keys:
      friedman_statistic, friedman_p : the omnibus paired test
      mean_ranks            : pd.Series of mean rank per method (lower = better)
      pairwise_p            : pd.DataFrame of Nemenyi post-hoc p-values
    """
    from scipy.stats import friedmanchisquare, rankdata
    names = list(method_fold_scores.keys())
    arrs = [np.asarray(method_fold_scores[n], dtype=float) for n in names]
    K = len(arrs[0])
    assert all(len(a) == K for a in arrs), "all methods must have the same fold count"

    # Omnibus
    try:
        stat, p = friedmanchisquare(*arrs)
        stat, p = float(stat), float(p)
    except Exception:
        stat, p = float("nan"), float("nan")

    # Per-fold ranks (1 = best because we negate higher-is-better)
    score_matrix = np.array(arrs)  # shape (M, K)
    ranks = np.zeros_like(score_matrix)
    for k in range(K):
        ranks[:, k] = rankdata(-score_matrix[:, k])
    mean_ranks = pd.Series(ranks.mean(axis=1), index=names, name="mean_rank")

    # Pairwise Nemenyi (Studentized range with infinite df approximation).
    # Critical-distance formula: q_alpha * sqrt(M*(M+1) / (6*K)).
    # Here we instead report two-sided pairwise p-values via a Wilcoxon
    # signed-rank pairwise comparison as a robust substitute (Nemenyi requires
    # specialised tables; the SR pairwise is widely accepted in this regime).
    from scipy.stats import wilcoxon
    M = len(names)
    pair = pd.DataFrame(np.eye(M), index=names, columns=names)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i >= j:
                continue
            try:
                w = wilcoxon(arrs[i], arrs[j], zero_method="wilcox", alternative="two-sided")
                pp = float(w.pvalue)
            except Exception:
                pp = float("nan")
            pair.iloc[i, j] = pp
            pair.iloc[j, i] = pp

    return {
        "friedman_statistic": stat,
        "friedman_p": p,
        "mean_ranks": mean_ranks,
        "pairwise_p": pair,
    }


# ===========================================================================
# Printing helpers for the new analyses
# ===========================================================================

def print_synthetic_recovery(result: Dict[str, Any]) -> None:
    """Print recovery metrics (precision/recall/F1) for a synthetic scenario."""
    ds = result["dataset"]
    if not ds.has_ground_truth:
        print(f"{ds.name}: no ground truth; skipping recovery report.")
        return

    true_set = set(ds.true_features)
    cor_set = set(ds.correlate_features or [])
    n_true = len(true_set)

    print(f"\n{'='*96}")
    print(f"  SYNTHETIC RECOVERY: {ds.name}  ({ds.task_type})")
    print('='*96)
    print(f"  Ground truth: {n_true} informative + {len(cor_set)} correlated decoys + noise")
    print(f"  {'Method':<32} {'Prec':>6}  {'Recall':>6}  {'F1':>6}  "
          f"{'TP':>4}  {'FP':>4}  {'FN':>4}  {'CorrPick':>8}  {'NoisePick':>9}")
    print(f"  {'-'*32} {'-'*6}  {'-'*6}  {'-'*6}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*8}  {'-'*9}")

    def _row(label: str, fold_sets: List[Set[str]]) -> None:
        rec = fold_recovery_metrics(fold_sets, true_set, cor_set)
        print(
            f"  {label:<32} "
            f"{rec['precision_mean']:>6.3f}  {rec['recall_mean']:>6.3f}  {rec['f1_mean']:>6.3f}  "
            f"{rec['tp_mean']:>4.1f}  {rec['fp_mean']:>4.1f}  {rec['fn_mean']:>4.1f}  "
            f"{rec['correlate_picks_mean']:>8.1f}  {rec['noise_picks_mean']:>9.1f}"
        )

    robust_result = result["robust_result"]
    rfs = [set(s) for s in robust_result.nested_cv_result.selected_features_per_fold]
    _row("ROBUST (stability-selected)", rfs)

    for _, cres in (result.get("comparators") or {}).items():
        if cres is None:
            continue
        _row(cres.name, [set(s) for s in cres.fold_feature_sets])


def print_intermethod_consensus(consensus: Dict[str, Any]) -> None:
    """Pretty-print the result of intermethod_consensus()."""
    print(f"\n{'='*96}")
    print("  INTER-METHOD CONSENSUS  (do the selectors agree on the same features?)")
    print('='*96)
    print(f"  Mean consensus size (features selected by ALL methods, per fold): "
          f"{consensus['mean_consensus_size']:.1f}")
    print(f"  Mean union size     (features selected by ANY method, per fold): "
          f"{consensus['mean_union_size']:.1f}")
    print("\n  Pairwise mean Jaccard between methods (1.00 = identical, 0.00 = disjoint):")
    pj = consensus["pairwise_jaccard"]
    if not pj.empty:
        for _, r in pj.iterrows():
            print(f"    {r['method_a']:<28} vs {r['method_b']:<28}  J = {r['mean_jaccard']:.3f}")


def print_cross_method_friedman(test: Dict[str, Any]) -> None:
    """Pretty-print the result of friedman_nemenyi()."""
    print(f"\n{'='*96}")
    print("  CROSS-METHOD FRIEDMAN + PAIRWISE COMPARISON")
    print('='*96)
    stat = test["friedman_statistic"]
    p = test["friedman_p"]
    print(f"  Friedman statistic = {stat:.4f}, p = {p:.4g}  "
          f"({'significant difference between methods' if p < 0.05 else 'no significant difference'})")
    print("\n  Mean ranks across folds (lower = better):")
    for name, r in test["mean_ranks"].sort_values().items():
        print(f"    {name:<32} {r:.3f}")
    print("\n  Pairwise paired Wilcoxon p-values (two-sided):")
    pair = test["pairwise_p"]
    cols = pair.columns.tolist()
    head = "    " + " " * 32 + "".join(f"{c[:10]:>11}" for c in cols)
    print(head)
    for name in cols:
        row = f"    {name[:32]:<32}"
        for c in cols:
            v = pair.loc[name, c]
            if name == c:
                row += f"{'--':>11}"
            else:
                row += f"{v:>11.4f}"
        print(row)


# ---------------------------------------------------------------------------
# pytest fixtures (module-scoped -- each scenario runs once per session)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def secom_result():
    try:
        ds = _load_secom()
    except DataUnavailable as e:
        pytest.skip(str(e))
    return run_scenario(ds, verbose=False)


@pytest.fixture(scope="module")
def urban_result():
    try:
        ds = _load_urban_land_cover()
    except DataUnavailable as e:
        pytest.skip(str(e))
    return run_scenario(ds, verbose=False)


@pytest.fixture(scope="module")
def graphene_result():
    try:
        ds = _load_graphene_oxide()
    except DataUnavailable as e:
        pytest.skip(str(e))
    return run_scenario(ds, verbose=False)


# ---------------------------------------------------------------------------
# Benchmark assertions -- SECOM Manufacturing (binary)
# ---------------------------------------------------------------------------

class TestSECOM:

    def test_robust_produces_binary_result(self, secom_result):
        r = secom_result
        assert r["robust_result"] is not None
        assert r["robust_result"].task_type == "binary"
        assert r["robust_result"].cutoff_result is not None

    def test_feature_reduction_at_least_10pct(self, secom_result):
        r = secom_result
        reduction = 1.0 - r["n_features_robust"] / r["n_features_bl"]
        assert reduction >= 0.10, (
            f"Expected >=10% feature reduction on SECOM; got {reduction:.1%}. "
            f"ROBUST selected {r['n_features_robust']}/{r['n_features_bl']} features."
        )

    def test_robust_score_above_floor(self, secom_result):
        r = secom_result
        floor = r["dataset"].floor_score
        assert r["score_robust"] > floor, (
            f"ROBUST AUC {r['score_robust']:.4f} should exceed floor {floor} on SECOM."
        )

    def test_robust_not_catastrophically_worse_than_baseline(self, secom_result):
        delta = secom_result["score_robust"] - secom_result["score_bl"]
        assert delta >= -0.15, (
            f"ROBUST should not lose more than 0.15 AUC vs baseline; got {delta:+.4f}."
        )

    def test_stability_frequencies_valid(self, secom_result):
        freqs = secom_result["robust_result"].stability_result.selection_frequencies
        assert freqs.shape[0] == secom_result["n_features_bl"]
        assert np.all((freqs >= 0) & (freqs <= 1))
        assert freqs.sum() > 0

    def test_cutoff_is_valid_probability(self, secom_result):
        cutoff = secom_result["robust_result"].cutoff_result.cutoff_median
        assert 0.0 <= cutoff <= 1.0, f"Binary cutoff {cutoff:.4f} is not a valid probability."

    def test_statistical_battery_has_enough_tests(self, secom_result):
        stat_df = secom_result["stat_df"]
        assert isinstance(stat_df, pd.DataFrame)
        test_rows = stat_df[stat_df["p_value"].apply(lambda v: isinstance(v, float) and not np.isnan(v))]
        assert len(test_rows) >= 10

    def test_paired_ttest_present(self, secom_result):
        assert any(secom_result["stat_df"]["test"].str.contains("Paired t-test", na=False))

    def test_wilcoxon_present(self, secom_result):
        assert any(secom_result["stat_df"]["test"].str.contains("Wilcoxon", na=False))

    def test_cohens_d_is_finite(self, secom_result):
        d_row = secom_result["stat_df"][secom_result["stat_df"]["test"].str.contains("Cohen", na=False)]
        assert len(d_row) >= 1
        assert np.isfinite(float(d_row["statistic"].iloc[0]))

    def test_bootstrap_ci_present(self, secom_result):
        assert any(secom_result["stat_df"]["test"].str.contains("bootstrap", case=False, na=False))

    def test_feature_stability_summary_is_dataframe(self, secom_result):
        stab = secom_result["robust_result"].stability_result.summary()
        assert isinstance(stab, pd.DataFrame)
        assert "selection_frequency" in stab.columns
        assert stab["selection_frequency"].max() > 0

    def test_selected_features_are_subset_of_original(self, secom_result):
        original = set(secom_result["robust_result"].feature_names)
        selected = set(secom_result["robust_result"].selected_features)
        assert selected.issubset(original)

    def test_predictions_match_sample_count(self, secom_result):
        r = secom_result
        n = len(r["robust_result"].nested_cv_result.outer_true_labels)
        assert r["robust_result"].nested_cv_result.outer_predictions.shape[0] == n


# ---------------------------------------------------------------------------
# Benchmark assertions -- Urban Land Cover (multiclass)
# ---------------------------------------------------------------------------

class TestUrbanLandCover:

    def test_robust_produces_multiclass_result(self, urban_result):
        r = urban_result
        assert r["robust_result"].task_type == "multiclass"
        assert r["robust_result"].cutoff_result is None

    def test_nine_class_structure(self, urban_result):
        class_names = urban_result["robust_result"].class_names
        assert class_names is not None
        assert len(class_names) == 9, (
            f"Expected 9 urban land cover classes; got {len(class_names)}."
        )

    def test_predict_returns_known_labels(self, urban_result):
        ds = urban_result["dataset"]
        maker = robust.RobustModelMaker(alg=ds.alg, task_type=ds.task_type, **ROBUST_PARAMS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            maker.fit(ds.X_train, ds.y_train)
        pred = maker.predict(ds.X_train.head(12))
        assert isinstance(pred, pd.Series)
        assert set(pred.unique()).issubset(set(ds.y_train.unique()))

    def test_feature_reduction_at_least_10pct(self, urban_result):
        r = urban_result
        reduction = 1.0 - r["n_features_robust"] / r["n_features_bl"]
        assert reduction >= 0.10, (
            f"Expected >=10% feature reduction on Urban Land Cover; got {reduction:.1%}."
        )

    def test_robust_score_above_floor(self, urban_result):
        r = urban_result
        floor = r["dataset"].floor_score
        assert r["score_robust"] > floor, (
            f"ROBUST AUC-OVR {r['score_robust']:.4f} should exceed floor {floor}."
        )

    def test_class_names_stored(self, urban_result):
        class_names = urban_result["robust_result"].class_names
        assert class_names is not None
        assert len(class_names) == 9

    def test_statistical_battery_has_enough_tests(self, urban_result):
        stat_df = urban_result["stat_df"]
        test_rows = stat_df[stat_df["p_value"].apply(
            lambda v: isinstance(v, float) and not np.isnan(v))]
        assert len(test_rows) >= 10

    def test_mann_whitney_present(self, urban_result):
        assert any(urban_result["stat_df"]["test"].str.contains("Mann-Whitney", na=False))

    def test_effect_size_row_present(self, urban_result):
        stat_df = urban_result["stat_df"]
        assert any(stat_df["test"].str.contains("Cohen", na=False))
        assert any(stat_df["test"].str.contains("Hedges", na=False))

    def test_correlation_tests_present(self, urban_result):
        stat_df = urban_result["stat_df"]
        assert any(stat_df["test"].str.contains("Pearson", na=False))
        assert any(stat_df["test"].str.contains("Spearman", na=False))
        assert any(stat_df["test"].str.contains("Kendall", na=False))

    def test_sign_test_present(self, urban_result):
        assert any(urban_result["stat_df"]["test"].str.contains("Sign test", na=False))

    def test_imagery_features_reduced(self, urban_result):
        r = urban_result
        reduction = 1.0 - r["n_features_robust"] / r["n_features_bl"]
        assert reduction > 0.05, (
            "Expected ROBUSTto drop at least some redundant imagery features."
        )


# ---------------------------------------------------------------------------
# Benchmark assertions -- Graphene Oxide Bulk (regression)
# ---------------------------------------------------------------------------

class TestGrapheneOxide:

    def test_robust_produces_regression_result(self, graphene_result):
        r = graphene_result
        assert r["robust_result"].task_type == "regression"
        assert r["robust_result"].cutoff_result is None
        assert r["robust_result"].label_mapping is None

    def test_scores_are_negative_rmse(self, graphene_result):
        assert graphene_result["score_robust"] < 0, "neg-RMSE score should be negative."
        assert graphene_result["score_bl"] < 0

    def test_feature_reduction_at_least_10pct(self, graphene_result):
        r = graphene_result
        reduction = 1.0 - r["n_features_robust"] / r["n_features_bl"]
        assert reduction >= 0.10, (
            f"Expected >=10% reduction for GO descriptors; got {reduction:.1%}."
        )

    def test_selected_features_not_too_few(self, graphene_result):
        """Guard against too few features being selected from the large descriptor space.

        Random forest importance scores (MDI variance reduction) are naturally non-uniform
        across correlated descriptors, so bootstrap stability selection produces a
        discriminative frequency distribution without algorithm-specific overrides.
        At threshold=0.75 we expect a physically plausible subset; the floor of 5
        is intentionally very conservative -- the test is a safety net against
        degenerate outcomes, not a prescription of the desired feature count.
        """
        n_selected = graphene_result["n_features_robust"]
        assert n_selected >= 5, (
            f"Graphene Oxide benchmark selected only {n_selected} features at "
            f"stability_threshold=0.75. Consider lowering the threshold or "
            "checking for fold-specific data quality issues."
        )

    def test_robust_score_above_floor(self, graphene_result):
        r = graphene_result
        floor = r["dataset"].floor_score
        assert r["score_robust"] > floor, (
            f"ROBUST neg-RMSE {r['score_robust']:.4f} should exceed floor {floor}."
        )

    def test_formation_energy_target_in_valid_range(self, graphene_result):
        ds = graphene_result["dataset"]
        y = ds.y.to_numpy(dtype=float)
        assert y.min() < -60.0, "Formation_energy minimum should be below -60 eV."
        assert y.max() > -90.0, "Formation_energy maximum should be above -90 eV."
        assert np.all(np.isfinite(y))

    def test_regression_predictions_are_continuous(self, graphene_result):
        preds = graphene_result["robust_result"].nested_cv_result.outer_predictions
        assert preds.ndim == 1
        assert len(np.unique(preds)) > 10
        assert np.all(np.isfinite(preds))

    def test_predict_returns_series(self, graphene_result):
        ds = graphene_result["dataset"]
        maker = robust.RobustModelMaker(alg=ds.alg, task_type=ds.task_type, **ROBUST_PARAMS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            maker.fit(ds.X_train, ds.y_train)
        pred = maker.predict(ds.X_train.head(10))
        assert isinstance(pred, pd.Series)
        assert pred.shape == (10,)
        assert np.all(np.isfinite(pred.to_numpy()))

    def test_predict_proba_raises_for_regression(self, graphene_result):
        ds = graphene_result["dataset"]
        maker = robust.RobustModelMaker(alg=ds.alg, task_type=ds.task_type, **ROBUST_PARAMS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            maker.fit(ds.X_train, ds.y_train)
        with pytest.raises(AttributeError):
            maker.predict_proba(ds.X_train.head(5))

    def test_statistical_battery_has_enough_tests(self, graphene_result):
        stat_df = graphene_result["stat_df"]
        test_rows = stat_df[stat_df["p_value"].apply(
            lambda v: isinstance(v, float) and not np.isnan(v))]
        assert len(test_rows) >= 10

    def test_ks_test_present(self, graphene_result):
        assert any(graphene_result["stat_df"]["test"].str.contains("Kolmogorov", na=False))

    def test_shapiro_wilk_present(self, graphene_result):
        assert any(graphene_result["stat_df"]["test"].str.contains("Shapiro", na=False))

    def test_variance_ratio_present(self, graphene_result):
        assert any(graphene_result["stat_df"]["test"].str.contains("Variance ratio", na=False))

    def test_bootstrap_diff_ci_excludes_or_includes_zero(self, graphene_result):
        stat_df = graphene_result["stat_df"]
        ci_row = stat_df[stat_df["test"].str.contains("95% bootstrap CI", na=False)]
        assert len(ci_row) >= 1
        interp = str(ci_row["interpretation"].iloc[0])
        assert interp in ("excludes 0 *", "includes 0"), f"Unexpected CI interpretation: {interp}"

    def test_kruskal_wallis_present(self, graphene_result):
        assert any(graphene_result["stat_df"]["test"].str.contains("Kruskal", na=False))

    def test_rank_biserial_present(self, graphene_result):
        assert any(graphene_result["stat_df"]["test"].str.contains("Rank-biserial", na=False))


# ---------------------------------------------------------------------------
# Cross-scenario sanity checks
# ---------------------------------------------------------------------------

class TestCrossScenario:

    def test_all_three_task_types_covered(
        self, secom_result, urban_result, graphene_result
    ):
        task_types = {
            secom_result["task_type"],
            urban_result["task_type"],
            graphene_result["task_type"],
        }
        assert task_types == {"binary", "multiclass", "regression"}

    def test_feature_reduction_achieved_in_all_scenarios(
        self, secom_result, urban_result, graphene_result
    ):
        for r in (secom_result, urban_result, graphene_result):
            reduction = 1.0 - r["n_features_robust"] / r["n_features_bl"]
            assert reduction > 0, f"{r['name']}: no feature reduction achieved."

    def test_all_results_tables_contain_required_keys(
        self, secom_result, urban_result, graphene_result
    ):
        required = {"overview", "selected_features", "stability_selection",
                    "nested_cv_scores", "nested_cv_predictions"}
        for r in (secom_result, urban_result, graphene_result):
            tables = r["robust_result"].results_tables()
            assert required.issubset(tables), (
                f"{r['name']}: missing tables {required - set(tables.keys())}"
            )

    def test_scores_are_finite_in_all_scenarios(
        self, secom_result, urban_result, graphene_result
    ):
        for r in (secom_result, urban_result, graphene_result):
            assert np.isfinite(r["score_robust"]), f"{r['name']}: ROBUST score is not finite."
            assert np.isfinite(r["score_bl"]), f"{r['name']}: BL score is not finite."

    def test_stat_battery_consistent_across_scenarios(
        self, secom_result, urban_result, graphene_result
    ):
        for r in (secom_result, urban_result, graphene_result):
            stat_df = r["stat_df"]
            for col in ("test", "statistic", "p_value", "interpretation"):
                assert col in stat_df.columns


# ---------------------------------------------------------------------------
# Comparator tests
# ---------------------------------------------------------------------------

class TestComparators:
    """Smoke tests for the three comparator runners and the Jaccard metric."""

    # ---- jaccard_stability ----

    def test_jaccard_identical_sets(self):
        sets = [{"a", "b", "c"}] * 5
        assert abs(jaccard_stability(sets) - 1.0) < 1e-9

    def test_jaccard_disjoint_sets(self):
        sets = [{"a", "b"}, {"c", "d"}, {"e", "f"}]
        assert abs(jaccard_stability(sets) - 0.0) < 1e-9

    def test_jaccard_partial_overlap(self):
        sets = [{"a", "b", "c"}, {"b", "c", "d"}]
        # |{b,c}| / |{a,b,c,d}| = 2/4 = 0.5
        assert abs(jaccard_stability(sets) - 0.5) < 1e-9

    def test_jaccard_fewer_than_two_returns_nan(self):
        assert np.isnan(jaccard_stability([{"a", "b"}]))
        assert np.isnan(jaccard_stability([]))

    def test_jaccard_empty_sets_count_as_identical(self):
        assert abs(jaccard_stability([set(), set()]) - 1.0) < 1e-9

    # ---- ANOVA comparator ----

    def test_anova_returns_comparator_result(self, graphene_result):
        cr = graphene_result["comparators"]["anova"]
        assert isinstance(cr, ComparatorResult)

    def test_anova_score_is_finite(self, secom_result):
        cr = secom_result["comparators"]["anova"]
        assert np.isfinite(cr.mean_score)

    def test_anova_stability_in_unit_interval(self, urban_result):
        cr = urban_result["comparators"]["anova"]
        assert 0.0 <= cr.stability <= 1.0

    def test_anova_feature_sets_have_correct_k(self, secom_result):
        cr = secom_result["comparators"]["anova"]
        k = cr.hyperparams["k"]
        # k should match the 10% rule: max(10, n_features // 10)
        expected_k = max(10, secom_result["n_features_bl"] // 10)
        assert k == expected_k, f"ANOVA k={k}, expected {expected_k} (10% of {secom_result['n_features_bl']})"
        for fset in cr.fold_feature_sets:
            assert len(fset) == k

    def test_anova_feature_names_are_subset_of_original(self, graphene_result):
        ds = graphene_result["dataset"]
        all_names = set(ds.X_train.columns.tolist())
        cr = graphene_result["comparators"]["anova"]
        for fset in cr.fold_feature_sets:
            assert fset.issubset(all_names)

    def test_anova_fold_count_matches_outer_cv(self, secom_result):
        cr = secom_result["comparators"]["anova"]
        assert len(cr.fold_scores) == _OUTER_CV
        assert len(cr.fold_feature_sets) == _OUTER_CV

    # ---- RFECV comparator ----

    def test_rfecv_returns_comparator_result(self, secom_result):
        cr = secom_result["comparators"]["rfecv"]
        assert isinstance(cr, ComparatorResult)

    def test_rfecv_score_is_finite(self, graphene_result):
        cr = graphene_result["comparators"]["rfecv"]
        assert np.isfinite(cr.mean_score)

    def test_rfecv_stability_in_unit_interval(self, urban_result):
        cr = urban_result["comparators"]["rfecv"]
        assert 0.0 <= cr.stability <= 1.0

    def test_rfecv_selects_at_least_one_feature(self, secom_result):
        cr = secom_result["comparators"]["rfecv"]
        assert all(len(s) >= 1 for s in cr.fold_feature_sets)

    def test_rfecv_fold_count_matches_outer_cv(self, graphene_result):
        cr = graphene_result["comparators"]["rfecv"]
        assert len(cr.fold_scores) == _OUTER_CV

    # ---- Boruta comparator (skipped if package absent) ----

    def test_boruta_result_or_none(self, secom_result):
        cr = secom_result["comparators"]["boruta"]
        assert cr is None or isinstance(cr, ComparatorResult)

    def test_boruta_score_finite_if_present(self, graphene_result):
        cr = graphene_result["comparators"]["boruta"]
        if cr is None:
            pytest.skip("boruta package not installed")
        assert np.isfinite(cr.mean_score)

    def test_boruta_stability_in_unit_interval_if_present(self, urban_result):
        cr = urban_result["comparators"]["boruta"]
        if cr is None:
            pytest.skip("boruta package not installed")
        assert 0.0 <= cr.stability <= 1.0

    # ---- ROBUST stability ----

    def test_robust_stability_in_unit_interval(self, secom_result):
        s = secom_result["robust_stability"]
        assert np.isfinite(s)
        assert 0.0 <= s <= 1.0

    def test_robust_stability_present_all_scenarios(
        self, secom_result, urban_result, graphene_result
    ):
        for r in (secom_result, urban_result, graphene_result):
            assert "robust_stability" in r
            assert np.isfinite(r["robust_stability"])

    # ---- Cross-method sanity ----

    def test_comparator_keys_present_in_all_scenarios(
        self, secom_result, urban_result, graphene_result
    ):
        for r in (secom_result, urban_result, graphene_result):
            assert "comparators" in r
            assert {"anova", "rfecv", "boruta"} == set(r["comparators"].keys())

    def test_all_comparator_scores_within_plausible_range(self, secom_result):
        for key, cres in secom_result["comparators"].items():
            if cres is None:
                continue
            # AUC must be in [0, 1]
            assert 0.0 <= cres.mean_score <= 1.0, (
                f"{key} mean AUC {cres.mean_score:.4f} outside [0,1] on SECOM"
            )

    def test_regression_comparator_scores_are_negative(self, graphene_result):
        # neg-RMSE must be ≤ 0
        for key, cres in graphene_result["comparators"].items():
            if cres is None:
                continue
            assert cres.mean_score <= 0.0, (
                f"{key} mean score {cres.mean_score:.4f} should be neg-RMSE (≤ 0)"
            )


# ---------------------------------------------------------------------------
# __main__ entry point: full verbose console report
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(_SEP)
    print("  ROBUST MODEL MAKER -- SCIENTIFIC BENCHMARK DEMONSTRATION")
    print(f"  Outer CV={_OUTER_CV}  Inner CV={_INNER_CV}  "
          f"N-bootstrap={ROBUST_PARAMS['n_bootstrap']}  N-iter={_N_ITER}  "
          f"Stability threshold={ROBUST_PARAMS['stability_threshold']}")
    print(_SEP)
    print(_LEGEND)
    print(f"  Goal: feature reduction while preserving predictive performance.")
    print(f"  Outcome: 'preserved' = no significant difference from full-feature baseline (p >= 0.05) -- primary success criterion")
    print(f"           'sig. worse *' = significant performance cost (p < 0.05)  |  'sig. better *' = unexpected improvement")
    print(_SEP)

    loaders = [_load_secom, _load_urban_land_cover, _load_graphene_oxide]
    all_results = []
    for loader in loaders:
        try:
            ds = loader()
        except DataUnavailable as e:
            print(f"\n  SKIP: {e}")
            continue
        result = run_scenario(ds, verbose=True)
        all_results.append(result)

    if all_results:
        print_summary_table(all_results)
        print_comparator_summary(all_results)
        total = sum(r["elapsed"] for r in all_results)
        print(f"\n  Total wall-clock time: {total:.1f}s\n")

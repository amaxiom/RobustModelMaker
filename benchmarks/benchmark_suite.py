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
from typing import Any, Dict, List, Optional, Tuple

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
    outer_cv=5,
    inner_cv=2,
    n_bootstrap=15,
    n_iter=8,
    stability_threshold=0.5,
    cutoff_n_bootstrap=100,
    random_state=42,
    n_jobs=1,
    verbose=False,
)

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
        alg="rdg",
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
        alg="rdg",
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
        alg="las",
        floor_score=-8.0,
        train_idx=train_idx,
        test_idx=test_idx,
    )


# ---------------------------------------------------------------------------
# Full-feature nested-CV baseline
# ---------------------------------------------------------------------------

def _build_baseline_estimator(
    task_type: str, alg: str, seed: int
) -> Tuple[Any, Dict, str]:
    """Return (sklearn_pipeline, param_distributions, scoring_string)."""
    pre = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    if task_type == "regression":
        if alg == "las":
            mdl = Lasso(max_iter=10_000, random_state=seed)
            params = {"model__alpha": loguniform(1e-4, 1e1)}
        else:
            mdl = Ridge(random_state=seed)
            params = {"model__alpha": loguniform(1e-4, 1e2)}
        pipe = Pipeline([("pre", pre), ("model", mdl)])
        scoring = "neg_root_mean_squared_error"
    else:
        if alg == "eln":
            mdl = LogisticRegression(
                penalty="elasticnet", solver="saga", l1_ratio=0.5,
                max_iter=5000, random_state=seed, class_weight="balanced",
            )
            params = {"model__C": loguniform(1e-3, 1e2), "model__l1_ratio": uniform(0, 1)}
        else:
            mdl = LogisticRegression(
                penalty="l2", solver="lbfgs",
                max_iter=5000, random_state=seed, class_weight="balanced",
            )
            params = {"model__C": loguniform(1e-3, 1e2)}
        pipe = Pipeline([("pre", pre), ("model", mdl)])
        scoring = "roc_auc" if task_type == "binary" else "roc_auc_ovr_weighted"
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
    try:
        p = float(v)
    except (TypeError, ValueError):
        return "            "
    if np.isnan(p):
        return "            "
    if p < 0.001:
        return f"{p:.3e} ***"
    if p < 0.01:
        return f"{p:.5f} ** "
    if p < 0.05:
        return f"{p:.5f} *  "
    return f"{p:.5f}    "


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
    predictive performance.  A binary 'winner' label based on a fixed score
    threshold is misleading when differences are small or noisy.  Instead,
    statistical significance of the fold-score difference determines the label:

      preserved   score difference is not significant (p >= 0.05) -- the
                  primary success criterion: fewer features, no real loss
      improved *  score is significantly higher (p < 0.05, delta > 0)
      degraded *  score is significantly lower  (p < 0.05, delta < 0)

    The p-value threshold (0.05) is the conventional two-sided alpha used
    across the rest of the test battery; the asterisk (*) flags significance.
    """
    p = _significance_p(stat_df)
    if np.isnan(p) or p >= 0.05:
        return "preserved"
    return "improved *" if delta > 0 else "degraded *"


def print_scenario_report(
    ds: BenchmarkDataset,
    robust_result: Any,
    baseline: Dict[str, Any],
    stat_df: pd.DataFrame,
    elapsed: float,
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
    print(f"  {'Baseline (BL)':<{lbl_w}}: {n_bl:5d} features   "
          f"score = {score_bl:+.4f} +/- {baseline['std']:.4f}")
    print(f"  {'ROBUST':<{lbl_w}}: {n_robust:5d} features   "
          f"score = {score_robust_mean:+.4f} +/- {robust_result.nested_cv_result.std_score:.4f}")
    print(f"  {'Feature reduction':<{lbl_w}}: {reduction:5.1f}%   "
          f"({n_bl - n_robust} features removed)")
    print(f"  {'Score delta (ROBUST - BL)':<{lbl_w}}: {delta:+.4f}   "
          f"p = {p_str}  ->  outcome: {outcome}")
    if abs(score_bl) > 1e-9 and n_bl > 0 and n_robust > 0:
        spf_robust = abs(score_robust_mean) / n_robust
        spf_bl = abs(score_bl) / n_bl
        print(f"  {'Efficiency gain':<{lbl_w}}: {spf_robust / max(spf_bl, 1e-15):.2f}x   "
              f"score-per-feature (ROBUST / BL)")
    print(f"\n  Outcome key: 'preserved' = no significant performance loss (p >= 0.05); "
          f"'improved *' / 'degraded *' = p < 0.05")

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
    print(f"\n  {'PER-FOLD SCORES  (outer CV, train split only)':^{_W - 4}}")
    print(_sep)
    robust_scores = robust_result.nested_cv_result.outer_scores
    bl_scores = baseline["fold_scores"]
    n_folds = len(robust_scores)
    deltas = robust_scores - bl_scores[:n_folds]
    fold_df = pd.DataFrame({
        "fold":         np.arange(1, n_folds + 1),
        "ROBUST_score": np.round(robust_scores, 5),
        "BL_score":     np.round(bl_scores[:n_folds], 5),
        "delta":        np.round(deltas, 5),
    })
    print(fold_df.to_string(index=False))
    wins = int(np.sum(robust_scores > bl_scores[:n_folds] + 1e-6))
    losses = int(np.sum(robust_scores < bl_scores[:n_folds] - 1e-6))
    print(f"  ROBUST higher on {wins}/{n_folds} folds, lower on {losses}/{n_folds} folds  "
          f"(fold-level counts; significance determined by paired test above)")

    # ---- Statistical test battery ----
    print(f"\n  {'STATISTICAL TEST BATTERY':^{_W - 4}}")
    print(_sep)
    print(f"  Significance threshold: p < 0.05 (two-sided).  "
          f"*** p<0.001  ** p<0.01  * p<0.05")
    hdr = f"  {'TEST':<52} {'STATISTIC':>13}  {'P-VALUE':>14}  INTERPRETATION"
    print(hdr)
    print(f"  {'-' * 51} {'-' * 13}  {'-' * 14}  {'-' * 24}")
    for _, r in stat_df.iterrows():
        stat_s = _fmt_stat(r["statistic"])
        p_s = _fmt_p(r["p_value"])
        interp = str(r.get("interpretation", ""))[:30]
        print(f"  {str(r['test']):<52} {stat_s}  {p_s}  {interp}")
    print()


def print_summary_table(results: List[Dict[str, Any]]) -> None:
    # Column widths: chosen to fit the widest expected value without wrapping.
    # BL = full-feature baseline (uses all p features), so no separate BL-feats column.
    C = dict(
        name=24, task=11, nxp=13,
        rob_n=12, red=5,
        bl_sc=9, rob_sc=12, delta=8, pval=7, outcome=11,
    )
    # Compute separator width to exactly match the table header
    _TW = (2 + C['name'] + 1 + C['task'] + 1 + C['nxp'] + 2
           + C['rob_n'] + 1 + C['red'] + 2
           + C['bl_sc'] + 1 + C['rob_sc'] + 1 + C['delta'] + 2
           + C['pval'] + 1 + C['outcome'])
    _TSEP = "=" * _TW
    _tsep = "-" * _TW

    print(f"\n{_TSEP}")
    print("  CROSS-SCENARIO SUMMARY")
    print(_TSEP)
    print(_LEGEND)
    print(f"  Outcome: 'preserved' = score not significantly different from BL (p >= 0.05, two-sided paired test)")
    print(f"           'improved *' / 'degraded *' = statistically significant difference (p < 0.05)")
    print(_tsep)
    hdr = (
        f"  {'Scenario':<{C['name']}} {'Task':<{C['task']}} "
        f"{'n_train x p':>{C['nxp']}}  "
        f"{'ROBUST feats':>{C['rob_n']}} {'Red%':>{C['red']}}  "
        f"{'BL score':>{C['bl_sc']}} {'ROBUST score':>{C['rob_sc']}} {'delta':>{C['delta']}}  "
        f"{'p-val':>{C['pval']}} {'Outcome':<{C['outcome']}}"
    )
    print(hdr)
    sep_row = (
        f"  {'-'*C['name']} {'-'*C['task']} "
        f"{'-'*C['nxp']}  "
        f"{'-'*C['rob_n']} {'-'*C['red']}  "
        f"{'-'*C['bl_sc']} {'-'*C['rob_sc']} {'-'*C['delta']}  "
        f"{'-'*C['pval']} {'-'*C['outcome']}"
    )
    print(sep_row)
    print()
    for r in results:
        n_bl = r["n_features_bl"]      # total features = what BL trained on
        n_robust = r["n_features_robust"]
        red = (1 - n_robust / n_bl) * 100
        delta = r["score_robust"] - r["score_bl"]
        n_train = r["n_samples"]
        nxp_str = f"{n_train} x {n_bl:4d}"
        p_val = _significance_p(r["stat_df"])
        p_str = f"{p_val:.3f}" if np.isfinite(p_val) else "  n/a"
        outcome = _outcome(delta, r["stat_df"])
        print(
            f"  {r['name']:<{C['name']}} {r['task_type']:<{C['task']}} "
            f"{nxp_str:>{C['nxp']}}  "
            f"{n_robust:>{C['rob_n']}} {red:>{C['red']}.0f}%  "
            f"{r['score_bl']:>{C['bl_sc']}.4f} {r['score_robust']:>{C['rob_sc']}.4f} "
            f"{delta:>{C['delta']}.4f}  "
            f"{p_str:>{C['pval']}} {outcome:<{C['outcome']}}"
        )
    print()
    print(_TSEP)


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

def run_scenario(ds: BenchmarkDataset, verbose: bool = True) -> Dict[str, Any]:
    """Fit ROBUST and baseline on one dataset; return results dict."""
    t0 = time.time()
    if verbose:
        print(f"\n>>> {ds.name}: running ROBUST({ds.alg.upper()}, {ds.task_type}) ...", flush=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        maker = robust.RobustModelMaker(alg=ds.alg, task_type=ds.task_type, **ROBUST_PARAMS)
        maker.fit(ds.X_train, ds.y_train)
    robust_result = maker.result_

    if verbose:
        print(f"    ROBUST done ({len(robust_result.selected_features)} features selected). Running baseline ...", flush=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        baseline = run_baseline_nested_cv(ds)

    elapsed = time.time() - t0

    robust_scores = robust_result.nested_cv_result.outer_scores
    bl_scores = baseline["fold_scores"]
    stat_df = run_statistical_battery(robust_scores, bl_scores, ds.task_type, ds.floor_score)

    if verbose:
        print_scenario_report(ds, robust_result, baseline, stat_df, elapsed)

    return {
        "name": ds.name,
        "task_type": ds.task_type,
        "alg": ds.alg,
        "n_samples": ds.X_train.shape[0],
        "n_features_bl": baseline["n_features"],
        "n_features_robust": len(robust_result.selected_features),
        "score_bl": baseline["mean"],
        "score_robust": robust_result.nested_cv_result.mean_score,
        "robust_result": robust_result,
        "baseline": baseline,
        "stat_df": stat_df,
        "elapsed": elapsed,
        "dataset": ds,
    }


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
    print(f"  Outcome: 'preserved' = ROBUST score not significantly different from BL (p >= 0.05)")
    print(f"           'improved *' / 'degraded *' = statistically significant change (p < 0.05)")
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
        total = sum(r["elapsed"] for r in all_results)
        print(f"\n  Total wall-clock time: {total:.1f}s\n")

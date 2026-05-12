"""
Performance test suite for RobustModelMaker v0.3.

Purpose
-------
This suite measures runtime, peak Python memory, scaling behaviour, save/reporting
overhead, permutation-importance cost, and repeated-run timing variance.

It is intentionally separate from unit_test_suite.py. Unit tests check correctness.
These tests produce benchmark records that can be compared across versions.

How to run
----------
Fast performance smoke test:
    pytest -q performance_test_suite.py -s

Full performance suite:
    set RUN_PERFORMANCE=1          # Windows cmd
    pytest -q performance_test_suite.py -s

Optional strict budget checks:
    set ROBUST_PERF_STRICT=1
    pytest -q performance_test_suite.py -s

Optional baseline comparison:
    set ROBUST_PERF_BASELINE=path\to\previous_perf_results.json
    pytest -q performance_test_suite.py -s

Outputs
-------
Writes JSONL records during the run and a summary JSON at session finish.
Default output directory:
    performance_results/

Notes
-----
Timings vary by CPU, BLAS, Python version, sklearn version, and background load.
The default thresholds are deliberately generous. Use baseline comparison for
serious regression tracking.
"""

from __future__ import annotations

import gc
import importlib
import importlib.util
import json
import math
import os
import platform
import statistics
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression


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
            spec = importlib.util.spec_from_file_location("robust_perf", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(
        "RobustModelMaker.py not found. Place it one level above tests/ "
        "or set ROBUST_MODEL_MAKER_PATH=/path/to/RobustModelMaker.py."
    )


robust = _load_robust_module()

RUN_PERFORMANCE = os.environ.get("RUN_PERFORMANCE", "0") == "1"
STRICT = os.environ.get("ROBUST_PERF_STRICT", "0") == "1"
RESULT_DIR = Path(os.environ.get("ROBUST_PERF_DIR", "performance_results"))
RESULT_DIR.mkdir(parents=True, exist_ok=True)
JSONL_PATH = RESULT_DIR / "performance_records.jsonl"
SUMMARY_PATH = RESULT_DIR / "performance_summary.json"
BASELINE_PATH = os.environ.get("ROBUST_PERF_BASELINE")

# Keep defaults modest so a normal development run is useful but not punishing.
FAST_KWARGS = dict(
    outer_cv=2,
    inner_cv=2,
    n_iter=1,
    n_bootstrap=2,
    cutoff_n_bootstrap=8,
    stability_threshold=0.10,
    random_state=123,
    n_jobs=1,
    verbose=False,
    save_results=False,
)

MEDIUM_KWARGS = dict(
    outer_cv=3,
    inner_cv=2,
    n_iter=2,
    n_bootstrap=4,
    cutoff_n_bootstrap=12,
    stability_threshold=0.10,
    random_state=123,
    n_jobs=1,
    verbose=False,
    save_results=False,
)

# Generous upper bounds. These are guards against accidental exponential blowups,
# not precise performance claims. Override with environment variables if needed.
DEFAULT_BUDGET_SECONDS = float(os.environ.get("ROBUST_PERF_BUDGET_SECONDS", "90"))
DEFAULT_MEMORY_MB = float(os.environ.get("ROBUST_PERF_MEMORY_MB", "750"))


@dataclass
class PerfRecord:
    name: str
    task_type: str
    algorithm: str
    n_samples: int
    n_features: int
    seconds: float
    peak_memory_mb: float
    selected_features: int
    mean_score: Optional[float]
    notes: str = ""


def _write_record(record: PerfRecord) -> None:
    with JSONL_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def _load_records() -> List[Dict[str, Any]]:
    if not JSONL_PATH.exists():
        return []
    rows = []
    for line in JSONL_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def pytest_sessionstart(session):  # pragma: no cover, pytest hook
    # Start each performance run with a fresh record file.
    if JSONL_PATH.exists():
        JSONL_PATH.unlink()


def pytest_sessionfinish(session, exitstatus):  # pragma: no cover, pytest hook
    records = _load_records()
    summary = {
        "module": MODULE_NAME,
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "strict": STRICT,
        "run_performance": RUN_PERFORMANCE,
        "n_records": len(records),
        "records": records,
    }
    if records:
        summary["total_seconds"] = float(sum(r["seconds"] for r in records))
        summary["max_seconds"] = float(max(r["seconds"] for r in records))
        summary["max_peak_memory_mb"] = float(max(r["peak_memory_mb"] for r in records))
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nPerformance summary written to: {SUMMARY_PATH}")


def _make_binary(n_samples: int = 90, n_features: int = 8, random_state: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, min(4, n_features // 2)),
        n_redundant=1 if n_features >= 4 else 0,
        n_repeated=0,
        n_classes=2,
        weights=[0.55, 0.45],
        class_sep=1.2,
        random_state=random_state,
    )
    cols = [f"b{i}" for i in range(n_features)]
    idx = [f"bin_{i:04d}" for i in range(n_samples)]
    return pd.DataFrame(X, columns=cols, index=idx), pd.Series(y, index=idx, name="target")


def _make_multiclass(n_samples: int = 96, n_features: int = 9, random_state: int = 2) -> Tuple[pd.DataFrame, pd.Series]:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(3, min(5, n_features // 2 + 1)),
        n_redundant=1 if n_features >= 5 else 0,
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.3,
        random_state=random_state,
    )
    labels = np.array(["alpha", "beta", "gamma"])[y]
    cols = [f"m{i}" for i in range(n_features)]
    idx = [f"mc_{i:04d}" for i in range(n_samples)]
    return pd.DataFrame(X, columns=cols, index=idx), pd.Series(labels, index=idx, name="class")


def _make_regression(n_samples: int = 90, n_features: int = 7, random_state: int = 3) -> Tuple[pd.DataFrame, pd.Series]:
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, min(4, n_features // 2 + 1)),
        noise=8.0,
        random_state=random_state,
    )
    cols = [f"r{i}" for i in range(n_features)]
    idx = [f"reg_{i:04d}" for i in range(n_samples)]
    return pd.DataFrame(X, columns=cols, index=idx), pd.Series(y, index=idx, name="response")


def _fit_timed(name: str, alg: str, task_type: str, X: pd.DataFrame, y: pd.Series, kwargs: Dict[str, Any]) -> Tuple[Any, PerfRecord]:
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    maker = robust.RobustModelMaker(alg=alg, task_type=task_type, **kwargs).fit(X, y)
    seconds = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result = maker.result_
    record = PerfRecord(
        name=name,
        task_type=result.task_type,
        algorithm=result.algorithm,
        n_samples=int(X.shape[0]),
        n_features=int(X.shape[1]),
        seconds=float(seconds),
        peak_memory_mb=float(peak / (1024 ** 2)),
        selected_features=int(len(result.selected_features)),
        mean_score=float(result.mean_score) if getattr(result, "mean_score", None) is not None else None,
    )
    _write_record(record)
    return maker, record


def _assert_budget(record: PerfRecord, seconds: float = DEFAULT_BUDGET_SECONDS, memory_mb: float = DEFAULT_MEMORY_MB) -> None:
    assert math.isfinite(record.seconds)
    assert record.seconds > 0
    assert math.isfinite(record.peak_memory_mb)
    assert record.peak_memory_mb >= 0
    if STRICT:
        assert record.seconds <= seconds, f"{record.name} took {record.seconds:.2f}s > {seconds:.2f}s"
        assert record.peak_memory_mb <= memory_mb, f"{record.name} used {record.peak_memory_mb:.1f} MB > {memory_mb:.1f} MB"


def _skip_xgb_if_missing(alg: str) -> None:
    if alg == "xgb" and not getattr(rmm, "_HAS_XGBOOST", False):
        pytest.skip("xgboost is not installed")


@pytest.mark.parametrize("alg", ["log", "rdg", "las", "svm", "eln", "rf"])
def test_binary_fit_predict_performance_fast(alg):
    _skip_xgb_if_missing(alg)
    X, y = _make_binary()
    maker, rec = _fit_timed(f"binary_fast_{alg}", alg, "binary", X, y, dict(FAST_KWARGS))
    preds = maker.predict(X.head(10))
    assert len(preds) == 10
    _assert_budget(rec)


@pytest.mark.parametrize("alg", ["log", "rdg", "las", "svm", "eln", "rf"])
def test_multiclass_fit_predict_performance_fast(alg):
    _skip_xgb_if_missing(alg)
    X, y = _make_multiclass()
    maker, rec = _fit_timed(f"multiclass_fast_{alg}", alg, "multiclass", X, y, dict(FAST_KWARGS))
    preds = maker.predict(X.head(10))
    assert len(preds) == 10
    tables = maker.result_.results_tables()
    assert "nested_cv_predictions" in tables
    _assert_budget(rec)


@pytest.mark.parametrize("alg", ["rdg", "las", "svm", "eln", "rf", "lin"])
def test_regression_fit_predict_performance_fast(alg):
    _skip_xgb_if_missing(alg)
    X, y = _make_regression()
    maker, rec = _fit_timed(f"regression_fast_{alg}", alg, "regression", X, y, dict(FAST_KWARGS))
    preds = maker.predict(X.head(10))
    assert len(preds) == 10
    _assert_budget(rec)


@pytest.mark.skipif(not RUN_PERFORMANCE, reason="Set RUN_PERFORMANCE=1 for scaling benchmarks")
def test_binary_scaling_logistic_runtime():
    records = []
    for n in [80, 160, 320]:
        X, y = _make_binary(n_samples=n, n_features=10, random_state=n)
        _, rec = _fit_timed(f"binary_scaling_log_n{n}", "log", "binary", X, y, dict(FAST_KWARGS))
        records.append(rec)
        _assert_budget(rec, seconds=DEFAULT_BUDGET_SECONDS * 2)

    # Runtime should not collapse into nonsense; this is deliberately weak because
    # small benchmarks are noisy and CV splits can change class balance.
    assert records[-1].seconds < max(1.0, records[0].seconds) * 20


@pytest.mark.skipif(not RUN_PERFORMANCE, reason="Set RUN_PERFORMANCE=1 for scaling benchmarks")
def test_feature_scaling_logistic_runtime():
    records = []
    for p in [8, 16, 32]:
        X, y = _make_binary(n_samples=120, n_features=p, random_state=100 + p)
        _, rec = _fit_timed(f"feature_scaling_log_p{p}", "log", "binary", X, y, dict(FAST_KWARGS))
        records.append(rec)
        _assert_budget(rec, seconds=DEFAULT_BUDGET_SECONDS * 2)
    assert records[-1].seconds < max(1.0, records[0].seconds) * 25


@pytest.mark.skipif(not RUN_PERFORMANCE, reason="Set RUN_PERFORMANCE=1 for repeated-run timing benchmarks")
def test_repeated_run_timing_variance_binary_logistic():
    X, y = _make_binary(n_samples=100, n_features=8, random_state=44)
    times = []
    selected = []
    for i in range(3):
        kwargs = dict(FAST_KWARGS)
        kwargs["random_state"] = 999
        maker, rec = _fit_timed(f"repeat_binary_log_{i}", "log", "binary", X, y, kwargs)
        times.append(rec.seconds)
        selected.append(tuple(maker.result_.selected_features))
    assert len(set(selected)) == 1
    if STRICT and statistics.mean(times) > 0:
        assert statistics.pstdev(times) / statistics.mean(times) < 0.75


@pytest.mark.skipif(not RUN_PERFORMANCE, reason="Set RUN_PERFORMANCE=1 for save/reporting overhead benchmark")
def test_save_reporting_overhead(tmp_path):
    X, y = _make_binary(n_samples=100, n_features=8, random_state=55)
    kwargs_no_save = dict(FAST_KWARGS)
    kwargs_no_save["save_results"] = False
    _, rec_no_save = _fit_timed("save_overhead_no_save", "log", "binary", X, y, kwargs_no_save)

    kwargs_save = dict(FAST_KWARGS)
    kwargs_save.update(save_results=True, output_dir=tmp_path / "saved", output_prefix="perf")
    maker, rec_save = _fit_timed("save_overhead_with_save", "log", "binary", X, y, kwargs_save)

    assert Path(maker.result_.results_dir).exists()
    assert (Path(maker.result_.results_dir) / "perf_metadata.json").exists()
    _assert_budget(rec_no_save)
    _assert_budget(rec_save)
    if STRICT:
        assert rec_save.seconds <= max(1.0, rec_no_save.seconds * 3.0)


@pytest.mark.skipif(not RUN_PERFORMANCE, reason="Set RUN_PERFORMANCE=1 for permutation importance benchmark")
def test_permutation_importance_overhead_binary_logistic():
    X, y = _make_binary(n_samples=120, n_features=10, random_state=66)
    maker, rec_fit = _fit_timed("perm_importance_fit", "log", "binary", X, y, dict(FAST_KWARGS))

    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    pi = maker.permutation_importance(X, y, n_repeats=2, random_state=42, n_jobs=1)
    seconds = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    record = PerfRecord(
        name="perm_importance_binary_log",
        task_type="binary",
        algorithm="log",
        n_samples=int(X.shape[0]),
        n_features=int(X.shape[1]),
        seconds=float(seconds),
        peak_memory_mb=float(peak / (1024 ** 2)),
        selected_features=int(len(maker.result_.selected_features)),
        mean_score=None,
        notes="permutation_importance only",
    )
    _write_record(record)
    # permutation_importance() returns a PermutationImportanceResult by default;
    # use as_frame=True to get a DataFrame, or call .summary() on the result.
    assert hasattr(pi, "importances_mean")
    assert len(pi.feature_names) >= 1
    summary = pi.summary()
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) >= 1
    _assert_budget(record)
    if STRICT:
        assert record.seconds <= max(1.0, rec_fit.seconds * 3.0)


@pytest.mark.skipif(not RUN_PERFORMANCE, reason="Set RUN_PERFORMANCE=1 for grouped CV benchmark")
def test_grouped_cv_performance_binary_logistic():
    X, y = _make_binary(n_samples=120, n_features=8, random_state=77)
    groups = np.repeat(np.arange(40), 3)
    kwargs = dict(FAST_KWARGS)
    kwargs.update(outer_cv=3, repeated_outer_cv=3)
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    maker = robust.RobustModelMaker(alg="log", task_type="binary", **kwargs).fit(X, y, groups=groups)
    seconds = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rec = PerfRecord(
        name="grouped_cv_binary_log",
        task_type="binary",
        algorithm="log",
        n_samples=int(X.shape[0]),
        n_features=int(X.shape[1]),
        seconds=float(seconds),
        peak_memory_mb=float(peak / (1024 ** 2)),
        selected_features=int(len(maker.result_.selected_features)),
        mean_score=float(maker.result_.mean_score),
        notes="groups supplied; repeated_outer_cv should be ignored safely",
    )
    _write_record(rec)
    _assert_budget(rec)


@pytest.mark.skipif(not RUN_PERFORMANCE, reason="Set RUN_PERFORMANCE=1 for slow algorithm benchmark")
@pytest.mark.parametrize("task_type,alg,make_data", [
    ("binary", "mlp", _make_binary),
    ("multiclass", "mlp", _make_multiclass),
    ("regression", "mlp", _make_regression),
])
def test_mlp_performance_opt_in(task_type, alg, make_data):
    X, y = make_data(n_samples=90, random_state=88)
    kwargs = dict(FAST_KWARGS)
    _, rec = _fit_timed(f"{task_type}_mlp_opt_in", alg, task_type, X, y, kwargs)
    _assert_budget(rec, seconds=DEFAULT_BUDGET_SECONDS * 2)


@pytest.mark.skipif(BASELINE_PATH is None, reason="Set ROBUST_PERF_BASELINE to compare against previous summary")
def test_compare_against_baseline_if_requested():
    baseline_path = Path(BASELINE_PATH)
    assert baseline_path.exists(), f"Baseline file does not exist: {baseline_path}"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    current_records = _load_records()
    baseline_records = baseline.get("records", [])
    base_by_name = {r["name"]: r for r in baseline_records}
    regressions = []
    for rec in current_records:
        old = base_by_name.get(rec["name"])
        if not old:
            continue
        old_s = float(old["seconds"])
        new_s = float(rec["seconds"])
        if old_s > 0 and new_s > old_s * float(os.environ.get("ROBUST_PERF_REGRESSION_FACTOR", "2.0")):
            regressions.append((rec["name"], old_s, new_s))
    assert not regressions, f"Runtime regressions detected: {regressions}"

"""
threshold_optimizer.py
======================
Multi-objective grid search over RobustModelMaker's ``stability_threshold``.

The three objectives are in natural tension:

  * A **higher** threshold is more conservative — only features that survive a
    large fraction of bootstrap resamples pass.  This removes more features
    (high compression, high stability) but risks excluding borderline-signal
    features (potentially lower predictive score).

  * A **lower** threshold is more permissive — more features survive (lower
    compression, potentially lower stability), and predictive score is usually
    higher because more signal is retained.

This optimizer sweeps a configurable grid of threshold values, runs a full
RobustModelMaker nested-CV fit at each point, and records three metrics:

  1. **Predictive score**  — mean outer-fold AUC (classification) or neg-RMSE
     (regression).  Higher is always better in both cases.
  2. **Jaccard stability** — mean pairwise feature-set similarity across outer
     folds (Nogueira et al. 2018).  1 = identical selection every fold.
  3. **Compression ratio** — fraction of input features removed
     (1 − mean_selected / total_features).  Higher = smaller subset.

After collecting raw metrics the optimizer:

  * Normalises each objective to [0, 1] within the evaluated grid.
  * Computes a weighted composite score for a single numeric recommendation.
  * Identifies the Pareto-non-dominated front for manual trade-off inspection.
  * Prints a formatted report and returns a fully-typed ``OptimiserResult``.

Location
--------
This file lives in ``tools/`` alongside RobustModelMaker.py's parent directory.
Import resolution searches ``tools/../RobustModelMaker.py`` automatically.

Quick start
-----------
    import sys
    sys.path.insert(0, r"path/to/RobustModelMaker/tools")
    from threshold_optimizer import ThresholdOptimizer

    result = ThresholdOptimizer(
        X_train, y_train,
        task_type   = "binary",
        base_params = dict(
            outer_cv=10, inner_cv=5, n_bootstrap=25, n_iter=10,
            cutoff_n_bootstrap=100, random_state=42, n_jobs=1, verbose=False,
        ),
    ).run()

    result.print_report()
    print("Best threshold:", result.best.threshold)

    # Merge directly into your ROBUST_PARAMS:
    ROBUST_PARAMS.update(result.best_params())

With your own data
------------------
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    result = ThresholdOptimizer(X_train, y_train, task_type='binary').run()

Speed tip
---------
Each threshold evaluation is a full nested-CV run; wall-clock time scales
linearly with ``len(thresholds)``.  For initial exploration, reduce
``n_bootstrap`` (e.g. 10) and ``n_iter`` (e.g. 5) in ``base_params``, then
re-run the best candidate(s) with full production settings.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Locate and import RobustModelMaker
# ---------------------------------------------------------------------------

def _import_robust_module() -> Any:
    """Find RobustModelMaker.py and return its module object.

    Search order:
      1. ROBUST_MODEL_MAKER_PATH env var (explicit override)
      2. Same directory as this file        (tools/)
      3. Parent directory of this file      (RobustModelMaker/ — standard layout)
      4. Grandparent directory              (fallback)
      5. Standard import via sys.path
    """
    _KEY = "RobustModelMaker"
    if _KEY in sys.modules:
        return sys.modules[_KEY]

    here = Path(__file__).resolve().parent
    candidates: List[Path] = []

    env = os.environ.get("ROBUST_MODEL_MAKER_PATH")
    if env:
        candidates.append(Path(env))

    candidates += [
        here            / f"{_KEY}.py",   # tools/RobustModelMaker.py    (unlikely)
        here.parent     / f"{_KEY}.py",   # RobustModelMaker/             (expected)
        here.parent.parent / f"{_KEY}.py",# one level higher              (fallback)
    ]

    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location(_KEY, path)
            mod  = importlib.util.module_from_spec(spec)           # type: ignore[arg-type]
            sys.modules[_KEY] = mod
            spec.loader.exec_module(mod)                           # type: ignore[union-attr]
            return mod

    try:
        import RobustModelMaker as _rm
        return _rm
    except ImportError:
        raise ImportError(
            "Cannot locate RobustModelMaker.py.\n"
            "  Option 1 — place threshold_optimizer.py one level above RobustModelMaker.py.\n"
            "  Option 2 — set the ROBUST_MODEL_MAKER_PATH env var to the full .py path.\n"
            "  Option 3 — add RobustModelMaker's directory to sys.path before importing."
        )


_rm_module       = _import_robust_module()
RobustModelMaker = _rm_module.RobustModelMaker


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Nine equally-spaced threshold candidates covering the typical operating range.
DEFAULT_GRID: List[float] = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

#: Conservative defaults for quick exploration.
#: For production use, raise n_bootstrap to ≥100 and n_iter to ≥50.
DEFAULT_BASE_PARAMS: Dict[str, Any] = dict(
    alg                = "rf",
    outer_cv           = 10,
    inner_cv           = 5,
    n_bootstrap        = 25,
    n_iter             = 10,
    cutoff_n_bootstrap = 100,
    random_state       = 42,
    n_jobs             = -1,
    verbose            = False,
)


# ---------------------------------------------------------------------------
# Jaccard stability
# ---------------------------------------------------------------------------

def jaccard_stability(feature_sets: List[Set[str]]) -> float:
    """Mean pairwise Jaccard similarity across K feature sets.

    J = (1 / C(K,2)) * Σ_{i<j}  |S_i ∩ S_j| / |S_i ∪ S_j|

    Returns NaN when fewer than two sets are provided.  Two identical empty
    sets return 1.0 by convention (union = 0 means no features to disagree on).
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
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class ThresholdResult:
    """All metrics recorded for a single threshold evaluation."""

    threshold:             float          # stability_threshold value
    mean_score:            float          # mean nested-CV score (AUC or neg-RMSE)
    std_score:             float          # std across outer folds
    stability:             float          # mean pairwise Jaccard (NaN if < 2 folds)
    mean_n_features:       float          # mean features selected per outer fold
    total_features:        int            # total input features
    compression:           float          # 1 - mean_n_features / total_features
    n_features_consensus:  int            # len(selected_features) from stability run
    elapsed:               float          # wall-clock seconds for this fit
    fold_scores:           List[float]    = field(repr=False, default_factory=list)
    fold_feature_sets:     List[Set[str]] = field(repr=False, default_factory=list)
    composite:             float          = 0.0    # set by _compute_composite()
    dominated:             bool           = False  # set by _find_pareto_front()

    @property
    def display_score(self) -> float:
        """Positive-convention score (flips neg-RMSE → +RMSE for display)."""
        return abs(self.mean_score)

    @property
    def retention(self) -> float:
        """Fraction of features retained (complement of compression)."""
        return max(0.0, 1.0 - self.compression)


@dataclass
class OptimiserResult:
    """Full result returned by :meth:`ThresholdOptimizer.run`."""

    results:                List[ThresholdResult]   # one per threshold, evaluated order
    pareto_front:           List[ThresholdResult]   # non-dominated solutions
    best:                   ThresholdResult          # highest composite score
    weights:                Dict[str, float]         # weights used for composite
    is_regression:          bool
    task_type:              str
    metric_name:            str
    total_elapsed:          float
    n_thresholds_evaluated: int

    # ---- public helpers -----------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """All evaluated thresholds as a DataFrame, sorted by composite desc.

        Columns: threshold, score, score_std, stability, mean_n_feat,
        compression, composite, pareto (bool), elapsed_s.
        """
        rows = [{
            "threshold":   r.threshold,
            "score":       r.display_score,
            "score_raw":   r.mean_score,
            "score_std":   r.std_score,
            "stability":   r.stability,
            "mean_n_feat": r.mean_n_features,
            "compression": r.compression,
            "composite":   r.composite,
            "pareto":      not r.dominated,
            "elapsed_s":   r.elapsed,
        } for r in self.results]
        return (
            pd.DataFrame(rows)
            .sort_values("composite", ascending=False)
            .reset_index(drop=True)
        )

    def best_params(self) -> Dict[str, float]:
        """Return ``{"stability_threshold": <best>}`` ready to merge into ROBUST_PARAMS."""
        return {"stability_threshold": self.best.threshold}

    def print_report(self) -> None:
        """Print the full formatted report to stdout."""
        _print_report(self)

    def plot(self, figsize: Tuple[float, float] = (12, 4)) -> Any:
        """Three-panel matplotlib figure: score / stability / compression vs threshold.

        Returns the Figure object, or None if matplotlib is not installed.
        """
        return _plot(self, figsize)


# ---------------------------------------------------------------------------
# Normalisation and Pareto helpers
# ---------------------------------------------------------------------------

def _normalize(values: List[float]) -> List[float]:
    """Min-max normalise to [0, 1].  All-equal input → all 1.0."""
    arr = np.asarray(values, dtype=float)
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi - lo < 1e-12:
        return [1.0] * len(values)
    return list((arr - lo) / (hi - lo))


def _compute_composite(
    results: List[ThresholdResult],
    weights: Dict[str, float],
) -> None:
    """Compute normalised weighted composite scores in-place.

    Each objective is min-max scaled to [0, 1] across the grid, then
    combined as a weighted mean.  ``mean_score`` is used as-is (neg-RMSE
    for regression is already higher-is-better).  NaN stability → 0.0.
    """
    raw_scores = [r.mean_score for r in results]
    raw_stab   = [r.stability if np.isfinite(r.stability) else 0.0 for r in results]
    raw_compr  = [r.compression for r in results]

    ns = _normalize(raw_scores)
    nj = _normalize(raw_stab)
    nc = _normalize(raw_compr)

    w_s = max(0.0, weights.get("score",       1.0))
    w_j = max(0.0, weights.get("stability",   1.0))
    w_c = max(0.0, weights.get("compression", 1.0))
    w_total = w_s + w_j + w_c
    if w_total < 1e-12:
        w_s = w_j = w_c = w_total = 1.0

    for i, r in enumerate(results):
        r.composite = (w_s * ns[i] + w_j * nj[i] + w_c * nc[i]) / w_total


def _find_pareto_front(results: List[ThresholdResult]) -> List[ThresholdResult]:
    """Mark dominated solutions; return the Pareto-non-dominated set.

    All three objectives are maximised: ``mean_score``, ``stability``,
    ``compression``.  NaN stability is treated as −∞ for dominance checks
    so that a finite-stability solution always dominates a NaN-stability one.
    """
    def _s(r: ThresholdResult) -> float:
        return r.stability if np.isfinite(r.stability) else -1e9

    for r in results:
        r.dominated = False

    for i, ri in enumerate(results):
        for j, rj in enumerate(results):
            if i == j:
                continue
            if (rj.mean_score  >= ri.mean_score  and
                _s(rj)         >= _s(ri)          and
                rj.compression >= ri.compression  and
               (rj.mean_score  >  ri.mean_score   or
                _s(rj)         >  _s(ri)           or
                rj.compression >  ri.compression)):
                ri.dominated = True
                break

    return [r for r in results if not r.dominated]


# ---------------------------------------------------------------------------
# Core optimizer
# ---------------------------------------------------------------------------

class ThresholdOptimizer:
    """
    Multi-objective grid search over RobustModelMaker's ``stability_threshold``.

    For each candidate value in the grid, a complete nested-CV
    RobustModelMaker fit is run on the supplied training data.  Three metrics
    are recorded per threshold:

    * **Predictive score** — mean outer-fold AUC (binary/multiclass) or
      neg-RMSE (regression).  Higher is always better.
    * **Jaccard stability** — mean pairwise feature-set similarity across outer
      folds.  1.0 = identical selection every fold; 0.0 = fully disjoint.
    * **Compression ratio** — fraction of features removed (1 − selected/total).

    After the sweep the optimizer normalises all three objectives to [0, 1],
    computes a configurable weighted composite score, identifies the
    Pareto-non-dominated front, and returns an :class:`OptimiserResult`.

    Parameters
    ----------
    X_train : array-like or DataFrame
        Training features.  Passed directly to RobustModelMaker; all
        preprocessing (imputation, scaling) is handled internally per fold.
    y_train : array-like or Series
        Training targets.
    task_type : {"auto", "binary", "multiclass", "regression"}
        Forwarded to RobustModelMaker.  ``"auto"`` infers from the target.
    thresholds : sequence of float, optional
        Threshold grid.  Each value must be in (0, 1).
        Default: ``[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]``.
    weights : dict, optional
        Relative importance weights for the composite score::

            {"score": 1.0, "stability": 1.0, "compression": 1.0}   # default

        Values are normalised internally — ``{2, 2, 2}`` ≡ ``{1, 1, 1}``.
        Increase ``"score"`` to emphasise predictive accuracy,
        ``"stability"`` for fold-to-fold consistency, ``"compression"``
        for the smallest possible feature subset.
    base_params : dict, optional
        Keyword arguments forwarded to every :class:`RobustModelMaker`
        instantiation.  ``stability_threshold`` and ``task_type`` are
        overridden per grid point; all other keys are passed through.

        Defaults (fast exploration — increase ``n_bootstrap``/``n_iter``
        for production)::

            alg="rf", outer_cv=10, inner_cv=5, n_bootstrap=25, n_iter=10,
            cutoff_n_bootstrap=100, random_state=42, n_jobs=-1, verbose=False

    verbose : bool
        Print per-threshold progress lines and a final summary (default True).

    Class methods
    -------------
    from_dataset(dataset, ...)
        Construct from any object with ``.X_train``, ``.y_train``,
        ``.task_type``, and optional ``.robust_params_override``.

    Examples
    --------
    Basic::

        result = ThresholdOptimizer(X_train, y_train, task_type="binary").run()
        result.print_report()
        ROBUST_PARAMS.update(result.best_params())

    Custom weights — prioritise stability::

        result = ThresholdOptimizer(
            X_train, y_train,
            weights={"score": 1.0, "stability": 2.0, "compression": 0.5},
        ).run()

    Fast coarse scan::

        result = ThresholdOptimizer(
            X_train, y_train,
            thresholds=[0.60, 0.70, 0.80, 0.90],
            base_params=dict(n_bootstrap=10, n_iter=5, outer_cv=5, n_jobs=-1),
        ).run()
        result.plot()           # three-panel matplotlib figure
        result.to_dataframe()   # pandas DataFrame for further analysis
    """

    _DEFAULT_WEIGHTS: Dict[str, float] = {
        "score": 1.0, "stability": 1.0, "compression": 1.0
    }

    def __init__(
        self,
        X_train: Any,
        y_train: Any,
        task_type: str = "auto",
        thresholds: Optional[Sequence[float]] = None,
        weights: Optional[Dict[str, float]] = None,
        base_params: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ) -> None:
        self.X_train     = X_train
        self.y_train     = y_train
        self.task_type   = task_type
        self.thresholds  = sorted({float(t) for t in (thresholds or DEFAULT_GRID)})
        self.weights     = {**self._DEFAULT_WEIGHTS, **(weights or {})}
        self.base_params = {**DEFAULT_BASE_PARAMS, **(base_params or {})}
        self.verbose     = verbose

        bad = [t for t in self.thresholds if not (0.0 < t < 1.0)]
        if bad:
            raise ValueError(
                f"All thresholds must be in the open interval (0, 1).  "
                f"Invalid: {bad}"
            )

    # ---- factory from any duck-typed dataset object -------------------------

    @classmethod
    def from_dataset(
        cls,
        dataset: Any,
        thresholds: Optional[Sequence[float]] = None,
        weights: Optional[Dict[str, float]] = None,
        base_params: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ) -> "ThresholdOptimizer":
        """
        Construct from any object exposing ``.X_train``, ``.y_train``,
        ``.task_type``, and (optionally) ``.robust_params_override``.

        ``robust_params_override`` — if present — is merged into
        ``base_params`` so that dataset-specific settings (e.g. a different
        algorithm or CV count) are picked up automatically.

        Parameters
        ----------
        dataset
            Any duck-typed object with the attributes described above.
        thresholds, weights, base_params, verbose
            Forwarded to :class:`ThresholdOptimizer.__init__`.
        """
        bp = dict(base_params or DEFAULT_BASE_PARAMS)
        override = getattr(dataset, "robust_params_override", None)
        if override:
            bp.update(override)
        # Respect dataset-level algorithm if not already set by caller
        ds_alg = getattr(dataset, "alg", None)
        if ds_alg and "alg" not in (base_params or {}):
            bp["alg"] = ds_alg
        return cls(
            X_train     = dataset.X_train,
            y_train     = dataset.y_train,
            task_type   = getattr(dataset, "task_type", "auto"),
            thresholds  = thresholds,
            weights     = weights,
            base_params = bp,
            verbose     = verbose,
        )

    # ---- main entry point ---------------------------------------------------

    def run(self) -> OptimiserResult:
        """
        Evaluate all threshold candidates and return an :class:`OptimiserResult`.

        Each candidate triggers a complete :class:`RobustModelMaker` nested-CV
        fit.  Total wall-clock time ≈ single-fit time × ``len(thresholds)``.
        """
        n = len(self.thresholds)
        width = len(str(n))

        if self.verbose:
            _hdr(f"ThresholdOptimizer  —  {n} threshold(s) to evaluate")
            _kv("Grid",     str(self.thresholds))
            _kv("Weights",  "  ".join(f"{k}={v:.2g}" for k, v in self.weights.items()))
            _kv("Base CV",  f"outer_cv={self.base_params.get('outer_cv')}  "
                            f"inner_cv={self.base_params.get('inner_cv')}  "
                            f"n_bootstrap={self.base_params.get('n_bootstrap')}  "
                            f"n_iter={self.base_params.get('n_iter')}")
            print()

        t0_total = time.perf_counter()
        results: List[ThresholdResult] = []
        detected_task = "unknown"
        metric_name   = "score"

        # Resolve total feature count once before the loop
        X_arr: np.ndarray = (
            self.X_train.values
            if hasattr(self.X_train, "values")
            else np.asarray(self.X_train, dtype=float)
        )
        total_feat = int(X_arr.shape[1])

        for idx, thr in enumerate(self.thresholds, 1):
            if self.verbose:
                print(f"  [{idx:>{width}}/{n}]  threshold={thr:.2f} … ",
                      end="", flush=True)

            t0 = time.perf_counter()

            # Override stability_threshold for this grid point
            params = {k: v for k, v in self.base_params.items()
                      if k != "stability_threshold"}
            params["stability_threshold"] = thr
            params["task_type"]           = self.task_type

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rmm = RobustModelMaker(**params)
                    rmm.fit(self.X_train, self.y_train)
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                if self.verbose:
                    print(f"FAILED  ({type(exc).__name__}: {exc})")
                warnings.warn(
                    f"ThresholdOptimizer: threshold={thr:.2f} raised "
                    f"{type(exc).__name__}: {exc}. Skipping this grid point.",
                    RuntimeWarning, stacklevel=2,
                )
                continue

            pipeline = rmm.result_              # PipelineResult
            ncv      = pipeline.nested_cv_result  # NestedCVResult

            if detected_task == "unknown":
                detected_task = str(ncv.task_type)
                metric_name   = str(ncv.metric_name)

            # Per-fold feature sets → Jaccard stability
            fold_sets: List[Set[str]] = [
                set(arr.tolist()) for arr in ncv.selected_features_per_fold
            ]
            stability   = jaccard_stability(fold_sets)
            n_per_fold  = [len(s) for s in fold_sets]
            mean_n      = float(np.mean(n_per_fold)) if n_per_fold else 0.0
            compression = 1.0 - (mean_n / total_feat) if total_feat > 0 else 0.0

            elapsed = time.perf_counter() - t0

            tr = ThresholdResult(
                threshold            = thr,
                mean_score           = float(ncv.mean_score),
                std_score            = float(ncv.std_score),
                stability            = stability,
                mean_n_features      = mean_n,
                total_features       = total_feat,
                compression          = compression,
                n_features_consensus = int(len(pipeline.selected_features)),
                elapsed              = elapsed,
                fold_scores          = list(ncv.outer_scores),
                fold_feature_sets    = fold_sets,
            )
            results.append(tr)

            if self.verbose:
                is_reg     = "regress" in detected_task.lower()
                sd         = abs(tr.mean_score) if is_reg else tr.mean_score
                ss         = f"{stability:.3f}" if np.isfinite(stability) else "n/a"
                print(f"score={sd:.4f}  stability={ss}  "
                      f"feats={mean_n:.0f}/{total_feat}  ({elapsed:.0f}s)")

        # Guard: at least one threshold must have succeeded
        if not results:
            raise RuntimeError(
                "Every threshold evaluation failed.  "
                "Check your data (NaN columns, constant columns, target dtype) "
                "and base_params (task_type, alg).  "
                "Re-run with verbose=True and inspect the FAILED lines above."
            )

        # Post-processing
        is_regression = "regress" in detected_task.lower()
        _compute_composite(results, self.weights)
        pareto = _find_pareto_front(results)
        # Tiebreaker: higher stability first, then lower threshold (conservative tie)
        best = max(
            results,
            key=lambda r: (r.composite,
                           r.stability if np.isfinite(r.stability) else 0.0)
        )

        total_elapsed = time.perf_counter() - t0_total

        if self.verbose:
            print(f"\n  Sweep complete.  Elapsed: {total_elapsed:.0f}s  |  "
                  f"Recommended threshold: {best.threshold:.2f}  "
                  f"(composite={best.composite:.3f})\n")

        return OptimiserResult(
            results                = results,
            pareto_front           = pareto,
            best                   = best,
            weights                = self.weights,
            is_regression          = is_regression,
            task_type              = detected_task,
            metric_name            = metric_name,
            total_elapsed          = total_elapsed,
            n_thresholds_evaluated = len(results),
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def optimise_threshold(
    X_train: Any,
    y_train: Any,
    task_type: str = "auto",
    thresholds: Optional[Sequence[float]] = None,
    weights: Optional[Dict[str, float]] = None,
    base_params: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> OptimiserResult:
    """One-shot convenience wrapper around :class:`ThresholdOptimizer`.

    Parameters
    ----------
    X_train, y_train
        Training data passed directly to RobustModelMaker.
    task_type
        ``"auto"`` | ``"binary"`` | ``"multiclass"`` | ``"regression"``.
    thresholds
        Threshold grid.  Default: nine points from 0.50 to 0.90.
    weights
        Objective weights.  Default: ``{"score": 1., "stability": 1.,
        "compression": 1.}``.
    base_params
        Extra RobustModelMaker kwargs, merged with :data:`DEFAULT_BASE_PARAMS`.
    verbose
        Print progress (default True).

    Returns
    -------
    OptimiserResult
    """
    return ThresholdOptimizer(
        X_train    = X_train,
        y_train    = y_train,
        task_type  = task_type,
        thresholds = thresholds,
        weights    = weights,
        base_params= base_params,
        verbose    = verbose,
    ).run()


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

_W = 88   # report width


def _hdr(title: str) -> None:
    print("=" * _W)
    print(f"  {title}")
    print("=" * _W)


def _kv(key: str, value: str, kw: int = 30) -> None:
    print(f"  {key:<{kw}}{value}")


def _print_report(opt: OptimiserResult) -> None:
    is_reg       = opt.is_regression
    metric       = opt.metric_name.upper()
    score_arrow  = "RMSE (↓ lower=better)" if is_reg else "AUC (↑ higher=better)"

    print()
    _hdr(f"ThresholdOptimizer Report  —  {opt.task_type}  |  {metric}")
    _kv("Thresholds evaluated", str(opt.n_thresholds_evaluated))
    _kv("Total elapsed",        f"{opt.total_elapsed:.0f}s")
    _kv("Objective weights",
        "  ".join(f"{k}={v:.2g}" for k, v in opt.weights.items()))
    print()

    # ---- main comparison table ----------------------------------------------
    HDR = (f"  {'Thr':>5}  {'Score':>8}  {'±Std':>6}  "
           f"{'Stability':>9}  {'Feats':>8}  {'Removed':>8}  "
           f"{'Composite':>9}  {'':>4}  {'s':>6}")
    SEP = "  " + "─" * (_W - 2)

    print(f"  Score column: {score_arrow}")
    print(SEP)
    print(HDR)
    print(SEP)

    for r in sorted(opt.results, key=lambda x: x.threshold):
        sd    = abs(r.mean_score) if is_reg else r.mean_score
        ss    = f"{r.stability:.3f}" if np.isfinite(r.stability) else "   n/a"
        flags = ("★" if not r.dominated else " ") + ("◆" if r is opt.best else " ")
        print(
            f"  {r.threshold:>5.2f}  {sd:>8.4f}  {r.std_score:>6.4f}  "
            f"{ss:>9}  {r.mean_n_features:>8.1f}  {r.compression:>7.1%}  "
            f"{r.composite:>9.3f}  {flags:>4}  {r.elapsed:>6.0f}"
        )

    print(SEP)
    print("  ★ = Pareto non-dominated   ◆ = recommended (highest composite score)")
    print()

    # ---- Pareto front -------------------------------------------------------
    pf = sorted(opt.pareto_front, key=lambda r: r.threshold)
    print(f"  Pareto-non-dominated front  ({len(pf)} solution(s))")
    print("  " + "─" * 62)
    for r in pf:
        sd  = abs(r.mean_score) if is_reg else r.mean_score
        ss  = f"{r.stability:.3f}" if np.isfinite(r.stability) else "n/a"
        tag = "  ← recommended" if r is opt.best else ""
        print(f"    thr={r.threshold:.2f}  score={sd:.4f}  "
              f"stability={ss}  compression={r.compression:.1%}  "
              f"composite={r.composite:.3f}{tag}")
    print()

    # ---- trade-off narrative ------------------------------------------------
    _narrative(opt)

    # ---- recommendation -----------------------------------------------------
    b  = opt.best
    sd = abs(b.mean_score) if is_reg else b.mean_score
    ss = f"{b.stability:.3f}" if np.isfinite(b.stability) else "n/a"

    print("  RECOMMENDATION")
    print("  " + "═" * 58)
    print(f"    stability_threshold  =  {b.threshold:.2f}")
    print(f"    Score       {sd:.4f} ± {b.std_score:.4f}  ({metric})")
    print(f"    Stability   {ss}  (mean pairwise Jaccard)")
    print(f"    Features    {b.mean_n_features:.0f} / {b.total_features}  "
          f"({b.compression:.1%} removed,  {b.retention:.1%} retained)")
    print(f"    Composite   {b.composite:.3f}  "
          f"(weights: {', '.join(f'{k}={v:.2g}' for k, v in opt.weights.items())})")
    print()
    print(f"    To apply:  ROBUST_PARAMS['stability_threshold'] = {b.threshold:.2f}")
    print("  " + "═" * 58)
    print()


def _narrative(opt: OptimiserResult) -> None:
    """Print a brief automated summary of how metrics move with the threshold."""
    if len(opt.results) < 2:
        return
    by_thr  = sorted(opt.results, key=lambda r: r.threshold)
    lo, hi  = by_thr[0], by_thr[-1]
    is_reg  = opt.is_regression

    # Score direction  (higher mean_score is always better, even for neg-RMSE)
    if hi.mean_score > lo.mean_score:
        score_dir = "improved"
    elif hi.mean_score < lo.mean_score:
        score_dir = "degraded"
    else:
        score_dir = "unchanged"

    sd_lo = abs(lo.mean_score) if is_reg else lo.mean_score
    sd_hi = abs(hi.mean_score) if is_reg else hi.mean_score
    sc_label = f"RMSE {sd_lo:.4f}→{sd_hi:.4f}" if is_reg else f"AUC {sd_lo:.4f}→{sd_hi:.4f}"

    stab_lo = lo.stability if np.isfinite(lo.stability) else None
    stab_hi = hi.stability if np.isfinite(hi.stability) else None
    if stab_lo is not None and stab_hi is not None:
        delta_s = stab_hi - stab_lo
        stab_dir = "↑" if delta_s > 0.01 else ("↓" if delta_s < -0.01 else "~")
        stab_line = f"{stab_lo:.3f}→{stab_hi:.3f}  {stab_dir}"
    else:
        stab_line = "n/a"

    compr_lo, compr_hi = lo.compression, hi.compression

    print("  Trade-off: lowest → highest threshold in grid")
    print("  " + "─" * 62)
    print(f"    Score       {sc_label}  ({score_dir})")
    print(f"    Stability   {stab_line}")
    print(f"    Compression {compr_lo:.1%}→{compr_hi:.1%}  "
          f"(+{compr_hi - compr_lo:.1%} more features removed at high threshold)")
    print()


# ---------------------------------------------------------------------------
# Optional matplotlib visualisation
# ---------------------------------------------------------------------------

def _plot(opt: OptimiserResult, figsize: Tuple[float, float] = (12, 4)) -> Any:
    """Three-panel figure: score / stability / compression vs threshold."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
    except ImportError:
        warnings.warn(
            "matplotlib is not installed — install with: pip install matplotlib",
            RuntimeWarning, stacklevel=3,
        )
        return None

    by_thr  = sorted(opt.results, key=lambda r: r.threshold)
    thrs    = [r.threshold           for r in by_thr]
    scores  = [r.display_score       for r in by_thr]
    stds    = [r.std_score           for r in by_thr]
    stabs   = [r.stability if np.isfinite(r.stability) else np.nan for r in by_thr]
    comprs  = [r.compression * 100   for r in by_thr]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(
        f"ThresholdOptimizer — {opt.task_type}  ({opt.metric_name.upper()})",
        fontsize=11, y=1.02,
    )

    best_thr = opt.best.threshold

    # Panel 1 — Score ± std
    ax = axes[0]
    lo_band = [s - e for s, e in zip(scores, stds)]
    hi_band = [s + e for s, e in zip(scores, stds)]
    ax.fill_between(thrs, lo_band, hi_band, alpha=0.20, label="±1 std")
    ax.plot(thrs, scores, "o-", linewidth=1.8, markersize=5)
    ax.axvline(best_thr, color="red", linestyle="--", linewidth=1.2,
               label=f"Best ({best_thr:.2f})")
    score_lbl = "RMSE (↓)" if opt.is_regression else "AUC (↑)"
    ax.set_xlabel("stability_threshold")
    ax.set_ylabel(score_lbl)
    ax.set_title("Predictive Score")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2 — Jaccard stability
    ax = axes[1]
    ax.plot(thrs, stabs, "s-", color="tab:green", linewidth=1.8, markersize=5)
    ax.axvline(best_thr, color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("stability_threshold")
    ax.set_ylabel("Jaccard stability (↑)")
    ax.set_title("Selection Stability")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Panel 3 — Compression %
    ax = axes[2]
    ax.plot(thrs, comprs, "^-", color="tab:orange", linewidth=1.8, markersize=5)
    ax.axvline(best_thr, color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("stability_threshold")
    ax.set_ylabel("Features removed (%)")
    ax.set_title("Compression")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI smoke test  (python tools/threshold_optimizer.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Smoke-test ThresholdOptimizer on synthetic data."
    )
    p.add_argument("--n-samples",    type=int,   default=300)
    p.add_argument("--n-features",   type=int,   default=30)
    p.add_argument("--n-informative",type=int,   default=8)
    p.add_argument("--task",         default="binary",
                   choices=["binary", "multiclass", "regression"])
    p.add_argument("--thresholds",   nargs="+",  type=float,
                   default=[0.50, 0.60, 0.70, 0.80])
    p.add_argument("--n-bootstrap",  type=int,   default=10)
    p.add_argument("--outer-cv",     type=int,   default=5)
    args = p.parse_args()

    print(f"\nSynthetic {args.task}  |  "
          f"{args.n_samples} samples × {args.n_features} features  "
          f"({args.n_informative} informative)\n")

    from sklearn.datasets import make_classification, make_regression

    rng = np.random.default_rng(42)
    if args.task == "regression":
        X_raw, y_raw = make_regression(
            n_samples=args.n_samples, n_features=args.n_features,
            n_informative=args.n_informative, noise=0.2, random_state=42,
        )
    elif args.task == "multiclass":
        X_raw, y_raw = make_classification(
            n_samples=args.n_samples, n_features=args.n_features,
            n_informative=args.n_informative, n_classes=3,
            n_clusters_per_class=1, random_state=42,
        )
    else:
        X_raw, y_raw = make_classification(
            n_samples=args.n_samples, n_features=args.n_features,
            n_informative=args.n_informative, random_state=42,
        )

    X_df = pd.DataFrame(X_raw, columns=[f"f{i}" for i in range(args.n_features)])
    y_s  = pd.Series(y_raw, name="target")

    result = ThresholdOptimizer(
        X_train     = X_df,
        y_train     = y_s,
        task_type   = args.task,
        thresholds  = args.thresholds,
        base_params = dict(
            outer_cv=args.outer_cv, inner_cv=3,
            n_bootstrap=args.n_bootstrap, n_iter=5,
            cutoff_n_bootstrap=50, random_state=42,
            n_jobs=1, verbose=False,
        ),
    ).run()

    result.print_report()

    print("DataFrame export (top rows by composite score):")
    print(result.to_dataframe().to_string(index=False))
    print()
    print(f"Apply:  ROBUST_PARAMS['stability_threshold'] = {result.best.threshold:.2f}")

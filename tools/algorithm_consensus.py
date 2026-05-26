"""
algorithm_consensus.py
======================
Multi-algorithm meta-consensus feature selection for RobustModelMaker.

Runs RobustModelMaker independently with several algorithms, then identifies
features that are consistently selected across all (or most) of them.

The motivation
--------------
Any single algorithm's bootstrap stability selection reflects its own inductive
bias.  A Random Forest latches onto non-linear interactions via MDI importance;
ElasticNet finds linear additive effects via |coefficient| magnitude; Lasso
aggressively zeroes out weak linear predictors.  A feature that survives *all*
of these fundamentally different selection criteria is extremely unlikely to be
an artefact of any one model family's assumptions — it is the closest thing to
algorithm-agnostic signal the data contains.

Output
------
The central result is a **feature table** with one row per feature selected by
at least one algorithm:

  feature         — feature name
  n_selected_by   — how many algorithms chose it
  coverage        — n_selected_by / n_algorithms  (0.0 – 1.0)
  mean_freq       — mean bootstrap selection frequency across selecting algorithms
  freq_{alg}      — per-algorithm bootstrap selection frequency (NaN = not selected)
  in_consensus    — True when coverage >= min_agreement

Features in the consensus set appear at the top, sorted by coverage then
mean_freq.  The full table also drives the heatmap visualisation.

Quick start
-----------
    from algorithm_consensus import AlgorithmConsensus

    result = AlgorithmConsensus(
        algorithms  = ['eln', 'rdg', 'las', 'rf'],
        base_params = dict(
            outer_cv=10, inner_cv=5, n_bootstrap=25, n_iter=10,
            stability_threshold=0.75, random_state=42, n_jobs=1, verbose=False,
        ),
        min_agreement = 1.0,   # all four algorithms must agree
    ).fit(X_train, y_train, task_type='regression')

    result.print_report()
    result.plot_heatmap()
    result.plot_agreement()

    print('Consensus features:', result.consensus_features)

With your own data
------------------
    from sklearn.model_selection import train_test_split
    from algorithm_consensus import AlgorithmConsensus

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    result = AlgorithmConsensus().fit(X_train, y_train, task_type='regression')
    result.print_report()
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Locate and import RobustModelMaker
# ---------------------------------------------------------------------------

def _import_robust_module() -> Any:
    """Find RobustModelMaker.py and return its module object."""
    _KEY = "RobustModelMaker"
    if _KEY in sys.modules:
        return sys.modules[_KEY]

    here = Path(__file__).resolve().parent
    candidates: List[Path] = []
    env = os.environ.get("ROBUST_MODEL_MAKER_PATH")
    if env:
        candidates.append(Path(env))
    candidates += [
        here            / f"{_KEY}.py",
        here.parent     / f"{_KEY}.py",
        here.parent.parent / f"{_KEY}.py",
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
            "Cannot locate RobustModelMaker.py.  Place algorithm_consensus.py "
            "in the same directory, or set ROBUST_MODEL_MAKER_PATH."
        )


_rm_module       = _import_robust_module()
RobustModelMaker = _rm_module.RobustModelMaker


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

#: Human-readable descriptions for each algorithm code.
ALGORITHM_LABELS: Dict[str, str] = {
    "eln": "ElasticNet (linear, L1+L2)",
    "rdg": "Ridge (linear, L2)",
    "las": "Lasso (linear, L1)",
    "lin": "OLS Linear Regression",
    "log": "Logistic Regression",
    "rf":  "Random Forest (tree, MDI)",
    "xgb": "XGBoost (boosted trees)",
    "svm": "SVM / LinearSVR",
    "mlp": "MLP Neural Network",
}

#: Sensible default cross-sections of model families, by task type.
#: Chosen to maximise diversity of inductive biases while remaining fast.
DEFAULT_ALGORITHMS: Dict[str, List[str]] = {
    "regression":   ["eln", "rdg", "las", "rf"],
    "binary":       ["eln", "rdg", "rf"],
    "multiclass":   ["eln", "rdg", "rf"],
}

#: Default RobustModelMaker params shared across all algorithm runs.
#: `alg` and `task_type` are set per-run and must NOT be included here.
DEFAULT_BASE_PARAMS: Dict[str, Any] = dict(
    outer_cv           = 10,
    inner_cv           = 5,
    n_bootstrap        = 25,
    n_iter             = 10,
    stability_threshold= 0.75,
    cutoff_n_bootstrap = 100,
    random_state       = 42,
    n_jobs             = -1,
    verbose            = False,
)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class SingleAlgorithmResult:
    """Stores everything extracted from one RobustModelMaker fit."""

    alg:               str
    label:             str              # human-readable description
    selected_features: List[str]        # consensus selected features
    n_selected:        int
    freq_map:          Dict[str, float] # feature -> bootstrap selection frequency
    elapsed:           float            # wall-clock seconds
    pipeline:          Any = field(repr=False)   # PipelineResult (full object)


@dataclass
class ConsensusResult:
    """Full result from :meth:`AlgorithmConsensus.fit`."""

    algorithms:        List[str]                            # successful algorithms
    n_algorithms:      int
    algorithm_results: Dict[str, SingleAlgorithmResult]    # alg -> result
    feature_table:     pd.DataFrame                        # see module docstring
    min_agreement:     float
    total_features:    int
    task_type:         str
    total_elapsed:     float

    # ---- derived properties -------------------------------------------------

    @property
    def consensus_features(self) -> List[str]:
        """Features selected by >= ``min_agreement`` fraction of algorithms."""
        if self.feature_table.empty:
            return []
        return self.feature_table.loc[
            self.feature_table["in_consensus"], "feature"
        ].tolist()

    @property
    def n_consensus_features(self) -> int:
        """Number of features in the consensus set."""
        return len(self.consensus_features)

    @property
    def agreement_counts(self) -> pd.Series:
        """Number of features selected by exactly k algorithms (k = 1..n)."""
        if self.feature_table.empty:
            return pd.Series(dtype=int)
        return (
            self.feature_table["n_selected_by"]
            .value_counts()
            .reindex(range(1, self.n_algorithms + 1), fill_value=0)
            .sort_index()
        )

    # ---- public methods -----------------------------------------------------

    def print_report(self) -> None:
        """Print the full formatted consensus report to stdout."""
        _print_report(self)

    def plot_heatmap(
        self,
        max_features: int = 50,
        figsize: Tuple[float, float] = (11, 7),
    ) -> Any:
        """
        Heatmap of bootstrap selection frequency: features × algorithms.

        Rows are features selected by at least one algorithm, sorted by coverage
        (most agreed-upon first).  Colour = per-algorithm bootstrap selection
        frequency; white = not selected.  Requires matplotlib + seaborn (or
        matplotlib alone for a simplified version).

        Parameters
        ----------
        max_features : int
            Show at most this many rows (top by coverage, then mean_freq).
        figsize : tuple
            Passed to ``plt.subplots``.
        """
        return _plot_heatmap(self, max_features=max_features, figsize=figsize)

    def plot_agreement(
        self,
        figsize: Tuple[float, float] = (8, 4),
    ) -> Any:
        """
        Bar chart: number of features selected by exactly k algorithms.

        Bars for k = ``n_algorithms`` (full consensus) are highlighted.
        """
        return _plot_agreement(self, figsize=figsize)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the feature table."""
        return self.feature_table.copy()


# ---------------------------------------------------------------------------
# Feature table construction
# ---------------------------------------------------------------------------

def _build_feature_table(
    algo_results: Dict[str, SingleAlgorithmResult],
    algorithms: List[str],
    min_agreement: float,
) -> pd.DataFrame:
    """Build the per-feature agreement table from individual algorithm results."""
    n_algs = len(algorithms)

    # Union: features selected by at least one algorithm
    all_selected: Set[str] = set()
    for ar in algo_results.values():
        all_selected.update(ar.selected_features)

    if not all_selected:
        return pd.DataFrame()

    rows = []
    for feat in sorted(all_selected):
        row: Dict[str, Any] = {"feature": feat}
        n_sel = 0
        freqs_of_selecting: List[float] = []

        for alg in algorithms:
            ar = algo_results.get(alg)
            if ar is None:
                row[f"freq_{alg}"] = np.nan
                continue
            raw_freq = ar.freq_map.get(feat, np.nan)
            row[f"freq_{alg}"] = raw_freq
            if feat in set(ar.selected_features):
                n_sel += 1
                if np.isfinite(raw_freq):
                    freqs_of_selecting.append(raw_freq)

        row["n_selected_by"] = n_sel
        row["coverage"]      = n_sel / n_algs if n_algs > 0 else 0.0
        row["mean_freq"]     = (float(np.mean(freqs_of_selecting))
                                if freqs_of_selecting else np.nan)
        row["in_consensus"]  = row["coverage"] >= min_agreement
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["in_consensus", "coverage", "mean_freq"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class AlgorithmConsensus:
    """
    Meta-consensus feature selection: run multiple algorithms, keep what all agree on.

    Each algorithm is run as a complete RobustModelMaker nested-CV fit with
    identical structural parameters (outer/inner CV, bootstrap count, threshold).
    The only thing that changes is the underlying model family — and therefore
    the feature importance criterion used during bootstrap stability selection:

    * ``"rf"``  — Random Forest MDI (captures non-linear interactions)
    * ``"eln"`` — ElasticNet |coefficient| (linear, L1+L2, handles collinearity)
    * ``"rdg"`` — Ridge |coefficient| (linear, L2, soft shrinkage)
    * ``"las"`` — Lasso |coefficient| (linear, L1, aggressive sparsity)
    * ``"xgb"`` — XGBoost gain (boosted trees; requires ``pip install xgboost``)

    Features surviving across all these criteria are the most robust available.

    Parameters
    ----------
    algorithms : list of str, optional
        Algorithm codes to run.  Default (task-dependent):
        ``["eln", "rdg", "las", "rf"]`` for regression,
        ``["eln", "rdg", "rf"]`` for classification.
    base_params : dict, optional
        Shared RobustModelMaker kwargs.  Do **not** include ``"alg"`` or
        ``"task_type"`` — those are set per-run.  Defaults::

            outer_cv=10, inner_cv=5, n_bootstrap=25, n_iter=10,
            stability_threshold=0.75, cutoff_n_bootstrap=100,
            random_state=42, n_jobs=-1, verbose=False

    min_agreement : float
        Fraction of algorithms that must select a feature for it to be in the
        consensus set.  Must be in (0, 1].  Default: ``1.0`` (all algorithms).
        Use ``0.75`` for "3 out of 4 must agree", etc.
    verbose : bool
        Print per-algorithm progress (default: True).

    Class methods
    -------------
    from_dataset(dataset, ...)
        Construct from any duck-typed dataset object with ``.X_train``,
        ``.y_train``, and ``.task_type`` attributes.
        Dataset-level ``robust_params_override`` (if present) is merged automatically.

    Examples
    --------
    Basic::

        result = AlgorithmConsensus(
            algorithms=['eln', 'rdg', 'las', 'rf'],
        ).fit(X_train, y_train, task_type='regression')
        result.print_report()
        result.plot_heatmap()
        print(result.consensus_features)

    Majority vote (3 of 4)::

        result = AlgorithmConsensus(min_agreement=0.75).fit(X, y, 'regression')

    From a duck-typed dataset object::

        result = AlgorithmConsensus.from_dataset(ds).fit(ds.X_train, ds.y_train)
    """

    def __init__(
        self,
        algorithms:    Optional[List[str]] = None,
        base_params:   Optional[Dict[str, Any]] = None,
        min_agreement: float = 1.0,
        verbose:       bool = True,
    ) -> None:
        self.algorithms    = algorithms
        self.base_params   = {k: v for k, v in
                              {**DEFAULT_BASE_PARAMS, **(base_params or {})}.items()
                              if k not in ("alg", "task_type")}
        self.min_agreement = float(min_agreement)
        self.verbose       = verbose

        if not (0.0 < self.min_agreement <= 1.0):
            raise ValueError(
                f"min_agreement must be in (0, 1].  Got: {min_agreement}"
            )

    # ---- factory from any duck-typed dataset object -------------------------

    @classmethod
    def from_dataset(
        cls,
        dataset:       Any,
        algorithms:    Optional[List[str]] = None,
        base_params:   Optional[Dict[str, Any]] = None,
        min_agreement: float = 1.0,
        verbose:       bool = True,
    ) -> "AlgorithmConsensus":
        """
        Construct from any object with ``.X_train``, ``.y_train``, ``.task_type``.

        ``robust_params_override`` (if present) is merged into ``base_params``,
        then ``"alg"`` and ``"task_type"`` keys are stripped — both are managed
        per-run by :meth:`fit`.
        """
        bp = {**DEFAULT_BASE_PARAMS, **(base_params or {})}
        override = getattr(dataset, "robust_params_override", None)
        if override:
            bp.update(override)
        # Strip keys that are set per-run
        for k in ("alg", "task_type"):
            bp.pop(k, None)

        # Default algorithms from dataset task type if not specified
        if algorithms is None:
            task_type = getattr(dataset, "task_type", "auto")
            algorithms = list(DEFAULT_ALGORITHMS.get(task_type,
                               DEFAULT_ALGORITHMS["regression"]))

        return cls(
            algorithms    = algorithms,
            base_params   = bp,
            min_agreement = min_agreement,
            verbose       = verbose,
        )

    # ---- main entry point ---------------------------------------------------

    def fit(
        self,
        X: Any,
        y: Any,
        task_type: str = "auto",
    ) -> ConsensusResult:
        """
        Run all algorithms and return a :class:`ConsensusResult`.

        Parameters
        ----------
        X : array-like or DataFrame
            Training features.  Passed directly to RobustModelMaker.
        y : array-like or Series
            Training targets.
        task_type : {"auto", "binary", "multiclass", "regression"}
            Forwarded to every RobustModelMaker instantiation.

        Returns
        -------
        ConsensusResult
        """
        # Resolve algorithm list
        algs = self._resolve_algorithms(task_type, y)
        n    = len(algs)

        # ── Pre-filter: drop columns with too few non-NaN values ─────────────
        # Two failure modes handled here:
        #
        # 1. All-NaN in the full X: RobustModelMaker._validate_inputs raises
        #    ValueError immediately.  Can occur when structured NaN values cluster
        #    in specific rows that all land in one side of a train/test split.
        #
        # 2. Near-all-NaN in X: the column is not all-NaN in the full training
        #    set, but bootstrap samples within RobustModelMaker's outer CV folds
        #    sometimes draw zero non-NaN values for the column.  RobustModelMaker
        #    drops the column inside that bootstrap iteration, producing importance
        #    arrays of different lengths across iterations, which then fail when
        #    concatenated (shape mismatch ValueError).
        #
        # Threshold: a column needs at least _min_nonnan non-NaN values in X so
        # that P(a bootstrap draw of fold-size samples is all-NaN) < ~0.1%.
        # Heuristic (outer_cv * 2 + 1): each fold training set has about
        # (outer_cv-1)/outer_cv * N samples; we want k >= 10 non-NaN per fold.
        _outer_cv    = self.base_params.get("outer_cv",    DEFAULT_BASE_PARAMS["outer_cv"])
        _n_bootstrap = self.base_params.get("n_bootstrap", DEFAULT_BASE_PARAMS["n_bootstrap"])
        _min_nonnan  = max(10, int(_outer_cv) * 2 + 1)

        _dropped_cols: List[str] = []
        if hasattr(X, "columns"):          # pandas DataFrame
            _nonnan_counts = X.notna().sum(axis=0)
            _drop_mask     = _nonnan_counts < _min_nonnan
            if _drop_mask.any():
                _dropped_cols = list(X.columns[_drop_mask])
                X = X.loc[:, ~_drop_mask]
        elif hasattr(X, "shape"):          # numpy array
            _nonnan_counts = (~np.isnan(X)).sum(axis=0)
            _drop_mask     = _nonnan_counts < _min_nonnan
            if _drop_mask.any():
                _dropped_cols = [f"col_{i}" for i in np.where(_drop_mask)[0]]
                X = X[:, ~_drop_mask]

        if _dropped_cols:
            _preview = _dropped_cols[:5]
            _extra   = len(_dropped_cols) - 5
            _suffix  = f" … and {_extra} more" if _extra > 0 else ""
            warnings.warn(
                f"AlgorithmConsensus: dropped {len(_dropped_cols)} column(s) with "
                f"fewer than {_min_nonnan} non-NaN values in the training set — "
                f"these would cause all-NaN bootstrap draws or shape mismatches "
                f"inside RobustModelMaker: {_dropped_cols}",
                UserWarning, stacklevel=2,
            )
            if self.verbose:
                print(
                    f"  [pre-filter] dropped {len(_dropped_cols)} sparse column(s) "
                    f"(< {_min_nonnan} non-NaN): {_preview}{_suffix}",
                    flush=True,
                )

        # Total features
        total_features = (X.shape[1] if hasattr(X, "shape")
                          else len(X.columns) if hasattr(X, "columns")
                          else 0)

        # Minimum number of algorithms that must agree (for header display)
        min_count = int(np.ceil(self.min_agreement * n))

        if self.verbose:
            _hdr(f"AlgorithmConsensus  —  {n} algorithm(s), "
                 f"min_agreement={self.min_agreement:.0%}")
            _kv("Algorithms",    "  |  ".join(
                f"{a} ({ALGORITHM_LABELS.get(a, a)})" for a in algs))
            _kv("min_agreement", f"{self.min_agreement:.2f}  "
                f"(feature must be selected by ≥ {min_count}/{n} algorithms)")
            _kv("threshold",     str(self.base_params.get("stability_threshold")))
            _kv("outer_cv / inner_cv",
                f"{self.base_params.get('outer_cv')} / "
                f"{self.base_params.get('inner_cv')}")
            print()

        t0_total = time.perf_counter()
        algo_results: Dict[str, SingleAlgorithmResult] = {}
        detected_task = "unknown"

        for idx, alg in enumerate(algs, 1):
            label = ALGORITHM_LABELS.get(alg, alg)
            if self.verbose:
                print(f"  [{idx}/{n}]  {alg:<4}  {label:<30} … ",
                      end="", flush=True)

            t0     = time.perf_counter()
            params = dict(self.base_params)
            params["alg"]       = alg
            params["task_type"] = task_type

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    rmm = RobustModelMaker(**params)
                    rmm.fit(X, y)
                    pipeline = rmm.result_
                except Exception as exc:
                    if self.verbose:
                        print(f"FAILED  ({type(exc).__name__}: {exc})")
                    warnings.warn(
                        f"Algorithm '{alg}' raised {type(exc).__name__}: {exc}",
                        RuntimeWarning, stacklevel=2,
                    )
                    continue

            if detected_task == "unknown":
                detected_task = str(pipeline.nested_cv_result.task_type)

            # Bootstrap selection frequencies for every feature
            stab_df  = pipeline.stability_result.summary()
            freq_map = {str(f): float(freq)
                        for f, freq in zip(stab_df["feature"],
                                           stab_df["selection_frequency"])}
            selected = [str(f) for f in pipeline.selected_features]
            elapsed  = time.perf_counter() - t0

            algo_results[alg] = SingleAlgorithmResult(
                alg               = alg,
                label             = label,
                selected_features = selected,
                n_selected        = len(selected),
                freq_map          = freq_map,
                elapsed           = elapsed,
                pipeline          = pipeline,
            )

            if self.verbose:
                print(f"{len(selected):>4} features selected  ({elapsed:.0f}s)")

        if not algo_results:
            raise RuntimeError(
                "Every algorithm failed.  Check base_params and task_type."
            )

        successful = list(algo_results.keys())
        feature_table = _build_feature_table(
            algo_results, successful, self.min_agreement
        )
        total_elapsed = time.perf_counter() - t0_total
        n_con = int(feature_table["in_consensus"].sum()) if not feature_table.empty else 0

        if self.verbose:
            print(f"\n  Complete.  Total: {total_elapsed:.0f}s  |  "
                  f"Consensus features: {n_con} / {total_features}\n")

        return ConsensusResult(
            algorithms        = successful,
            n_algorithms      = len(successful),
            algorithm_results = algo_results,
            feature_table     = feature_table,
            min_agreement     = self.min_agreement,
            total_features    = total_features,
            task_type         = detected_task,
            total_elapsed     = total_elapsed,
        )

    # ---- internal -----------------------------------------------------------

    def _resolve_algorithms(self, task_type: str, y: Any) -> List[str]:
        if self.algorithms is not None:
            return list(self.algorithms)
        # Infer task type from y when task_type="auto"
        if task_type == "auto":
            y_arr    = np.asarray(y)
            n_unique = len(np.unique(y_arr))
            if y_arr.dtype.kind == "f" or n_unique > 20:
                task_type = "regression"
            elif n_unique == 2:
                task_type = "binary"
            else:
                task_type = "multiclass"
        return list(DEFAULT_ALGORITHMS.get(task_type,
                    DEFAULT_ALGORITHMS["regression"]))


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def run_algorithm_consensus(
    X: Any,
    y: Any,
    task_type: str = "auto",
    algorithms: Optional[List[str]] = None,
    base_params: Optional[Dict[str, Any]] = None,
    min_agreement: float = 1.0,
    verbose: bool = True,
) -> ConsensusResult:
    """One-shot wrapper around :class:`AlgorithmConsensus`.

    Parameters
    ----------
    X, y
        Training data.
    task_type
        ``"auto"`` | ``"binary"`` | ``"multiclass"`` | ``"regression"``.
    algorithms
        Algorithm codes.  Default is task-dependent (see :data:`DEFAULT_ALGORITHMS`).
    base_params
        Shared RobustModelMaker kwargs (merged with :data:`DEFAULT_BASE_PARAMS`).
    min_agreement
        Fraction of algorithms that must select a feature.  Default: 1.0 (all).
    verbose
        Print progress (default: True).
    """
    return AlgorithmConsensus(
        algorithms    = algorithms,
        base_params   = base_params,
        min_agreement = min_agreement,
        verbose       = verbose,
    ).fit(X, y, task_type=task_type)


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

_W = 90


def _hdr(title: str) -> None:
    print("=" * _W)
    print(f"  {title}")
    print("=" * _W)


def _kv(key: str, value: str, kw: int = 28) -> None:
    print(f"  {key:<{kw}}{value}")


def _print_report(res: ConsensusResult) -> None:
    print()
    _hdr(f"AlgorithmConsensus Report  —  {res.task_type}")
    _kv("Algorithms run",      str(res.algorithms))
    _kv("Total elapsed",       f"{res.total_elapsed:.0f}s")
    _kv("min_agreement",       f"{res.min_agreement:.2f}  "
        f"(≥ {int(np.ceil(res.min_agreement * res.n_algorithms))}"
        f"/{res.n_algorithms} algorithms must agree)")
    print()

    # ---- Per-algorithm summary ---------------------------------------------
    SEP = "  " + "─" * (_W - 2)
    print("  Per-algorithm results")
    print(SEP)
    print(f"  {'Alg':<5}  {'Description':<32}  {'Selected':>8}  "
          f"{'Reduction':>10}  {'Time(s)':>7}")
    print(SEP)
    for alg in res.algorithms:
        ar  = res.algorithm_results[alg]
        red = (1.0 - ar.n_selected / res.total_features) if res.total_features > 0 else 0.0
        print(f"  {alg:<5}  {ar.label:<32}  {ar.n_selected:>8}  "
              f"{red:>9.1%}  {ar.elapsed:>7.0f}")
    print(SEP)
    print()

    # ---- Agreement distribution --------------------------------------------
    counts = res.agreement_counts
    print("  Agreement distribution  (features selected by k algorithms)")
    print(SEP)
    for k, cnt in counts.items():
        bar  = "█" * min(cnt, 50)
        mark = "  ← consensus" if k == res.n_algorithms else ""
        print(f"  k={k}  {cnt:>4}  {bar}{mark}")
    print(SEP)
    print()

    # ---- Consensus features ------------------------------------------------
    consensus_df = res.feature_table[res.feature_table["in_consensus"]]
    freq_cols    = [c for c in res.feature_table.columns if c.startswith("freq_")]

    n_con = len(consensus_df)
    red_con = (1.0 - n_con / res.total_features) * 100 if res.total_features > 0 else 0.0
    print(f"  Consensus feature set  ({n_con} features, "
          f"{red_con:.1f}% of {res.total_features} removed)")
    print(SEP)

    if n_con == 0:
        print("  (no features met the agreement threshold)")
        print()
    else:
        # Header
        alg_cols = "  ".join(f"{c.replace('freq_', ''):>7}" for c in freq_cols)
        print(f"  {'Feature':<24}  {'Coverage':>8}  {'MeanFreq':>8}  {alg_cols}")
        print(f"  {'':24}  {'':>8}  {'':>8}  " +
              "  ".join(f"{'(' + c.replace('freq_', '') + ')':>7}" for c in freq_cols))
        print(SEP)

        for _, row in consensus_df.iterrows():
            freq_str = "  ".join(
                f"{row[c]:>7.3f}" if np.isfinite(row[c]) else f"{'—':>7}"
                for c in freq_cols
            )
            print(f"  {str(row['feature']):<24}  {row['coverage']:>8.2f}  "
                  f"{row['mean_freq']:>8.3f}  {freq_str}")

        print(SEP)
        print(f"  MeanFreq = mean bootstrap selection frequency across selecting algorithms")
        print()

    # ---- Interpretation note -----------------------------------------------
    print("  INTERPRETATION")
    print("  " + "═" * 56)
    print(f"    {n_con} feature(s) selected by all {res.n_algorithms} algorithms.")
    print(f"    These are the most algorithmically-robust signals in the data —")
    print(f"    features whose importance survives across fundamentally different")
    print(f"    model families and importance criteria.")
    if n_con == 0:
        print()
        print(f"    Tip: lower min_agreement (e.g. 0.75) to include features")
        print(f"    agreed on by most but not all algorithms.")
    print("  " + "═" * 56)
    print()


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def _plot_heatmap(
    res: ConsensusResult,
    max_features: int = 50,
    figsize: Tuple[float, float] = (11, 7),
) -> Any:
    """Heatmap: bootstrap selection frequency, features × algorithms."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import matplotlib.patches as mpatches
    except ImportError:
        warnings.warn("matplotlib not installed — install with: pip install matplotlib",
                      RuntimeWarning, stacklevel=2)
        return None

    ft   = res.feature_table
    if ft.empty:
        warnings.warn("No features to plot.", RuntimeWarning, stacklevel=2)
        return None

    # Limit rows
    ft_plot = ft.head(max_features)
    freq_cols = [c for c in ft.columns if c.startswith("freq_")]
    alg_labels = [ALGORITHM_LABELS.get(c.replace("freq_", ""), c.replace("freq_", ""))
                  for c in freq_cols]

    # Matrix: rows = features, cols = algorithms
    mat = ft_plot[freq_cols].to_numpy(dtype=float)

    # Colourmap: white (NaN / not selected) → blue (low freq) → dark blue (high freq)
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color="#f0f0f0")   # NaN → light grey
    mat_masked = np.ma.masked_invalid(mat)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat_masked, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                   interpolation="nearest")

    # Axes labels
    ax.set_xticks(range(len(freq_cols)))
    ax.set_xticklabels(alg_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(ft_plot)))
    ax.set_yticklabels(ft_plot["feature"], fontsize=7.5)
    ax.set_xlabel("Algorithm", fontsize=10)
    ax.set_ylabel("Feature", fontsize=10)

    n_con = int(res.feature_table["in_consensus"].sum())
    title_suffix = f"  (top {len(ft_plot)} of {len(ft)} selected)" if len(ft) > max_features else ""
    ax.set_title(
        f"Algorithm Consensus — selection frequency heatmap{title_suffix}\n"
        f"min_agreement={res.min_agreement:.0%}  |  "
        f"{n_con} consensus features  |  "
        f"{res.n_algorithms} algorithms  |  "
        f"{res.total_features} total features",
        fontsize=10,
    )

    # Colourbar
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Bootstrap selection frequency", fontsize=9)

    # Highlight consensus rows with a left-side bar
    consensus_mask = ft_plot["in_consensus"].values
    for row_idx, is_con in enumerate(consensus_mask):
        if is_con:
            ax.add_patch(mpatches.FancyBboxPatch(
                (-0.5, row_idx - 0.5), len(freq_cols), 1,
                linewidth=1.8, edgecolor="#d62728", facecolor="none",
                boxstyle="square,pad=0",
            ))

    # Legend
    con_patch = mpatches.Patch(facecolor="none", edgecolor="#d62728",
                               linewidth=1.8, label="In consensus set")
    na_patch  = mpatches.Patch(facecolor="#f0f0f0", edgecolor="grey",
                               linewidth=0.8, label="Not selected by this algorithm")
    ax.legend(handles=[con_patch, na_patch], loc="lower right",
              fontsize=8, framealpha=0.9)

    fig.tight_layout()
    return fig


def _plot_agreement(
    res: ConsensusResult,
    figsize: Tuple[float, float] = (8, 4),
) -> Any:
    """Bar chart: number of features selected by exactly k algorithms."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not installed.", RuntimeWarning, stacklevel=2)
        return None

    counts = res.agreement_counts
    if counts.empty:
        warnings.warn("No agreement data to plot.", RuntimeWarning, stacklevel=2)
        return None

    ks     = list(counts.index)
    vals   = list(counts.values)
    colors = ["#d62728" if k == res.n_algorithms else "#4878d0" for k in ks]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(ks, vals, color=colors, edgecolor="white", linewidth=0.8)

    # Count labels on top of each bar
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(val), ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(ks)
    ax.set_xticklabels(
        [f"k={k}\n({'all' if k == res.n_algorithms else str(k)})" for k in ks],
        fontsize=9,
    )
    ax.set_xlabel("Number of algorithms agreeing", fontsize=10)
    ax.set_ylabel("Number of features", fontsize=10)
    ax.set_title(
        f"Algorithm agreement distribution  "
        f"(min_agreement={res.min_agreement:.0%}, "
        f"n_consensus={res.n_consensus_features})",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.3)

    import matplotlib.patches as mpatches
    con_patch = mpatches.Patch(color="#d62728", label=f"Consensus (k={res.n_algorithms})")
    oth_patch  = mpatches.Patch(color="#4878d0", label="Partial agreement")
    ax.legend(handles=[con_patch, oth_patch], fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Smoke-test AlgorithmConsensus on synthetic data."
    )
    p.add_argument("--n-samples",     type=int,  default=300)
    p.add_argument("--n-features",    type=int,  default=30)
    p.add_argument("--n-informative", type=int,  default=8)
    p.add_argument("--task",          default="regression",
                   choices=["binary", "multiclass", "regression"])
    p.add_argument("--algorithms",    nargs="+",
                   default=["eln", "rdg", "rf"])
    p.add_argument("--min-agreement", type=float, default=1.0)
    p.add_argument("--n-bootstrap",   type=int,  default=10)
    p.add_argument("--outer-cv",      type=int,  default=5)
    args = p.parse_args()

    print(f"\nSynthetic {args.task}  |  "
          f"{args.n_samples} samples x {args.n_features} features  "
          f"({args.n_informative} informative)\n")

    from sklearn.datasets import make_classification, make_regression

    if args.task == "regression":
        X_raw, y_raw = make_regression(
            n_samples=args.n_samples, n_features=args.n_features,
            n_informative=args.n_informative, noise=0.1, random_state=42,
        )
    else:
        X_raw, y_raw = make_classification(
            n_samples=args.n_samples, n_features=args.n_features,
            n_informative=args.n_informative, random_state=42,
        )

    X_df = pd.DataFrame(X_raw, columns=[f"f{i}" for i in range(args.n_features)])
    y_s  = pd.Series(y_raw, name="target")

    result = AlgorithmConsensus(
        algorithms    = args.algorithms,
        min_agreement = args.min_agreement,
        base_params   = dict(
            outer_cv=args.outer_cv, inner_cv=3,
            n_bootstrap=args.n_bootstrap, n_iter=5,
            stability_threshold=0.70,
            cutoff_n_bootstrap=50, random_state=42,
            n_jobs=1, verbose=False,
        ),
    ).fit(X_df, y_s, task_type=args.task)

    result.print_report()
    print("Consensus features:", result.consensus_features)

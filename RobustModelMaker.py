
"""
RobustModelMaker v0.3
======================

A reproducible model-building pipeline for small-to-medium scientific datasets.

v0.3 adds:
- binary and multiclass classification
- regression
- external validation evaluation
- optional probability calibration for classification: sigmoid/Platt or isotonic
- permutation importance
- SHAP-ready exports
- grouped CV
- repeated nested CV for non-grouped workflows

The design keeps the v0.2 public entry points where possible:
    result = run_pipeline(X, y, ...)
    maker = RobustModelMaker(...).fit(X, y)

Notes
-----
- Preprocessing is fitted inside each training fold during validation.
- Final preprocessing is fitted only after model assessment, on the full training set.
- For binary classification, target-specificity cutoff handling is retained.
- For multiclass classification, predict() returns the most probable class.
- For regression, predict() returns continuous values.
"""

from __future__ import annotations

import json
import os
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from scipy.stats import loguniform, randint, uniform

from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance as sklearn_permutation_importance
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGBOOST = True
except Exception:  # pragma: no cover
    XGBClassifier = None
    XGBRegressor = None
    _HAS_XGBOOST = False

Algorithm = Literal["eln", "rf", "xgb"]
TaskType = Literal["auto", "binary", "multiclass", "regression"]
ResolvedTask = Literal["binary", "multiclass", "regression"]
PreprocessMode = Literal["auto", "standard", "none"]
CalibrationMode = Literal["none", "sigmoid", "isotonic"]


def set_global_seed(random_state: Optional[int] = 42) -> None:
    """Set deterministic environment-level seeds where possible."""
    if random_state is None:
        return
    os.environ.setdefault("PYTHONHASHSEED", str(random_state))
    np.random.seed(int(random_state))


@dataclass
class StabilitySelectionResult:
    """Feature selection frequencies from bootstrap stability selection."""

    feature_names: np.ndarray
    selection_frequencies: np.ndarray
    selected_features: np.ndarray
    selected_indices: np.ndarray
    threshold: float
    n_bootstrap: int
    task_type: ResolvedTask

    def summary(self) -> pd.DataFrame:
        return (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "selection_frequency": self.selection_frequencies,
                    "selected": self.selection_frequencies >= self.threshold,
                }
            )
            .sort_values("selection_frequency", ascending=False)
            .reset_index(drop=True)
        )


@dataclass
class CutoffResult:
    """Binary target-specificity cutoff result."""

    cutoff_median: float
    cutoff_ci_lower: float
    cutoff_ci_upper: float
    cutoff_distribution: np.ndarray
    target_specificity: float
    achieved_specificity: float
    achieved_sensitivity: float

    def summary(self) -> str:
        return (
            f"Cutoff: {self.cutoff_median:.4f} "
            f"(95% CI: {self.cutoff_ci_lower:.4f} - {self.cutoff_ci_upper:.4f})\n"
            f"Target specificity: {self.target_specificity:.1%}\n"
            f"Achieved specificity: {self.achieved_specificity:.1%}\n"
            f"Achieved sensitivity: {self.achieved_sensitivity:.1%}"
        )


@dataclass
class NestedCVResult:
    """Results from nested or repeated nested cross-validation."""

    outer_scores: np.ndarray
    outer_predictions: np.ndarray
    outer_true_labels: np.ndarray
    mean_score: float
    std_score: float
    metric_name: str
    best_params_per_fold: List[Dict[str, Any]]
    selected_features_per_fold: List[np.ndarray]
    feature_stability: pd.DataFrame
    repeats: int
    task_type: ResolvedTask

    @property
    def mean_auc(self) -> float:
        """Backward-compatible alias for classification AUC."""
        return self.mean_score

    @property
    def std_auc(self) -> float:
        """Backward-compatible alias for classification AUC std."""
        return self.std_score

    def summary(self) -> str:
        return (
            f"Nested CV {self.metric_name}: {self.mean_score:.4f} +/- {self.std_score:.4f}\n"
            f"Per-fold scores: {[f'{s:.3f}' for s in self.outer_scores]}"
        )


@dataclass
class VerificationResult:
    """External validation result for classification or regression."""

    task_type: ResolvedTask
    metrics: Dict[str, float]
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None
    cutoff: Optional[float] = None
    confusion: Optional[np.ndarray] = None

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([self.metrics])


@dataclass
class PermutationImportanceResult:
    """Permutation importance result."""

    importances_mean: np.ndarray
    importances_std: np.ndarray
    importances: np.ndarray
    feature_names: np.ndarray
    scoring: str

    def summary(self) -> pd.DataFrame:
        return (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance_mean": self.importances_mean,
                    "importance_std": self.importances_std,
                }
            )
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )


@dataclass
class PipelineResult:
    """Complete fitted RobustModelMaker result."""

    nested_cv_result: NestedCVResult
    stability_result: StabilitySelectionResult
    cutoff_result: Optional[CutoffResult]
    robust_model: Any
    selected_features: np.ndarray
    selected_feature_indices: np.ndarray
    algorithm: str
    task_type: ResolvedTask
    preprocessor: Optional[Pipeline] = None
    feature_names: Optional[np.ndarray] = None
    label_mapping: Optional[Dict[Any, int]] = None
    inverse_label_mapping: Optional[Dict[int, Any]] = None
    class_names: Optional[np.ndarray] = None
    calibration: CalibrationMode = "none"
    validation_result: Optional[VerificationResult] = None
    preserve_nans: bool = True
    nan_dropping_col_mask: Optional[np.ndarray] = None
    nan_dropping_result: Optional[Dict[str, Any]] = None

    def _prepare_X_selected(self, X: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, Optional[pd.Index]]:
        if self.preprocessor is None:
            raise NotFittedError("PipelineResult has no fitted preprocessor.")
        index = X.index if isinstance(X, pd.DataFrame) else None
        if isinstance(X, pd.DataFrame):
            if self.feature_names is not None:
                if all(col in X.columns for col in self.feature_names):
                    X_arr = X.loc[:, self.feature_names].to_numpy(dtype=float)
                else:
                    string_to_col = {str(col): col for col in X.columns}
                    missing = [col for col in self.feature_names if str(col) not in string_to_col]
                    if missing:
                        raise ValueError(f"X is missing required columns: {missing[:10]}")
                    ordered_cols = [string_to_col[str(col)] for col in self.feature_names]
                    X_arr = X.loc[:, ordered_cols].to_numpy(dtype=float)
            else:
                X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)
            # When preserve_nans=False was used, apply the column-dropping mask
            # so that callers can pass the original full feature matrix.
            if (
                self.nan_dropping_col_mask is not None
                and X_arr.ndim == 2
                and X_arr.shape[1] == len(self.nan_dropping_col_mask)
            ):
                X_arr = X_arr[:, self.nan_dropping_col_mask]
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array or DataFrame.")
        if self.feature_names is not None and X_arr.shape[1] != len(self.feature_names):
            raise ValueError(f"X has {X_arr.shape[1]} features, expected {len(self.feature_names)}.")
        X_proc = self.preprocessor.transform(X_arr)
        return X_proc[:, self.selected_feature_indices], index

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame, pd.Series]:
        """Return probabilities for classification.

        Binary classification returns a positive-class Series for DataFrame input
        and a 1D array for array input, matching v0.2 behaviour. Multiclass returns
        a probability matrix or DataFrame.
        """
        if self.task_type == "regression":
            raise AttributeError("Regression models do not support predict_proba.")
        X_sel, index = self._prepare_X_selected(X)
        if not hasattr(self.robust_model, "predict_proba"):
            raise AttributeError("The fitted model does not support predict_proba.")
        proba = self.robust_model.predict_proba(X_sel)
        if self.task_type == "binary":
            out = proba[:, 1]
            if index is not None:
                return pd.Series(out, index=index, name="probability")
            return out
        columns = [str(c) for c in (self.class_names if self.class_names is not None else np.arange(proba.shape[1]))]
        if index is not None:
            return pd.DataFrame(proba, index=index, columns=columns)
        return proba

    def predict(self, X: Union[np.ndarray, pd.DataFrame], cutoff: Optional[float] = None) -> Union[np.ndarray, pd.Series]:
        """Predict class labels or regression values."""
        X_sel, index = self._prepare_X_selected(X)
        if self.task_type == "binary":
            if cutoff is None:
                cutoff = self.cutoff_result.cutoff_median if self.cutoff_result is not None else 0.5
            proba = np.asarray(self.predict_proba(X), dtype=float)
            pred_int = (proba >= float(cutoff)).astype(int)
            values = _decode_labels(pred_int, self.inverse_label_mapping)
            if index is not None:
                return pd.Series(values, index=index, name="prediction")
            return values
        pred = self.robust_model.predict(X_sel)
        if self.task_type == "multiclass":
            pred = _decode_labels(pred.astype(int), self.inverse_label_mapping)
        if index is not None:
            return pd.Series(pred, index=index, name="prediction")
        return pred

    def evaluate_verification(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, List[Any]],
        cutoff: Optional[float] = None,
    ) -> VerificationResult:
        """Evaluate an external validation dataset."""
        y_encoded = _prepare_y(y, self.task_type, label_mapping=self.label_mapping)
        pred = self.predict(X, cutoff=cutoff)
        pred_encoded = _prepare_y(pred, self.task_type, label_mapping=self.label_mapping) if self.task_type != "regression" else np.asarray(pred, dtype=float)

        if self.task_type == "regression":
            metrics = {
                "r2": float(r2_score(y_encoded, pred_encoded)),
                "rmse": float(np.sqrt(mean_squared_error(y_encoded, pred_encoded))),
                "mae": float(mean_absolute_error(y_encoded, pred_encoded)),
            }
            return VerificationResult(self.task_type, metrics, pred_encoded)

        if self.task_type == "binary":
            proba = np.asarray(self.predict_proba(X), dtype=float)
            cut = float(cutoff if cutoff is not None else (self.cutoff_result.cutoff_median if self.cutoff_result else 0.5))
            cm = confusion_matrix(y_encoded, pred_encoded, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            metrics = {
                "auc": float(roc_auc_score(y_encoded, proba)) if len(np.unique(y_encoded)) == 2 else np.nan,
                "accuracy": float(accuracy_score(y_encoded, pred_encoded)),
                "balanced_accuracy": float(balanced_accuracy_score(y_encoded, pred_encoded)),
                "sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
                "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
                "tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp),
                "cutoff": cut,
            }
            return VerificationResult(self.task_type, metrics, pred_encoded, probabilities=proba, cutoff=cut, confusion=cm)

        proba = np.asarray(self.predict_proba(X), dtype=float)
        labels = np.arange(len(self.class_names)) if self.class_names is not None else np.unique(y_encoded)
        cm = confusion_matrix(y_encoded, pred_encoded, labels=labels)
        try:
            auc = float(roc_auc_score(y_encoded, proba, multi_class="ovr", average="weighted"))
        except ValueError:
            auc = np.nan
        metrics = {
            "auc_ovr_weighted": auc,
            "accuracy": float(accuracy_score(y_encoded, pred_encoded)),
            "balanced_accuracy": float(balanced_accuracy_score(y_encoded, pred_encoded)),
            "f1_weighted": float(f1_score(y_encoded, pred_encoded, average="weighted")),
            "macro_f1": float(f1_score(y_encoded, pred_encoded, average="macro")),
        }
        return VerificationResult(self.task_type, metrics, pred_encoded, probabilities=proba, confusion=cm)

    def permutation_importance(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, List[Any]],
        n_repeats: int = 20,
        scoring: Optional[str] = None,
        random_state: Optional[int] = 42,
        n_jobs: int = -1,
    ) -> PermutationImportanceResult:
        """Compute permutation importance on processed selected features."""
        X_sel, _ = self._prepare_X_selected(X)
        y_encoded = _prepare_y(y, self.task_type, label_mapping=self.label_mapping)
        if scoring is None:
            scoring = _default_scoring(self.task_type)
        imp = sklearn_permutation_importance(
            self.robust_model,
            X_sel,
            y_encoded,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring=scoring,
            n_jobs=n_jobs,
        )
        return PermutationImportanceResult(
            importances_mean=imp.importances_mean,
            importances_std=imp.importances_std,
            importances=imp.importances,
            feature_names=self.selected_features,
            scoring=scoring,
        )

    def export_shap_ready(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.Series, List[Any]]] = None) -> Dict[str, Any]:
        """Return model and processed selected matrix suitable for SHAP explainers."""
        X_sel, index = self._prepare_X_selected(X)
        X_df = pd.DataFrame(X_sel, index=index, columns=self.selected_features)
        out: Dict[str, Any] = {
            "model": self.robust_model,
            "X": X_df,
            "feature_names": self.selected_features,
            "task_type": self.task_type,
            "algorithm": self.algorithm,
            "class_names": self.class_names,
            "predict_function": self.robust_model.predict,
        }
        if self.task_type != "regression" and hasattr(self.robust_model, "predict_proba"):
            out["predict_proba_function"] = self.robust_model.predict_proba
        if y is not None:
            out["y"] = _prepare_y(y, self.task_type, label_mapping=self.label_mapping)
        return out

    def plot_feature_stability(self, top_n: int = 30):
        if top_n <= 0:
            raise ValueError("top_n must be a positive integer.")
        import matplotlib.pyplot as plt
        df = self.stability_result.summary().head(top_n).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(df))))
        ax.barh(df["feature"].astype(str), df["selection_frequency"])
        ax.axvline(self.stability_result.threshold, linestyle="--", linewidth=1)
        ax.set_xlabel("Selection frequency")
        ax.set_ylabel("Feature")
        ax.set_xlim(0, 1)
        ax.set_title("Feature stability")
        fig.tight_layout()
        return ax

    def summary(self) -> str:
        cut = self.cutoff_result.summary() if self.cutoff_result is not None else "Not applicable"
        val = ""
        if self.validation_result is not None:
            val = "\n\nEXTERNAL VALIDATION:\n" + self.validation_result.summary().to_string(index=False)
        nan_info = ""
        if not self.preserve_nans and self.nan_dropping_result is not None:
            d = self.nan_dropping_result
            nan_info = (
                f"\nNaN STRATEGY (preserve_nans=False):\n"
                f"  Original: {d['original_n_samples']} rows x {d['original_n_features']} features\n"
                f"  Retained: {d['retained_n_samples']} rows x {d['retained_n_features']} features\n"
            )
        return (
            f"{'=' * 60}\n"
            f"ROBUST MODEL MAKER v0.3 RESULTS\n"
            f"{'=' * 60}\n"
            f"Task: {self.task_type}\n"
            f"Algorithm: {self.algorithm}\n"
            f"Calibration: {self.calibration}\n"
            f"Selected features ({len(self.selected_features)}): {list(self.selected_features)}\n"
            f"{nan_info}\n"
            f"NESTED CV PERFORMANCE:\n{self.nested_cv_result.summary()}\n\n"
            f"CUTOFF DETERMINATION:\n{cut}"
            f"{val}\n"
            f"{'=' * 60}"
        )


def _extract_feature_names(X: Any, feature_names: Optional[np.ndarray]) -> np.ndarray:
    if feature_names is not None:
        names = np.asarray(feature_names)
    elif isinstance(X, pd.DataFrame):
        names = X.columns.to_numpy()
    else:
        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array or DataFrame.")
        names = np.array([f"feature_{i}" for i in range(X_arr.shape[1])])
    return names.astype(str)


def _to_numpy_X(X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    X_arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array or DataFrame.")
    if X_arr.shape[0] < 4:
        raise ValueError("X must contain at least 4 samples.")
    if X_arr.shape[1] < 1:
        raise ValueError("X must contain at least 1 feature.")
    return X_arr


def _resolve_task_type(y: Union[np.ndarray, pd.Series, List[Any]], task_type: TaskType) -> ResolvedTask:
    if task_type != "auto":
        return task_type  # type: ignore[return-value]
    y_arr = np.asarray(y).ravel()
    if pd.api.types.is_float_dtype(y_arr) and len(np.unique(y_arr)) > min(20, max(3, int(0.2 * len(y_arr)))):
        return "regression"
    n_classes = len(np.unique(y_arr))
    if n_classes == 2:
        return "binary"
    if 2 < n_classes <= max(20, int(0.2 * len(y_arr))):
        return "multiclass"
    return "regression"


def _make_label_mapping(y: Union[np.ndarray, pd.Series, List[Any]], task_type: ResolvedTask) -> Tuple[Optional[Dict[Any, int]], Optional[Dict[int, Any]], Optional[np.ndarray]]:
    if task_type == "regression":
        return None, None, None
    y_arr = np.asarray(y).ravel()
    if np.any(pd.isna(y_arr)):
        raise ValueError("y contains missing values.")
    classes = np.unique(y_arr)
    if task_type == "binary" and len(classes) != 2:
        raise ValueError(f"Binary classification requires exactly 2 classes. Found: {classes}")
    if task_type == "multiclass" and len(classes) < 3:
        raise ValueError(f"Multiclass classification requires at least 3 classes. Found: {classes}")
    mapping = {cls: i for i, cls in enumerate(classes)}
    inverse = {i: cls for cls, i in mapping.items()}
    return mapping, inverse, classes.astype(str)


def _prepare_y(y: Union[np.ndarray, pd.Series, List[Any]], task_type: ResolvedTask, label_mapping: Optional[Dict[Any, int]] = None) -> np.ndarray:
    y_arr = np.asarray(y).ravel()
    if y_arr.ndim != 1:
        raise ValueError("y must be a 1D vector.")
    if np.any(pd.isna(y_arr)):
        raise ValueError("y contains missing values.")
    if task_type == "regression":
        y_num = y_arr.astype(float)
        if np.any(~np.isfinite(y_num)):
            raise ValueError("Regression target contains non-finite values.")
        return y_num
    if label_mapping is None:
        label_mapping, _, _ = _make_label_mapping(y_arr, task_type)
    try:
        return np.array([label_mapping[v] for v in y_arr], dtype=int)  # type: ignore[index]
    except KeyError as exc:
        raise ValueError(f"Unknown label in y: {exc}") from exc


def _decode_labels(values: np.ndarray, inverse_label_mapping: Optional[Dict[int, Any]]) -> np.ndarray:
    if inverse_label_mapping is None:
        return values
    return np.array([inverse_label_mapping[int(v)] for v in values], dtype=object)


def _validate_inputs(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series, List[Any]],
    feature_names: Optional[np.ndarray],
    outer_cv: int,
    inner_cv: int,
    task_type: TaskType,
    groups: Optional[Union[np.ndarray, pd.Series, List[Any]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, ResolvedTask, Optional[Dict[Any, int]], Optional[Dict[int, Any]], Optional[np.ndarray], Optional[np.ndarray]]:
    names = _extract_feature_names(X, feature_names)
    X_arr = _to_numpy_X(X)
    resolved = _resolve_task_type(y, task_type)
    mapping, inverse_mapping, class_names = _make_label_mapping(y, resolved)
    y_arr = _prepare_y(y, resolved, mapping)
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(f"X has {X_arr.shape[0]} rows, but y has {y_arr.shape[0]} values.")
    if len(names) != X_arr.shape[1]:
        raise ValueError(f"feature_names has length {len(names)}, but X has {X_arr.shape[1]} columns.")
    if len(np.unique(names)) != len(names):
        raise ValueError("feature_names contains duplicates.")
    if np.isinf(X_arr).any():
        raise ValueError("X contains infinite values. Replace or remove infinities before fitting.")
    all_missing = np.isnan(X_arr).all(axis=0)
    if all_missing.any():
        raise ValueError(f"X contains all-missing feature columns: {names[all_missing][:10].tolist()}")
    groups_arr = None
    if groups is not None:
        groups_arr = np.asarray(groups).ravel()
        if len(groups_arr) != X_arr.shape[0]:
            raise ValueError("groups must have the same length as y.")
        if len(np.unique(groups_arr)) < 2:
            raise ValueError("Grouped CV requires at least 2 distinct groups.")
    if resolved in {"binary", "multiclass"}:
        counts = np.bincount(y_arr.astype(int))
        if groups_arr is None and counts.min() < max(2, min(outer_cv, inner_cv)):
            raise ValueError(
                "Each class must contain at least as many samples as the requested CV folds. "
                f"Smallest class has {int(counts.min())}; requested folds include {min(outer_cv, inner_cv)}."
            )
    else:
        if X_arr.shape[0] < max(outer_cv, inner_cv) * 2 and groups_arr is None:
            raise ValueError("Regression requires enough samples for the requested outer and inner folds.")
    return X_arr, y_arr, names, resolved, mapping, inverse_mapping, class_names, groups_arr


def _make_preprocessor(preprocess: PreprocessMode, alg: Algorithm) -> Pipeline:
    steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    use_scaler = preprocess == "standard" or (preprocess == "auto" and alg == "eln")
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def _default_scoring(task_type: ResolvedTask) -> str:
    if task_type == "binary":
        return "roc_auc"
    if task_type == "multiclass":
        return "roc_auc_ovr_weighted"
    return "neg_root_mean_squared_error"


def _score_predictions(task_type: ResolvedTask, y_true: np.ndarray, pred: np.ndarray) -> float:
    if task_type == "binary":
        return float(roc_auc_score(y_true, pred))
    if task_type == "multiclass":
        try:
            return float(roc_auc_score(y_true, pred, multi_class="ovr", average="weighted"))
        except ValueError:
            return float(accuracy_score(y_true, np.argmax(pred, axis=1)))
    return float(-np.sqrt(mean_squared_error(y_true, pred)))


def _make_outer_splitter(task_type: ResolvedTask, n_splits: int, random_state: Optional[int], groups: Optional[np.ndarray]):
    if groups is not None:
        n_groups = len(np.unique(groups))
        if n_splits > n_groups:
            warnings.warn(f"outer_cv reduced from {n_splits} to {n_groups} because only {n_groups} groups are available.")
            n_splits = n_groups
        return GroupKFold(n_splits=n_splits), n_splits
    if task_type in {"binary", "multiclass"}:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state), n_splits
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state), n_splits


def _make_inner_splitter(task_type: ResolvedTask, n_splits: int, random_state: Optional[int], groups: Optional[np.ndarray]):
    if groups is not None:
        n_groups = len(np.unique(groups))
        n_splits = min(n_splits, n_groups)
        if n_splits < 2:
            raise ValueError("Inner grouped CV requires at least 2 groups in the training fold.")
        return GroupKFold(n_splits=n_splits)
    if task_type in {"binary", "multiclass"}:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _stratified_or_random_subsample(y: np.ndarray, task_type: ResolvedTask, n_samples: int, rng: np.random.RandomState) -> np.ndarray:
    if task_type == "regression":
        idx = rng.choice(np.arange(len(y)), size=min(n_samples, len(y)), replace=False)
        rng.shuffle(idx)
        return idx
    classes, counts = np.unique(y.astype(int), return_counts=True)
    proportions = counts / len(y)
    indices: List[int] = []
    for cls, prop in zip(classes, proportions):
        cls_indices = np.where(y == cls)[0]
        n_cls = max(1, int(round(n_samples * prop)))
        n_cls = min(n_cls, len(cls_indices))
        indices.extend(rng.choice(cls_indices, size=n_cls, replace=False).tolist())
    out = np.array(indices, dtype=int)
    rng.shuffle(out)
    return out


def get_algorithm_config(
    alg: Algorithm,
    task_type: ResolvedTask = "binary",
    random_state: Optional[int] = 42,
    n_jobs: int = -1,
    n_classes: Optional[int] = None,
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """Get estimator and RandomizedSearchCV parameter distributions."""
    if alg == "eln":
        if task_type == "regression":
            model = ElasticNet(random_state=random_state, max_iter=10000)
            return model, {"alpha": loguniform(1e-4, 1e2), "l1_ratio": uniform(0, 1)}
        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            max_iter=5000,
            random_state=random_state,
            n_jobs=1,
        )
        return model, {"C": loguniform(1e-4, 1e2), "l1_ratio": uniform(0, 1)}

    if alg == "rf":
        if task_type == "regression":
            model = RandomForestRegressor(random_state=random_state, n_jobs=n_jobs)
            param = {
                "n_estimators": randint(100, 500),
                "max_depth": randint(2, 20),
                "min_samples_split": randint(2, 20),
                "min_samples_leaf": randint(1, 10),
                "max_features": ["sqrt", "log2", None],
            }
            return model, param
        model = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs, class_weight="balanced_subsample")
        param = {
            "n_estimators": randint(100, 500),
            "max_depth": randint(2, 20),
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 10),
            "max_features": ["sqrt", "log2", None],
        }
        return model, param

    if alg == "xgb":
        if not _HAS_XGBOOST:
            raise ImportError("xgboost is not installed. Install xgboost or use alg='eln' or alg='rf'.")
        common = {
            "n_estimators": randint(100, 500),
            "max_depth": randint(2, 10),
            "learning_rate": loguniform(1e-2, 3e-1),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "reg_alpha": loguniform(1e-4, 1),
            "reg_lambda": loguniform(1e-3, 10),
        }
        if task_type == "regression":
            return XGBRegressor(random_state=random_state, n_jobs=n_jobs, tree_method="hist", objective="reg:squarederror"), common
        objective = "binary:logistic" if task_type == "binary" else "multi:softprob"
        model = XGBClassifier(
            random_state=random_state,
            n_jobs=n_jobs,
            eval_metric="auc" if task_type == "binary" else "mlogloss",
            tree_method="hist",
            objective=objective,
            num_class=n_classes if task_type == "multiclass" else None,
        )
        return model, common
    raise ValueError("alg must be one of 'eln', 'rf', or 'xgb'.")


def stability_selection(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series, List[Any]],
    feature_names: Optional[np.ndarray] = None,
    alg: Algorithm = "xgb",
    task_type: TaskType = "auto",
    n_bootstrap: int = 100,
    sample_fraction: float = 0.7,
    threshold: float = 0.7,
    random_state: Optional[int] = 42,
    n_jobs: int = -1,
) -> StabilitySelectionResult:
    """Bootstrap stability selection on already preprocessed numeric data."""
    set_global_seed(random_state)
    names = _extract_feature_names(X, feature_names)
    X_arr = _to_numpy_X(X)
    resolved = _resolve_task_type(y, task_type)
    mapping, _, _ = _make_label_mapping(y, resolved)
    y_arr = _prepare_y(y, resolved, mapping)
    if not (0 < sample_fraction <= 1):
        raise ValueError("sample_fraction must be in (0, 1].")
    if not (0 < threshold <= 1):
        raise ValueError("threshold must be in (0, 1].")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1.")
    rng = np.random.RandomState(random_state)
    n_samples, n_features = X_arr.shape
    subsample_size = max(2, int(round(n_samples * sample_fraction)))
    selection_counts = np.zeros(n_features, dtype=float)

    for b in range(n_bootstrap):
        seed_b = None if random_state is None else int(random_state + 10000 + b)
        idx = _stratified_or_random_subsample(y_arr, resolved, subsample_size, rng)
        X_sub, y_sub = X_arr[idx], y_arr[idx]
        model, _ = get_algorithm_config(alg, resolved, random_state=seed_b, n_jobs=n_jobs, n_classes=len(np.unique(y_arr)) if resolved != "regression" else None)
        if alg == "eln":
            if resolved == "regression":
                model.set_params(alpha=0.01, l1_ratio=0.7)
            else:
                model.set_params(C=1.0, l1_ratio=0.7)
        elif alg == "rf":
            model.set_params(n_estimators=80, max_depth=10)
        elif alg == "xgb":
            model.set_params(n_estimators=80, max_depth=6, learning_rate=0.05)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_sub, y_sub)
        if hasattr(model, "coef_"):
            coef = np.asarray(model.coef_)
            if coef.ndim == 1:
                importance = np.abs(coef)
            else:
                importance = np.max(np.abs(coef), axis=0)
        elif hasattr(model, "feature_importances_"):
            importance = np.asarray(model.feature_importances_)
        else:
            raise RuntimeError("Model does not expose coefficients or feature_importances_.")
        selected = importance > np.median(importance)
        if not np.any(selected):
            selected[np.argmax(importance)] = True
        selection_counts += selected.astype(float)

    freqs = selection_counts / float(n_bootstrap)
    selected_mask = freqs >= threshold
    return StabilitySelectionResult(
        feature_names=names,
        selection_frequencies=freqs,
        selected_features=names[selected_mask],
        selected_indices=np.where(selected_mask)[0],
        threshold=threshold,
        n_bootstrap=n_bootstrap,
        task_type=resolved,
    )


def _fit_calibrated_if_needed(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    task_type: ResolvedTask,
    calibration: CalibrationMode,
    cv: Any,
    groups: Optional[np.ndarray] = None,
) -> BaseEstimator:
    if task_type == "regression" or calibration == "none":
        model.fit(X, y)
        return model
    try:
        calibrated = CalibratedClassifierCV(estimator=model, method=calibration, cv=cv, ensemble=True)
    except TypeError:  # older sklearn
        calibrated = CalibratedClassifierCV(base_estimator=model, method=calibration, cv=cv, ensemble=True)
    if groups is not None:
        # Some sklearn versions do not pass groups through CalibratedClassifierCV.
        # Prefer a fitted uncalibrated model rather than silently leaking groups.
        warnings.warn("Calibration is skipped for grouped CV because CalibratedClassifierCV does not reliably pass groups across sklearn versions.")
        model.fit(X, y)
        return model
    calibrated.fit(X, y)
    return calibrated


def nested_cross_validation(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series, List[Any]],
    feature_names: Optional[np.ndarray] = None,
    alg: Algorithm = "eln",
    task_type: TaskType = "auto",
    outer_cv: int = 10,
    inner_cv: int = 10,
    repeated_outer_cv: int = 1,
    n_iter: int = 100,
    stability_threshold: float = 0.7,
    n_bootstrap: int = 100,
    random_state: Optional[int] = 42,
    preprocess: PreprocessMode = "auto",
    calibration: CalibrationMode = "none",
    groups: Optional[Union[np.ndarray, pd.Series, List[Any]]] = None,
    n_jobs: int = -1,
    verbose: bool = True,
) -> NestedCVResult:
    """Run leakage-safe nested or repeated nested CV."""
    set_global_seed(random_state)
    X_arr, y_arr, names, resolved, mapping, _, class_names, groups_arr = _validate_inputs(X, y, feature_names, outer_cv, inner_cv, task_type, groups)
    n_classes = len(class_names) if class_names is not None else None
    if groups_arr is not None and repeated_outer_cv > 1:
        warnings.warn("Grouped CV is deterministic GroupKFold; repeated_outer_cv is set to 1 for grouped workflows.")
        repeated_outer_cv = 1
    if repeated_outer_cv < 1:
        raise ValueError("repeated_outer_cv must be >= 1.")

    prediction_shape = (len(y_arr), n_classes) if resolved == "multiclass" else (len(y_arr),)
    prediction_sums = np.zeros(prediction_shape, dtype=float)
    prediction_counts = np.zeros(len(y_arr), dtype=float)
    outer_scores: List[float] = []
    best_params: List[Dict[str, Any]] = []
    selected_features_per_fold: List[np.ndarray] = []
    all_freqs: List[np.ndarray] = []

    for rep in range(repeated_outer_cv):
        rep_seed = None if random_state is None else int(random_state + rep * 1000)
        outer_splitter, outer_splits = _make_outer_splitter(resolved, outer_cv, rep_seed, groups_arr)
        split_iter = outer_splitter.split(X_arr, y_arr, groups_arr) if groups_arr is not None else outer_splitter.split(X_arr, y_arr)
        for fold_idx, (train_idx, test_idx) in enumerate(split_iter):
            if verbose:
                print(f"  Repeat {rep + 1}/{repeated_outer_cv}, outer fold {fold_idx + 1}/{outer_splits}...")
            fold_seed = None if rep_seed is None else int(rep_seed + fold_idx)
            X_train_raw, X_test_raw = X_arr[train_idx], X_arr[test_idx]
            y_train, y_test = y_arr[train_idx], y_arr[test_idx]
            groups_train = groups_arr[train_idx] if groups_arr is not None else None
            pre = _make_preprocessor(preprocess, alg)
            X_train = pre.fit_transform(X_train_raw)
            X_test = pre.transform(X_test_raw)

            stab = stability_selection(
                X_train, y_train, names, alg=alg, task_type=resolved,
                n_bootstrap=n_bootstrap, threshold=stability_threshold,
                random_state=fold_seed, n_jobs=n_jobs,
            )
            all_freqs.append(stab.selection_frequencies)
            selected_idx = stab.selected_indices
            if len(selected_idx) == 0:
                warnings.warn(f"Fold {fold_idx + 1}: no features selected. Using top 5 features.")
                selected_idx = np.argsort(stab.selection_frequencies)[-min(5, len(names)):]
            selected_features_per_fold.append(names[selected_idx])
            X_train_sel, X_test_sel = X_train[:, selected_idx], X_test[:, selected_idx]

            inner_splitter = _make_inner_splitter(resolved, inner_cv, fold_seed, groups_train)
            model, param_dist = get_algorithm_config(alg, resolved, random_state=fold_seed, n_jobs=n_jobs, n_classes=n_classes)
            search = RandomizedSearchCV(
                model, param_dist, n_iter=n_iter, cv=inner_splitter,
                scoring=_default_scoring(resolved), n_jobs=n_jobs,
                random_state=fold_seed, refit=True,
            )
            fit_kwargs = {"groups": groups_train} if groups_train is not None else {}
            search.fit(X_train_sel, y_train, **fit_kwargs)
            best_params.append(search.best_params_)
            best_estimator = clone(search.best_estimator_)
            best_estimator.set_params(**search.best_params_)
            calibration_cv = _make_inner_splitter(resolved, min(3, inner_cv), fold_seed, groups_train) if calibration != "none" else None
            fitted = _fit_calibrated_if_needed(best_estimator, X_train_sel, y_train, resolved, calibration, calibration_cv, groups_train)

            if resolved == "regression":
                pred = fitted.predict(X_test_sel)
                prediction_sums[test_idx] += pred
            elif resolved == "binary":
                pred = fitted.predict_proba(X_test_sel)[:, 1]
                prediction_sums[test_idx] += pred
            else:
                pred = fitted.predict_proba(X_test_sel)
                prediction_sums[test_idx] += pred
            prediction_counts[test_idx] += 1
            outer_scores.append(_score_predictions(resolved, y_test, pred))
            if verbose:
                print(f"    Fold {_default_scoring(resolved)}: {outer_scores[-1]:.4f}, features selected: {len(selected_idx)}")

    if np.any(prediction_counts == 0):
        raise RuntimeError("Some samples did not receive out-of-fold predictions.")
    if resolved == "multiclass":
        outer_predictions = prediction_sums / prediction_counts[:, None]
    else:
        outer_predictions = prediction_sums / prediction_counts
    freq_arr = np.vstack(all_freqs)
    feature_stability = (
        pd.DataFrame(
            {
                "feature": names,
                "mean_frequency": freq_arr.mean(axis=0),
                "std_frequency": freq_arr.std(axis=0),
                "selected_in_n_folds": (freq_arr >= stability_threshold).sum(axis=0),
            }
        )
        .sort_values("mean_frequency", ascending=False)
        .reset_index(drop=True)
    )
    return NestedCVResult(
        outer_scores=np.asarray(outer_scores, dtype=float),
        outer_predictions=outer_predictions,
        outer_true_labels=y_arr,
        mean_score=float(np.mean(outer_scores)),
        std_score=float(np.std(outer_scores)),
        metric_name=_default_scoring(resolved),
        best_params_per_fold=best_params,
        selected_features_per_fold=selected_features_per_fold,
        feature_stability=feature_stability,
        repeats=repeated_outer_cv,
        task_type=resolved,
    )


def determine_cutoff(
    y_true: Union[np.ndarray, pd.Series, List[Any]],
    y_scores: Union[np.ndarray, pd.Series, List[float]],
    target_specificity: float = 0.98,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = 42,
) -> CutoffResult:
    """Determine binary cutoff using bootstrap controls."""
    if not (0 < target_specificity < 1):
        raise ValueError("target_specificity must be in (0, 1).")
    rng = np.random.RandomState(random_state)
    mapping, _, _ = _make_label_mapping(y_true, "binary")
    y_arr = _prepare_y(y_true, "binary", mapping)
    scores = np.asarray(y_scores, dtype=float).ravel()
    control_scores = scores[y_arr == 0]
    case_scores = scores[y_arr == 1]
    cutoffs = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        boot = rng.choice(control_scores, size=len(control_scores), replace=True)
        cutoffs[i] = np.quantile(boot, target_specificity, method="linear")
    alpha = 1 - confidence_level
    cutoff = float(np.median(cutoffs))
    return CutoffResult(
        cutoff_median=cutoff,
        cutoff_ci_lower=float(np.quantile(cutoffs, alpha / 2, method="linear")),
        cutoff_ci_upper=float(np.quantile(cutoffs, 1 - alpha / 2, method="linear")),
        cutoff_distribution=cutoffs,
        target_specificity=target_specificity,
        achieved_specificity=float(np.mean(control_scores < cutoff)),
        achieved_sensitivity=float(np.mean(case_scores >= cutoff)),
    )


def _find_optimal_missingness_thresholds(X: np.ndarray) -> Tuple[float, float]:
    """
    Grid-search column and row missingness thresholds that maximise a score
    balancing data density, row retention, and column retention.

    Score = (non-missing density in retained region)
            * sqrt(retained row fraction)
            * sqrt(retained col fraction)

    Using the retained region as the denominator penalises high-missingness
    columns: keeping a nearly-empty column reduces density without proportionally
    improving the col-fraction term.

    Returns (col_threshold, row_threshold).
    """
    n_rows, n_cols = X.shape
    col_missing = np.isnan(X).mean(axis=0)
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best_score = -1.0
    best_col_t, best_row_t = 0.9, 0.9

    for col_t in thresholds:
        col_mask = col_missing <= col_t
        if not col_mask.any():
            continue
        n_col_kept = int(col_mask.sum())
        X_col = X[:, col_mask]
        row_missing = np.isnan(X_col).mean(axis=1)
        for row_t in thresholds:
            row_mask = row_missing <= row_t
            n_kept = int(row_mask.sum())
            if n_kept < 4:
                continue
            retained = float((~np.isnan(X_col[row_mask])).sum())
            region = max(n_kept * n_col_kept, 1)
            density = retained / region
            score = (density
                     * np.sqrt(n_kept / n_rows)
                     * np.sqrt(n_col_kept / n_cols))
            if score > best_score:
                best_score = score
                best_col_t, best_row_t = col_t, row_t

    return best_col_t, best_row_t


def _smart_drop_nans(
    X: np.ndarray,
    names: np.ndarray,
    random_state: Optional[int] = 42,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Drop columns and rows whose missingness fraction exceeds optimal thresholds.

    Thresholds are chosen by grid search to maximise retained non-missing cells
    while keeping enough rows for cross-validation.  Feature selection is left
    entirely to stability selection downstream.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features), may contain NaNs.
    names : feature name array of length n_features.

    Returns
    -------
    X_clean : array after missingness dropping (remaining NaNs kept for imputer).
    names_clean : feature names for retained columns.
    row_mask : bool array over original rows (True = kept).
    col_mask : bool array over original columns (True = kept).
    """
    n_rows, n_cols = X.shape

    col_t, row_t = _find_optimal_missingness_thresholds(X)

    col_missing = np.isnan(X).mean(axis=0)
    col_mask = col_missing <= col_t
    if not col_mask.any():
        col_mask = np.ones(n_cols, dtype=bool)
    X_col = X[:, col_mask]

    row_missing = np.isnan(X_col).mean(axis=1)
    row_mask = row_missing <= row_t
    if row_mask.sum() < 4:
        sorted_idx = np.argsort(row_missing)
        keep_n = max(4, int(n_rows * 0.5))
        row_mask = np.zeros(n_rows, dtype=bool)
        row_mask[sorted_idx[:keep_n]] = True

    X_clean = X_col[row_mask]
    names_clean = names[col_mask]

    if verbose:
        print(
            f"  [preserve_nans=False] col_threshold={col_t:.2f}, row_threshold={row_t:.2f}: "
            f"kept {col_mask.sum()}/{n_cols} cols, {row_mask.sum()}/{n_rows} rows"
        )

    return X_clean, names_clean, row_mask, col_mask


def run_pipeline(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series, List[Any]],
    feature_names: Optional[np.ndarray] = None,
    alg: Algorithm = "eln",
    task_type: TaskType = "auto",
    spec: float = 0.98,
    outer_cv: int = 10,
    inner_cv: int = 10,
    repeated_outer_cv: int = 1,
    n_iter: int = 100,
    stability_threshold: float = 0.7,
    n_bootstrap: int = 100,
    cutoff_n_bootstrap: int = 1000,
    random_state: Optional[int] = 42,
    preprocess: PreprocessMode = "auto",
    calibration: CalibrationMode = "none",
    groups: Optional[Union[np.ndarray, pd.Series, List[Any]]] = None,
    X_validation: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    y_validation: Optional[Union[np.ndarray, pd.Series, List[Any]]] = None,
    n_jobs: int = -1,
    verbose: bool = True,
    preserve_nans: bool = True,
) -> PipelineResult:
    """Run the full v0.3 pipeline."""
    set_global_seed(random_state)

    _nan_col_mask: Optional[np.ndarray] = None
    _nan_dropping_result: Optional[Dict[str, Any]] = None

    if not preserve_nans:
        names_pre = _extract_feature_names(X, feature_names)
        X_pre = _to_numpy_X(X)
        X_dropped, names_dropped, _nan_row_mask, _nan_col_mask = _smart_drop_nans(
            X_pre, names_pre, random_state=random_state, verbose=verbose
        )
        _nan_dropping_result = {
            "original_n_samples": int(X_pre.shape[0]),
            "original_n_features": int(X_pre.shape[1]),
            "retained_n_samples": int(X_dropped.shape[0]),
            "retained_n_features": int(X_dropped.shape[1]),
            "retained_feature_names": names_dropped.tolist(),
        }
        y_raw = np.asarray(y).ravel()[_nan_row_mask]
        groups_raw = np.asarray(groups).ravel()[_nan_row_mask] if groups is not None else None
        X_arr, y_arr, names, resolved, mapping, inverse_mapping, class_names, groups_arr = _validate_inputs(
            X_dropped, y_raw, names_dropped, outer_cv, inner_cv, task_type, groups_raw
        )
    else:
        X_arr, y_arr, names, resolved, mapping, inverse_mapping, class_names, groups_arr = _validate_inputs(
            X, y, feature_names, outer_cv, inner_cv, task_type, groups
        )

    n_classes = len(class_names) if class_names is not None else None
    if verbose:
        print("Running RobustModelMaker v0.3")
        print(f"  Task: {resolved}")
        print(f"  Algorithm: {alg}")
        print(f"  Samples: {X_arr.shape[0]}, features: {X_arr.shape[1]}")
        print(f"  Repeats: {repeated_outer_cv}")
        print(f"  Calibration: {calibration}")
        if not preserve_nans and _nan_dropping_result is not None:
            d = _nan_dropping_result
            print(f"  NaN strategy: {d['original_n_samples']}→{d['retained_n_samples']} rows, "
                  f"{d['original_n_features']}→{d['retained_n_features']} features")
        print()

    nested = nested_cross_validation(
        X_arr, y_arr, names, alg=alg, task_type=resolved,
        outer_cv=outer_cv, inner_cv=inner_cv, repeated_outer_cv=repeated_outer_cv,
        n_iter=n_iter, stability_threshold=stability_threshold, n_bootstrap=n_bootstrap,
        random_state=random_state, preprocess=preprocess, calibration=calibration,
        groups=groups_arr, n_jobs=n_jobs, verbose=verbose,
    )

    final_pre = _make_preprocessor(preprocess, alg)
    X_proc = final_pre.fit_transform(X_arr)
    stability = stability_selection(
        X_proc, y_arr, names, alg=alg, task_type=resolved,
        n_bootstrap=n_bootstrap, threshold=stability_threshold,
        random_state=random_state, n_jobs=n_jobs,
    )
    selected_idx = stability.selected_indices
    selected_features = stability.selected_features
    if len(selected_idx) == 0:
        warnings.warn(f"No features selected at threshold {stability_threshold}. Using top 5 features.")
        selected_idx = np.argsort(stability.selection_frequencies)[-min(5, len(names)):]
        selected_features = names[selected_idx]
    X_sel = X_proc[:, selected_idx]
    inner_splitter = _make_inner_splitter(resolved, inner_cv, random_state, groups_arr)
    model, param_dist = get_algorithm_config(alg, resolved, random_state=random_state, n_jobs=n_jobs, n_classes=n_classes)
    search = RandomizedSearchCV(
        model, param_dist, n_iter=n_iter, cv=inner_splitter,
        scoring=_default_scoring(resolved), n_jobs=n_jobs,
        random_state=random_state, refit=True,
    )
    fit_kwargs = {"groups": groups_arr} if groups_arr is not None else {}
    search.fit(X_sel, y_arr, **fit_kwargs)
    final_base = clone(search.best_estimator_)
    final_base.set_params(**search.best_params_)
    calibration_cv = _make_inner_splitter(resolved, min(3, inner_cv), random_state, groups_arr) if calibration != "none" else None
    robust_model = _fit_calibrated_if_needed(final_base, X_sel, y_arr, resolved, calibration, calibration_cv, groups_arr)

    cutoff = None
    if resolved == "binary":
        cutoff = determine_cutoff(y_arr, nested.outer_predictions, spec, cutoff_n_bootstrap, random_state=random_state)

    result = PipelineResult(
        nested_cv_result=nested,
        stability_result=stability,
        cutoff_result=cutoff,
        robust_model=robust_model,
        selected_features=selected_features,
        selected_feature_indices=selected_idx,
        algorithm=alg,
        task_type=resolved,
        preprocessor=final_pre,
        feature_names=names,
        label_mapping=mapping,
        inverse_label_mapping=inverse_mapping,
        class_names=class_names,
        calibration=calibration,
        preserve_nans=preserve_nans,
        nan_dropping_col_mask=_nan_col_mask,
        nan_dropping_result=_nan_dropping_result,
    )
    if (X_validation is None) ^ (y_validation is None):
        raise ValueError("Provide both X_validation and y_validation, or neither.")
    if X_validation is not None and y_validation is not None:
        result.validation_result = result.evaluate_verification(X_validation, y_validation)
    if verbose:
        print(result.summary())
    return result


class RobustModelMaker:
    """Estimator-style interface for RobustModelMaker v0.3."""

    def __init__(
        self,
        alg: Algorithm = "eln",
        task_type: TaskType = "auto",
        spec: float = 0.98,
        outer_cv: int = 10,
        inner_cv: int = 10,
        repeated_outer_cv: int = 1,
        n_iter: int = 100,
        stability_threshold: float = 0.7,
        n_bootstrap: int = 100,
        cutoff_n_bootstrap: int = 1000,
        random_state: Optional[int] = 42,
        preprocess: PreprocessMode = "auto",
        calibration: CalibrationMode = "none",
        n_jobs: int = -1,
        verbose: bool = True,
    ) -> None:
        self.alg = alg
        self.task_type = task_type
        self.spec = spec
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
        self.repeated_outer_cv = repeated_outer_cv
        self.n_iter = n_iter
        self.stability_threshold = stability_threshold
        self.n_bootstrap = n_bootstrap
        self.cutoff_n_bootstrap = cutoff_n_bootstrap
        self.random_state = random_state
        self.preprocess = preprocess
        self.calibration = calibration
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.result_: Optional[PipelineResult] = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, List[Any]],
        feature_names: Optional[np.ndarray] = None,
        groups: Optional[Union[np.ndarray, pd.Series, List[Any]]] = None,
        X_validation: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_validation: Optional[Union[np.ndarray, pd.Series, List[Any]]] = None,
    ) -> "RobustModelMaker":
        self.result_ = run_pipeline(
            X=X, y=y, feature_names=feature_names, alg=self.alg,
            task_type=self.task_type, spec=self.spec, outer_cv=self.outer_cv,
            inner_cv=self.inner_cv, repeated_outer_cv=self.repeated_outer_cv,
            n_iter=self.n_iter, stability_threshold=self.stability_threshold,
            n_bootstrap=self.n_bootstrap, cutoff_n_bootstrap=self.cutoff_n_bootstrap,
            random_state=self.random_state, preprocess=self.preprocess,
            calibration=self.calibration, groups=groups,
            X_validation=X_validation, y_validation=y_validation,
            n_jobs=self.n_jobs, verbose=self.verbose,
        )
        return self

    def _check_fitted(self) -> PipelineResult:
        if self.result_ is None:
            raise NotFittedError("Call fit(X, y) before prediction or evaluation.")
        return self.result_

    def predict(self, X: Union[np.ndarray, pd.DataFrame], cutoff: Optional[float] = None):
        return self._check_fitted().predict(X, cutoff=cutoff)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]):
        return self._check_fitted().predict_proba(X)

    def evaluate_verification(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series, List[Any]], cutoff: Optional[float] = None) -> VerificationResult:
        return self._check_fitted().evaluate_verification(X, y, cutoff=cutoff)

    def permutation_importance(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series, List[Any]], **kwargs) -> PermutationImportanceResult:
        return self._check_fitted().permutation_importance(X, y, **kwargs)

    def export_shap_ready(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.Series, List[Any]]] = None) -> Dict[str, Any]:
        return self._check_fitted().export_shap_ready(X, y)

    def plot_feature_stability(self, top_n: int = 30):
        return self._check_fitted().plot_feature_stability(top_n=top_n)

    def summary(self) -> str:
        return self._check_fitted().summary()


def print_pipeline_results(result: PipelineResult) -> None:
    """Compact console summary."""
    print(result.summary())
    print("\nFeature stability:")
    print(result.stability_result.summary().head(20).to_string(index=False))
    if result.validation_result is not None:
        print("\nExternal validation:")
        print(result.validation_result.summary().to_string(index=False))

# -----------------------------------------------------------------------------
# v0.3 compatibility and reporting/performance patches
# -----------------------------------------------------------------------------
# These assignments keep the modelling path intact while restoring the reporting,
# saving and performance-test API expected by the v0.3 test suites.

_ROBUST_ORIGINAL_RUN_PIPELINE = run_pipeline
_ROBUST_ORIGINAL_INIT = RobustModelMaker.__init__

def _robust_pipeline_result_mean_score(self):
    return float(self.nested_cv_result.mean_score)

def _robust_pipeline_result_std_score(self):
    return float(self.nested_cv_result.std_score)

PipelineResult.mean_score = property(_robust_pipeline_result_mean_score)
PipelineResult.std_score = property(_robust_pipeline_result_std_score)


def _robust_result_tables(self) -> Dict[str, pd.DataFrame]:
    rows: List[Dict[str, Any]] = [
        {"section": "top_level", "attribute": "task_type", "value": self.task_type, "description": "Resolved modelling task"},
        {"section": "top_level", "attribute": "algorithm", "value": self.algorithm, "description": "Algorithm used"},
        {"section": "top_level", "attribute": "calibration", "value": self.calibration, "description": "Probability calibration method"},
        {"section": "top_level", "attribute": "n_selected_features", "value": len(self.selected_features), "description": "Number of consensus selected features"},
        {"section": "nested_cv", "attribute": "metric", "value": self.nested_cv_result.metric_name, "description": "Primary nested CV scoring metric"},
        {"section": "nested_cv", "attribute": "mean_score", "value": self.nested_cv_result.mean_score, "description": "Mean outer-fold score"},
        {"section": "nested_cv", "attribute": "std_score", "value": self.nested_cv_result.std_score, "description": "Standard deviation across outer folds"},
        {"section": "nested_cv", "attribute": "repeats", "value": getattr(self.nested_cv_result, "repeats", getattr(self.nested_cv_result, "repeated_outer_cv", 1)), "description": "Number of repeated outer CV runs"},
    ]
    if self.cutoff_result is not None:
        rows.extend([
            {"section": "cutoff", "attribute": "target_specificity", "value": self.cutoff_result.target_specificity, "description": "Requested binary specificity"},
            {"section": "cutoff", "attribute": "cutoff_median", "value": self.cutoff_result.cutoff_median, "description": "Median bootstrap decision cutoff"},
            {"section": "cutoff", "attribute": "cutoff_ci_lower", "value": self.cutoff_result.cutoff_ci_lower, "description": "Lower bootstrap confidence bound"},
            {"section": "cutoff", "attribute": "cutoff_ci_upper", "value": self.cutoff_result.cutoff_ci_upper, "description": "Upper bootstrap confidence bound"},
            {"section": "cutoff", "attribute": "achieved_specificity", "value": self.cutoff_result.achieved_specificity, "description": "Specificity at median cutoff"},
            {"section": "cutoff", "attribute": "achieved_sensitivity", "value": self.cutoff_result.achieved_sensitivity, "description": "Sensitivity at median cutoff"},
        ])
    overview = pd.DataFrame(rows)

    outer_pred = np.asarray(self.nested_cv_result.outer_predictions)
    nested_pred_df = pd.DataFrame({"y_true": np.asarray(self.nested_cv_result.outer_true_labels)})
    if outer_pred.ndim == 1:
        nested_pred_df["prediction_or_score"] = outer_pred
    elif outer_pred.ndim == 2:
        for i in range(outer_pred.shape[1]):
            label = self.class_names[i] if getattr(self, "class_names", None) is not None and i < len(self.class_names) else i
            nested_pred_df[f"score_class_{label}"] = outer_pred[:, i]
    else:
        nested_pred_df["prediction_or_score"] = outer_pred.reshape(len(nested_pred_df), -1).tolist()

    tables: Dict[str, pd.DataFrame] = {
        "overview": overview,
        "selected_features": pd.DataFrame({"rank": np.arange(1, len(self.selected_features) + 1), "feature": self.selected_features}),
        "stability_selection": self.stability_result.summary(),
        "feature_stability_cv": self.nested_cv_result.feature_stability.reset_index(drop=True),
        "nested_cv_scores": pd.DataFrame({"fold_or_repeat_fold": np.arange(1, len(self.nested_cv_result.outer_scores) + 1), self.nested_cv_result.metric_name: self.nested_cv_result.outer_scores}),
        "nested_cv_predictions": nested_pred_df,
    }
    if self.cutoff_result is not None:
        tables["cutoff_distribution"] = pd.DataFrame({"cutoff": self.cutoff_result.cutoff_distribution})
    if self.validation_result is not None:
        val = self.validation_result
        tables["external_validation"] = val.summary()
        tables["external_validation_metrics"] = val.summary()
        pred_df = pd.DataFrame({"prediction": val.predictions})
        if val.probabilities is not None:
            probs = np.asarray(val.probabilities)
            if probs.ndim == 1:
                pred_df["score_or_probability"] = probs
            elif probs.ndim == 2:
                for i in range(probs.shape[1]):
                    label = self.class_names[i] if getattr(self, "class_names", None) is not None and i < len(self.class_names) else i
                    pred_df[f"probability_class_{label}"] = probs[:, i]
        tables["external_validation_predictions"] = pred_df
        if val.confusion is not None:
            tables["external_validation_confusion_matrix"] = pd.DataFrame(val.confusion)
    return tables

PipelineResult.results_tables = _robust_result_tables


def _robust_pipeline_result_save_results(self, output_dir: Union[str, Path] = "robust_model_results", prefix: str = "robust_model") -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for name, table in self.results_tables().items():
        table.to_csv(output_path / f"{prefix}_{name}.csv", index=False)
    metadata = {
        "task_type": self.task_type,
        "algorithm": self.algorithm,
        "calibration": self.calibration,
        "metric_name": self.nested_cv_result.metric_name,
        "mean_score": float(self.nested_cv_result.mean_score),
        "std_score": float(self.nested_cv_result.std_score),
        "nested_cv_metric": self.nested_cv_result.metric_name,
        "nested_cv_mean_score": float(self.nested_cv_result.mean_score),
        "nested_cv_std_score": float(self.nested_cv_result.std_score),
        "n_selected_features": int(len(self.selected_features)),
        "selected_features": [str(f) for f in self.selected_features],
        "has_external_validation": self.validation_result is not None,
    }
    with open(output_path / f"{prefix}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(output_path / f"{prefix}_summary.txt", "w", encoding="utf-8") as f:
        f.write(self.summary())
    with open(output_path / f"{prefix}_result.pkl", "wb") as f:
        pickle.dump(self, f)
    self.results_dir = str(output_path)
    return str(output_path)

PipelineResult.save_results = _robust_pipeline_result_save_results


def _robust_pipeline_result_print_results(self, top_n: int = 20) -> None:
    print("=" * 60)
    print("ROBUST MODEL MAKER RESULTS SUMMARY")
    print("=" * 60)
    print(self.summary())
    tables = self.results_tables()
    print("\n[TOP LEVEL]")
    print(tables["overview"].to_string(index=False))
    print("\n[NESTED CV RESULTS]")
    print(tables["nested_cv_scores"].to_string(index=False))
    print("\n[SELECTED FEATURES]")
    print(tables["selected_features"].head(top_n).to_string(index=False))

PipelineResult.print_results = _robust_pipeline_result_print_results


def _robust_pipeline_result_permutation_importance(self, X, y, n_repeats: int = 20, scoring: Optional[str] = None, random_state: Optional[int] = 42, n_jobs: int = -1, as_frame: bool = False):
    X_sel, _ = self._prepare_X_selected(X)
    y_encoded = _prepare_y(y, self.task_type, label_mapping=self.label_mapping)
    if scoring is None:
        scoring = _default_scoring(self.task_type)
    imp = sklearn_permutation_importance(
        self.robust_model,
        X_sel,
        y_encoded,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=n_jobs,
    )
    result = PermutationImportanceResult(
        importances_mean=imp.importances_mean,
        importances_std=imp.importances_std,
        importances=imp.importances,
        feature_names=self.selected_features,
        scoring=scoring,
    )
    if as_frame:
        return result.summary()
    return result

PipelineResult.permutation_importance = _robust_pipeline_result_permutation_importance


def run_pipeline(*args, save_results: bool = False, output_dir: Union[str, Path] = "robust_model_results", output_prefix: str = "robust_model", preserve_nans: bool = True, **kwargs):
    result = _ROBUST_ORIGINAL_RUN_PIPELINE(*args, preserve_nans=preserve_nans, **kwargs)
    if save_results:
        result.save_results(output_dir=output_dir, prefix=output_prefix)
    return result


def _robust_init(self, alg: Algorithm = "eln", task_type: TaskType = "auto", spec: float = 0.98, outer_cv: int = 10, inner_cv: int = 10, repeated_outer_cv: int = 1, n_iter: int = 100, stability_threshold: float = 0.7, n_bootstrap: int = 100, cutoff_n_bootstrap: int = 1000, random_state: Optional[int] = 42, preprocess: PreprocessMode = "auto", calibration: CalibrationMode = "none", n_jobs: int = -1, verbose: bool = True, save_results: bool = False, output_dir: Union[str, Path] = "robust_model_results", output_prefix: str = "robust_model", preserve_nans: bool = True) -> None:
    self.alg = alg
    self.task_type = task_type
    self.spec = spec
    self.outer_cv = outer_cv
    self.inner_cv = inner_cv
    self.repeated_outer_cv = repeated_outer_cv
    self.n_iter = n_iter
    self.stability_threshold = stability_threshold
    self.n_bootstrap = n_bootstrap
    self.cutoff_n_bootstrap = cutoff_n_bootstrap
    self.random_state = random_state
    self.preprocess = preprocess
    self.calibration = calibration
    self.n_jobs = n_jobs
    self.verbose = verbose
    self.save_results_auto = save_results
    self.output_dir = output_dir
    self.output_prefix = output_prefix
    self.preserve_nans = preserve_nans
    self.result_: Optional[PipelineResult] = None

RobustModelMaker.__init__ = _robust_init


def _robust_fit(self, X, y, feature_names: Optional[np.ndarray] = None, groups: Optional[Union[np.ndarray, pd.Series, List[Any]]] = None, X_validation: Optional[Union[np.ndarray, pd.DataFrame]] = None, y_validation: Optional[Union[np.ndarray, pd.Series, List[Any]]] = None):
    self.result_ = run_pipeline(
        X=X, y=y, feature_names=feature_names, alg=self.alg,
        task_type=self.task_type, spec=self.spec, outer_cv=self.outer_cv,
        inner_cv=self.inner_cv, repeated_outer_cv=self.repeated_outer_cv,
        n_iter=self.n_iter, stability_threshold=self.stability_threshold,
        n_bootstrap=self.n_bootstrap, cutoff_n_bootstrap=self.cutoff_n_bootstrap,
        random_state=self.random_state, preprocess=self.preprocess,
        calibration=self.calibration, groups=groups,
        X_validation=X_validation, y_validation=y_validation,
        n_jobs=self.n_jobs, verbose=self.verbose,
        save_results=self.save_results_auto,
        output_dir=self.output_dir,
        output_prefix=self.output_prefix,
        preserve_nans=self.preserve_nans,
    )
    return self

RobustModelMaker.fit = _robust_fit


def _robust_maker_permutation_importance(self, X, y, **kwargs):
    return self._check_fitted().permutation_importance(X, y, **kwargs)

RobustModelMaker.permutation_importance = _robust_maker_permutation_importance


def _robust_maker_results_tables(self):
    return self._check_fitted().results_tables()

RobustModelMaker.results_tables = _robust_maker_results_tables


def _robust_maker_save(self, output_dir: Union[str, Path] = "robust_model_results", prefix: str = "robust_model") -> str:
    return self._check_fitted().save_results(output_dir=output_dir, prefix=prefix)

RobustModelMaker.save = _robust_maker_save
RobustModelMaker.save_results = _robust_maker_save


def _robust_maker_print_results(self, top_n: int = 20) -> None:
    self._check_fitted().print_results(top_n=top_n)

RobustModelMaker.print_results = _robust_maker_print_results


def print_pipeline_results(result: PipelineResult, top_n: int = 20) -> None:
    result.print_results(top_n=top_n)


# -----------------------------------------------------------------------------
# Expanded algorithm registry and stability-selection compatibility patch
# -----------------------------------------------------------------------------
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, LinearSVR


def get_algorithm_config(alg: str, task_type: ResolvedTask = "binary", random_state: Optional[int] = 42, n_jobs: int = -1, n_classes: Optional[int] = None) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """Return estimator and RandomizedSearchCV search space for all v0.3 algorithms."""
    if alg == "lin" and task_type != "regression":
        raise ValueError("alg='lin' is only valid for regression.")
    if alg == "log" and task_type == "regression":
        raise ValueError("alg='log' is only valid for classification.")

    if alg == "eln":
        if task_type == "regression":
            return ElasticNet(random_state=random_state, max_iter=10000), {"alpha": loguniform(1e-4, 1e2), "l1_ratio": uniform(0, 1)}
        return LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=5000, random_state=random_state, n_jobs=1), {"C": loguniform(1e-4, 1e2), "l1_ratio": uniform(0, 1)}

    if alg == "rf":
        params = {
            "n_estimators": randint(20, 100),
            "max_depth": randint(2, 12),
            "min_samples_split": randint(2, 12),
            "min_samples_leaf": randint(1, 6),
            "max_features": ["sqrt", "log2", None],
        }
        if task_type == "regression":
            return RandomForestRegressor(random_state=random_state, n_jobs=n_jobs), params
        return RandomForestClassifier(random_state=random_state, n_jobs=n_jobs, class_weight="balanced_subsample"), params

    if alg == "xgb":
        if not _HAS_XGBOOST:
            raise ImportError("xgboost is not installed.")
        params = {
            "n_estimators": randint(20, 100),
            "max_depth": randint(2, 8),
            "learning_rate": loguniform(1e-2, 3e-1),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
        }
        if task_type == "regression":
            return XGBRegressor(random_state=random_state, n_jobs=n_jobs, tree_method="hist", objective="reg:squarederror"), params
        objective = "binary:logistic" if task_type == "binary" else "multi:softprob"
        return XGBClassifier(random_state=random_state, n_jobs=n_jobs, eval_metric="auc" if task_type == "binary" else "mlogloss", tree_method="hist", objective=objective, num_class=n_classes if task_type == "multiclass" else None), params

    if alg == "mlp":
        params = {"hidden_layer_sizes": [(16,), (32,)], "alpha": loguniform(1e-5, 1e-2), "learning_rate_init": loguniform(1e-4, 1e-2)}
        if task_type == "regression":
            return MLPRegressor(max_iter=300, random_state=random_state, early_stopping=False), params
        return MLPClassifier(max_iter=300, random_state=random_state, early_stopping=False), params

    if alg == "svm":
        if task_type == "regression":
            return LinearSVR(random_state=random_state, max_iter=5000), {"C": loguniform(1e-3, 1e2), "epsilon": loguniform(1e-3, 1)}
        return SVC(kernel="linear", probability=True, random_state=random_state, class_weight="balanced"), {"C": loguniform(1e-3, 1e2)}

    if alg == "rdg":
        if task_type == "regression":
            return Ridge(random_state=random_state), {"alpha": loguniform(1e-4, 1e2)}
        return LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000, random_state=random_state, class_weight="balanced"), {"C": loguniform(1e-3, 1e2)}

    if alg == "las":
        if task_type == "regression":
            return Lasso(random_state=random_state, max_iter=10000), {"alpha": loguniform(1e-4, 1e1)}
        return LogisticRegression(penalty="l1", solver="saga", max_iter=5000, random_state=random_state, class_weight="balanced"), {"C": loguniform(1e-3, 1e2)}

    if alg == "log":
        if task_type == "regression":
            raise ValueError("alg='log' is only valid for classification.")
        return LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000, random_state=random_state, class_weight="balanced"), {"C": loguniform(1e-3, 1e2)}

    if alg == "lin":
        return LinearRegression(), {"fit_intercept": [True]}

    raise ValueError("alg must be one of 'eln', 'rf', 'xgb', 'mlp', 'svm', 'rdg', 'las', 'log', or 'lin'.")


def _robust_model_importance(model: BaseEstimator, X_eval: np.ndarray, y_eval: np.ndarray, n_features: int, seed: Optional[int]) -> np.ndarray:
    """Native importance where available, permutation fallback otherwise."""
    base = model
    if isinstance(base, CalibratedClassifierCV):
        try:
            base = base.calibrated_classifiers_[0].estimator
        except Exception:
            pass
    if hasattr(base, "feature_importances_"):
        imp = np.asarray(base.feature_importances_, dtype=float)
    elif hasattr(base, "coef_"):
        coef = np.asarray(base.coef_, dtype=float)
        imp = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
    elif hasattr(base, "coefs_"):
        imp = np.mean(np.abs(base.coefs_[0]), axis=1)
    else:
        perm = sklearn_permutation_importance(base, X_eval, y_eval, n_repeats=2, random_state=seed, n_jobs=1)
        imp = np.asarray(perm.importances_mean, dtype=float)
    if imp.shape[0] != n_features:
        imp = np.resize(imp, n_features)
    return np.nan_to_num(imp)


def stability_selection(X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series, List[Any]], feature_names: Optional[np.ndarray] = None, alg: str = "xgb", task_type: TaskType = "auto", n_bootstrap: int = 100, sample_fraction: float = 0.7, threshold: float = 0.7, random_state: Optional[int] = 42, n_jobs: int = -1) -> StabilitySelectionResult:
    """Bootstrap stability selection supporting all v0.3 algorithms."""
    set_global_seed(random_state)
    names = _extract_feature_names(X, feature_names)
    X_arr = _to_numpy_X(X)
    resolved = _resolve_task_type(y, task_type)
    label_mapping, _, _ = _make_label_mapping(y, resolved)
    y_arr = _prepare_y(y, resolved, label_mapping)
    if not (0 < sample_fraction <= 1):
        raise ValueError("sample_fraction must be in (0, 1].")
    if not (0 < threshold <= 1):
        raise ValueError("threshold must be in (0, 1].")
    rng = np.random.RandomState(random_state)
    n_samples, n_features = X_arr.shape
    subsample_size = max(2, int(round(n_samples * sample_fraction)))
    selection_counts = np.zeros(n_features, dtype=float)

    for b in range(int(n_bootstrap)):
        seed_b = None if random_state is None else int(random_state + 10000 + b)
        if resolved == "regression":
            idx = rng.choice(np.arange(n_samples), size=min(subsample_size, n_samples), replace=False)
        else:
            idx_parts = []
            for cls in np.unique(y_arr):
                cls_idx = np.where(y_arr == cls)[0]
                take = max(1, int(round(subsample_size * len(cls_idx) / n_samples)))
                idx_parts.extend(rng.choice(cls_idx, size=min(take, len(cls_idx)), replace=False))
            idx = np.asarray(idx_parts, dtype=int)
        X_sub, y_sub = X_arr[idx], y_arr[idx]
        pre = _make_preprocessor("auto", alg).fit(X_sub)
        X_fit = pre.transform(X_sub)
        model, _ = get_algorithm_config(alg, resolved, random_state=seed_b, n_jobs=n_jobs, n_classes=len(np.unique(y_arr)) if resolved != "regression" else None)
        # Fixed lightweight settings for stability resamples.
        try:
            if alg == "eln":
                if resolved == "regression":
                    model.set_params(alpha=0.01, l1_ratio=0.7)
                else:
                    model.set_params(C=1.0, l1_ratio=0.7)
            elif alg == "rf":
                model.set_params(n_estimators=80, max_depth=10)
            elif alg == "xgb":
                model.set_params(n_estimators=80, max_depth=6, learning_rate=0.05)
        except ValueError:
            pass
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_fit, y_sub)
        importance = _robust_model_importance(model, X_fit, y_sub, n_features, seed_b)
        if np.allclose(importance, 0):
            selected = np.zeros(n_features, dtype=bool)
            selected[int(np.argmax(importance))] = True
        else:
            selected = importance > np.median(importance)
            if not np.any(selected):
                selected[int(np.argmax(importance))] = True
        selection_counts += selected.astype(float)

    frequency = selection_counts / float(n_bootstrap)
    selected_indices = np.where(frequency >= threshold)[0]
    if len(selected_indices) == 0:
        selected_indices = np.argsort(frequency)[-min(3, n_features):]
    return StabilitySelectionResult(
        feature_names=names,
        selection_frequencies=frequency,
        selected_features=names[selected_indices],
        selected_indices=selected_indices,
        threshold=threshold,
        n_bootstrap=int(n_bootstrap),
        task_type=resolved,
    )

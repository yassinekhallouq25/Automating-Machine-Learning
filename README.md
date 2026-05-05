"""
pricing_checks/checks/model_evaluation/model_performance_comparison.py
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.core.checks import DatasetKind
from deepchecks.tabular import SingleDatasetCheck
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOWER_IS_BETTER: set = {"log_loss"}

_DISPLAY_NAMES: Dict[str, str] = {
    "roc_auc":   "ROC AUC",
    "log_loss":  "Log Loss",
    "f1":        "F1 Score (macro)",
    "precision": "Precision (macro)",
    "recall":    "Recall (macro)",
    "accuracy":  "Accuracy",
}

_DEFAULT_METRICS: List[str] = list(_DISPLAY_NAMES.keys())

BaselineType = Literal["logistic_regression", "random_forest"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_baseline(baseline: BaselineType):
    """Return an unfitted baseline estimator based on the chosen type."""
    if baseline == "logistic_regression":
        return CalibratedClassifierCV(
            LogisticRegression(max_iter=1000, random_state=42),
            method="sigmoid",   # Platt scaling
            cv=5,
        )
    elif baseline == "random_forest":
        return CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=100, random_state=42),
            method="sigmoid",   # Platt scaling
            cv=5,
        )
    else:
        raise ValueError(
            f"Unknown baseline '{baseline}'. "
            "Choose 'logistic_regression' or 'random_forest'."
        )


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    metrics: List[str],
    classes: np.ndarray,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    multiclass = len(classes) > 2

    for m in metrics:
        if m == "accuracy":
            out[m] = accuracy_score(y_true, y_pred)
        elif m == "f1":
            out[m] = f1_score(y_true, y_pred, average="macro", zero_division=0)
        elif m == "precision":
            out[m] = precision_score(y_true, y_pred, average="macro", zero_division=0)
        elif m == "recall":
            out[m] = recall_score(y_true, y_pred, average="macro", zero_division=0)
        elif m == "roc_auc":
            if y_proba is None:
                raise ValueError("roc_auc requires predict_proba().")
            out[m] = (
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                if multiclass
                else roc_auc_score(y_true, y_proba[:, 1])
            )
        elif m == "log_loss":
            if y_proba is None:
                raise ValueError("log_loss requires predict_proba().")
            out[m] = log_loss(y_true, y_proba)
        else:
            raise ValueError(
                f"Unknown metric '{m}'. Supported: {list(_DISPLAY_NAMES.keys())}"
            )

    return out


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------

class ModelPerformanceComparison(SingleDatasetCheck):
    """Compare a candidate model's classification performance to a built-in baseline.

    The baseline is trained on the same dataset at run time (fit + evaluate on
    the same split — useful for a quick sanity check). For a proper evaluation
    pass pre-split train/test datasets via a Suite.

    Parameters
    ----------
    baseline : 'logistic_regression' | 'random_forest'
        Which baseline model to use. Both are Platt-scaled (CalibratedClassifierCV
        with method='sigmoid') so they produce well-calibrated probabilities for
        ROC AUC and Log Loss.
    metrics :
        Metrics to evaluate. Defaults to all six:
        roc_auc, log_loss, f1, precision, recall, accuracy.
    threshold :
        Max relative degradation allowed before a condition fails (default 0.05 = 5%).
        For log_loss (lower-is-better): candidate <= base * (1 + threshold).
        For all others: candidate >= base * (1 - threshold).

    Usage
    -----
    from deepchecks.core import Suite
    from pricing_checks.checks.model_evaluation.model_performance_comparison import (
        ModelPerformanceComparison,
    )

    suite = Suite(
        "Pricing Model Suite",
        ModelPerformanceComparison(
            baseline="logistic_regression",   # or "random_forest"
            threshold=0.05,
        ).add_condition_performance_not_degraded()
    )
    suite.run(ds_test, model=candidate_model).show()
    """

    def __init__(
        self,
        baseline: BaselineType = "logistic_regression",
        metrics: Optional[List[str]] = None,
        threshold: float = 0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.baseline      = baseline
        self.metrics       = metrics if metrics is not None else _DEFAULT_METRICS
        self.threshold     = threshold

    # ------------------------------------------------------------------
    # run_logic
    # ------------------------------------------------------------------

    def run_logic(self, context, dataset_kind=DatasetKind.TRAIN) -> CheckResult:
        dataset = context.train
        model   = context.model

        X      = dataset.features_columns
        y_true = np.asarray(dataset.data[dataset.label_name])
        classes = np.unique(y_true)

        # ── Fit baseline on the same data ────────────────────────────
        base_model = _build_baseline(self.baseline)
        base_model.fit(X, y_true)

        # ── Predictions ──────────────────────────────────────────────
        cand_pred = np.asarray(model.predict(X))
        base_pred = np.asarray(base_model.predict(X))

        needs_proba = any(m in self.metrics for m in ("roc_auc", "log_loss"))
        cand_proba = base_proba = None

        if needs_proba:
            if not hasattr(model, "predict_proba"):
                raise ValueError("Candidate model has no predict_proba().")
            cand_proba = np.asarray(model.predict_proba(X))
            base_proba = np.asarray(base_model.predict_proba(X))

        # ── Compute scores ───────────────────────────────────────────
        cand_scores = _compute_metrics(y_true, cand_pred, cand_proba, self.metrics, classes)
        base_scores = _compute_metrics(y_true, base_pred, base_proba, self.metrics, classes)

        return CheckResult(
            value={
                "candidate_scores": cand_scores,
                "base_scores":      base_scores,
                "baseline_type":    self.baseline,
            },
            display=[],   # no additional output — conditions summary is enough
            header="Model Performance Comparison vs Base Model",
        )

    # ------------------------------------------------------------------
    # Conditions — one per metric
    # ------------------------------------------------------------------

    def add_condition_performance_not_degraded(
        self, threshold: Optional[float] = None
    ) -> "ModelPerformanceComparison":
        """One condition per metric → one row each in the Conditions Summary table."""
        _t = threshold if threshold is not None else self.threshold

        for metric in self.metrics:
            def _make_condition(m: str, t: float):
                lower = m in _LOWER_IS_BETTER

                def _condition(value: dict) -> ConditionResult:
                    cv = value["candidate_scores"][m]
                    bv = value["base_scores"][m]
                    passed = (cv <= bv * (1 + t)) if lower else (cv >= bv * (1 - t))
                    delta  = (bv - cv) if lower else (cv - bv)
                    more_info = (
                        f"base={bv:.4f}  candidate={cv:.4f}  Δ={delta:+.4f}"
                    )
                    return ConditionResult(
                        ConditionCategory.PASS if passed else ConditionCategory.FAIL,
                        more_info,
                    )

                return _condition

            condition_name = (
                f"{_DISPLAY_NAMES.get(metric, metric)}: "
                f"candidate not worse than base by >{_t*100:.0f}% "
                f"({'lower' if metric in _LOWER_IS_BETTER else 'higher'} is better)"
            )
            self.add_condition(condition_name, _make_condition(metric, _t))

        return self

"""
pricing_checks/checks/model_evaluation/model_performance_comparison.py
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from deepchecks.core import BaseCheck, CheckResult, ConditionCategory, ConditionResult
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


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

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
            raise ValueError(f"Unknown metric '{m}'. Supported: {list(_DISPLAY_NAMES.keys())}")

    return out


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------

class ModelPerformanceComparison(BaseCheck):
    """Compare a candidate model's classification performance to a base model.

    Each metric appears as its own row in Deepchecks' native
    Conditions Summary table (Status / Check / Condition / More Info).

    Call ``add_condition_performance_not_degraded()`` to register all
    per-metric conditions before running.

    Parameters
    ----------
    base_model :
        Fitted sklearn-compatible classifier.
    metrics :
        Metrics to evaluate. Defaults to all six:
        roc_auc, log_loss, f1, precision, recall, accuracy.
    threshold :
        Max relative degradation allowed (default 0.05 = 5 %).
    """

    def __init__(
        self,
        base_model,
        metrics: Optional[List[str]] = None,
        threshold: float = 0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.metrics: List[str] = metrics if metrics is not None else _DEFAULT_METRICS
        self.threshold = threshold

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def run(self, dataset, model=None, **kwargs) -> CheckResult:
        if model is None:
            raise ValueError("Pass the candidate model via model=<your_model>.")

        X       = dataset.features_columns
        y_true  = np.asarray(dataset.data[dataset.label_name])
        classes = np.unique(y_true)

        cand_pred = np.asarray(model.predict(X))
        base_pred = np.asarray(self.base_model.predict(X))

        needs_proba = any(m in self.metrics for m in ("roc_auc", "log_loss"))
        cand_proba = base_proba = None

        if needs_proba:
            if not hasattr(model, "predict_proba"):
                raise ValueError("Candidate model has no predict_proba().")
            if not hasattr(self.base_model, "predict_proba"):
                raise ValueError("Base model has no predict_proba().")
            cand_proba = np.asarray(model.predict_proba(X))
            base_proba = np.asarray(self.base_model.predict_proba(X))

        cand_scores = _compute_metrics(y_true, cand_pred, cand_proba, self.metrics, classes)
        base_scores = _compute_metrics(y_true, base_pred, base_proba, self.metrics, classes)

        # Plain dataframe shown under "Check With Conditions Output"
        rows = []
        for metric in self.metrics:
            cv, bv = cand_scores[metric], base_scores[metric]
            lower  = metric in _LOWER_IS_BETTER
            delta  = (bv - cv) if lower else (cv - bv)
            rows.append({
                "Metric":             _DISPLAY_NAMES.get(metric, metric),
                "Base Model":         round(bv, 6),
                "Candidate Model":    round(cv, 6),
                "Δ (Candidate−Base)": round(delta, 6),
                "Direction":          "↓ lower is better" if lower else "↑ higher is better",
            })

        display_df = pd.DataFrame(rows).set_index("Metric")

        result = CheckResult(
            value={"candidate_scores": cand_scores, "base_scores": base_scores},
            display=[display_df],
            header="Model Performance Comparison vs Base Model",
        )
        result.check = self
        return result

    # ------------------------------------------------------------------
    # Conditions — ONE per metric → one row each in Conditions Summary
    # ------------------------------------------------------------------

    def add_condition_performance_not_degraded(
        self, threshold: Optional[float] = None
    ) -> "ModelPerformanceComparison":
        """
        Register one condition per metric.
        Each metric gets its own row in the native Deepchecks
        Conditions Summary table:

            Status | Check                              | Condition                                      | More Info
            ✓      | Model Performance Comparison ...   | ROC AUC: candidate not worse than base by >5%  | base=0.8821  candidate=0.9012  Δ=+0.0191
            ✗      | Model Performance Comparison ...   | Log Loss: candidate not worse than base by >5% | base=0.3210  candidate=0.3890  Δ=-0.0680
            ...
        """
        _t = threshold if threshold is not None else self.threshold

        for metric in self.metrics:
            # We need a default-argument capture to avoid the closure-loop problem
            def _make_condition(m: str, t: float) -> callable:
                lower       = m in _LOWER_IS_BETTER
                name        = _DISPLAY_NAMES.get(m, m)
                direction   = "lower" if lower else "higher"

                def _condition(value: dict) -> ConditionResult:
                    cv = value["candidate_scores"][m]
                    bv = value["base_scores"][m]
                    passed = (
                        cv <= bv * (1 + t) if lower else cv >= bv * (1 - t)
                    )
                    delta = (bv - cv) if lower else (cv - bv)
                    more_info = (
                        f"base={bv:.4f}  candidate={cv:.4f}  "
                        f"Δ={delta:+.4f}  "
                        f"({'improvement' if delta >= 0 else 'degradation'})"
                    )
                    category = ConditionCategory.PASS if passed else ConditionCategory.FAIL
                    return ConditionResult(category, more_info)

                return _condition

            condition_name = (
                f"{_DISPLAY_NAMES.get(metric, metric)}: "
                f"candidate not worse than base by >{_t*100:.0f}% "
                f"({'lower' if metric in _LOWER_IS_BETTER else 'higher'} is better)"
            )
            self.add_condition(condition_name, _make_condition(metric, _t))

        return self

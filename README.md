"""
pricing_checks/checks/model_evaluation/model_performance_comparison.py

A custom Deepchecks check that compares a candidate model's classification
performance against a base/reference model and passes only when the candidate
meets or beats the base model on every required metric.

Suggested metrics for a classification task
--------------------------------------------
| Metric            | Why it matters                                          |
|-------------------|---------------------------------------------------------|
| ROC AUC           | Threshold-independent discriminative power              |
| Log Loss          | Calibration quality (lower = better)                   |
| F1 Score (macro)  | Balanced precision/recall across all classes           |
| Precision (macro) | How many predicted positives are truly positive        |
| Recall (macro)    | How many actual positives are captured                 |
| Accuracy          | Overall correctness baseline                           |

Usage
-----
from deepchecks.tabular import Dataset
from pricing_checks.checks.model_evaluation.model_performance_comparison import (
    ModelPerformanceComparison,
)

check = ModelPerformanceComparison(
    base_model=base_clf,
    metrics=["roc_auc", "log_loss", "f1", "precision", "recall", "accuracy"],
    # For log_loss: candidate must be <= base * (1 + threshold)
    # For all other metrics: candidate must be >= base * (1 - threshold)
    threshold=0.05,   # allow up to 5% degradation before failing
)
result = check.run(dataset, model=candidate_clf)
result.show()
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.tabular import Context, SingleDatasetCheck
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LOWER_IS_BETTER = {"log_loss"}  # all other supported metrics: higher is better

_METRIC_DISPLAY_NAMES: Dict[str, str] = {
    "roc_auc": "ROC AUC",
    "log_loss": "Log Loss",
    "f1": "F1 Score (macro)",
    "precision": "Precision (macro)",
    "recall": "Recall (macro)",
    "accuracy": "Accuracy",
}

_DEFAULT_METRICS = list(_METRIC_DISPLAY_NAMES.keys())


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    metrics: List[str],
    classes: np.ndarray,
) -> Dict[str, float]:
    """Return a dict of {metric_name: value} for the requested metrics."""
    results: Dict[str, float] = {}
    multi_class = len(classes) > 2

    for metric in metrics:
        if metric == "accuracy":
            results[metric] = accuracy_score(y_true, y_pred)

        elif metric == "f1":
            results[metric] = f1_score(y_true, y_pred, average="macro", zero_division=0)

        elif metric == "precision":
            results[metric] = precision_score(
                y_true, y_pred, average="macro", zero_division=0
            )

        elif metric == "recall":
            results[metric] = recall_score(
                y_true, y_pred, average="macro", zero_division=0
            )

        elif metric == "roc_auc":
            if y_proba is None:
                raise ValueError(
                    "ROC AUC requires probability estimates. "
                    "Make sure your model exposes predict_proba()."
                )
            if multi_class:
                results[metric] = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="macro"
                )
            else:
                results[metric] = roc_auc_score(y_true, y_proba[:, 1])

        elif metric == "log_loss":
            if y_proba is None:
                raise ValueError(
                    "Log Loss requires probability estimates. "
                    "Make sure your model exposes predict_proba()."
                )
            results[metric] = log_loss(y_true, y_proba)

        else:
            raise ValueError(
                f"Unsupported metric '{metric}'. "
                f"Supported metrics: {list(_METRIC_DISPLAY_NAMES.keys())}"
            )

    return results


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------


class ModelPerformanceComparison(SingleDatasetCheck):
    """Compare a candidate model's classification performance to a base model.

    The check computes a configurable set of metrics for both models on the
    same dataset and builds a comparison table.  An optional condition
    (``add_condition_performance_not_degraded``) will FAIL if the candidate
    model is worse than the base model by more than ``threshold`` on any
    metric.

    Parameters
    ----------
    base_model:
        A fitted scikit-learn–compatible classifier that implements
        ``predict()`` and, if ROC AUC / Log Loss are used, ``predict_proba()``.
    metrics:
        List of metric names to evaluate.  Defaults to all six metrics:
        ``["roc_auc", "log_loss", "f1", "precision", "recall", "accuracy"]``.
    threshold:
        Maximum *relative* degradation allowed before the condition fails.
        For higher-is-better metrics: candidate >= base * (1 - threshold).
        For lower-is-better metrics (log_loss): candidate <= base * (1 + threshold).
        Default is ``0.05`` (5 %).
    label_col:
        Name of the label column in the dataset.  If ``None``, the Deepchecks
        ``Dataset`` label is used automatically.
    """

    def __init__(
        self,
        base_model,
        metrics: Optional[List[str]] = None,
        threshold: float = 0.05,
        label_col: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.metrics: List[str] = metrics if metrics is not None else _DEFAULT_METRICS
        self.threshold = threshold
        self.label_col = label_col

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        dataset = context.get_dataset(dataset_kind)
        model = context.model  # candidate model injected by Deepchecks

        # ---- Extract features & labels --------------------------------
        label_name = self.label_col or dataset.label_name
        X = dataset.features_columns  # pd.DataFrame
        y_true = np.asarray(dataset.data[label_name])
        classes = np.unique(y_true)

        # ---- Predictions ----------------------------------------------
        cand_pred = np.asarray(model.predict(X))
        base_pred = np.asarray(self.base_model.predict(X))

        cand_proba: Optional[np.ndarray] = None
        base_proba: Optional[np.ndarray] = None

        needs_proba = any(m in self.metrics for m in ("roc_auc", "log_loss"))
        if needs_proba:
            if hasattr(model, "predict_proba"):
                cand_proba = np.asarray(model.predict_proba(X))
            else:
                raise ValueError(
                    "Candidate model does not have predict_proba(), "
                    "which is required for roc_auc / log_loss."
                )
            if hasattr(self.base_model, "predict_proba"):
                base_proba = np.asarray(self.base_model.predict_proba(X))
            else:
                raise ValueError(
                    "Base model does not have predict_proba(), "
                    "which is required for roc_auc / log_loss."
                )

        # ---- Compute metrics ------------------------------------------
        cand_scores = _compute_metrics(y_true, cand_pred, cand_proba, self.metrics, classes)
        base_scores = _compute_metrics(y_true, base_pred, base_proba, self.metrics, classes)

        # ---- Build results table --------------------------------------
        rows = []
        for metric in self.metrics:
            cand_val = cand_scores[metric]
            base_val = base_scores[metric]
            lower_is_better = metric in _LOWER_IS_BETTER

            if lower_is_better:
                delta = base_val - cand_val           # positive = candidate improved
                passed = cand_val <= base_val * (1 + self.threshold)
            else:
                delta = cand_val - base_val           # positive = candidate improved
                passed = cand_val >= base_val * (1 - self.threshold)

            rows.append(
                {
                    "Metric": _METRIC_DISPLAY_NAMES.get(metric, metric),
                    "Base Model": round(base_val, 6),
                    "Candidate Model": round(cand_val, 6),
                    "Delta (Candidate − Base)": round(delta, 6),
                    "Direction": "↓ lower is better" if lower_is_better else "↑ higher is better",
                    "Pass": "✅" if passed else "❌",
                }
            )

        result_df = pd.DataFrame(rows).set_index("Metric")

        display_value = {
            "candidate_scores": cand_scores,
            "base_scores": base_scores,
            "table": result_df,
        }

        return CheckResult(
            value=display_value,
            display=result_df,
            header="Model Performance Comparison vs Base Model",
        )

    # ------------------------------------------------------------------
    # Built-in condition
    # ------------------------------------------------------------------

    def add_condition_performance_not_degraded(
        self, threshold: Optional[float] = None
    ) -> "ModelPerformanceComparison":
        """Add a condition that fails if the candidate degrades beyond ``threshold``.

        Parameters
        ----------
        threshold:
            Override the instance-level threshold for this condition only.
            If ``None``, the instance ``self.threshold`` is used.
        """
        _threshold = threshold if threshold is not None else self.threshold

        def condition_fn(check_result_value: dict) -> ConditionResult:
            cand = check_result_value["candidate_scores"]
            base = check_result_value["base_scores"]
            failed_metrics: List[str] = []

            for metric, cand_val in cand.items():
                base_val = base[metric]
                lower_is_better = metric in _LOWER_IS_BETTER

                if lower_is_better:
                    passed = cand_val <= base_val * (1 + _threshold)
                else:
                    passed = cand_val >= base_val * (1 - _threshold)

                if not passed:
                    display = _METRIC_DISPLAY_NAMES.get(metric, metric)
                    failed_metrics.append(
                        f"{display}: base={base_val:.4f}, candidate={cand_val:.4f}"
                    )

            if failed_metrics:
                return ConditionResult(
                    ConditionCategory.FAIL,
                    "Candidate model underperforms base model beyond allowed threshold "
                    f"({_threshold*100:.1f}%) on: " + " | ".join(failed_metrics),
                )

            return ConditionResult(
                ConditionCategory.PASS,
                f"Candidate model meets or beats base model on all metrics "
                f"(threshold={_threshold*100:.1f}%).",
            )

        return self.add_condition(
            f"Candidate not degraded vs base model (threshold={_threshold*100:.1f}%)",
            condition_fn,
        )

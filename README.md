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
# Internal helper
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
                raise ValueError(
                    "roc_auc requires predict_proba(). "
                    "Make sure both models expose it."
                )
            if multiclass:
                out[m] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            else:
                out[m] = roc_auc_score(y_true, y_proba[:, 1])
 
        elif m == "log_loss":
            if y_proba is None:
                raise ValueError(
                    "log_loss requires predict_proba(). "
                    "Make sure both models expose it."
                )
            out[m] = log_loss(y_true, y_proba)
 
        else:
            raise ValueError(
                f"Unknown metric '{m}'. "
                f"Supported: {list(_DISPLAY_NAMES.keys())}"
            )
 
    return out
 
 
# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------
 
class ModelPerformanceComparison(BaseCheck):
    """Compare a candidate model's classification performance to a base model.
 
    Inherits from ``BaseCheck`` and overrides ``run()`` directly so that it
    never touches Deepchecks' internal Context / dataset_kind pipeline —
    making it robust across deepchecks versions.
 
    Parameters
    ----------
    base_model :
        A fitted sklearn-compatible classifier (needs predict, and
        predict_proba if roc_auc / log_loss are requested).
    metrics :
        Metric names to evaluate. Defaults to all six.
    threshold :
        Max *relative* degradation tolerated before the condition fails.
        For higher-is-better: candidate >= base * (1 - threshold).
        For lower-is-better (log_loss): candidate <= base * (1 + threshold).
        Default 0.05 (5 %).
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
    # Public run() — called directly by the user or by a Suite
    # ------------------------------------------------------------------
 
    def run(self, dataset, model=None, **kwargs) -> CheckResult:  # type: ignore[override]
        """
        Parameters
        ----------
        dataset : deepchecks.tabular.Dataset
        model   : fitted sklearn-compatible candidate classifier
        """
        if model is None:
            raise ValueError("Pass the candidate model via model=<your_model>.")
 
        # ── Extract X and y ──────────────────────────────────────────
        X = dataset.features_columns          # pd.DataFrame
        y_true = np.asarray(dataset.data[dataset.label_name])
        classes = np.unique(y_true)
 
        # ── Predictions ──────────────────────────────────────────────
        cand_pred = np.asarray(model.predict(X))
        base_pred = np.asarray(self.base_model.predict(X))
 
        needs_proba = any(m in self.metrics for m in ("roc_auc", "log_loss"))
        cand_proba: Optional[np.ndarray] = None
        base_proba: Optional[np.ndarray] = None
 
        if needs_proba:
            if not hasattr(model, "predict_proba"):
                raise ValueError("Candidate model has no predict_proba().")
            if not hasattr(self.base_model, "predict_proba"):
                raise ValueError("Base model has no predict_proba().")
            cand_proba = np.asarray(model.predict_proba(X))
            base_proba = np.asarray(self.base_model.predict_proba(X))
 
        # ── Compute scores ───────────────────────────────────────────
        cand_scores = _compute_metrics(y_true, cand_pred, cand_proba, self.metrics, classes)
        base_scores = _compute_metrics(y_true, base_pred, base_proba, self.metrics, classes)
 
        # ── Build display table ──────────────────────────────────────
        rows = []
        for metric in self.metrics:
            cv, bv = cand_scores[metric], base_scores[metric]
            lower  = metric in _LOWER_IS_BETTER
            delta  = (bv - cv) if lower else (cv - bv)   # positive = improvement
            passed = (cv <= bv * (1 + self.threshold)) if lower else (cv >= bv * (1 - self.threshold))
            rows.append({
                "Metric":             _DISPLAY_NAMES.get(metric, metric),
                "Base Model":         round(bv, 6),
                "Candidate Model":    round(cv, 6),
                "Δ (Candidate−Base)": round(delta, 6),
                "Direction":          "↓ lower is better" if lower else "↑ higher is better",
                "Pass":               "✅" if passed else "❌",
            })
 
        result_df = pd.DataFrame(rows).set_index("Metric")
 
        return CheckResult(
            value={"candidate_scores": cand_scores, "base_scores": base_scores},
            display=result_df,
            header="Model Performance Comparison vs Base Model",
        )
 
    # ------------------------------------------------------------------
    # Condition
    # ------------------------------------------------------------------
 
    def add_condition_performance_not_degraded(
        self, threshold: Optional[float] = None
    ) -> "ModelPerformanceComparison":
        """Fail if the candidate degrades beyond *threshold* on any metric."""
        _t = threshold if threshold is not None else self.threshold
 
        def _condition(value: dict) -> ConditionResult:
            cand, base = value["candidate_scores"], value["base_scores"]
            failures: List[str] = []
 
            for metric, cv in cand.items():
                bv    = base[metric]
                lower = metric in _LOWER_IS_BETTER
                passed = (cv <= bv * (1 + _t)) if lower else (cv >= bv * (1 - _t))
                if not passed:
                    name = _DISPLAY_NAMES.get(metric, metric)
                    failures.append(f"{name}: base={bv:.4f} candidate={cv:.4f}")
 
            if failures:
                return ConditionResult(
                    ConditionCategory.FAIL,
                    f"Candidate underperforms base (threshold={_t*100:.1f}%): "
                    + " | ".join(failures),
                )
            return ConditionResult(
                ConditionCategory.PASS,
                f"Candidate meets or beats base model on all metrics "
                f"(threshold={_t*100:.1f}%).",
            )
 
        return self.add_condition(
            f"Candidate not degraded vs base model (threshold={_t*100:.1f}%)",
            _condition,
        )

el calibration · PY
Copy

"""
pricing_checks/checks/custom/model_evaluation/model_calibration.py
"""
 
from __future__ import annotations
 
from typing import Dict, List, Optional
 
import numpy as np
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
 
from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.core.checks import DatasetKind
from deepchecks.core.errors import DeepchecksNotSupportedError
from deepchecks.tabular import SingleDatasetCheck
 
__all__ = ["ModelCalibration"]
 
DEFAULT_ECE_THRESHOLD   = 0.10
DEFAULT_BRIER_THRESHOLD = 0.20
 
 
# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
 
def _ece(y_true_binary: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """Reliability-weighted mean |confidence - accuracy| across bins."""
    bins, ece_val, n = np.linspace(0.0, 1.0, n_bins + 1), 0.0, len(y_true_binary)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_proba >= lo) & (y_proba < hi)
        if mask.sum() == 0:
            continue
        ece_val += (mask.sum() / n) * abs(y_true_binary[mask].mean() - y_proba[mask].mean())
    return float(ece_val)
 
 
def _compute_calibration_metrics(
    y_true: np.ndarray,
    y_proba_matrix: np.ndarray,
    classes: List,
    n_bins: int = 10,
) -> Dict:
    """ECE and Brier Score per class + macro average."""
    binary, per_class = len(classes) == 2, {}
 
    if binary:
        y_bin = (y_true == classes[1]).astype(int)
        p     = y_proba_matrix[:, 1]
        per_class[classes[1]] = {
            "ece":   _ece(y_bin, p, n_bins),
            "brier": float(brier_score_loss(y_bin, p)),
        }
    else:
        for idx, cls in enumerate(classes):
            y_bin = (y_true == cls).astype(int)
            p     = y_proba_matrix[:, idx]
            per_class[cls] = {
                "ece":   _ece(y_bin, p, n_bins),
                "brier": float(brier_score_loss(y_bin, p)),
            }
 
    avg = {
        metric: float(np.mean([v[metric] for v in per_class.values()]))
        for metric in ("ece", "brier")
    }
    return {"per_class": per_class, "avg": avg, "binary": binary}
 
 
# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------
 
class ModelCalibration(SingleDatasetCheck):
    """Evaluate model probability calibration using ECE and Brier Score.
 
    - ECE  : measures pure calibration — are the probability values trustworthy?
    - Brier: measures overall probabilistic quality — calibration + discrimination.
 
    Together they let you diagnose the type of problem:
      ECE fail + Brier fail  -> miscalibration issue
      ECE pass + Brier fail  -> discrimination issue (ranks poorly, not miscalibrated)
      ECE fail + Brier pass  -> miscalibrated but discriminates well (dangerous in pricing)
 
    Parameters
    ----------
    n_bins : int, default 10
        Bins for calibration curve and ECE.
    ece_threshold : float, default 0.10
        Max ECE (macro avg) to pass.
    brier_threshold : float, default 0.20
        Max Brier Score (macro avg) to pass.
    n_samples : int, default 1_000_000
    random_state : int, default 42
 
    Usage
    -----
    from deepchecks.core import Suite
    from pricing_checks.checks.custom.model_evaluation.model_calibration import ModelCalibration
 
    suite = Suite(
        "Pricing Model Suite",
        ModelCalibration(
            ece_threshold=0.10,
            brier_threshold=0.20,
        ).add_condition_calibration_score()
    )
    suite.run(ds_test, model=candidate_model).show()
    """
 
    def __init__(
        self,
        n_bins: int = 10,
        ece_threshold: float   = DEFAULT_ECE_THRESHOLD,
        brier_threshold: float = DEFAULT_BRIER_THRESHOLD,
        n_samples: int = 1_000_000,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_bins          = n_bins
        self.ece_threshold   = ece_threshold
        self.brier_threshold = brier_threshold
        self.n_samples       = n_samples
        self.random_state    = random_state
 
    def run_logic(self, context, dataset_kind=DatasetKind.TRAIN) -> CheckResult:
        context.assert_classification_task()
        dataset = context.get_data_by_kind(dataset_kind).sample(
            self.n_samples, random_state=self.random_state
        )
        model = context.model
 
        if not hasattr(model, "predict_proba"):
            raise DeepchecksNotSupportedError(
                "ModelCalibration requires predict_proba(). "
                "The check evaluates predicted probabilities, not class labels."
            )
 
        X               = dataset.features_columns
        y_true          = np.asarray(dataset.label_col)
        dataset_classes = list(dataset.classes_in_label_col)
        y_proba         = model.predict_proba(X)
 
        metrics = _compute_calibration_metrics(y_true, y_proba, dataset_classes, self.n_bins)
 
        display = []
        if context.with_display:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                line_width=2, line_dash="dash",
                name="Perfectly calibrated",
            ))
 
            if metrics["binary"]:
                cls      = dataset_classes[1]
                y_mapped = (y_true == cls).astype(int)
                frac_pos, mean_pred = calibration_curve(
                    y_mapped, y_proba[:, 1], n_bins=self.n_bins
                )
                m = metrics["per_class"][cls]
                fig.add_trace(go.Scatter(
                    x=mean_pred, y=frac_pos, mode="lines+markers",
                    name=f"ECE={m['ece']:.4f}  Brier={m['brier']:.4f}",
                ))
            else:
                for idx, cls in enumerate(dataset_classes):
                    y_bin    = (y_true == cls).astype(int)
                    frac_pos, mean_pred = calibration_curve(
                        y_bin, y_proba[:, idx], n_bins=self.n_bins
                    )
                    m = metrics["per_class"][cls]
                    fig.add_trace(go.Scatter(
                        x=mean_pred, y=frac_pos, mode="lines+markers",
                        name=f"{cls}  ECE={m['ece']:.4f}  Brier={m['brier']:.4f}",
                    ))
 
            fig.update_layout(title_text="Calibration plots (reliability curve)", height=500)
            fig.update_yaxes(title="Fraction of positives")
            fig.update_xaxes(title="Mean predicted value")
            display = [
                "Calibration curves compare how well probabilistic predictions are calibrated. "
                "The x-axis shows mean predicted probability per bin; the y-axis shows the true "
                "fraction of positives. A perfectly calibrated model follows the dashed diagonal.",
                fig,
            ]
 
        return CheckResult(value=metrics, display=display, header="Model Calibration")
 
    def add_condition_calibration_score(
        self,
        ece_threshold: Optional[float]   = None,
        brier_threshold: Optional[float] = None,
    ) -> "ModelCalibration":
        """Two conditions (ECE + Brier) -> one row each in Conditions Summary."""
        _ece_t   = ece_threshold   if ece_threshold   is not None else self.ece_threshold
        _brier_t = brier_threshold if brier_threshold is not None else self.brier_threshold
 
        def _cond_ece(value: dict) -> ConditionResult:
            score = value["avg"]["ece"]
            return ConditionResult(
                ConditionCategory.PASS if score <= _ece_t else ConditionCategory.FAIL,
                f"ECE={score:.4f}  (threshold={_ece_t:.4f})",
            )
        self.add_condition(
            f"Expected Calibration Error (ECE) <= {_ece_t:.2f} (macro avg)", _cond_ece
        )
 
        def _cond_brier(value: dict) -> ConditionResult:
            score = value["avg"]["brier"]
            return ConditionResult(
                ConditionCategory.PASS if score <= _brier_t else ConditionCategory.FAIL,
                f"Brier={score:.4f}  (threshold={_brier_t:.4f})",
            )
        self.add_condition(
            f"Brier Score <= {_brier_t:.2f} (macro avg)", _cond_brier
        )
 
        return self

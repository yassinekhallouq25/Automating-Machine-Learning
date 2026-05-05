# pricing_checks/checks/custom/model_evaluation/base_model_comparison.py

from typing import Dict, Optional

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    log_loss,
)

from deepchecks.core import CheckResult, ConditionResult, ConditionCategory
from deepchecks.tabular import Dataset
from deepchecks.tabular.base_checks import SingleDatasetBaseCheck


class BaseModelComparison(SingleDatasetBaseCheck):
    """
    Compare a candidate classification model against a base model.

    Supports normal prediction metrics and probability-based metrics:
    accuracy, f1_macro, precision_macro, recall_macro, roc_auc, log_loss.
    """

    HIGHER_IS_BETTER = {
        "accuracy": True,
        "f1_macro": True,
        "precision_macro": True,
        "recall_macro": True,
        "roc_auc": True,
        "log_loss": False,
    }

    def __init__(
        self,
        base_model,
        candidate_model,
        metrics: Optional[list[str]] = None,
        min_improvement: Optional[Dict[str, float]] = None,
        label_col: Optional[str] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.candidate_model = candidate_model
        self.metrics = metrics or [
            "accuracy",
            "f1_macro",
            "precision_macro",
            "recall_macro",
            "roc_auc",
            "log_loss",
        ]
        self.min_improvement = min_improvement or {}
        self.label_col = label_col

    def run_logic(self, dataset: Dataset, context=None) -> CheckResult:
        df = dataset.data.copy()

        label_col = self.label_col or dataset.label_name

        if label_col is None:
            raise ValueError(
                "No label column was found. Pass label_col=... or create "
                "the Deepchecks Dataset with label=..."
            )

        y_true = df[label_col]
        X = df.drop(columns=[label_col])

        base_pred = self.base_model.predict(X)
        candidate_pred = self.candidate_model.predict(X)

        base_proba = self._safe_predict_proba(self.base_model, X)
        candidate_proba = self._safe_predict_proba(self.candidate_model, X)

        rows = []

        for metric_name in self.metrics:
            base_score = self._score_metric(
                metric_name=metric_name,
                y_true=y_true,
                y_pred=base_pred,
                y_proba=base_proba,
            )

            candidate_score = self._score_metric(
                metric_name=metric_name,
                y_true=y_true,
                y_pred=candidate_pred,
                y_proba=candidate_proba,
            )

            higher_is_better = self.HIGHER_IS_BETTER[metric_name]

            if higher_is_better:
                improvement = candidate_score - base_score
            else:
                improvement = base_score - candidate_score

            required_improvement = self.min_improvement.get(metric_name, 0.0)
            passed = improvement >= required_improvement

            rows.append(
                {
                    "metric": metric_name,
                    "base_model": base_score,
                    "candidate_model": candidate_score,
                    "higher_is_better": higher_is_better,
                    "improvement": improvement,
                    "required_improvement": required_improvement,
                    "passed": passed,
                }
            )

        result_df = pd.DataFrame(rows)

        return CheckResult(
            value=result_df,
            display=[result_df],
        )

    @staticmethod
    def _safe_predict_proba(model, X):
        if not hasattr(model, "predict_proba"):
            return None
        return model.predict_proba(X)

    def _score_metric(self, metric_name, y_true, y_pred, y_proba):
        if metric_name == "accuracy":
            return accuracy_score(y_true, y_pred)

        if metric_name == "f1_macro":
            return f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )

        if metric_name == "precision_macro":
            return precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )

        if metric_name == "recall_macro":
            return recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )

        if metric_name == "roc_auc":
            if y_proba is None:
                raise ValueError(
                    "roc_auc requires models that implement predict_proba."
                )

            if len(pd.Series(y_true).unique()) == 2:
                return roc_auc_score(y_true, y_proba[:, 1])

            return roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="macro",
            )

        if metric_name == "log_loss":
            if y_proba is None:
                raise ValueError(
                    "log_loss requires models that implement predict_proba."
                )

            return log_loss(y_true, y_proba)

        raise ValueError(f"Unsupported metric: {metric_name}")

    def add_condition_candidate_better_or_equal(self):
        """
        Pass if candidate model is better than or equal to base model
        for all configured metrics.
        """

        def condition(result_df: pd.DataFrame):
            failed = result_df[result_df["passed"] == False]

            if failed.empty:
                return ConditionResult(
                    ConditionCategory.PASS,
                    "Candidate model passed all metric comparisons.",
                )

            failed_metrics = failed["metric"].tolist()

            return ConditionResult(
                ConditionCategory.FAIL,
                f"Candidate model failed these metrics: {failed_metrics}",
            )

        return self.add_condition(
            "Candidate model should be better than or equal to base model",
            condition,
        )

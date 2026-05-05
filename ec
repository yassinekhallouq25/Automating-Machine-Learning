def run_logic(self, context, dataset_kind) -> CheckResult:  # ← rename + fix signature
    from deepchecks.tabular.context import Context
    from deepchecks.utils.strings import format_header

    dataset = context.get_dataset(dataset_kind)
    df = dataset.data.copy()

    label_col = self.label_col or dataset.label_name

    if label_col is None:
        raise ValueError(
            "Label column was not found. Pass label_col=... or create "
            "the Deepchecks Dataset with label=..."
        )

    X = df.drop(columns=[label_col])
    y_true = df[label_col]

    base_pred = self.base_model.predict(X)
    candidate_pred = self.candidate_model.predict(X)

    base_proba = self._safe_predict_proba(self.base_model, X)
    candidate_proba = self._safe_predict_proba(self.candidate_model, X)

    rows = []

    for metric_name in self.metrics:
        base_score = self._score_metric(metric_name, y_true, base_pred, base_proba)
        candidate_score = self._score_metric(metric_name, y_true, candidate_pred, candidate_proba)

        higher_is_better = self.HIGHER_IS_BETTER[metric_name]
        improvement = candidate_score - base_score if higher_is_better else base_score - candidate_score
        required_improvement = self.min_improvement.get(metric_name, 0.0)
        passed = improvement >= required_improvement

        rows.append({
            "metric": metric_name,
            "base_model": base_score,
            "candidate_model": candidate_score,
            "higher_is_better": higher_is_better,
            "improvement": improvement,
            "required_improvement": required_improvement,
            "passed": passed,
        })

    result_df = pd.DataFrame(rows)

    return CheckResult(value=result_df, display=[result_df])

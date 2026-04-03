import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deepchecks.core import CheckResult, ConditionCategory, ConditionResult, DatasetKind
from deepchecks.tabular import Dataset, Context, SingleDatasetCheck

class QuantileTargetJumpWithPlots(SingleDatasetCheck):
    def __init__(self, features=None, top_n=10, n_bins=10, min_samples=100, min_bin_size=10, **kwargs):
        super().__init__(**kwargs)
        self.features = features
        self.top_n = top_n
        self.n_bins = n_bins
        self.min_samples = min_samples
        self.min_bin_size = min_bin_size

    def run_logic(self, context: Context, dataset_kind: DatasetKind) -> CheckResult:
        dataset = context.get_data_by_kind(dataset_kind)
        data = dataset.data.copy()
        y = pd.Series(dataset.label_col).copy()

        if y.nunique(dropna=True) != 2:
            msg = pd.DataFrame({"message": ["This check requires a binary target"]})
            return CheckResult(
                {"max_jump": np.nan, "n_tested_features": 0, "top_features": []},
                display=[msg]
            )

        if not pd.api.types.is_numeric_dtype(y):
            y = pd.Series(pd.Categorical(y).codes, index=y.index)
        else:
            uniq = sorted(pd.Series(y.dropna()).unique())
            y = y.map({uniq[0]: 0, uniq[1]: 1})

        features = self.features if self.features is not None else list(dataset.features)
        rows = []
        per_feature_bins = {}

        for col in features:
            if col not in data.columns or col == dataset.label_name:
                continue

            x = pd.to_numeric(data[col], errors="coerce")
            valid = x.notna() & y.notna()

            if valid.sum() < self.min_samples:
                continue
            if x[valid].nunique() < max(5, self.n_bins // 2):
                continue

            try:
                bins = pd.qcut(x[valid], q=self.n_bins, duplicates="drop")
            except Exception:
                continue

            tmp = pd.DataFrame({"x": x[valid], "y": y[valid], "bin": bins})
            grouped = (
                tmp.groupby("bin", observed=False)
                .agg(
                    count=("y", "size"),
                    target_rate=("y", "mean"),
                    x_mean=("x", "mean"),
                    x_min=("x", "min"),
                    x_max=("x", "max")
                )
                .reset_index(drop=True)
            )

            grouped = grouped[grouped["count"] >= self.min_bin_size].reset_index(drop=True)

            if len(grouped) < 3:
                continue

            jumps = grouped["target_rate"].diff().abs().dropna()
            if len(jumps) == 0:
                continue

            max_jump = float(jumps.max())
            mean_jump = float(jumps.mean())
            weighted_std = float(np.sqrt(np.average((grouped["target_rate"] - grouped["target_rate"].mean()) ** 2, weights=grouped["count"])))
            rows.append(
                {
                    "feature": col,
                    "n_bins_used": int(len(grouped)),
                    "samples_used": int(grouped["count"].sum()),
                    "max_jump": max_jump,
                    "mean_jump": mean_jump,
                    "weighted_target_rate_std": weighted_std
                }
            )
            per_feature_bins[col] = grouped

        if not rows:
            msg = pd.DataFrame({"message": ["No valid numeric features were tested"]})
            return CheckResult(
                {"max_jump": np.nan, "n_tested_features": 0, "top_features": []},
                display=[msg]
            )

        results_df = pd.DataFrame(rows).sort_values("max_jump", ascending=False).reset_index(drop=True)
        top_df = results_df.head(self.top_n).copy()
        worst_feature = top_df.iloc[0]["feature"]
        worst_bins = per_feature_bins[worst_feature].copy()
        worst_bins["bin_label"] = [f"Q{i+1}" for i in range(len(worst_bins))]

        def top_jump_bar_plot():
            plot_df = top_df.iloc[::-1]
            plt.figure(figsize=(10, max(4, 0.5 * len(plot_df))))
            plt.barh(plot_df["feature"], plot_df["max_jump"])
            plt.xlabel("Max adjacent target-rate jump")
            plt.ylabel("Feature")
            plt.title("Top Features by Quantile Target-Rate Jump")
            plt.tight_layout()

        def worst_feature_line_plot():
            plt.figure(figsize=(10, 5))
            plt.plot(worst_bins["bin_label"], worst_bins["target_rate"], marker="o")
            plt.ylim(0, 1)
            plt.xlabel("Quantile bin")
            plt.ylabel("Positive class rate")
            plt.title(f"Target Rate Across Quantiles: {worst_feature}")
            plt.tight_layout()

        def worst_feature_count_plot():
            plt.figure(figsize=(10, 5))
            plt.bar(worst_bins["bin_label"], worst_bins["count"])
            plt.xlabel("Quantile bin")
            plt.ylabel("Count")
            plt.title(f"Samples per Quantile Bin: {worst_feature}")
            plt.tight_layout()

        return CheckResult(
            {
                "max_jump": float(results_df["max_jump"].iloc[0]),
                "n_tested_features": int(len(results_df)),
                "top_features": top_df.to_dict(orient="records"),
                "worst_feature": worst_feature
            },
            display=[top_df, top_jump_bar_plot, worst_feature_line_plot, worst_feature_count_plot]
        )

    def add_condition_max_jump_not_greater_than(self, threshold=0.2):
        def condition(result):
            max_jump = result["max_jump"]
            passed = pd.isna(max_jump) or max_jump <= threshold
            return ConditionResult(
                ConditionCategory.PASS if passed else ConditionCategory.FAIL,
                f"max quantile target-rate jump = {max_jump:.4f}" if not pd.isna(max_jump) else "no valid features were tested"
            )
        return self.add_condition(f"Max quantile target-rate jump <= {threshold}", condition)

target = "y"

candidate_features = [
    c for c in df.columns
    if c != target and (
        c.startswith("TAUX_NOMINAL_PRECONISE")
        or c in [
            "MENSUALITE_MAXIMALE_CLIENT",
            "IMPOTS",
            "CREDIT_MONTANT",
            "ENCOURS_AUTRES_CREDITS_HORS_IMMO",
            "MENSUALITES_AUTRES_CREDITS",
            "MONTANT_DECOUVERT_AUTORISE",
            "CREDIT_DUREE",
            "EPARGNE_BNP",
            "BCOM-LOT1-SITLOCCEPT_0_1648_P",
            "BCOM-LOT1-SITLOCCEPT_0_1648_L",
            "BCOM-LOT1-SITLOCCEPT_0_1648_F",
            "BCOM-LOT1-SITLOCCEPT_0_1648_X",
            "BCOM-LOT1-SITLOCCEPT_0_1648_E",
            "TOTAL_REVENUS",
            "EPARGNE_BNP LIQUIDE",
        ]
    )
]

df_custom = df[[target] + candidate_features].dropna(subset=[target]).copy()
custom_ds = Dataset(df_custom, label=target)

custom_check = QuantileTargetJumpWithPlots(
    features=candidate_features,
    top_n=10,
    n_bins=10,
    min_samples=100,
    min_bin_size=10
).add_condition_max_jump_not_greater_than(0.2)

custom_result = custom_check.run(custom_ds)
custom_result.show(as_widget=False)
print(custom_result.value)

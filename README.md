import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from deepchecks.core import CheckResult, ConditionCategory, ConditionResult, DatasetKind
from deepchecks.tabular import Dataset, Context, SingleDatasetCheck

class SingleFeatureLeakageAUC(SingleDatasetCheck):
    def __init__(self, features=None, top_n=10, min_samples=30, **kwargs):
        super().__init__(**kwargs)
        self.features = features
        self.top_n = top_n
        self.min_samples = min_samples

    def run_logic(self, context: Context, dataset_kind: DatasetKind) -> CheckResult:
        dataset = context.get_data_by_kind(dataset_kind)
        data = dataset.data.copy()
        y = pd.Series(dataset.label_col).copy()

        if y.nunique(dropna=True) != 2:
            return CheckResult(
                {"max_auc": np.nan, "n_tested_features": 0, "top_features": []},
                display=[pd.DataFrame({"message": ["This check requires a binary target"]})]
            )

        if not pd.api.types.is_numeric_dtype(y):
            y = pd.Series(pd.Categorical(y).codes, index=y.index)
        else:
            uniq = sorted(pd.Series(y.dropna()).unique())
            y = y.map({uniq[0]: 0, uniq[1]: 1})

        features = self.features if self.features is not None else list(dataset.features)
        rows = []

        for col in features:
            if col not in data.columns or col == dataset.label_name:
                continue
            x = pd.to_numeric(data[col], errors="coerce")
            valid = x.notna() & y.notna()
            if valid.sum() < self.min_samples:
                continue
            if x[valid].nunique() < 2 or y[valid].nunique() < 2:
                continue
            auc = roc_auc_score(y[valid], x[valid])
            auc = max(auc, 1 - auc)
            rows.append(
                {
                    "feature": col,
                    "auc": float(auc),
                    "samples_used": int(valid.sum()),
                    "unique_values": int(x[valid].nunique())
                }
            )

        results_df = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True) if rows else pd.DataFrame(columns=["feature", "auc", "samples_used", "unique_values"])
        max_auc = float(results_df["auc"].iloc[0]) if not results_df.empty else np.nan

        return CheckResult(
            {
                "max_auc": max_auc,
                "n_tested_features": int(len(results_df)),
                "top_features": results_df.head(self.top_n).to_dict(orient="records")
            },
            display=[results_df.head(self.top_n)]
        )

    def add_condition_max_auc_not_greater_than(self, threshold=0.9):
        def condition(result):
            max_auc = result["max_auc"]
            passed = pd.isna(max_auc) or max_auc <= threshold
            return ConditionResult(
                ConditionCategory.PASS if passed else ConditionCategory.FAIL,
                f"max single-feature AUC = {max_auc:.4f}" if not pd.isna(max_auc) else "no valid numeric features were tested"
            )
        return self.add_condition(f"Max single-feature AUC <= {threshold}", condition)

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

custom_check = SingleFeatureLeakageAUC(
    features=candidate_features,
    top_n=10,
    min_samples=30
).add_condition_max_auc_not_greater_than(0.9)

custom_result = custom_check.run(custom_ds)
custom_result.show(as_widget=False)
print(custom_result.value)

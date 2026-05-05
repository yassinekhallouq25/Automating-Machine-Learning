from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from deepchecks.tabular import Dataset

from pricing_checks.checks.model_evaluation.model_performance_comparison import (
    ModelPerformanceComparison,
)

# ── 1. Toy dataset ──────────────────────────────────────────────────────────
X, y = make_classification(n_samples=2_000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

feature_cols = [f"feature_{i}" for i in range(X.shape[1])]
df_test = pd.DataFrame(X_test, columns=feature_cols)
df_test["target"] = y_test

# ── 2. Train two models ─────────────────────────────────────────────────────
base_model = RandomForestClassifier(n_estimators=50, random_state=0).fit(X_train, y_train)
candidate_model = GradientBoostingClassifier(n_estimators=100, random_state=0).fit(X_train, y_train)

# ── 3. Wrap in a Deepchecks Dataset ─────────────────────────────────────────
ds_test = Dataset(df_test, label="target", cat_features=[])

# ── 4. Build and run the check ───────────────────────────────────────────────
check = (
    ModelPerformanceComparison(
        base_model=base_model,
        # Explicitly list the metrics you care about (all six shown here)
        metrics=["roc_auc", "log_loss", "f1", "precision", "recall", "accuracy"],
        threshold=0.05,  # 5 % max degradation
    )
    .add_condition_performance_not_degraded()
)

result = check.run(ds_test, model=candidate_model)
result.show()   # renders an interactive table in Jupyter / HTML

# ── 5. Access raw values programmatically ───────────────────────────────────
print("\nCandidate scores:", result.value["candidate_scores"])
print("Base scores     :", result.value["base_scores"])

# ── 6. Embed in a Suite (optional) ──────────────────────────────────────────
from deepchecks.core import Suite

suite = Suite(
    "Pricing Model Evaluation Suite",
    ModelPerformanceComparison(base_model=base_model).add_condition_performance_not_degraded(),
)
suite_result = suite.run(ds_test, model=candidate_model)
suite_result.show()

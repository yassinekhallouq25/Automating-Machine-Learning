from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.core import Suite

from pricing_checks.checks.model_evaluation.model_performance_comparison import (
    ModelPerformanceComparison,
)

# ── 1. Generate data ─────────────────────────────────────────────────────────
X, y = make_classification(
    n_samples=2000, n_features=20, n_informative=10, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

feature_cols = [f"feature_{i}" for i in range(X.shape[1])]
df_test = pd.DataFrame(X_test, columns=feature_cols)
df_test["target"] = y_test

# ── 2. Train candidate model ──────────────────────────────────────────────────
candidate_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
candidate_model.fit(X_train, y_train)

# ── 3. Wrap in Deepchecks Dataset ─────────────────────────────────────────────
ds_test = Dataset(df_test, label="target", cat_features=[])

# ── 4. Run inside a Suite ─────────────────────────────────────────────────────
suite = Suite(
    "Pricing Model Suite",
    ModelPerformanceComparison(
        baseline="logistic_regression",  # or "random_forest"
        threshold=0.05,
    ).add_condition_performance_not_degraded()
)

suite.run(ds_test, model=candidate_model).show()

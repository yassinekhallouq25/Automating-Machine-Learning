import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from deepchecks.tabular import Dataset, Suite

from pricing_checks.checks.custom.model_evaluation.base_model_comparison import (
    BaseModelComparison,
)


# --------------------------------------------------
# 1. Generate fake classification data
# --------------------------------------------------
X, y = make_classification(
    n_samples=1000,
    n_features=12,
    n_informative=8,
    n_redundant=2,
    n_classes=2,
    weights=[0.65, 0.35],
    random_state=42,
)

feature_names = [f"feature_{i}" for i in range(X.shape[1])]

df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

target_col = "target"


# --------------------------------------------------
# 2. Split into train and test
# --------------------------------------------------
train_df, test_df = train_test_split(
    df,
    test_size=0.25,
    random_state=42,
    stratify=df[target_col],
)

X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]


# --------------------------------------------------
# 3. Train base model
# --------------------------------------------------
base_model = LogisticRegression(
    max_iter=2000,
    solver="lbfgs",
)

base_model.fit(X_train, y_train)


# --------------------------------------------------
# 4. Train candidate model
# --------------------------------------------------
candidate_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=6,
    random_state=42,
)

candidate_model.fit(X_train, y_train)


# --------------------------------------------------
# 5. Create Deepchecks Dataset
# --------------------------------------------------
test_dataset = Dataset(
    test_df,
    label=target_col,
    cat_features=[],
)


# --------------------------------------------------
# 6. Create custom Deepchecks check
# --------------------------------------------------
check = BaseModelComparison(
    base_model=base_model,
    candidate_model=candidate_model,
    label_col=target_col,
    metrics=[
        "accuracy",
        "f1_macro",
        "precision_macro",
        "recall_macro",
        "roc_auc",
        "log_loss",
    ],
    min_improvement={
        "accuracy": 0.0,
        "f1_macro": 0.0,
        "precision_macro": 0.0,
        "recall_macro": 0.0,
        "roc_auc": 0.0,
        "log_loss": 0.0,
    },
).add_condition_candidate_passes_all_metrics()


# --------------------------------------------------
# 7. Run as a Deepchecks suite
# --------------------------------------------------
suite = Suite(
    "Generated Data - Base Model vs Candidate Model",
    check,
)

result = suite.run(test_dataset)

result.show()


# --------------------------------------------------
# 8. Also print raw result table
# --------------------------------------------------
single_result = check.run(test_dataset)

print(single_result.value)

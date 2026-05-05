from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from deepchecks.tabular import Dataset, Suite
from pricing_checks.checks.custom import BaseModelComparison


# 1. Load binary classification data
data = load_breast_cancer(as_frame=True)
df = data.frame

target_col = "target"


# 2. Train/test split
train_df, test_df = train_test_split(
    df,
    test_size=0.25,
    random_state=42,
    stratify=df[target_col],
)

X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]


# 3. Train base model
base_model = LogisticRegression(max_iter=1000)
base_model.fit(X_train, y_train)


# 4. Train candidate model
candidate_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)
candidate_model.fit(X_train, y_train)


# 5. Create Deepchecks dataset
test_dataset = Dataset(
    test_df,
    label=target_col,
    cat_features=[],
)


# 6. Create the custom comparison check
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
).add_condition_candidate_better_or_equal()


# 7. Run the check inside a Deepchecks suite
suite = Suite(
    "Candidate vs Base Model Classification Suite",
    check,
)

result = suite.run(test_dataset)


# 8. Display result
result.show()

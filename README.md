from sklearn.ensemble import RandomForestRegressor
from deepchecks.tabular.checks.model_evaluation import SimpleModelComparison

# Use only the columns in your Deepchecks Dataset, excluding the label
feature_cols = [c for c in train_df.columns if c != label_col]

X_train = train_df[feature_cols]
y_train = train_df[label_col]

# Tree model
tree_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)

tree_model.fit(X_train, y_train)

# Deepchecks Simple Model Comparison
# simple_model_type='tree' makes Deepchecks compare your model
# against a simple tree baseline
check = SimpleModelComparison(
    simple_model_type="tree",
    max_depth=5
)

result = check.run(
    train_dataset=train_ds,
    test_dataset=test_ds,
    model=tree_model
)

result.show()

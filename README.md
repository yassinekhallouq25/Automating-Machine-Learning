import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import classification_performance

# Create datasets
train_ds = Dataset(
    pd.concat(
        [
            train[["X1", "X2", "r"]],
            train["Y_label"].rename("target")
        ],
        axis=1
    ),
    label="target",
    cat_features=[],
    task_type="binary"   # or "multiclass"
)

test_ds = Dataset(
    pd.concat(
        [
            test[["X1", "X2", "r"]],
            test["Y_label"].rename("target")
        ],
        axis=1
    ),
    label="target",
    cat_features=[],
    task_type="binary"   # or "multiclass"
)

# Classification suite
suite = classification_performance()

# Run
result = suite.run(
    train_dataset=train_ds,
    test_dataset=test_ds,
    model=wrapper
)

result.show(as_widget=False)

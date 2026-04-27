import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import model_evaluation

# Create train dataset
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
    task_type="binary"   # change to "multiclass" if more than 2 classes
)

# Create test dataset
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
    task_type="binary"   # change to "multiclass" if more than 2 classes
)

# Default model evaluation suite
suite = model_evaluation()

# Run suite
result = suite.run(
    train_dataset=train_ds,
    test_dataset=test_ds,
    model=wrapper
)

result.show(as_widget=False)

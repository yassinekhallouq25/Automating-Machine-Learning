# 1. Create full folder structure
New-Item -ItemType Directory -Force -Path pricing_checks\pricing_checks\checks\custom\data_integrity
New-Item -ItemType Directory -Force -Path pricing_checks\pricing_checks\checks\custom\model_evaluation
New-Item -ItemType Directory -Force -Path pricing_checks\pricing_checks\checks\custom\train_test_validation
New-Item -ItemType Directory -Force -Path pricing_checks\pricing_checks\suites

# 2. Create all files
New-Item -ItemType File -Force -Path pricing_checks\pricing_checks\__init__.py
New-Item -ItemType File -Force -Path pricing_checks\pricing_checks\checks\__init__.py
New-Item -ItemType File -Force -Path pricing_checks\pricing_checks\checks\custom\__init__.py
New-Item -ItemType File -Force -Path pricing_checks\pricing_checks\checks\custom\data_integrity\__init__.py
New-Item -ItemType File -Force -Path pricing_checks\pricing_checks\checks\custom\data_integrity\descriptive_stats.py
New-Item -ItemType File -Force -Path pricing_checks\pricing_checks\checks\custom\model_evaluation\__init__.py
New-Item -ItemType File -Force -Path pricing_checks\pricing_checks\checks\custom\train_test_validation\__init__.py
New-Item -ItemType File -Force -Path pricing_checks\pricing_checks\suites\__init__.py
New-Item -ItemType File -Force -Path pricing_checks\setup.py
New-Item -ItemType File -Force -Path pricing_checks\README.md

# 3. Install in editable mode
cd pricing_checks
pip install -e .



pricing_checks/pricing_checks/__init__.py
pythonfrom pricing_checks.checks import *
from pricing_checks.suites import *

from deepchecks.tabular import Dataset, Suite
from deepchecks.core import CheckResult, ConditionResult
from deepchecks.tabular import SingleDatasetCheck, TrainTestCheck




pricing_checks/pricing_checks/checks/__init__.py
pythonfrom deepchecks.tabular.checks import *
from pricing_checks.checks.custom import *

pricing_checks/pricing_checks/checks/custom/__init__.py
pythonfrom pricing_checks.checks.custom.data_integrity import *
from pricing_checks.checks.custom.model_evaluation import *
from pricing_checks.checks.custom.train_test_validation import *

pricing_checks/pricing_checks/checks/custom/data_integrity/__init__.py
pythonfrom .descriptive_stats import DescriptiveStats

__all__ = ["DescriptiveStats"]

pricing_checks/pricing_checks/checks/custom/model_evaluation/__init__.py
python# empty for now — add imports here when you create model evaluation checks

pricing_checks/pricing_checks/checks/custom/train_test_validation/__init__.py
python# empty for now — add imports here when you create train/test validation checks

pricing_checks/pricing_checks/checks/custom/data_integrity/descriptive_stats.py
pythonimport pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from deepchecks.core import CheckResult, ConditionResult, ConditionCategory
from deepchecks.tabular import SingleDatasetCheck, Context


class DescriptiveStats(SingleDatasetCheck):
    """
    Displays descriptive statistics and distribution plots for all numeric columns.

    Conditions available:
        - add_condition_null_ratio_less_than(max_ratio)
        - add_condition_no_high_skewness(max_skew)
        - add_condition_no_high_kurtosis(max_kurt)
        - add_condition_std_not_zero()
        - add_condition_mean_in_range(col, min_val, max_val)
        - add_condition_coefficient_of_variation_less_than(max_cv)
    """

    def __init__(self, columns=None, **kwargs):
        super().__init__(**kwargs)
        self.columns = columns

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        df = context.get_data_by_kind(dataset_kind).data

        numeric_df = (
            df[self.columns]
            if self.columns
            else df.select_dtypes(include="number")
        )

        # ---- stats table ----
        stats = pd.DataFrame({
            "count":    numeric_df.count(),
            "null":     numeric_df.isnull().sum(),
            "null %":   (numeric_df.isnull().mean() * 100).round(2),
            "mean":     numeric_df.mean().round(4),
            "std":      numeric_df.std().round(4),
            "min":      numeric_df.min().round(4),
            "25%":      numeric_df.quantile(0.25).round(4),
            "median":   numeric_df.median().round(4),
            "75%":      numeric_df.quantile(0.75).round(4),
            "max":      numeric_df.max().round(4),
            "skewness": numeric_df.skew().round(4),
            "kurtosis": numeric_df.kurt().round(4),
        })

        # ---- plotly distribution plots ----
        cols   = numeric_df.columns.tolist()
        n_cols = 3
        n_rows = int(np.ceil(len(cols) / n_cols))

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[
                f"{col}<br><sup>skew={numeric_df[col].skew():.2f}  kurt={numeric_df[col].kurt():.2f}</sup>"
                for col in cols
            ]
        )

        for i, col in enumerate(cols):
            row     = i // n_cols + 1
            col_pos = i % n_cols + 1
            data    = numeric_df[col].dropna()

            fig.add_trace(
                go.Histogram(
                    x=data,
                    nbinsx=30,
                    name=col,
                    marker_color="#4C72B0",
                    opacity=0.85,
                    showlegend=False,
                ),
                row=row, col=col_pos
            )

            fig.add_vline(
                x=data.mean(),
                line_dash="dash",
                line_color="#DD4444",
                line_width=2,
                annotation_text=f"Mean: {data.mean():.2f}",
                annotation_font_size=9,
                row=row, col=col_pos
            )

            fig.add_vline(
                x=data.median(),
                line_dash="dot",
                line_color="#44AA44",
                line_width=2,
                annotation_text=f"Median: {data.median():.2f}",
                annotation_font_size=9,
                row=row, col=col_pos
            )

        fig.update_layout(
            title_text="Distribution of Numeric Variables",
            title_font_size=16,
            height=400 * n_rows,
            bargap=0.05,
            template="plotly_white",
        )

        return CheckResult(
            value=stats,
            display=[stats, fig],
            header="Descriptive Statistics",
        )

    def add_condition_null_ratio_less_than(self, max_ratio: float = 0.05):
        def condition(stats_df):
            failing = stats_df[stats_df["null %"] > max_ratio * 100]
            if failing.empty:
                return ConditionResult(ConditionCategory.PASS, "All columns within null threshold.")
            return ConditionResult(ConditionCategory.FAIL, f"Columns exceeding {max_ratio:.0%} nulls: {failing.index.tolist()}")
        return self.add_condition(f"Null ratio < {max_ratio:.0%}", condition)

    def add_condition_no_high_skewness(self, max_skew: float = 2.0):
        def condition(stats_df):
            failing = stats_df[stats_df["skewness"].abs() > max_skew]
            if failing.empty:
                return ConditionResult(ConditionCategory.PASS, f"No column exceeds skewness of {max_skew}.")
            return ConditionResult(ConditionCategory.FAIL, f"Highly skewed columns: {failing.index.tolist()}")
        return self.add_condition(f"|Skewness| < {max_skew}", condition)

    def add_condition_no_high_kurtosis(self, max_kurt: float = 5.0):
        def condition(stats_df):
            failing = stats_df[stats_df["kurtosis"].abs() > max_kurt]
            if failing.empty:
                return ConditionResult(ConditionCategory.PASS, f"No column exceeds kurtosis of {max_kurt}.")
            return ConditionResult(ConditionCategory.FAIL, f"High kurtosis columns: {failing.index.tolist()}")
        return self.add_condition(f"Kurtosis < {max_kurt}", condition)

    def add_condition_std_not_zero(self):
        def condition(stats_df):
            failing = stats_df[stats_df["std"] < 1e-6]
            if failing.empty:
                return ConditionResult(ConditionCategory.PASS, "No constant columns detected.")
            return ConditionResult(ConditionCategory.FAIL, f"Constant columns: {failing.index.tolist()}")
        return self.add_condition("No constant columns (std > 0)", condition)

    def add_condition_mean_in_range(self, col: str, min_val: float, max_val: float):
        def condition(stats_df):
            if col not in stats_df.index:
                return ConditionResult(ConditionCategory.PASS, f"Column '{col}' not found, skipping.")
            mean_val = stats_df.loc[col, "mean"]
            passed = min_val <= mean_val <= max_val
            return ConditionResult(
                ConditionCategory.PASS if passed else ConditionCategory.FAIL,
                f"Mean of '{col}' = {mean_val:.4f} (expected: [{min_val}, {max_val}])"
            )
        return self.add_condition(f"Mean of '{col}' in [{min_val}, {max_val}]", condition)

    def add_condition_coefficient_of_variation_less_than(self, max_cv: float = 1.0):
        def condition(stats_df):
            cv = (stats_df["std"] / stats_df["mean"].abs()).round(4)
            failing = cv[cv > max_cv].index.tolist()
            if not failing:
                return ConditionResult(ConditionCategory.PASS, f"All columns have CV <= {max_cv}.")
            return ConditionResult(ConditionCategory.FAIL, f"High variability columns (CV > {max_cv}): {failing}")
        return self.add_condition(f"Coefficient of Variation < {max_cv}", condition)

pricing_checks/pricing_checks/suites/__init__.py
pythonfrom deepchecks.tabular.suites import (
    data_integrity,
    train_test_validation,
    model_evaluation,
    full_suite,
)

pricing_checks/setup.py
pythonfrom setuptools import setup, find_packages

setup(
    name="pricing_checks",
    version="0.1.0",
    description="A wrapper around deepchecks with custom pricing checks.",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "deepchecks>=0.18.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "plotly>=5.0.0",
    ],
    python_requires=">=3.8",
)

pricing_checks/README.md
markdown# PricingChecks

A wrapper around [deepchecks](https://docs.deepchecks.com/) that exposes all built-in checks and suites under the `pricing_checks` namespace, with additional custom pricing-specific checks.

## Installation
```bash
pip install -e .
```

## Usage
```python
import pricing_checks as pc

# wrap your data
train_ds = pc.Dataset(train_df, label="target_price", cat_features=["category"])
test_ds  = pc.Dataset(test_df,  label="target_price", cat_features=["category"])

# built-in suites
pc.data_integrity().run(train_ds).show()
pc.train_test_validation().run(train_dataset=train_ds, test_dataset=test_ds).show()
pc.model_evaluation().run(train_dataset=train_ds, test_dataset=test_ds, y_pred_train=y_pred_train, y_pred_test=y_pred_test).show()

# built-in individual checks
pc.MixedNulls().run(train_ds).show()
pc.FeatureDrift().run(train_dataset=train_ds, test_dataset=test_ds).show()

# custom checks
pc.DescriptiveStats()\
    .add_condition_null_ratio_less_than(0.05)\
    .add_condition_no_high_skewness(2.0)\
    .add_condition_no_high_kurtosis(5.0)\
    .add_condition_std_not_zero()\
    .add_condition_mean_in_range("price", 10, 500)\
    .add_condition_coefficient_of_variation_less_than(1.0)\
    .run(train_ds)\
    .show()
```

## Adding a new custom check
1. Create your file in the right folder:
   - Data quality → `pricing_checks/checks/custom/data_integrity/`
   - Model performance → `pricing_checks/checks/custom/model_evaluation/`
   - Train vs test → `pricing_checks/checks/custom/train_test_validation/`

2. Register it in that folder's `__init__.py`
3. Restart the Jupyter kernel — no reinstall needed

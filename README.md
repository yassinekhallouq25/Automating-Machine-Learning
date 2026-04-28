import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from deepchecks.core import CheckResult, ConditionResult, ConditionCategory
from deepchecks.tabular import SingleDatasetCheck, Context
 
 
class DescriptiveStats(SingleDatasetCheck):
    """
    Displays descriptive statistics and distribution plots for ALL column types:
    numeric (int/float) and categorical (object/category/bool).
 
    Parameters
    ----------
    columns : list[str] | None
        Explicit list of columns to analyse. If None, all columns are used.
    max_categories_display : int
        Maximum number of top categories shown in bar charts (default 20).
 
    Numeric conditions available:
        - add_condition_null_ratio_less_than(max_ratio)
        - add_condition_no_high_skewness(max_skew)
        - add_condition_no_high_kurtosis(max_kurt)
        - add_condition_std_not_zero()
        - add_condition_mean_in_range(col, min_val, max_val)
        - add_condition_coefficient_of_variation_less_than(max_cv)
 
    Categorical conditions available:
        - add_condition_max_categories_less_than(max_n)
        - add_condition_dominant_category_ratio_less_than(max_ratio)
        - add_condition_rare_category_ratio_less_than(min_ratio)
    """
 
    def __init__(self, columns=None, max_categories_display: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.columns = columns
        self.max_categories_display = max_categories_display
 
    # ----------------------------------------------------------------
    # MAIN LOGIC
    # ----------------------------------------------------------------
 
    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        df = context.get_data_by_kind(dataset_kind).data
 
        if self.columns:
            df = df[self.columns]
 
        numeric_df     = df.select_dtypes(include="number")
        categorical_df = df.select_dtypes(include=["object", "category", "bool"])
 
        display_items = []
        result_value  = {}
 
        # ---- numeric stats ----
        if not numeric_df.empty:
            num_stats = self._numeric_stats(numeric_df)
            result_value["numeric"] = num_stats
            display_items.append(num_stats)
            display_items.append(self._numeric_plots(numeric_df))
 
        # ---- categorical stats ----
        if not categorical_df.empty:
            cat_stats = self._categorical_stats(categorical_df)
            result_value["categorical"] = cat_stats
            display_items.append(cat_stats)
            display_items.append(self._categorical_plots(categorical_df))
 
        return CheckResult(
            value=result_value,
            display=display_items,
            header="Descriptive Statistics (Numeric & Categorical)",
        )
 
    # ----------------------------------------------------------------
    # NUMERIC HELPERS
    # ----------------------------------------------------------------
 
    def _numeric_stats(self, numeric_df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
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
 
    def _numeric_plots(self, numeric_df: pd.DataFrame) -> go.Figure:
        cols   = numeric_df.columns.tolist()
        n_cols = 3
        n_rows = int(np.ceil(len(cols) / n_cols))
 
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[
                f"{col}<br><sup>skew={numeric_df[col].skew():.2f}  "
                f"kurt={numeric_df[col].kurt():.2f}</sup>"
                for col in cols
            ],
        )
 
        for i, col in enumerate(cols):
            row     = i // n_cols + 1
            col_pos = i %  n_cols + 1
            data    = numeric_df[col].dropna()
 
            fig.add_trace(
                go.Histogram(
                    x=data, nbinsx=30, name=col,
                    marker_color="#4C72B0", opacity=0.85, showlegend=False,
                ),
                row=row, col=col_pos,
            )
            fig.add_vline(
                x=data.mean(), line_dash="dash", line_color="#DD4444",
                line_width=2,
                annotation_text=f"Mean: {data.mean():.2f}",
                annotation_font_size=9,
                row=row, col=col_pos,
            )
            fig.add_vline(
                x=data.median(), line_dash="dot", line_color="#44AA44",
                line_width=2,
                annotation_text=f"Median: {data.median():.2f}",
                annotation_font_size=9,
                row=row, col=col_pos,
            )
 
        fig.update_layout(
            title_text="Distribution of Numeric Variables",
            title_font_size=16,
            height=400 * n_rows,
            bargap=0.05,
            template="plotly_white",
        )
        return fig
 
    # ----------------------------------------------------------------
    # CATEGORICAL HELPERS
    # ----------------------------------------------------------------
 
    def _categorical_stats(self, categorical_df: pd.DataFrame) -> pd.DataFrame:
        rows = {}
        for col in categorical_df.columns:
            series      = categorical_df[col].astype(str).replace("nan", pd.NA)
            n_total     = len(series)
            n_null      = series.isnull().sum()
            n_non_null  = n_total - n_null
            vc          = series.dropna().value_counts()
            n_unique    = vc.nunique()
            top_cat     = vc.index[0]  if n_unique > 0 else pd.NA
            top_count   = vc.iloc[0]   if n_unique > 0 else 0
            top_ratio   = round(top_count / n_non_null, 4) if n_non_null > 0 else pd.NA
            # rare = categories appearing only once
            n_rare      = int((vc == 1).sum())
            rare_ratio  = round(n_rare / n_unique, 4) if n_unique > 0 else pd.NA
 
            rows[col] = {
                "count":       n_non_null,
                "null":        n_null,
                "null %":      round(n_null / n_total * 100, 2),
                "n_unique":    n_unique,
                "top_category": top_cat,
                "top_count":   top_count,
                "top_ratio %": round(top_ratio * 100, 2) if pd.notna(top_ratio) else pd.NA,
                "n_rare (freq=1)": n_rare,
                "rare_ratio %": round(rare_ratio * 100, 2) if pd.notna(rare_ratio) else pd.NA,
            }
        return pd.DataFrame(rows).T
 
    def _categorical_plots(self, categorical_df: pd.DataFrame) -> go.Figure:
        cols   = categorical_df.columns.tolist()
        n_cols = 2
        n_rows = int(np.ceil(len(cols) / n_cols))
 
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=cols,
        )
 
        palette = [
            "#4C72B0", "#DD8452", "#55A868", "#C44E52",
            "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
            "#CCB974", "#64B5CD",
        ]
 
        for i, col in enumerate(cols):
            row     = i // n_cols + 1
            col_pos = i %  n_cols + 1
 
            series = categorical_df[col].astype(str).replace("nan", pd.NA).dropna()
            vc     = series.value_counts().head(self.max_categories_display)
 
            colors = [palette[j % len(palette)] for j in range(len(vc))]
 
            fig.add_trace(
                go.Bar(
                    x=vc.index.tolist(),
                    y=vc.values.tolist(),
                    name=col,
                    marker_color=colors,
                    showlegend=False,
                    text=[f"{v/series.shape[0]*100:.1f}%" for v in vc.values],
                    textposition="outside",
                ),
                row=row, col=col_pos,
            )
 
        fig.update_layout(
            title_text=f"Distribution of Categorical Variables "
                       f"(top {self.max_categories_display} categories each)",
            title_font_size=16,
            height=420 * n_rows,
            template="plotly_white",
            bargap=0.15,
        )
        fig.update_xaxes(tickangle=-35)
        return fig
 
    # ----------------------------------------------------------------
    # CONDITIONS — NUMERIC
    # ----------------------------------------------------------------
 
    def add_condition_null_ratio_less_than(self, max_ratio: float = 0.05):
        """Fail if any column (numeric or categorical) has more than max_ratio nulls."""
        def condition(value):
            failing = []
            for dtype_key in ("numeric", "categorical"):
                stats_df = value.get(dtype_key)
                if stats_df is not None:
                    bad = stats_df[stats_df["null %"].astype(float) > max_ratio * 100]
                    failing.extend(bad.index.tolist())
            if not failing:
                return ConditionResult(
                    ConditionCategory.PASS,
                    "All columns are within the null ratio threshold."
                )
            return ConditionResult(
                ConditionCategory.FAIL,
                f"Columns exceeding null ratio of {max_ratio:.0%}: {failing}"
            )
        return self.add_condition(f"Null ratio < {max_ratio:.0%}", condition)
 
    def add_condition_no_high_skewness(self, max_skew: float = 2.0):
        """Fail if any numeric column has absolute skewness above max_skew."""
        def condition(value):
            stats_df = value.get("numeric")
            if stats_df is None:
                return ConditionResult(ConditionCategory.PASS, "No numeric columns.")
            failing = stats_df[stats_df["skewness"].abs() > max_skew]
            if failing.empty:
                return ConditionResult(
                    ConditionCategory.PASS,
                    f"No column exceeds skewness magnitude of {max_skew}."
                )
            return ConditionResult(
                ConditionCategory.FAIL,
                f"Highly skewed columns (|skew| > {max_skew}): {failing.index.tolist()}"
            )
        return self.add_condition(f"|Skewness| < {max_skew}", condition)
 
    def add_condition_no_high_kurtosis(self, max_kurt: float = 5.0):
        """Fail if any numeric column has kurtosis above max_kurt."""
        def condition(value):
            stats_df = value.get("numeric")
            if stats_df is None:
                return ConditionResult(ConditionCategory.PASS, "No numeric columns.")
            failing = stats_df[stats_df["kurtosis"].abs() > max_kurt]
            if failing.empty:
                return ConditionResult(
                    ConditionCategory.PASS, f"No column exceeds kurtosis of {max_kurt}."
                )
            return ConditionResult(
                ConditionCategory.FAIL,
                f"High kurtosis columns (> {max_kurt}): {failing.index.tolist()}"
            )
        return self.add_condition(f"Kurtosis < {max_kurt}", condition)
 
    def add_condition_std_not_zero(self):
        """Fail if any numeric column is constant (std ≈ 0)."""
        def condition(value):
            stats_df = value.get("numeric")
            if stats_df is None:
                return ConditionResult(ConditionCategory.PASS, "No numeric columns.")
            failing = stats_df[stats_df["std"] < 1e-6]
            if failing.empty:
                return ConditionResult(ConditionCategory.PASS, "No constant columns detected.")
            return ConditionResult(
                ConditionCategory.FAIL,
                f"Constant (zero std) columns: {failing.index.tolist()}"
            )
        return self.add_condition("No constant columns (std > 0)", condition)
 
    def add_condition_mean_in_range(self, col: str, min_val: float, max_val: float):
        """Fail if a specific numeric column mean is outside [min_val, max_val]."""
        def condition(value):
            stats_df = value.get("numeric")
            if stats_df is None or col not in stats_df.index:
                return ConditionResult(
                    ConditionCategory.PASS, f"Column '{col}' not found, skipping."
                )
            mean_val = stats_df.loc[col, "mean"]
            passed   = min_val <= mean_val <= max_val
            return ConditionResult(
                ConditionCategory.PASS if passed else ConditionCategory.FAIL,
                f"Mean of '{col}' = {mean_val:.4f} (expected: [{min_val}, {max_val}])"
            )
        return self.add_condition(f"Mean of '{col}' in [{min_val}, {max_val}]", condition)
 
    def add_condition_coefficient_of_variation_less_than(self, max_cv: float = 1.0):
        """Fail if any numeric column has a high std/mean ratio."""
        def condition(value):
            stats_df = value.get("numeric")
            if stats_df is None:
                return ConditionResult(ConditionCategory.PASS, "No numeric columns.")
            cv      = (stats_df["std"] / stats_df["mean"].abs()).round(4)
            failing = cv[cv > max_cv].index.tolist()
            if not failing:
                return ConditionResult(
                    ConditionCategory.PASS, f"All columns have CV <= {max_cv}."
                )
            return ConditionResult(
                ConditionCategory.FAIL,
                f"High variability columns (CV > {max_cv}): {failing}"
            )
        return self.add_condition(f"Coefficient of Variation < {max_cv}", condition)
 
    # ----------------------------------------------------------------
    # CONDITIONS — CATEGORICAL
    # ----------------------------------------------------------------
 
    def add_condition_max_categories_less_than(self, max_n: int = 50):
        """Fail if any categorical column has more than max_n unique categories."""
        def condition(value):
            stats_df = value.get("categorical")
            if stats_df is None:
                return ConditionResult(ConditionCategory.PASS, "No categorical columns.")
            failing = stats_df[stats_df["n_unique"].astype(int) > max_n]
            if failing.empty:
                return ConditionResult(
                    ConditionCategory.PASS,
                    f"All categorical columns have <= {max_n} unique categories."
                )
            return ConditionResult(
                ConditionCategory.FAIL,
                f"High-cardinality columns (> {max_n} categories): {failing.index.tolist()}"
            )
        return self.add_condition(f"Number of categories < {max_n}", condition)
 
    def add_condition_dominant_category_ratio_less_than(self, max_ratio: float = 0.9):
        """Fail if any categorical column's top category accounts for more than max_ratio of rows."""
        def condition(value):
            stats_df = value.get("categorical")
            if stats_df is None:
                return ConditionResult(ConditionCategory.PASS, "No categorical columns.")
            failing = stats_df[
                stats_df["top_ratio %"].astype(float) > max_ratio * 100
            ]
            if failing.empty:
                return ConditionResult(
                    ConditionCategory.PASS,
                    f"No dominant category exceeds {max_ratio:.0%} of rows."
                )
            return ConditionResult(
                ConditionCategory.FAIL,
                f"Columns with a dominant category (> {max_ratio:.0%}): {failing.index.tolist()}"
            )
        return self.add_condition(
            f"Dominant category ratio < {max_ratio:.0%}", condition
        )
 
    def add_condition_rare_category_ratio_less_than(self, max_ratio: float = 0.1):
        """Fail if more than max_ratio of unique categories appear only once (sparse labels)."""
        def condition(value):
            stats_df = value.get("categorical")
            if stats_df is None:
                return ConditionResult(ConditionCategory.PASS, "No categorical columns.")
            failing = stats_df[
                stats_df["rare_ratio %"].astype(float) > max_ratio * 100
            ]
            if failing.empty:
                return ConditionResult(
                    ConditionCategory.PASS,
                    f"No column has excessive rare categories (> {max_ratio:.0%})."
                )
            return ConditionResult(
                ConditionCategory.FAIL,
                f"Columns with many rare categories (freq=1 ratio > {max_ratio:.0%}): "
                f"{failing.index.tolist()}"
            )
        return self.add_condition(
            f"Rare category ratio < {max_ratio:.0%}", condition
        )

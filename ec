import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pricing_checks as pc

# ================================================================
# 1. GENERATE DATA
# ================================================================
np.random.seed(42)
n = 500

df = pd.DataFrame({
    "price":            np.random.uniform(10, 200, n),
    "cost":             np.random.uniform(5, 100, n),
    "competitor_price": np.random.uniform(10, 200, n),
    "discount":         np.random.uniform(0, 0.5, n),
    "units_sold":       np.random.randint(1, 500, n),
    "category":         np.random.choice(["Electronics", "Clothing", "Food", "Home"], n),
    "region":           np.random.choice(["North", "South", "East", "West"], n),
    "margin":           np.random.uniform(0.1, 0.6, n),
})

df["target_price"] = (
    df["cost"] * 1.3
    + df["competitor_price"] * 0.2
    + np.random.normal(0, 5, n)
)

# introduce dirty data to make checks interesting
df.loc[0:5,  "price"]    = None           # nulls
df.loc[6:8,  "category"] = "electronics"  # string mismatch
df = pd.concat([df, df.iloc[:3]], ignore_index=True)  # duplicates

# ================================================================
# 2. TRAIN / TEST SPLIT
# ================================================================
label_col    = "target_price"
cat_features = ["category", "region"]
feature_cols = ["price", "cost", "competitor_price", "discount", "units_sold", "margin"]

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# ================================================================
# 3. TRAIN MODEL
# ================================================================
X_train = train_df[feature_cols].fillna(train_df[feature_cols].mean())
X_test  = test_df[feature_cols].fillna(test_df[feature_cols].mean())
y_train = train_df[label_col]
y_test  = test_df[label_col]

model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

# ================================================================
# 4. WRAP IN pc.Dataset
# ================================================================
train_ds = pc.Dataset(train_df, label=label_col, cat_features=cat_features)
test_ds  = pc.Dataset(test_df,  label=label_col, cat_features=cat_features)

# ================================================================
# 5. BUILT-IN SUITES
# ================================================================
print("=" * 60)
print("DATA INTEGRITY SUITE")
print("=" * 60)
pc.data_integrity().run(train_ds).show()

print("=" * 60)
print("TRAIN TEST VALIDATION SUITE")
print("=" * 60)
pc.train_test_validation().run(
    train_dataset=train_ds,
    test_dataset=test_ds
).show()

print("=" * 60)
print("MODEL EVALUATION SUITE")
print("=" * 60)
pc.model_evaluation().run(
    train_dataset=train_ds,
    test_dataset=test_ds,
    y_pred_train=y_pred_train,
    y_pred_test=y_pred_test
).show()

# ================================================================
# 6. BUILT-IN INDIVIDUAL CHECKS
# ================================================================
print("=" * 60)
print("MIXED NULLS")
print("=" * 60)
pc.MixedNulls().run(train_ds).show()

print("=" * 60)
print("DATA DUPLICATES")
print("=" * 60)
pc.DataDuplicates().run(train_ds).show()

print("=" * 60)
print("STRING MISMATCH")
print("=" * 60)
pc.StringMismatch().run(train_ds).show()

print("=" * 60)
print("FEATURE DRIFT")
print("=" * 60)
pc.FeatureDrift().run(
    train_dataset=train_ds,
    test_dataset=test_ds
).show()

print("=" * 60)
print("FEATURE LABEL CORRELATION")
print("=" * 60)
pc.FeatureLabelCorrelation().run(train_ds).show()

print("=" * 60)
print("TRAIN TEST PERFORMANCE")
print("=" * 60)
pc.TrainTestPerformance().run(
    train_dataset=train_ds,
    test_dataset=test_ds,
    y_pred_train=y_pred_train,
    y_pred_test=y_pred_test
).show()

# ================================================================
# 7. CUSTOM CHECK — DESCRIPTIVE STATS
# ================================================================
print("=" * 60)
print("DESCRIPTIVE STATS (CUSTOM CHECK)")
print("=" * 60)
pc.DescriptiveStats()\
    .add_condition_null_ratio_less_than(0.05)\
    .add_condition_no_high_skewness(2.0)\
    .add_condition_no_high_kurtosis(5.0)\
    .add_condition_std_not_zero()\
    .add_condition_mean_in_range("price", 10, 500)\
    .add_condition_coefficient_of_variation_less_than(1.0)\
    .run(train_ds)\
    .show()

print("=" * 60)
print("ALL CHECKS COMPLETED SUCCESSFULLY")
print("=" * 60)

# =========================
# Deepchecks 0.19.1 - One-cell classification pipeline
# =========================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from deepchecks.tabular import Dataset, Suite
from deepchecks.tabular.suites import data_integrity, train_test_validation, model_evaluation
from deepchecks.tabular.checks import (
    ConflictingLabels,
    DataDuplicates,
    FeatureLabelCorrelation,
    TrainTestSamplesMix,
    IsSingleValue,
    MixedNulls,
    PercentOfNulls,
    SimpleModelComparison,
    CalibrationScore,
    ConfusionMatrixReport,
    RocReport,
    SegmentPerformance,
    WeakSegmentPerformance
)

# -------------------------------------------------
# 1) Select target and columns
# -------------------------------------------------
target = "y"

price_features = [
    "TAUX_NOMINAL_PRECONISE",
    "TAUX_NOMINAL_PRECONISE - pred_all_features",
    "TAUX_NOMINAL_PRECONISE - pred_restricted_features",
    "TAUX_NOMINAL_PRECONISE - TCI",
]

beta_features = [
    "MENSUALITE_MAXIMALE_CLIENT",
    "IMPOTS",
    "CREDIT_MONTANT",
    "ENCOURS_AUTRES_CREDITS_HORS_IMMO",
    "MENSUALITES_AUTRES_CREDITS",
    "MONTANT_DECOUVERT_AUTORISE",
    "CREDIT_DUREE",
    "EPARGNE_BNP",
    "BCOM-LOT1-SITLOCCEPT_0_1648_P",
    "BCOM-LOT1-SITLOCCEPT_0_1648_L",
    "BCOM-LOT1-SITLOCCEPT_0_1648_F",
    "BCOM-LOT1-SITLOCCEPT_0_1648_X",
    "BCOM-LOT1-SITLOCCEPT_0_1648_E",
    "TOTAL_REVENUS",
    "EPARGNE_BNP LIQUIDE",
]

columns_to_keep = [target] + price_features + beta_features

# Keep only existing columns
existing_cols = [c for c in columns_to_keep if c in df.columns]
missing_cols = [c for c in columns_to_keep if c not in df.columns]

print("Missing columns:", missing_cols)
print("Using columns:", existing_cols)

df2 = df[existing_cols].copy()

# -------------------------------------------------
# 2) Basic cleaning
# -------------------------------------------------
# Drop rows with missing target
df2 = df2.dropna(subset=[target]).copy()

# Convert object columns to string so Deepchecks/sklearn handle them better
for col in df2.columns:
    if df2[col].dtype == "object":
        df2[col] = df2[col].astype(str)

# Optional categorical columns
candidate_cat_features = [
    "BCOM-LOT1-SITLOCCEPT_0_1648_P",
    "BCOM-LOT1-SITLOCCEPT_0_1648_L",
    "BCOM-LOT1-SITLOCCEPT_0_1648_F",
    "BCOM-LOT1-SITLOCCEPT_0_1648_X",
    "BCOM-LOT1-SITLOCCEPT_0_1648_E",
]

cat_features = [c for c in candidate_cat_features if c in df2.columns]

# -------------------------------------------------
# 3) Encode categorical/object columns for the model
#    Deepchecks can still receive cat_features metadata
# -------------------------------------------------
X = df2.drop(columns=[target]).copy()
y = df2[target].copy()

# Make target categorical if needed
if y.dtype == "object":
    y = y.astype(str)

# Encode feature columns for sklearn model
X_model = X.copy()
for col in X_model.columns:
    if X_model[col].dtype == "object":
        X_model[col] = X_model[col].astype("category").cat.codes

# Also encode explicitly declared categorical columns if not numeric
for col in cat_features:
    if col in X_model.columns and not pd.api.types.is_numeric_dtype(X_model[col]):
        X_model[col] = X_model[col].astype("category").cat.codes

# Fill missing values for model training
for col in X_model.columns:
    if pd.api.types.is_numeric_dtype(X_model[col]):
        X_model[col] = X_model[col].fillna(X_model[col].median())
    else:
        X_model[col] = X_model[col].fillna("missing").astype("category").cat.codes

# Encode target if needed
if not pd.api.types.is_numeric_dtype(y):
    y = y.astype("category").cat.codes

# -------------------------------------------------
# 4) Train/test split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_model, y, test_size=0.2, random_state=42, stratify=y
)

train_df = pd.concat([X_train, y_train.rename(target)], axis=1)
test_df  = pd.concat([X_test, y_test.rename(target)], axis=1)

train_ds = Dataset(
    df=train_df,
    label=target,
    cat_features=[c for c in cat_features if c in train_df.columns]
)

test_ds = Dataset(
    df=test_df,
    label=target,
    cat_features=[c for c in cat_features if c in test_df.columns]
)

full_df_for_integrity = pd.concat([X_model, y.rename(target)], axis=1)
full_ds = Dataset(
    df=full_df_for_integrity,
    label=target,
    cat_features=[c for c in cat_features if c in full_df_for_integrity.columns]
)

# -------------------------------------------------
# 5) Train baseline classification model
# -------------------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# -------------------------------------------------
# 6) Run Deepchecks - Data Integrity
# -------------------------------------------------
print("\n==============================")
print("RUNNING DATA INTEGRITY SUITE")
print("==============================")
integrity_suite = data_integrity()
integrity_result = integrity_suite.run(full_ds)
integrity_result.show()
integrity_result.save_as_html("deepchecks_data_integrity.html")

# -------------------------------------------------
# 7) Run important individual checks
# -------------------------------------------------
print("\n=========================================")
print("RUNNING IMPORTANT INDIVIDUAL CHECKS")
print("=========================================")
important_checks = [
    ConflictingLabels(),
    DataDuplicates(),
    FeatureLabelCorrelation(),
    IsSingleValue(),
    MixedNulls(),
    PercentOfNulls(),
]

for check in important_checks:
    print(f"\n--- {check.name()} ---")
    try:
        result = check.run(full_ds)
        result.show()
    except Exception as e:
        print(f"Failed on {check.name()}: {e}")

# -------------------------------------------------
# 8) Run Deepchecks - Train/Test Validation
# -------------------------------------------------
print("\n===================================")
print("RUNNING TRAIN-TEST VALIDATION SUITE")
print("===================================")
tt_suite = train_test_validation()
tt_result = tt_suite.run(train_dataset=train_ds, test_dataset=test_ds)
tt_result.show()
tt_result.save_as_html("deepchecks_train_test_validation.html")

# Extra train/test leakage check
print("\n--- TrainTestSamplesMix ---")
try:
    mix_result = TrainTestSamplesMix().run(train_dataset=train_ds, test_dataset=test_ds)
    mix_result.show()
except Exception as e:
    print(f"Failed on TrainTestSamplesMix: {e}")

# -------------------------------------------------
# 9) Run Deepchecks - Model Evaluation
# -------------------------------------------------
print("\n==============================")
print("RUNNING MODEL EVALUATION SUITE")
print("==============================")
eval_suite = model_evaluation()
eval_result = eval_suite.run(
    train_dataset=train_ds,
    test_dataset=test_ds,
    model=model
)
eval_result.show()
eval_result.save_as_html("deepchecks_model_evaluation.html")

# -------------------------------------------------
# 10) Run useful classification checks individually
# -------------------------------------------------
print("\n==========================================")
print("RUNNING CLASSIFICATION-SPECIFIC CHECKS")
print("==========================================")
classification_checks = [
    SimpleModelComparison(),
    CalibrationScore(),
    ConfusionMatrixReport(),
    RocReport(),
    SegmentPerformance(),
    WeakSegmentPerformance(),
]

for check in classification_checks:
    print(f"\n--- {check.name()} ---")
    try:
        result = check.run(
            train_dataset=train_ds,
            test_dataset=test_ds,
            model=model
        )
        result.show()
    except Exception as e:
        print(f"Failed on {check.name()}: {e}")

# -------------------------------------------------
# 11) Quick summary
# -------------------------------------------------
print("\n==============================")
print("DONE")
print("==============================")
print("Saved reports:")
print("- deepchecks_data_integrity.html")
print("- deepchecks_train_test_validation.html")
print("- deepchecks_model_evaluation.html")
print("\nTrain shape:", train_df.shape)
print("Test shape :", test_df.shape)
print("Target distribution in train:")
print(pd.Series(y_train).value_counts(normalize=True).sort_index())
print("\nTarget distribution in test:")
print(pd.Series(y_test).value_counts(normalize=True).sort_index())

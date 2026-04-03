import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from deepchecks.tabular import Dataset
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
existing_cols = [c for c in columns_to_keep if c in df.columns]
df2 = df[existing_cols].copy()
df2 = df2.dropna(subset=[target]).copy()

for col in df2.columns:
    if df2[col].dtype == "object":
        df2[col] = df2[col].astype(str)

candidate_cat_features = [
    "BCOM-LOT1-SITLOCCEPT_0_1648_P",
    "BCOM-LOT1-SITLOCCEPT_0_1648_L",
    "BCOM-LOT1-SITLOCCEPT_0_1648_F",
    "BCOM-LOT1-SITLOCCEPT_0_1648_X",
    "BCOM-LOT1-SITLOCCEPT_0_1648_E",
]

cat_features = [c for c in candidate_cat_features if c in df2.columns]

X = df2.drop(columns=[target]).copy()
y = df2[target].copy()

if y.dtype == "object":
    y = y.astype(str)

X_model = X.copy()
for col in X_model.columns:
    if X_model[col].dtype == "object":
        X_model[col] = X_model[col].astype("category").cat.codes

for col in cat_features:
    if col in X_model.columns and not pd.api.types.is_numeric_dtype(X_model[col]):
        X_model[col] = X_model[col].astype("category").cat.codes

for col in X_model.columns:
    if pd.api.types.is_numeric_dtype(X_model[col]):
        X_model[col] = X_model[col].fillna(X_model[col].median())
    else:
        X_model[col] = X_model[col].fillna("missing").astype("category").cat.codes

if not pd.api.types.is_numeric_dtype(y):
    y = y.astype("category").cat.codes

X_train, X_test, y_train, y_test = train_test_split(
    X_model, y, test_size=0.2, random_state=42, stratify=y
)

train_df = pd.concat([X_train, y_train.rename(target)], axis=1)
test_df = pd.concat([X_test, y_test.rename(target)], axis=1)

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

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

integrity_suite = data_integrity()
integrity_result = integrity_suite.run(full_ds)
integrity_result.show()
integrity_result.save_as_html("deepchecks_data_integrity.html")

important_checks = [
    ConflictingLabels(),
    DataDuplicates(),
    FeatureLabelCorrelation(),
    IsSingleValue(),
    MixedNulls(),
    PercentOfNulls(),
]

for check in important_checks:
    try:
        result = check.run(full_ds)
        result.show()
    except Exception as e:
        print(f"Failed on {check.name()}: {e}")

tt_suite = train_test_validation()
tt_result = tt_suite.run(train_dataset=train_ds, test_dataset=test_ds)
tt_result.show()
tt_result.save_as_html("deepchecks_train_test_validation.html")

try:
    mix_result = TrainTestSamplesMix().run(train_dataset=train_ds, test_dataset=test_ds)
    mix_result.show()
except Exception as e:
    print(f"Failed on TrainTestSamplesMix: {e}")

eval_suite = model_evaluation()
eval_result = eval_suite.run(
    train_dataset=train_ds,
    test_dataset=test_ds,
    model=model
)
eval_result.show()
eval_result.save_as_html("deepchecks_model_evaluation.html")

classification_checks = [
    SimpleModelComparison(),
    CalibrationScore(),
    ConfusionMatrixReport(),
    RocReport(),
    SegmentPerformance(),
    WeakSegmentPerformance(),
]

for check in classification_checks:
    try:
        result = check.run(
            train_dataset=train_ds,
            test_dataset=test_ds,
            model=model
        )
        result.show()
    except Exception as e:
        print(f"Failed on {check.name()}: {e}")

print("Saved reports:")
print("deepchecks_data_integrity.html")
print("deepchecks_train_test_validation.html")
print("deepchecks_model_evaluation.html")
print(train_df.shape)
print(test_df.shape)
print(pd.Series(y_train).value_counts(normalize=True).sort_index())
print(pd.Series(y_test).value_counts(normalize=True).sort_index())

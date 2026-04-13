import mlflow
from sklearn.ensemble import RandomForestClassifier
from deepchecks.tabular.suites import data_integrity, train_test_validation, full_suite
from deepchecks.tabular.checks import FeatureLabelCorrelation


def log_suite_result(result, html_name, metric_prefix):
    # Save HTML report
    result.save_as_html(html_name)
    mlflow.log_artifact(html_name)

    # Deepchecks API: use methods, not .failed_checks
    not_passed = result.get_not_passed_checks()
    not_ran = result.get_not_ran_checks()

    mlflow.log_metric(f"{metric_prefix}_not_passed_checks", len(not_passed))
    mlflow.log_metric(f"{metric_prefix}_not_ran_checks", len(not_ran))
    mlflow.log_metric(f"{metric_prefix}_passed", int(result.passed()))


with mlflow.start_run():
    # --- STEP 1: Data Integrity Check ---
    print("🔍 Running Data Integrity Suite...")
    integrity_suite = data_integrity()
    integrity_result = integrity_suite.run(train_dataset)

    log_suite_result(
        integrity_result,
        "data_integrity.html",
        "data_integrity"
    )

    # --- STEP 2: Train-Test Validation ---
    print("🧪 Running Train-Test Validation...")
    train_test_suite = train_test_validation()
    train_test_result = train_test_suite.run(train_dataset, test_dataset)

    log_suite_result(
        train_test_result,
        "train_test_validation.html",
        "train_test_validation"
    )

    # --- STEP 3: Train Model ---
    print("🤖 Training RandomForest model...")
    features = [col for col in train_df.columns if col != "y"]
    model = RandomForestClassifier(random_state=42)
    model.fit(train_df[features], train_df["y"])

    mlflow.sklearn.log_model(model, "model")

    # --- STEP 4: Full Suite (Evaluation + Monitoring) ---
    print("📊 Running Full Suite (Evaluation + Monitoring)...")
    full_suite_result = full_suite().run(train_dataset, test_dataset, model)

    log_suite_result(
        full_suite_result,
        "full_suite.html",
        "full_suite"
    )

    # Optional: log summary counts instead of result.metrics()
    mlflow.log_metric(
        "full_suite_not_passed_checks",
        len(full_suite_result.get_not_passed_checks())
    )
    mlflow.log_metric(
        "full_suite_not_ran_checks",
        len(full_suite_result.get_not_ran_checks())
    )

    # --- STEP 5: Feature-Label Correlation (Optional) ---
    print("🔗 Running Feature-Label Correlation...")
    fld_check = FeatureLabelCorrelation()
    fld_result = fld_check.run(train_dataset)
    fld_result.save_as_html("feature_label_correlation.html")
    mlflow.log_artifact("feature_label_correlation.html")

    print("✅ MLflow run completed! Check the UI at http://localhost:5000")

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from deepchecks.tabular.suites import data_integrity, train_test_validation, model_evaluation
from deepchecks.tabular.checks import FeatureLabelCorrelation
from deepchecks.tabular import Dataset


label_col = "y"
features = [col for col in train_df.columns if col != label_col]

train_dataset = Dataset(train_df, label=label_col)
test_dataset = Dataset(test_df, label=label_col)


def log_suite_result(result, html_name, metric_prefix):
    result.save_as_html(html_name, as_widget=False)
    mlflow.log_artifact(html_name)

    not_passed = result.get_not_passed_checks()
    not_ran = result.get_not_ran_checks()

    mlflow.log_metric(f"{metric_prefix}_not_passed_checks", len(not_passed))
    mlflow.log_metric(f"{metric_prefix}_not_ran_checks", len(not_ran))
    mlflow.log_metric(f"{metric_prefix}_passed", int(result.passed()))


def log_check_result(result, html_name, metric_prefix):
    result.save_as_html(html_name, as_widget=False)
    mlflow.log_artifact(html_name)

    if hasattr(result, "passed_conditions"):
        mlflow.log_metric(f"{metric_prefix}_passed", int(result.passed_conditions()))
    else:
        mlflow.log_metric(f"{metric_prefix}_passed", 1)


def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    return model, metrics


models = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
}


with mlflow.start_run(run_name="deepchecks_multi_model_comparison"):

    print("Running shared data validation...")

    integrity_result = data_integrity().run(train_dataset)
    log_suite_result(integrity_result, "data_integrity.html", "data_integrity")

    ttv_result = train_test_validation().run(train_dataset, test_dataset)
    log_suite_result(ttv_result, "train_test_validation.html", "train_test_validation")

    model_results = []

    for model_name, model in models.items():
        print(f"Training and evaluating: {model_name}")

        with mlflow.start_run(run_name=model_name, nested=True):
            mlflow.log_param("model_name", model_name)

            trained_model, metrics = evaluate_model(
                model,
                train_df[features],
                train_df[label_col],
                test_df[features],
                test_df[label_col],
            )

            mlflow.sklearn.log_model(trained_model, "model")

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            eval_result = model_evaluation().run(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                model=trained_model
            )
            log_suite_result(
                eval_result,
                f"{model_name}_model_evaluation.html",
                "model_eval"
            )

            fld_result = FeatureLabelCorrelation().run(train_dataset)
            log_check_result(
                fld_result,
                f"{model_name}_feature_label_correlation.html",
                "feature_label_correlation"
            )

            model_results.append({"model_name": model_name, **metrics})

    best_model = max(model_results, key=lambda x: x["f1_score"])

    mlflow.log_param("best_model_name", best_model["model_name"])
    mlflow.log_metric("best_model_f1_score", best_model["f1_score"])

    print("\nModel comparison summary:")
    for row in model_results:
        print(row)

    print(f"\nBest model: {best_model['model_name']} with F1 = {best_model['f1_score']:.4f}")

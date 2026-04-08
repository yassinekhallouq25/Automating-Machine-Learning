from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity

deepchecks_train = Dataset(train_processed_df, label=TARGET_COL, cat_features=[])

integrity_suite = data_integrity()
integrity_result = integrity_suite.run(deepchecks_train)

# show in VS Code / notebook
integrity_result.show_in_iframe()

# save
integrity_html = paths.deepchecks / "data_integrity.html"
integrity_result.save_as_html(
    str(integrity_html),
    as_widget=False,
    requirejs=False,
    connected=True
)

print(integrity_html.resolve())

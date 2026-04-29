print(train_ds.data.dtypes)
print(test_ds.data.dtypes)

print(type(train_ds.features[0]), train_ds.features)
print(type(train_ds.label_name), train_ds.label_name)

pred = mapper.predict(test_ds.data[test_ds.features].head())
proba = mapper.predict_proba(test_ds.data[test_ds.features].head())

print("pred:", pred, type(pred), np.asarray(pred).dtype)
print("proba:", proba, type(proba), np.asarray(proba).dtype)



def predict_proba(self, X):
    X_input = X[["X1", "X2", "r"]].to_numpy(dtype=float)

    p1 = self.model.predict(X_input)
    p1 = np.asarray(p1, dtype=float).reshape(-1)

    return np.column_stack([1 - p1, p1]).astype(float)


    def predict(self, X):
    proba = self.predict_proba(X)
    return np.array([0, 1])[np.argmax(proba, axis=1)]


    result = suite.run(
    train_dataset=train_ds,
    test_dataset=test_ds,
    model=mapper,
    model_classes=[0, 1]
)

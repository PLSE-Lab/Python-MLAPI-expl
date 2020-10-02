import pandas as pd

train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
prepared_data = train_data.append(test_data).copy()
categorical_vars = prepared_data.columns[prepared_data.dtypes == "object"]
prepared_data[categorical_vars] = prepared_data[categorical_vars].astype(str)
prepared_data = prepared_data.fillna(0)
prepared_data = pd.get_dummies(prepared_data, prefix=categorical_vars)
from sklearn.linear_model import LinearRegression

prepared_train_data = prepared_data[prepared_data["SalePrice"] != 0]
model = LinearRegression() \
    .fit(prepared_train_data.drop("SalePrice", axis=1), prepared_train_data["SalePrice"])
prepared_test_data = prepared_data[prepared_data["SalePrice"] == 0].drop("SalePrice", axis=1)
predictions = model.predict(prepared_test_data)
predictions[predictions < 0] = 0
prepared_test_data["SalePrice"] = predictions
prepared_test_data[["Id", "SalePrice"]].to_csv("submission.csv", index=False)

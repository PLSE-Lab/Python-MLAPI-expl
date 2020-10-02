

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib

pd.options.mode.chained_assignment = None


data = pd.read_csv("../input/train.csv")

median_age = data['Age'].median()
data['Age'].fillna(median_age, inplace=True)
data_inputs = data[["Pclass", "Sex", "Age"]]
data_inputs["Sex"] = np.where(data_inputs["Sex"] == "female", 0, 1)
expected_output = data[["Survived"]]


inputs_train, inputs_test, expected_output_train, expected_output_test   = \
train_test_split (data_inputs, expected_output, test_size = 0.33, random_state = 42)

rf = RandomForestClassifier (n_estimators=100)
rf.fit(inputs_train, expected_output_train)


test = pd.read_csv("../input/test.csv")

median_age = test['Age'].median()
test['Age'].fillna(median_age, inplace=True)
test_inputs = test[["Pclass", "Sex", "Age"]]
test_inputs["Sex"] = np.where(test_inputs["Sex"] == "female", 0, 1)

prediction = rf.predict(test_inputs)

results = pd.DataFrame(data=test["PassengerId"])
results["Survived"] = prediction

results.to_csv("results.csv", index=False)

print(results.tail(1))
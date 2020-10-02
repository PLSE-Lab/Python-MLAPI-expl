import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

numerical_features = train[['Fare', 'Pclass', 'Age']]
median_features = numerical_features.dropna().median()
features_train = numerical_features.fillna(median_features)

target_train = train[['Survived']]

logreg = LogisticRegression(C=1)
logreg.fit(features_train, target_train)

numerical_test = test[['Fare', 'Pclass', 'Age']]
median_features = numerical_test.dropna().median()
features_test = numerical_test.fillna(median_features)
predictions = logreg.predict(features_test)

passengerId = list(range(892, 1310))
final_data = {'PassengerId' : passengerId, 'Survived' : predictions}
df = pd.DataFrame(final_data)
df.to_csv('submission_titanic_logistic_regression.csv', index=False)
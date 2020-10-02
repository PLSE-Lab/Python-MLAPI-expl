import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import (Imputer, LabelEncoder)
from sklearn.preprocessing import StandardScaler

import sklearn

print('The scikit-learn version is {}.'.format(sklearn.__version__))

# Importing data set
dataset = pd.read_csv('../input/titanic/train.csv')
dataset_test = pd.read_csv('../input/titanic/test.csv')

X = dataset.iloc[:, [2, 4, 5, 6, 7, 9, 11]].values
y = dataset.iloc[:, 1].values

X_test = dataset_test.iloc[:, [1, 3, 4, 5, 6, 8, 10]].values

# Label Encoder
label_encoder_X = LabelEncoder()

# Encoding label Male/Female
X[:, 1] = label_encoder_X.fit_transform(X[:, 1])
X_test[:, 1] = label_encoder_X.fit_transform(X_test[:, 1])

# Encoding label Embarked
# Missing values
X[pd.isnull(X[:,6]), 6] = 'C'

X[:, 6] = label_encoder_X.fit_transform(X[:, 6])
X_test[:, 6] = label_encoder_X.fit_transform(X_test[:, 6])

# Substitute NaN by values
# Mean
imputer_mean = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer_mean = imputer_mean.fit(X[:, 2:4])

X[:, 2:4] = imputer_mean.transform(X[:, 2:4])

imputer_mean_test = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer_mean_test = imputer_mean.fit(X_test[:, 2:4])

X_test[:, 2:4] = imputer_mean_test.transform(X_test[:, 2:4])

# Most frequent
imputer_frequent = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imputer_frequent = imputer_frequent.fit(X[:, 5:7])

X[:, 5:7] = imputer_frequent.transform(X[:, 5:7])

imputer_frequent_test = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imputer_frequent_test = imputer_frequent_test.fit(X_test[:, 5:7])

X_test[:, 5:7] = imputer_frequent.transform(X_test[:, 5:7])

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
X_train, y_train = X, y

# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting logistic regression to the training set
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predicting the test set result
passenger_ids = dataset_test.iloc[:, 0].values
y_pred = classifier.predict(X_test)

result = []

for ind, passenger_id in enumerate(passenger_ids):
    result.append([passenger_id, y_pred[ind]])

# Saving on a CSV file
np_result = np.array(result).astype(int)
np.savetxt('result.csv', np_result, fmt='%s', delimiter=',')


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset_train = pd.read_csv('../input/train.csv')
dataset_test = pd.read_csv('../input/test.csv')
X_train = dataset_train.iloc[:,[2,4,5,6,7,9]].values
y_train = dataset_train.iloc[:, 1].values
X_test = dataset_test.iloc[:,[1,3,4,5,6,8]].values

                
# Replacing NaN in Ages
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="median", axis=0, verbose=0, copy=True)
imputer.fit(X_train[:, 2:3])
X_train[:, 2:3] = imputer.transform(X_train[:, 2:3])
imputer.fit(X_test[:, 2:9])
X_test[:, 2:9] = imputer.transform(X_test[:, 2:9])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X_train[:, 1] = labelencoder.fit_transform(X_train[:, 1])
X_test[:, 1] = labelencoder.fit_transform(X_test[:, 1])

onehotencoder = OneHotEncoder(categorical_features = [1])
X_test = onehotencoder.fit_transform(X_test).toarray()
X_train = onehotencoder.fit_transform(X_train).toarray()

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

y_pred_export = pd.DataFrame(y_pred)
y_pred_export.to_csv('prediction.csv', index=True + 892, header=True, index_label=["PassengerId","Survived"])

# Any results you write to the current directory are saved as output.
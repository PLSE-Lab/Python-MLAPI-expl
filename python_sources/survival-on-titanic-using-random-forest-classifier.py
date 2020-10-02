# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Importing the dataset. Importing Pclass, Sex, Age, SibSp, Parch columns for train.csv and test.csv. Also importing Survived column in variable y.
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, [2,4,5,6,7]].values
y = dataset.iloc[:, 1].values

testdataset = pd.read_csv('test.csv')
X_test = testdataset.iloc[:, [1,3,4,5,6]].values
passengerId = testdataset.iloc[:,0]

# Replacing the missing values in Age column with mean of the age column
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer_train = imputer.fit(X[:, 2:3])
X[:, 2:3] = imputer_train.transform(X[:,2:3])

imputer_test = imputer.fit(X_test[:, 2:3])
X_test[:, 2:3] = imputer_test.transform(X_test[:,2:3])

# Taking care of categorical data (sex)
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:,1])

X_test[:, 1] = labelencoder_X.fit_transform(X_test[:,1])

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
# Create your classifier here
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'gini', random_state = 0)
classifier.fit(X, y)

# Predicting the Test set results and generating results.csv file.
y_pred = classifier.predict(X_test)
y_pred = pd.DataFrame(y_pred)
y_pred.columns = ['Survived']
results = pd.concat([passengerId, y_pred], axis = 1)

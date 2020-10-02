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

# Applying Logistics Regression Model

#import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Get the Training Dataset
dataset = pd.read_csv('train.csv')
dataset.fillna(dataset.mean()[5], inplace = True)
y = dataset.iloc[:, 1].values
X = dataset.iloc[:, [2,4,5,6,7]].values

#Get the test dataset
test_dataset = pd.read_csv('test.csv')
test_dataset.fillna(test_dataset.mean()[4], inplace=True)
X_test = test_dataset.iloc[:, [1,3,4,5,6]].values
y_test = test_dataset.iloc[:, 0].values

#Encoding Categorical data i.e Sex field in Training Dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_X = LabelEncoder()
X[:, 1] = le_X.fit_transform(X[:, 1])
hotencoder = OneHotEncoder(categorical_features = [1])
X = hotencoder.fit_transform(X).toarray()

#Encoding Categorical data i.e Sex field in Test Dataset
le_X_test = LabelEncoder()
X_test[:, 1] = le_X_test.fit_transform(X_test[:, 1])
hotencoder = OneHotEncoder(categorical_features = [1])
X_test = hotencoder.fit_transform(X_test).toarray()

# Feature Scaling for Age field
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
X_test = sc_X.transform(X_test)

#Fit the training data to the Logistic Regression Model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X,y)

#Test the model
y_pred = classifier.predict(X_test)

#Format the data into expected format
y_result = np.append(np.reshape(y_test, [418,1]), np.reshape(y_pred, [418,1]), axis=1)


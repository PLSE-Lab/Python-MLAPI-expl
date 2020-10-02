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

import numpy as np
import pandas as pd

#Importing the dataset

dataset = pd.read_csv("../input/train.csv")
y = dataset.iloc[:,1].values
X_train = dataset.iloc[:,[2,4,5,6,7,11]].values
test_data= pd.read_csv("../input/test.csv")
X_test = test_data.iloc[:,[1,3,4,5,6,10]].values
#Data preprocessing
#Taking care of the Missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis =0)
imputer = imputer.fit(X_train[:,2:3])#Imputer class accepts only 2D arrays
X_train[:,2:3]= imputer.transform(X_train[:,2:3])
imputer = imputer.fit(X_test[:,2:3])  #imputer class accepts only 2-D array
X_test[:,2:3] = imputer.transform(X_test[:,2:3])

#Find and replace the missing values in the Embark feature

for i in range(len(X_train)):
     if (pd.isnull(X_train[i,5])):
          X_train[i,5] = "null"

#Encoding the categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
X_train[:,1] = le.fit_transform(X_train[:,1])
X_train[:,5] = le.fit_transform(X_train[:,5])
X_test[:,1] = le.fit_transform(X_test[:,1])
X_test[:,5] = le.fit_transform(X_test[:,5])
#Replacing the encoded value of the missing feature values by the most frequent value
from statistics import mode
for i in range(len(X_train)):
     if (X_train[i,5] == 3):
          X_train[i,5] = mode(X_train[:,5])
         

ohe = OneHotEncoder(categorical_features = [0])
X_train = ohe.fit_transform(X_train).toarray()
X_test = ohe.fit_transform(X_test).toarray()
X_train = X_train[:,1:]
X_test = X_test[:,1:]
ohe2 = OneHotEncoder(categorical_features = [6])
X_train = ohe2.fit_transform(X_train).toarray()
X_test = ohe2.fit_transform(X_test).toarray()
X_train = X_train[:,1:]
X_test = X_test[:,1:]

#Feature Scaling for the age column
from sklearn.preprocessing import  StandardScaler

std_X = StandardScaler()

X_train_std = std_X.fit_transform(X_train)
X_test_std = std_X.fit_transform(X_test)

#Classification model

"""from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', gamma = 0.2)
classifier.fit(X_train,y)
"""
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train_std,y)
#Predicting the testset survivals

y_pred = classifier.predict(X_test_std)
print(y_pred)

my_submission = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_pred})
# you could use any filename. We choose submission here
my_submission.to_csv('survival_2.csv', index=False)
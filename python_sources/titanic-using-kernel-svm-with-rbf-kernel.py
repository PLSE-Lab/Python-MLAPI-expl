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
dataset=pd.read_csv('../input/train.csv')
dataset_y = pd.read_csv('../input/test.csv')
X_train = dataset.iloc[:,[2,4,5,6,7,9,11]]
X_test = dataset_y.iloc[:,[1,3,4,5,6,8,10]]
Y_train = dataset.iloc[:,[1]]

# replace non numeric variables by numeric ones
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

#aligning train and test predictors
X_train, X_test = X_train.align(X_test, join='left', axis=1)

from sklearn.impute import SimpleImputer
impu = SimpleImputer()
X_train = pd.DataFrame(impu.fit_transform(X_train))
X_test = pd.DataFrame(impu.fit_transform(X_test))

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state=0)
classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_test)
dataframe_y_pred = pd.DataFrame(y_pred)

#Creating PassengerID column
e =[]
for num in range(892, 1310):
    e.append(num)
# adding the new column to y_pred DataFrame
dataframe_y_pred['e'] = e

#adding headers after being deleted during imputation
dataframe_y_pred.columns=['Survived', 'PassengerId']

#swiching colums to the right order to match the needed output formula
dataframe_y_pred = dataframe_y_pred[['PassengerId', 'Survived']]

dataframe_y_pred.to_csv('Titanic_Kernel_svm.csv',index=False)
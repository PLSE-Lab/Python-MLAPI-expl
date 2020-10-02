# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
dataset = pd.read_csv('../input/insurance.csv')

#Getting X and y
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

X=pd.DataFrame(X)

#Converting Sex, Smoker and Region into integers using Label Encoder
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X[1]=le.fit_transform(X[1])
X[4]=le.fit_transform(X[4])
X[5]=le.fit_transform(X[5])

#Scalling the data
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

#Splitting into train and test
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8)

#Applying linear Regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)

#Getting the prediction from the model
y_pred=lr.predict(X_test)


#Using various methods to calculate accuracy
from sklearn.metrics import accuracy_score,explained_variance_score,r2_score
r2_score(y_test,y_pred)
explained_variance_score(y_test,y_pred)



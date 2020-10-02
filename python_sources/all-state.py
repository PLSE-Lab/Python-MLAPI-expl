# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

"""
Created on Sat Sep  7 14:54:15 2019

@author: NAVEEN
"""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_csv('E:/Data/allstate/train.csv')

label = LabelEncoder()
label.fit(np.unique(df.iloc[:,1:116].values))
#1-116 category
#117-131 continous
df.iloc[:,1:116] = df.iloc[:,1:116].apply(label.transform)

corr = df.corr()


mapping = dict(zip(label.classes_, label.transform(label.classes_)))

X = df.iloc[:,[7,10,12,57,79,80,81,87,101]]

Y = df['loss']

model = LinearRegression()
X_train, X_val, y_train, y_val = train_test_split(
    X, Y , train_size = 0.8 ,random_state=10)
#kfold = KFold(n_splits=3, shuffle=True, random_state=42)

reg = LinearRegression().fit(X, Y)
print(reg.score(X, Y))
y_pred = reg.predict(X_val)


dr = pd.read_csv('E:/Data/allstate/test.csv')

dr.iloc[:,[7,10,12,57,79,80,81,87,101]] = dr.iloc[:,[7,10,12,57,79,80,81,87,101]].apply(label.transform)

X_test = dr.iloc[:,[7,10,12,57,79,80,81,87,101]]
y_pred = reg.predict(X_test)


pd.DataFrame(y_pred).to_csv('E:/Data/allstate/result.csv')
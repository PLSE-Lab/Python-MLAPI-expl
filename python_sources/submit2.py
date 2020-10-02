# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 20:21:48 2017

@author: zahra
"""
from math import sqrt
import numpy as np
import pandas as pd 
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

# load data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train2 = train
test2 = test
train = train.loc[:,'MSSubClass':'SaleCondition']
test = test.loc[:,'MSSubClass':'SaleCondition']


#filling NA's with the mean of the column:
test = test.fillna(test.median())
train = train.fillna(train.median())

#change categorial features to numerical
train = pd.get_dummies(train)
test = pd.get_dummies(test)


#drop the features in train that doesn't exist in test
for a in list(train):
    flag =0
    for b in list(test):
        if a == b:
            flag = 1
    if flag==0:
        train = train.drop(a, axis=1)
#drop the features in test that doesn't exist in train
for a in list(test):
    flag =0
    for b in list(train):
        if a == b:
            flag = 1
    if flag==0:
        test = test.drop(a, axis=1)
        
# Normalize data to N(0, 1)
train = normalize(train)
test = normalize(test)

#train2 = normalize(train2)
test = test.drop(train.std()[train.std() < 0.05].index.values, axis=1)
train = train.drop(train.std()[train.std() < 0.05].index.values, axis=1)


lso = linear_model.Lasso(alpha=100, max_iter=50000)
lso.fit(train, train2['SalePrice'])

x1 = train.values
y1 = train2['SalePrice']  
kf = KFold(n_splits=10)
sumrms = 0
xv = x1
yv = y1.values

for train_i, test_i in kf.split(x1):
     x_train, x_test = xv[train_i,:], xv[test_i,:]            
     y_train, y_test = yv[train_i], yv[test_i]
     lso.fit(x_train,y_train)
     p = lso.predict(x_test)   
     rms = sqrt(mean_squared_error(y_test, p))
     sumrms += (rms ** 2)
      
print('\n RMSE(10-fold cross validation) is : ', sqrt(sumrms/10),'\n')

# predict values for test set
X_test = test.values
prediction_test = lso.predict(X_test)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
HouseId =np.array(test2["Id"]).astype(int)
my_solution = pd.DataFrame(prediction_test, HouseId, columns = ["SalePrice"])
print(my_solution)
# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("P2_submission.csv", index_label = ["Id"])

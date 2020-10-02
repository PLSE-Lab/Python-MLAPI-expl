# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 13:33:48 2020

@author: Ashish.Gupta
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input/house-prices-advanced-regression-techniques'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))
        train=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
        test=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")






#Filling NA Values
variable=[['LotFrontage','MasVnrArea','GarageYrBlt']]

for i in variable:
    train[i] = train[i].fillna(train[i].mean())


#Dropping Columns
train = train.drop(columns=['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'])

#Defining Categorical Variables
categorical=['object']
categorical_var = train.select_dtypes(include=categorical).columns.tolist()


#Filling null vallues of Categorical Variables 
for x in categorical_var:
    train[x] = train[x].fillna('No Value')
    

#Label Encoding categorical values
label=LabelEncoder()
for x in categorical_var:
    train[x] = label.fit_transform(train[x])


train['Neighborhood'] = label.fit_transform(train['Neighborhood'])
train['SalePrice'] = label.fit_transform(train['SalePrice'])
train['GarageYrBlt'] = label.fit_transform(train['GarageYrBlt'])


train['SalePrice']=train['SalePrice'].astype(int)

#train['MasVnrArea'] = pd.to_numeric(train['MasVnrArea'])
y1=train['SalePrice']
x1=train.drop(['SalePrice'],axis=1)
#print(x_train.head(5))



#Normalizing variables
X_scaled = scale(x1)
Y_scaled = scale(y1)

#Train and Test
x_train,x_test,y_train,y_test = train_test_split(X_scaled,Y_scaled,test_size=0.3,random_state=101)





#Gradient Boost
#print("Gradient Boost")
GBModel=GradientBoostingRegressor(
            random_state=0, 
            n_estimators=500, max_features=20, max_depth=5,
            learning_rate=0.05, subsample=0.8
        )
GBModel.fit(x_train,y_train)
score = GBModel.score(x_test, y_test)
#print(score)



#Working on Test Dataset


#Filling NA Values
variable=[['LotFrontage','MasVnrArea','GarageYrBlt','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageCars','GarageArea']]

for i in variable:
    test[i] = test[i].fillna(test[i].mean())
    
#Dropping Columns
test = test.drop(columns=['Alley','FireplaceQu','PoolQC','Fence','MiscFeature',])

#Defining Categorical Variables
categorical=['object']
categorical_var = test.select_dtypes(include=categorical).columns.tolist()

#Filling null vallues of Categorical Variables 
for x in categorical_var:
    test[x] = test[x].fillna('No Value')
    
    
#Label Encoding categorical values

for x in categorical_var:
    test[x] = label.fit_transform(test[x])


test['Neighborhood'] = label.fit_transform(test['Neighborhood'])
test['GarageYrBlt'] = label.fit_transform(test['GarageYrBlt'])

#train['MasVnrArea'] = pd.to_numeric(train['MasVnrArea'])

X_scaled2 = scale(test)


#print("Test Predictions")
test_predictions = GBModel.predict(X_scaled2)

df_submit = pd.DataFrame(test_predictions, columns=['SalePrice']).set_index(test.index)

df_submit.to_csv("../working/ashish_house_price.csv")

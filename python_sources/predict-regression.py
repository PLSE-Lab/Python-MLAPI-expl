#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/boston-house-prices/housing.csv',names = ['data'])


# In[ ]:


df.head()


# In[ ]:


d = ['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','b','lstat','medv']


# In[ ]:


a = list(range(14))
a = [str(i) for i in a]
a


# In[ ]:


df1 = pd.DataFrame(df,columns = a)
#df1 = df1.fillna(0)


# In[ ]:


df1['data'] = df.data


# In[ ]:


for i in a:
    df1[i] = df1['data'].apply(lambda x : x.split()[int(i)])


# In[ ]:


df1 = df1[a]
df1.columns = d


# In[ ]:


df1.head()


# # Types of Regressions:
# * Linear Regression
# * Ada boosting
# * Gradient boosting regression
# * Logistic Regression
# * Polynomial Regression
# * Stepwise Regression
# * Ridge Regression
# * Lasso Regression
# * ElasticNet Regression
# * Regressor chain
# 
# 
# 

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[ ]:


X = df1[['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','b','lstat']]
y = df1['medv']


# # 1 . Linear regression

# In[ ]:


linR = LinearRegression(normalize=True)


# In[ ]:


trainX , testX ,trainY, testY = train_test_split( X, y, test_size=0.33, random_state=42)


# In[ ]:


linR.fit(trainX,trainY)


# # accuracy of linear regression

# In[ ]:


#accuracy of linear regression
print(linR.score(testX,testY))


# # 2 . Ada Boost Algorithm

# In[ ]:


from sklearn.ensemble import AdaBoostRegressor

regr = AdaBoostRegressor(random_state=0, n_estimators=100)
regr.fit(trainX, trainY)


# Accuracy of Ada Boost Algorithm

# In[ ]:


regr.score(testX,testY)


# # 3 . Gradient boosting regression

# In[ ]:


from sklearn import ensemble

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
xg_model = ensemble.GradientBoostingRegressor(**params)


xg_model.fit(trainX,trainY)


# # accuracy gradient boosting regression

# In[ ]:


#accuracy of gradient boosting regression
print(xg_model.score(testX,testY))


# # 4 . Polynomial Equation

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2)
poly.fit_transform(trainX)
linR.fit(trainX,trainY)


# In[ ]:


print(linR.score(testX,testY))


# We will try to check how the increase in degree of polynomial changes the accuracy score of the predictor

# In[ ]:


dict = { }


# In[ ]:


for i in range(2,10):
    
    poly.fit_transform(trainX)
    linR.fit(trainX,trainY)
    dict.update({ i: linR.score(testX,testY)})


# In[ ]:


dict


# # 5.  Stepwise Regression
# The aim of this modeling technique is to maximize the prediction power with minimum number of predictor variables. It is one of the method to handle higher dimensionality of data set.
# 
# The method is to determine that how much a feature can contribute to the accuracy of the prediction.

# In[ ]:


trainX , testX ,trainY, testY = train_test_split( X, y, test_size=0.33, random_state=42)


# In[ ]:


features = set(X.columns)


# In[ ]:


from sklearn import ensemble as en

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
xg_model = en.GradientBoostingRegressor(**params)


xg_model.fit(trainX,trainY)


# In[ ]:


dict={ }


# In[ ]:


for feat in features:
    train = trainX[list(features - set(feat))]
    test = testX[list(features - set(feat))]
    xg_model.fit(train ,trainY)
    dict.update({ feat : xg_model.score(test,testY)})
    


# In[ ]:


dict


# In[ ]:


sorted_x =sorted(dict.items(), key=lambda x: x[1], reverse=True)


# In[ ]:


sorted_x


# # 6 . Ridge Regression

# In[ ]:


trainX , testX ,trainY, testY = train_test_split( X, y, test_size=0.33, random_state=42)


# In[ ]:


from sklearn import linear_model as lin
reg = lin.Ridge(alpha = 0.001)
reg.fit(trainX,trainY)


# In[ ]:


#accuracy of the ridge regression
print(reg.score(testX,testY))


# # 7 . Lasso Regression

# In[ ]:


from sklearn import linear_model as lin
las = lin.Lasso(alpha = 0.001)
las.fit(trainX,trainY)


# In[ ]:


#accuracy of the ridge regression
print(las.score(testX,testY))


# # 8 . ElasticNet Regression

# In[ ]:


from sklearn.linear_model import ElasticNet
regr = ElasticNet(alpha = 0.5)
regr.fit(trainX,trainY)
#accuracy of the ridge regression
print(regr.score(testX,testY))


# In[ ]:





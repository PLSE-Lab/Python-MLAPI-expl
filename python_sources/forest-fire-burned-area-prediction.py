#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import BaggingRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


dataset = pd.read_csv('../input/forestfires.csv')
dataset.head()


# In[ ]:


dataset.describe(include='all')


# In[ ]:


dataset.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
dataset.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7), inplace =True)


# In[ ]:


dataset.head()


# In[ ]:


corr = dataset.corr(method='pearson')
print("Correlation of the Dataset:",corr)


# In[ ]:


f,ax = plt.subplots(figsize=(18, 18))
print("Plotting correlation:")
sns.heatmap(corr,annot= True, linewidths=.5)


# In[ ]:


data = dataset.values

X = data[:,0:12]
Y = data[:,12]


# In[ ]:


extraTreesRegressor = ExtraTreesRegressor()
rfe = RFE(extraTreesRegressor,5)
fit = rfe.fit(X,Y)

print("The number of features:", fit.n_features_)
print("Selected Features:", fit.support_)
print("Feature Rankings:", fit.ranking_)


# In[ ]:


dataset.plot(kind='density', subplots=True, layout=(4,4))


# In[ ]:


features_train, features_test, target_train, target_test = train_test_split(X, Y, test_size=0.10, random_state=45)


# In[ ]:


print("Linear Regression")
Lreg = LinearRegression()
Lreg.fit(features_train,target_train)
prediction = Lreg.predict(features_test)
score = explained_variance_score(target_test, prediction)
mae = mean_absolute_error(prediction, target_test)
print('Variance score: %.2f' % r2_score(target_test, prediction))

print("Score:", score)
print("Mean Absolute Error:", mae)


# In[ ]:


print("Lasso Regression")
lasso = Lasso()
lasso.fit(features_train,target_train)
prediction_lasso = lasso.predict(features_test)
score = explained_variance_score(target_test, prediction_lasso)
mae = mean_absolute_error(prediction_lasso, target_test)

print('Variance score: %.2f' % r2_score(target_test, prediction_lasso))
print("Score:", score)
print("Mean Absolute Error:", mae)


# In[ ]:


print("Ridge Regression")
ridge = Ridge()
ridge.fit(features_test,target_test)
prediction_ridge = ridge.predict(features_test)
score = explained_variance_score(target_test, prediction_ridge)
mae = mean_absolute_error(prediction_ridge, target_test)

print('Variance score: %.2f' % r2_score(target_test, prediction_ridge))
print("Score:", score)
print("Mean Absolute Error:", mae)


# In[ ]:


print('K-Neighbors Regressor')
knreg = KNeighborsRegressor(n_neighbors=16)
knreg.fit(features_train,target_train)
prediction_knreg = knreg.predict(features_test)
score = explained_variance_score(target_test, prediction_knreg)
mae = mean_absolute_error(prediction_knreg, target_test)

print('Variance score: %.2f' % r2_score(target_test, prediction_knreg))
print("Score:", score)
print("Mean Absolute Error:", mae)


# In[ ]:


print('Random Forest Regressor')
rfreg = RandomForestRegressor()
rfreg.fit(features_train,target_train)
prediction_rfreg = rfreg.predict(features_test)
score = explained_variance_score(target_test, prediction_rfreg)
mae = mean_absolute_error(prediction_rfreg, target_test)
print('Variance score: %.2f' % r2_score(target_test, prediction_rfreg))
print("Score:", score)
print("Mean Absolute Error:", mae)


# In[ ]:


print('BaggingRegressor')
Breg = BaggingRegressor()
Breg.fit(features_train,target_train)
prediction_Breg = Breg.predict(features_test)
score = explained_variance_score(target_test, prediction_Breg)
mae = mean_absolute_error(prediction_Breg, target_test)
print('Variance score: %.2f' % r2_score(target_test, prediction_Breg))
print("Score:", score)
print("Mean Absolute Error:", mae)


# In[ ]:


print('Support Vector Regressor')
svr = SVR(kernel='poly')
svr.fit(features_train,target_train)
prediction_svr = svr.predict(features_train)
score = explained_variance_score(target_test, prediction_svr)
mae = mean_absolute_error(prediction_svr, target_test)
print('Variance score: %.2f' % r2_score(target_test, prediction_svr))
print("Score:", score)
print("Mean Absolute Error:", mae)


# In[ ]:


poly = PolynomialFeatures(degree=2)
features_train_poly = poly.fit_transform(features_train)
features_test_poly = poly.fit_transform(features_test)
Lreg_polu = LinearRegression()
Lreg.fit(features_train_poly,target_train)
prediction_poly = Lreg.predict(features_test_poly)
score = explained_variance_score(target_test, prediction_poly)
mae = mean_absolute_error(prediction_poly, target_test)
print('Variance score: %.2f' % r2_score(target_test, prediction_poly))

print("Score:", score)
print("Mean Absolute Error:", mae)


# In[ ]:


# print('Random Forest Regressor')
# Ireg = IsotonicRegression()
# Ireg.fit(features_train,target_train)
# prediction_Ireg = Ireg.predict(features_test)
# score = explained_variance_score(target_test, prediction_Ireg)
# mae = mean_absolute_error(prediction_Ireg, target_test)

# print("Score:", score)
# print("Mean Absolute Error:", mae)


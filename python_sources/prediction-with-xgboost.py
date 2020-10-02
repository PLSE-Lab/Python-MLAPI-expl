#!/usr/bin/env python
# coding: utf-8

# This notebook uses the output database from https://www.kaggle.com/nickrood/can-star-power-predict-box-office-revenue/output
# In that notebook I made predictions with a linear and random forest regressor model, but I also want to see how well a XGBoost model works on the data becasue it seems to outperform a lot of other algorithms. 

# In[1]:


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
#insert submission file
df_submission=pd.read_csv('../input/tmdb-box-office-prediction/sample_submission.csv')
df_submission.head(1)


# In[134]:


#insert commbined dataset with 78 features and target
df_tmdb = pd.read_csv("../input/tmdb-2/tmdb.csv")
df_tmdb.head(5)


# In[135]:


#insert some libraries
import json
import ast
from pprint import pprint
import seaborn as sns 
from scipy.stats import norm,skew
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from math import sqrt
pd.set_option('display.max_columns', None)


# In[136]:


#split tmdb into train and test set
df_train= df_tmdb.iloc[0:3000] # first 3000 rows of the tmdb dataframe
df_test=df_tmdb.iloc[3000:7398]


# In[137]:


df_train.shape, df_test.shape


# In[138]:


#split data into features and target. All features are either integer, float or dummy.
features = df_train.select_dtypes(include=['int64', 'float64', 'uint8', 'int8']).columns.tolist()
features.remove('log_revenue')
features_unseen = df_test.select_dtypes(include=['int64', 'float64', 'uint8', 'int8']).columns.tolist()
features_unseen.remove('log_revenue')

X, y = df_train[features], df_train['log_revenue']


# In[139]:


#split data into train and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


# First settings, R2=0,36, RMSE 2.45
# xg_reg = xgb.XGBRegressor(objective ='reg:linear', 
#                           #colsample_bytree = 0.3, #colsample_bytree= percentage of features user per tree. Higher values can lead to overfitting
#                           #submsample=0.5, #percentage of samples used per tree. Lower values can lead to underfitting 
#                           learning_rate= 0.01,
#                           max_depth = 5, 
#                           alpha = 0.01, #L1 regularization
#                           n_estimators = 10)
# model_xgb=xgb.XGBRegressor(max_depth=5,
#                            learning_rate=0.1, 
#                            n_estimators=2000, 
#                            objective='reg:linear', 
#                            gamma=1.45, 
#                            verbosity=3,
#                            subsample=0.25, 
#                            colsample_bytree=0.8, 
#                            colsample_bylevel=0.50)
# 
# 

# In[104]:


#final settings of model
model_xgb=xgb.XGBRegressor(max_depth=5,
                           learning_rate=0.10, 
                           n_estimators=115, 
                           objective='reg:linear', 
                           gamma=10,
                           alpha=0.5,
                           verbosity=3,
                           subsample=0.5, #percentage of samples used per tree. Lower values can lead to underfitting 
                           colsample_bytree=0.8) #percentage of features user per tree. Higher values can lead to overfitting


# eval_set = [(X_train, y_train), (X_test, y_test)]
# eval_metric = ["rmse"]
# %time model_xgb.fit(X_train, y_train, eval_metric=eval_metric, eval_set=eval_set, verbose=True)

# In[106]:


#fit model and make predictions 
model_xgb.fit(X_train,y_train)

y_pred=model_xgb.predict(X_test)


# In[107]:


#look at rmse and R-Squared 
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(y_test, y_pred))


print ("R-Squared is:", metrics.r2_score(y_test, y_pred))
print ("The rmse is:", rmse)


# In[108]:


#look at actual and predicted values 
compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
compare.head(5)


# In[109]:


# look at actual and predicted values of first 50 entries in the dataset
compare1 = compare.head(50)
compare1.plot(kind='bar',figsize=(30,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[110]:


#make prediction on unseen test dataset
X_unseen=df_test[features_unseen]
prediction_unseen= model_xgb.predict(X_unseen)


# In[111]:


prediction_unseen


# In[112]:


df_submission['revenue'] = np.expm1(prediction_unseen)


# In[113]:


df_submission.head(5)


# In[114]:


df_submission[['id','revenue']].to_csv('submission_xgb1.csv', index=False)


# In[140]:


#run model with Kfold splits
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import loadtxt


# In[141]:


# k-fold cross validation evaluation of xgboost model

import xgboost
from sklearn.model_selection import KFold

#final settings of model
model_xgb2=xgb.XGBRegressor(max_depth=5,
                           learning_rate=0.10, 
                           n_estimators=115, 
                           objective='reg:linear', 
                           gamma=10,
                           alpha=0.5,
                           verbosity=3,
                           subsample=0.5, #percentage of samples used per tree. Lower values can lead to underfitting 
                           colsample_bytree=0.8) #percentage of features user per tree. Higher values can lead to overfitting
kfold=KFold(n_splits=5, random_state=7)
results = cross_val_score(model_xgb2, X, y, cv=kfold)




# In[142]:


results


# In[143]:


#fit model and make predictions 
model_xgb2.fit(X_train,y_train)

y_pred2=model_xgb2.predict(X_test)


# In[144]:


#look at rmse and R-Squared 
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(y_test, y_pred2))


print ("R-Squared is:", metrics.r2_score(y_test, y_pred2))
print ("The rmse is:", rmse)


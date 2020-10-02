#!/usr/bin/env python
# coding: utf-8

# **Covid-19 Week 4 **

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


df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')
df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
df_submit = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')


# In[ ]:


df_train.describe()


# In[ ]:


df_train.head()


# In[ ]:


#Converting the object type column into datetime type
df_train['Date'] = df_train.Date.apply(pd.to_datetime)
df_test['Date'] = df_test.Date.apply(pd.to_datetime)


# In[ ]:


df_train.insert(1,'Month',df_train['Date'].dt.month)

df_train.insert(2,'Day',df_train['Date'].dt.day)


# In[ ]:


df_test.insert(1,'Month',df_test['Date'].dt.month)

df_test.insert(2,'Day',df_test['Date'].dt.day)


# In[ ]:


df_train['Province_State'].fillna(df_train['Country_Region'],inplace=True)
df_test['Province_State'].fillna(df_test['Country_Region'],inplace=True)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df_train.Country_Region = le.fit_transform(df_train.Country_Region)
df_train['Province_State'] = le.fit_transform(df_train['Province_State'])

df_test.Country_Region = le.fit_transform(df_test.Country_Region)
df_test['Province_State'] = le.fit_transform(df_test['Province_State'])


# In[ ]:


#Avoiding duplicated data.
df_train = df_train.loc[:,~df_train.columns.duplicated()]
df_test = df_test.loc[:,~df_test.columns.duplicated()]
print (df_test.shape)


# In[ ]:


# Dropping the object type columns

objList = df_train.select_dtypes(include = "object").columns
df_train.drop(objList, axis=1, inplace=True)
df_test.drop(objList, axis=1, inplace=True)
print (df_train.shape)


# In[ ]:


df_train.drop('Date',axis=1,inplace=True)
df_test.drop('Date',axis=1,inplace=True)


# In[ ]:


X = df_train.drop(['Id','ConfirmedCases', 'Fatalities'], axis=1)
y = df_train[['ConfirmedCases', 'Fatalities']]


# In[ ]:


from sklearn.model_selection import ShuffleSplit, cross_val_score,train_test_split
from sklearn.metrics import make_scorer, r2_score, mean_squared_log_error
skfold = ShuffleSplit(random_state=7)
import xgboost as xgb


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[ ]:


clf_CC = xgb.XGBRegressor(n_estimators = 10000)
clf_Fat = xgb.XGBRegressor(n_estimators = 9000)


# In[ ]:


xgb_cc = cross_val_score(clf_CC, X_train, y_train['ConfirmedCases'], cv = skfold)
xgb_fat = cross_val_score(clf_Fat, X_train, y_train['Fatalities'], cv = skfold)

print (xgb_cc.mean(), xgb_fat.mean())


# In[ ]:


X_test_CC = df_test.drop(['ForecastId'],axis=1)
X_test_Fat = df_test.drop(['ForecastId'],axis=1)


# In[ ]:


clf_CC.fit(X_train, y_train['ConfirmedCases'])
Y_pred_CC = clf_CC.predict(X_test_CC) 

clf_Fat.fit(X_train, y_train['Fatalities'])
Y_pred_Fat = clf_Fat.predict(X_test_Fat) 


# In[ ]:


df_cc = pd.DataFrame(Y_pred_CC)
df_fat = pd.DataFrame(Y_pred_Fat)


# In[ ]:


import warnings
warnings.filterwarnings('ignore')

# Calling DataFrame constructor on list 
df_results = pd.DataFrame(columns=['ForecastId','ConfirmedCases','Fatalities']) 
df_results


# In[ ]:


df_results['ForecastId'] = df_test['ForecastId']
df_results['ConfirmedCases'] = df_cc.astype(int)
df_results['Fatalities'] = df_fat.astype(int)

df_results.head()


# In[ ]:


df_results.to_csv('submission.csv', index=False)


# In[ ]:





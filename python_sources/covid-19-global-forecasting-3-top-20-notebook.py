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


from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from fbprophet import Prophet


# In[ ]:


train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')
test=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')
submission=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/submission.csv')


# In[ ]:


train['Date'] = pd.to_datetime(train['Date'], infer_datetime_format=True)
test['Date'] = pd.to_datetime(test['Date'], infer_datetime_format=True)

train.loc[:, 'Date'] = train.Date.dt.strftime("%m%d")
train["Date"]  = train["Date"].astype(int)

test.loc[:, 'Date'] = test.Date.dt.strftime("%m%d")
test["Date"]  = test["Date"].astype(int)


# In[ ]:


X_train=train.drop(columns=['Id','ConfirmedCases','Fatalities'])
y_train_cc=train.ConfirmedCases
y_train_ft=train.Fatalities
X_test=test.drop(columns=['ForecastId'])


# In[ ]:


EMPTY_VAL = "EMPTY_VAL"

def fillState(state, country):
    if state == EMPTY_VAL: return country
    return state


# In[ ]:


X_train['Province_State'].fillna(EMPTY_VAL, inplace=True)
X_train['Province_State'] = X_train.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)

X_test['Province_State'].fillna(EMPTY_VAL, inplace=True)
X_test['Province_State'] = X_test.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)


# In[ ]:


le = LabelEncoder()

X_train['Country_Region'] = le.fit_transform(X_train['Country_Region'])
X_train['Province_State'] = le.fit_transform(X_train['Province_State'])

X_test['Country_Region'] = le.fit_transform(X_test['Country_Region'])
X_test['Province_State'] = le.fit_transform(X_test['Province_State'])


# In[ ]:


model=DecisionTreeRegressor()


# In[ ]:


model.fit(X_train, y_train_cc)
print(model.score(X_train, y_train_cc))
y_pred_cc=model.predict(X_test)


# In[ ]:


model.fit(X_train,y_train_ft)
print(model.score(X_train, y_train_ft))
y_pred_ft=model.predict(X_test)


# In[ ]:


result=pd.DataFrame({'ForecastId':submission.ForecastId, 'ConfirmedCases':y_pred_cc, 'Fatalities':y_pred_ft})
result.to_csv('/kaggle/working/submission.csv', index=False)
data=pd.read_csv('/kaggle/working/submission.csv')
data.head(20)


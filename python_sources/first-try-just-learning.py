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


import seaborn as sns


# In[ ]:


train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')


# In[ ]:


train_original = train.copy()
test_original = test.copy()


# In[ ]:


def fill_province(row):
    province = row.Province_State
    country = row.Country_Region
    if pd.isnull(province):
        return country
    else:
        return province


# In[ ]:


train['Province_State'] = train.apply(fill_province, axis=1)
test['Province_State']= test.apply(fill_province, axis=1)


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


label_encoder1 = LabelEncoder()
label_encoder2 = LabelEncoder()


# In[ ]:


train['Province_State_encoded'] = label_encoder1.fit_transform(train['Province_State'])
test['Province_State_encoded'] = label_encoder1.transform(test['Province_State'])


# In[ ]:


train['Country_Region_encoded'] = label_encoder2.fit_transform(train['Country_Region'])
test['Country_Region_encoded'] = label_encoder2.transform(test['Country_Region'])


# In[ ]:


train.head()


# In[ ]:


train['Date'] = pd.to_datetime(train_original.Date, infer_datetime_format=True)
train['Date'] = train.Date.dt.strftime('%y%m%d')
train['Date'] = train['Date'].astype(int)
test['Date'] = pd.to_datetime(test_original.Date, infer_datetime_format=True)
test['Date'] = test.Date.dt.strftime('%y%m%d')
test['Date'] = test['Date'].astype(int)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.tail()


# In[ ]:


test.tail()


# In[ ]:


from xgboost import XGBRegressor


# In[ ]:


X = train[['Province_State_encoded', 'Country_Region_encoded', 'Date']]
y = train[['ConfirmedCases', 'Fatalities']]


# In[ ]:


y_confirmed = y['ConfirmedCases']
y_fatalities = y['Fatalities']


# In[ ]:


model1 = XGBRegressor(n_estimators=40000)


# In[ ]:


model1.fit(X,y_confirmed)


# In[ ]:


X_test = test[['Province_State_encoded', 'Country_Region_encoded', 'Date']]


# In[ ]:


y_confirmed_pred = model1.predict(X_test)


# In[ ]:


model2 = XGBRegressor(n_estimators=30000)


# In[ ]:


model2.fit(X,y_fatalities)


# In[ ]:


y_fatalities_pred = model2.predict(X_test)


# In[ ]:


submission_dict = {'ForecastId': test['ForecastId'], 'ConfirmedCases':y_confirmed_pred, 'Fatalities':y_fatalities_pred}


# In[ ]:


submission_df = pd.DataFrame(submission_dict)


# In[ ]:


submission_df.head()


# In[ ]:


submission_df = submission_df.clip(0)


# In[ ]:


submission_df.to_csv('submission.csv', index=False)


# In[ ]:





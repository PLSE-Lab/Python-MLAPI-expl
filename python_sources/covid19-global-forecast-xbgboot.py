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


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly_express as px

import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import OrdinalEncoder
from sklearn import metrics
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import plot_importance, plot_tree
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm


# In[ ]:


train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/train.csv")
train.head()


# In[ ]:


train.describe()


# In[ ]:


train.isnull().sum()


# In[ ]:


del train['Id']


# In[ ]:


train.head()


# In[ ]:


test = pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")
test.head()


# In[ ]:


test.head()


# In[ ]:


last_date = train.Date.max()
top_countries = train[train['Date']==last_date]
top_countries = top_countries.groupby('Country_Region')['ConfirmedCases','Fatalities'].sum()


# In[ ]:


top_countries = top_countries.nlargest(10,'ConfirmedCases').reset_index()


# In[ ]:


fig1 = px.bar(top_countries, x='Country_Region', y='ConfirmedCases',color='Country_Region')
fig1.show()


# In[ ]:


top_trend = train.groupby(['Date','Country_Region'], as_index=False)['ConfirmedCases','Fatalities'].sum()
top_trend = top_trend.merge(top_countries, on='Country_Region')
top_trend.drop(['ConfirmedCases_y','Fatalities_y'],axis=1, inplace=True)


# In[ ]:


fig1 = px.line(top_trend, x='Date', y='ConfirmedCases_x',color='Country_Region')
fig1.show()


# In[ ]:


fig1 = px.line(top_trend, x='Date', y='Fatalities_x',color='Country_Region')
fig1.show()


# In[ ]:


train.head()
train_df = train.copy()


# In[ ]:


test_df = test.copy()
test_df.head()


# In[ ]:


train_df['Date'] = pd.to_datetime(train_df['Date'])
test_df['Date'] = pd.to_datetime(test_df['Date'])


# In[ ]:


import datetime as dt
train_df['Date'] = train_df['Date'].dt.strftime("%m%d")
train_df['Date'] = train_df['Date'].astype(int) 

test_df['Date'] = test_df['Date'].dt.strftime("%m%d")
test_df['Date'] = test_df['Date'].astype(int) 


# In[ ]:


train_df['Province_State'] = train_df['Province_State'].fillna('Nan')
test_df['Province_State'] = test_df['Province_State'].fillna('Nan')


# In[ ]:


train_df.head()


# In[ ]:


train_df['ConfirmedCases'] = train_df['ConfirmedCases'].apply(int)
train_df['Fatalities'] = train_df['Fatalities'].apply(int)


# In[ ]:


from sklearn import preprocessing

LE = preprocessing.LabelEncoder()
train_df.Country_Region = LE.fit_transform(train_df.Country_Region)
train_df['Province_State'] = LE.fit_transform(train_df['Province_State'])

test_df.Country_Region = LE.fit_transform(test_df.Country_Region)
test_df['Province_State'] = LE.fit_transform(test_df['Province_State'])


# In[ ]:


features = ['Date','Country_Region','Province_State']
submission = pd.DataFrame(columns=['ForecastId', 'ConfirmedCases', 'Fatalities'])

for i in tqdm(train_df.Country_Region.unique()):
    c_train = train_df[train_df['Country_Region'] == i]
    c_test = test_df[test_df['Country_Region'] == i]
    for j in c_train.Province_State.unique():
        p_train = c_train[c_train['Province_State'] == j]
        p_test = c_test[c_test['Province_State'] == j]
        x_train = p_train[features]
        y1 = p_train['ConfirmedCases']
        y2 = p_train['Fatalities']
        model = xgb.XGBRegressor(n_estimators=1000)
        model.fit(x_train, y1)
        #Confirmed Cases Prediction
        ConfirmedCasesPreds = model.predict(p_test[features])
        model.fit(x_train, y2)
        #Fatalities Prediction
        FatalitiesPreds = model.predict(p_test[features])
        
        p_test['ConfirmedCases'] = ConfirmedCasesPreds
        p_test['Fatalities'] = FatalitiesPreds
        submission = pd.concat([submission, p_test[['ForecastId', 'ConfirmedCases', 'Fatalities']]], axis=0)


# In[ ]:


submission.to_csv('submission.csv', index=False)


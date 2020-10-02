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


from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
from sklearn import metrics


# In[ ]:


df_train = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv', infer_datetime_format=True)
df_test = pd.read_csv('../input/covid19-global-forecasting-week-1/test.csv', infer_datetime_format=True)


# In[ ]:


datetime_array = np.array([df_train['Date'][0]])
datetime_array1 = np.array([df_test['Date'][0]])
df_train = df_train.drop(['Province/State','Country/Region'], axis=1)
df_test = df_test.drop(['Province/State', 'Country/Region'], axis=1)
df_train = df_train.dropna()

df_test["Lat"]  = df_test["Lat"].fillna(12.5211)
df_test["Long"]  = df_test["Long"].fillna(69.9683)


# In[ ]:


inferred_datetime_format =         pd.core.tools.datetimes._guess_datetime_format_for_array(
            datetime_array)
inferred_datetime_format1 =         pd.core.tools.datetimes._guess_datetime_format_for_array(
            datetime_array1)


# In[ ]:


df_master_data_dt = pd.to_datetime(df_train['Date'],
                                       format=inferred_datetime_format)
df_test_data_dt = pd.to_datetime(df_test['Date'],
                                       format=inferred_datetime_format1)


# In[ ]:


df_train['Year'] = df_master_data_dt.dt.year
df_train['Origin Day of Week'] = df_master_data_dt.dt.dayofweek
df_train['Origin Month'] = df_master_data_dt.dt.month
df_train['Origin Week'] = df_master_data_dt.dt.week

df_test['Year'] = df_test_data_dt.dt.year
df_test['Origin Day of Week'] = df_test_data_dt.dt.dayofweek
df_test['Origin Month'] = df_test_data_dt.dt.month
df_test['Origin Week'] = df_test_data_dt.dt.week


# In[ ]:


df_train["Date"] = df_train["Date"].apply(lambda x: x.replace("-",""))
df_train["Date"]  = df_train["Date"].astype(int)

df_test["Date"] = df_test["Date"].apply(lambda x: x.replace("-",""))
df_test["Date"]  = df_test["Date"].astype(int)


# In[ ]:


features = ["Lat","Long","Date","Year","Origin Day of Week","Origin Month","Origin Week"]

x = df_train[features]
y = df_train['ConfirmedCases']
y1 = df_train['Fatalities']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23, shuffle=False)

x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y1, test_size=0.2, random_state=23, shuffle=False)


# In[ ]:


cb_model = CatBoostRegressor(iterations=5000, learning_rate=0.0001, max_depth=15)
model = cb_model.fit(x_train, y_train, use_best_model=True, verbose=True)

predictions = model.predict(x_test)
pred1 = model.predict(df_test)
pred1 = pd.DataFrame(pred1)
pred1.columns = ["ConfirmedCases_prediction"]


# In[ ]:


cb_model1 = CatBoostRegressor(iterations=2000, learning_rate=0.0001, max_depth=15)
model1 = cb_model1.fit(x_train1, y_train1, use_best_model=True, verbose=True)
predictions1 = model1.predict(x_test1)

pred2 = model1.predict(df_test)
pred2 = pd.DataFrame(pred2)
pred2.columns = ["Death_prediction"]

Sub = pd.read_csv("../input/covid19-global-forecasting-week-1/submission.csv")
Sub.columns
sub_new = Sub[["ForecastId"]]

OP = pd.concat([pred1,pred2,sub_new],axis=1)
OP.head()
OP.columns = ['ConfirmedCases', 'Fatalities', 'ForecastId']
OP = OP[['ForecastId','ConfirmedCases', 'Fatalities']]

OP["ConfirmedCases"] = OP["ConfirmedCases"].astype(int)
OP["Fatalities"] = OP["Fatalities"].astype(int)

OP.head()


# In[ ]:


OP.to_csv("submission.csv",index=False)


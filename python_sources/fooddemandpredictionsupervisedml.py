#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train = pd.read_csv('/kaggle/input/av-genpact-hack-dec2018/train.csv')
test = pd.read_csv('/kaggle/input/av-genpact-hack-dec2018/test.csv')
center = pd.read_csv('/kaggle/input/av-genpact-hack-dec2018/fulfilment_center_info.csv')
meal = pd.read_csv('/kaggle/input/av-genpact-hack-dec2018/meal_info.csv')


# In[ ]:


train = train.merge(center,on='center_id')
train = train.merge(meal,on='meal_id')
train.head()


# In[ ]:


train = train.groupby('week').sum()
train.head()


# In[ ]:


import datetime
from dateutil.relativedelta import relativedelta

year = 2000
def change_week_to_datetime(data):
    return datetime.date(year,1,1)+relativedelta(weeks=+data)


# In[ ]:


data = train['num_orders']
data.index = list(map(change_week_to_datetime,data.index))


# In[ ]:


data.head()


# In[ ]:


fig = plt.figure(1,(15,10))
plt.plot(data.index,data.values)
plt.show()


# In[ ]:


data


# In[ ]:


def convert_to_supervised(data,lags,predicts):
    data = pd.DataFrame(data)
    columns_list = []
    headers_list = []
    for i in range(lags,0,-1):
        columns_list.append(data.shift(i))
        headers_list.append("t-{}".format(i))
    for i in range(0,predicts+1):
        columns_list.append(data.shift(-i))
        headers_list.append("t+{}".format(i))
    
    supervised_data = pd.concat(columns_list,axis = 1)
    supervised_data.dropna(inplace=True)
    supervised_data.columns = headers_list
    return supervised_data


# In[ ]:


supervised_data = convert_to_supervised(data,10,1)
print(supervised_data)


# In[ ]:


x,y = supervised_data.drop('t+1',axis=1),supervised_data['t+1']


# In[ ]:


x


# In[ ]:


y


# In[ ]:


def split_data(x,y,percentage):
    x_train,x_test = x[:int(percentage*x.shape[0])],x[int(percentage*x.shape[0]):] 
    y_train,y_test = y[:int(percentage*x.shape[0])],y[int(percentage*x.shape[0]):]
    return x_train,x_test,y_train,y_test


# In[ ]:


x_train,x_test,y_train,y_test = split_data(x,y,0.8)


# In[ ]:


x_train


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

linear_model = LinearRegression()
linear_model.fit(x_train,y_train)
predictions = linear_model.predict(x_test)


# In[ ]:


fig = plt.figure(1,(7,5))
plt.title(math.sqrt(mean_squared_error(y_test,predictions)))
plt.plot(predictions,color='r')
plt.plot(y_test.values,color='g')


# In[ ]:


from xgboost import XGBRegressor

xgboost_regressor = XGBRegressor(booster = 'gblinear')
xgboost_regressor.fit(x_train,y_train)
xg_preds = xgboost_regressor.predict(x_test)


# In[ ]:


fig = plt.figure(1,(7,5))
plt.title(math.sqrt(mean_squared_error(y_test,xg_preds)))
plt.plot(xg_preds,color='r')
plt.plot(y_test.values,color='g')


# In[ ]:


x_train


# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x_train,x_test,y_train,y_test = scaler.fit_transform(x_train),scaler.fit_transform(x_test),scaler.fit_transform(y_train.values.reshape(-1,1)),scaler.fit_transform(y_test.values.reshape(-1,1))


# In[ ]:


linear_model = LinearRegression()
linear_model.fit(x_train,y_train)
predictions = linear_model.predict(x_test)


# In[ ]:


fig = plt.figure(1,(7,5))
plt.title(math.sqrt(mean_squared_error(y_test,xg_preds)))
plt.plot(predictions,color='r')
plt.plot(y_test,color='g')


# In[ ]:





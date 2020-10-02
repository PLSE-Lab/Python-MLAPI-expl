#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.offline as py
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import os


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train = pd.read_csv('/kaggle/input/av-genpact-hack-dec2018/train.csv')
meal = pd.read_csv('/kaggle/input/av-genpact-hack-dec2018/meal_info.csv')
center = pd.read_csv('/kaggle/input/av-genpact-hack-dec2018/fulfilment_center_info.csv')


# In[ ]:


train.head()


# In[ ]:


len(train)


# In[ ]:


meal.head()


# In[ ]:


center.head()


# In[ ]:


train = train.merge(center,on='center_id')


# In[ ]:


train.head()


# In[ ]:


train = train.merge(meal,on='meal_id')


# In[ ]:


train.head(20)


# In[ ]:


len(train)


# In[ ]:


train.sort_values(by='week',inplace=True)


# In[ ]:


train.head()


# Plotting the data as it is.

# In[ ]:


fig = plt.figure(1,(15,10))
overall_data = train.groupby('week')['num_orders'].sum()
plt.plot(overall_data.index,overall_data)
plt.title('Week wise food demand')
plt.ylabel('Total Demand')
plt.show()


# Plotting the data center-wise

# In[ ]:


def plot_center_wise(center_id):
    center_wise_data = train[train['center_id']==center_id]
    center_wise_orders = center_wise_data['num_orders']
    center_wise_dates = center_wise_data['week']
    return center_wise_orders,center_wise_dates

fig = make_subplots(rows=20, cols=2)
for index,_id in enumerate(list(set(train['center_id']))):
    if index == 21:
        break
    center_wise_orders,center_wise_dates = plot_center_wise(_id)
    fig.add_trace(
        go.Scatter(x=center_wise_dates, y=center_wise_orders,name=_id),
        row=(index//2)+1, col=(index%2)+1
    )
fig.update_layout(height =1500 ,width =1500 ,title_text="Center wise food demand")
fig.show()


# Meal wise food demand

# In[ ]:


def plot_meal_wise(meal_id):
    meal_wise_data = train[train['meal_id']==meal_id]
    meal_wise_orders = meal_wise_data['num_orders']
    meal_wise_dates = meal_wise_data['week']
    return meal_wise_orders,meal_wise_dates

fig = make_subplots(rows=len(list(set(train['meal_id'])))+2, cols=2)
for index,_id in enumerate(list(set(train['meal_id']))):
    if index == len(list(set(train['meal_id']))):
        break
    meal_wise_orders,meal_wise_dates = plot_meal_wise(_id)
    fig.add_trace(
        go.Scatter(x=meal_wise_dates, y=meal_wise_orders,name=_id),
        row=(index//2)+1, col=(index%2)+1
    )
fig.update_layout(height =4500 ,width =1500 ,title_text="Meal wise food demand")
fig.show()


# In[ ]:


def plot_cuisine_wise(cuisine_id):
    cuisine_wise_data = train[train['cuisine']==cuisine_id]
    cuisine_wise_orders = cuisine_wise_data['num_orders']
    cuisine_wise_dates = cuisine_wise_data['week']
    return cuisine_wise_orders,cuisine_wise_dates

fig = make_subplots(rows=len(list(set(train['cuisine'])))+2, cols=2)
for index,_id in enumerate(list(set(train['cuisine']))):
    if index == len(list(set(train['cuisine']))):
        break
    cuisine_wise_orders,cuisine_wise_dates = plot_cuisine_wise(_id)
    fig.add_trace(
        go.Scatter(x=cuisine_wise_dates, y=cuisine_wise_orders,name=_id),
        row=(index//2)+1, col=(index%2)+1
    )
fig.update_layout(height =1100 ,width =900 ,title_text="Cuisine wise food demand")
fig.show()


# Category wise demand for food.
# 

# In[ ]:


def plot_category_wise(category_id):
    category_wise_data = train[train['category']==category_id]
    category_wise_orders = category_wise_data['num_orders']
    category_wise_dates = category_wise_data['week']
    return category_wise_orders,category_wise_dates

fig = make_subplots(rows=len(list(set(train['category'])))+2, cols=2)
for index,_id in enumerate(list(set(train['category']))):
    if index == len(list(set(train['category']))):
        break
    category_wise_orders,category_wise_dates = plot_category_wise(_id)
    fig.add_trace(
        go.Scatter(x=category_wise_dates, y=category_wise_orders,name=_id),
        row=(index//2)+1, col=(index%2)+1
    )
fig.update_layout(height =1500 ,width =900 ,title_text="Category wise food demand")
fig.show()


# Checkout Price v/s Number of orders

# In[ ]:


fig = plt.figure(1,(15,10))
priced_data = train.groupby('checkout_price').sum()
num_orders_by_price = priced_data['num_orders']
priced_data = priced_data.index
plt.plot(priced_data,num_orders_by_price)
plt.title('Checkout price vs number of orders')
plt.xlabel('Price of the food item')
plt.ylabel('Number of orders')
plt.show()


# Average food cost per week

# In[ ]:


fig = plt.figure(1,(15,10))
weekly_avg = train.groupby('week').mean()
prices_avg = weekly_avg['checkout_price']
plt.plot(weekly_avg.index,prices_avg,color='green')
plt.title('weekly price')
plt.xlabel('Weeks')
plt.ylabel('Price of the food items')
plt.show()


# In[ ]:


fig = plt.figure(1,(15,10))
overall_data = train.groupby('week')['num_orders'].sum()
plt.plot(overall_data.index,overall_data)
plt.title('Week wise food demand')
plt.ylabel('Total Demand')
plt.show()


# In[ ]:


from statsmodels.tsa.stattools import adfuller

def check_stationarity(data):
    rolling_mean = data.rolling(window = 4).mean()
    rolling_std = data.rolling(window = 4).std() 
        
    fig = plt.figure(1,(9,6))
    plt.plot(rolling_mean , color = 'r')
    plt.plot(rolling_std , color ='g')
    plt.plot(data,color='b')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show()
    
    result = adfuller(data)
    
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    


# In[ ]:


check_stationarity(overall_data)


# In[ ]:


ts_log = np.log(overall_data)
check_stationarity(ts_log)


# In[ ]:


ts_sqrt = np.sqrt(overall_data)
check_stationarity(ts_sqrt)


# In[ ]:


moving_average = ts_log.rolling(window = 12).mean()
plt.plot(ts_log)
plt.plot(moving_average , color = 'r')


# In[ ]:


ts_log_moving_average = ts_log - moving_average
ts_log_moving_average.dropna(inplace=True)
check_stationarity(ts_log_moving_average)


# In[ ]:


exp_mva = pd.DataFrame.ewm(ts_log,halflife=4).mean()
plt.plot(exp_mva,color='r')
plt.plot(ts_log)


# In[ ]:


ts_log_exp_mva = ts_log-exp_mva
ts_log_exp_mva.dropna(inplace=True)
check_stationarity(ts_log_exp_mva)


# In[ ]:


ts_log_diff = ts_log - ts_log.shift()
ts_log_diff.dropna(inplace=True)
check_stationarity(ts_log_diff)


# In[ ]:


ts_log


# In[ ]:


import datetime
from dateutil.relativedelta import relativedelta

year = 2000
def change_week_to_datetime(data):
    return datetime.date(year,1,1)+relativedelta(weeks=+data)


# In[ ]:


ts_log_index = list(ts_log.index)
ts_log_index = list(map(change_week_to_datetime,ts_log_index))
ts_log_index


# In[ ]:


ts_log.index = ts_log_index


# In[ ]:


ts_log


# In[ ]:


from statsmodels.tsa.seasonal import seasonal_decompose

decomposed_data = seasonal_decompose(ts_log,period=4)
trend = decomposed_data.trend
_seasonal = decomposed_data.seasonal
resid = decomposed_data.resid


# In[ ]:


plt.plot(trend)


# In[ ]:


plt.plot(_seasonal)
plt.show()


# In[ ]:


def difference_series(data,interval = 1):
    difference = []
    for i in range(interval,len(data)):
        difference.append(data[i]-data[i-interval])
    return pd.Series(difference)


# In[ ]:


def apply_inverse(history,prediction,interval = 1):
    return prediction + history[-interval]


# In[ ]:


seasonally_differenced_data = difference_series(ts_log,4)
seasonally_differenced_data.index = ts_log.index[4:]


# In[ ]:


check_stationarity(seasonally_differenced_data)


# In[ ]:


from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

plt.figure()
plt.subplot(211)
plot_acf(seasonally_differenced_data, ax=plt.gca())
plt.subplot(212)
plot_pacf(seasonally_differenced_data, ax=plt.gca())
plt.show()


# In[ ]:


from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import math

train_data = overall_data.values
training,validation = train_data[:int(len(train_data)*0.7)],train_data[int(len(train_data)*0.7):] 
history = [item for item in training]
outputs = []
for i in range(len(validation)):
    differenced_data = difference_series(history,4)
    model = ARIMA(differenced_data,(4,0,6))
    model = model.fit(trend='nc',disp = 0)
    output = model.forecast()[0]
    output = apply_inverse(history,output,4)
    outputs.append(output)
    history.append(validation[i])
    print("Actual Value: {:.2f} Predicted value: {:.2f}".format(validation[i],output))

print("Root mean squared error: ",math.sqrt(mean_squared_error(validation,outputs)))


# In[ ]:


residual_overall_data = pd.DataFrame([validation[i]-outputs[i] for i in range(len(validation))])
residual_overall_data.describe()


# In[ ]:


mean_to_be_added = -15994.344968
plt.figure()
plt.subplot(211)
residual_overall_data.hist(ax = plt.gca())
plt.subplot(212)
residual_overall_data.plot(kind='kde',ax = plt.gca())
plt.show()


# In[ ]:


train_data = overall_data.values
training,validation = train_data[:int(len(train_data)*0.7)],train_data[int(len(train_data)*0.7):] 
history = [item for item in training]
outputs = []            
for i in range(len(validation)):
    differenced_data = difference_series(history,4)
    model = ARIMA(differenced_data,(4,0,5))
    model = model.fit(trend='nc',disp = 0)
    output = model.forecast()[0]
    output = mean_to_be_added + apply_inverse(history,output,4)
    outputs.append(output)
    history.append(validation[i])
#     print("Actual Value: {:.2f} Predicted value: {:.2f}".format(validation[i],output))

print("Root mean squared error: ",math.sqrt(mean_squared_error(validation,outputs)))


# In[ ]:


residual_overall_data = pd.DataFrame([validation[i]-outputs[i] for i in range(len(validation))])
plt.figure()
plt.subplot(211)
residual_overall_data.hist(ax = plt.gca())
plt.subplot(212)
residual_overall_data.plot(kind='kde',ax = plt.gca())
plt.show()


# In[ ]:


plt.figure()
plt.subplot(211)
plot_acf(residual_overall_data, ax=plt.gca())
plt.subplot(212)
plot_pacf(residual_overall_data, ax=plt.gca())
plt.show()


# In[ ]:


plt.plot(validation,color='g')
plt.plot(outputs,color='r')


# In[ ]:





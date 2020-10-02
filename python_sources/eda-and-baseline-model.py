#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Import widgets
from ipywidgets import widgets, interactive, interact
import ipywidgets as widgets
from IPython.display import display

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_sales = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
calendar_df = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
submission_file = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')


# In[ ]:


days = range(1, 1913 + 1)
time_series_columns = [f'd_{i}' for i in days]

ids = np.random.choice(train_sales['id'].unique().tolist(), 1000)

series_ids = widgets.Dropdown(
    options=ids,
    value=ids[0],
    description='series_ids:'
)

def plot_data(series_ids):
    df = train_sales.loc[train_sales['id'] == series_ids][time_series_columns]
    df = pd.Series(df.values.flatten())

    df.plot(figsize=(20, 10), lw=2, marker='*')
    df.rolling(7).mean().plot(figsize=(20, 10), lw=2, marker='o', color='orange')
    plt.axhline(df.mean(), lw=3, color='red')
    plt.grid()


# # Visualizing the Time Series
# 
# Below I make a simple plot of the first time series in the data. Going through the different time series data we can see that a lot of the items have intermittent demand. These are series that have many zeros with bursts of demand inbetween. This will be one of the biggest challenges in this competition.

# In[ ]:


w = interactive(
    plot_data,
    series_ids=series_ids
)
display(w)


# In the analysis above it looks like a lot of the time series data start with leading zeros. I believe we can characterize these leading zeros as items that were not selling or available to sell for those periods of time. This might not be a good assumption for every series. We can investigate the distribution of leading zeros, this could help us bring down the large data size (although may not be a good choice for algorightms such as ARIMA).

# In[ ]:


series_data = train_sales[time_series_columns].values
pd.Series((series_data != 0).argmax(axis=1)).hist(figsize=(25, 5), bins=100)


# What is the distribution of zeros per series? Wow, the distribution of zeros for each of the series has a mean around 0.8 which means there is a lot of intermittent data!

# In[ ]:


pd.Series((series_data == 0).sum(axis=1) / series_data.shape[1]).hist(figsize=(25, 5), color='red')


# What is the distribution of max number of sales for each of the series?
# 
# Alot of the items have a max number of sales between 2 and 12. There are also some items with a very high number of sales for a particular item. It might be fruitful to investigate these items and whether it was a holiday or not.

# In[ ]:


pd.Series(series_data.max(axis=1)).value_counts().head(20).plot(kind='bar', figsize=(25, 10))


# In[ ]:


pd.Series(series_data.max(axis=1)).value_counts().tail(20)


# # Simple Mean Model
# 
# For the first bench mark model I will just take the average sales from the last 28 days for each of the time series in the data.

# In[ ]:


forecast = pd.DataFrame(series_data[:, -28:]).mean(axis=1)
forecast = pd.concat([forecast] * 28, axis=1)
forecast.columns = [f'F{i}' for i in range(1, forecast.shape[1] + 1)]
forecast.head()


# # Predictions
# 
# We need to provide predictions for the next 28 days for each of the series. For the validation series that is days 1914 - 1941 and for the evaluation that is days 1942 - 1969.

# In[ ]:


validation_ids = train_sales['id'].values
evaluation_ids = [i.replace('validation', 'evaluation') for i in validation_ids]


# In[ ]:


ids = np.concatenate([validation_ids, evaluation_ids])


# In[ ]:


predictions = pd.DataFrame(ids, columns=['id'])
forecast = pd.concat([forecast] * 2).reset_index(drop=True)
predictions = pd.concat([predictions, forecast], axis=1)


# In[ ]:


predictions.to_csv('submission.csv', index=False)


# This is as simple as it can get for a forecasting method. There are others as well such as choosing the last known value and propigating it forward as a forecast. I plan to do more analysis of the data and present different methods! Stay tuned :) I would love feedback of what others might like to see, so please let me know in the comments!

# In[ ]:





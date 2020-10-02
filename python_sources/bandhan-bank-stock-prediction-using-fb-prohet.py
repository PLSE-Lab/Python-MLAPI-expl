#!/usr/bin/env python
# coding: utf-8

# ** Let's Predict the Opening Stock of Bandhan Bank NSE. **
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/bandhan-bank-nse-data/BANDHANBNK.NS (2).csv')


# Checking the data

# In[ ]:


data.head()


# In[ ]:


data.shape


# 498 rows and 7 columns are present in data

# Now, let us check for the null value in the dta

# In[ ]:


data.isnull().any()


# Above shows apart from Data column, other column has null data.

# Let's check the number of null values in each column.

# In[ ]:


data.isnull().sum()


# Let's drop the null value row

# In[ ]:


data.dropna(axis = 0, how ='any')


# In[ ]:


data = data.dropna(axis = 0, how ='any')


# In[ ]:


data.isnull().sum()


# plt.plot(data['Date'], data['Open'])

# In[ ]:


plt.figure(figsize=(14, 10))
plt.plot(data.index/15,data['Open'])
plt.title('Bandhan Bank Stock Price ')
plt.ylabel('Stock');
plt.xlabel('0-30 Months: from 27th March 2018 til 8th April 2020');
plt.show()


# Let's apply fbprophet 

# In[ ]:


import fbprophet
# Prophet requires columns ds (Date) and y (value)
data = data.rename(columns={'Date': 'ds', 'Open': 'y'})

# Make the prophet model and fit on the data
data_prophet = fbprophet.Prophet(changepoint_prior_scale=0.15)
data_prophet.fit(data)


# In[ ]:


# Make a future dataframe for 2 years or 90 days
data_forecast = data_prophet.make_future_dataframe(periods=90, freq='D')
# Make predictions
data_forecast = data_prophet.predict(data_forecast)


# In[ ]:


data_prophet.plot(data_forecast, xlabel = 'Date', ylabel = 'Stock')
plt.title('Stock Prediction of Bandhan Bank');


# In[ ]:


data_prophet.plot_components(data_forecast)
plt.title('Stock Prediction of Bandhan Bank');


# This plot shows the predition. Use your mouse on graph after 8th April to see the predicted Stock.

# In[ ]:


from fbprophet.plot import plot_plotly
import plotly.offline as py
py.init_notebook_mode()

fig = plot_plotly(data_prophet, data_forecast)  # This returns a plotly Figure
py.iplot(fig)


# Conclusion
# 
# Opening Stock of Bandhan Bank has been predicted. 
# 
# Note 
# 
# The data for training has been used from finance.yahoo.com.
# 
# 

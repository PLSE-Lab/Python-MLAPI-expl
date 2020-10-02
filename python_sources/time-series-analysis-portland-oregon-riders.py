#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


pwd


# In[ ]:


ls ../input/portland-oregon-avg-rider-monthly-data/


# In[ ]:


df = pd.read_csv('../input/portland-oregon-avg-rider-monthly-data/portland-oregon-average-monthly-.csv')


# In[ ]:


df.head(10)


# In[ ]:


len(df)


# In[ ]:


df.info(verbose=True)


# In[ ]:


df.dtypes


# In[ ]:


# rename df columns
df.columns = ['Date','Riders']


# In[ ]:


df


# In[ ]:


# the amount of riders shouldn't be an object type, much be there are other types mixed in 
df.Riders.unique()


# In[ ]:


df = df[df['Riders'] != ' n=114']


# In[ ]:


# convert columns to correct types 
df.Riders = df.Riders.astype(int)
df.Date = pd.to_datetime(df['Date'])#, format='%Y-%m')


# In[ ]:


df


# In[ ]:


df.dtypes


# In[ ]:





# # time series analysis

# ### monthly average trends 

# In[ ]:


sns.lineplot(df['Date'],df['Riders'])
plt.show()


# In[ ]:


# the amount of riders are increasing over the course of time 


# In[ ]:


df_months=df.groupby(by=df['Date'].dt.month).sum().reset_index()


# In[ ]:


sns.barplot(x='Date',y='Riders',data=df_months)
plt.show()


# In[ ]:


sns.scatterplot(x='Date',y='Riders',data=df_months)
plt.show()


# In[ ]:


# can see here the variation over the year for the average number of riders. the summer had the least amount 


# In[ ]:





# ### analyzing trend in seaons 

# In[ ]:


# Autocorrelation plots are often used for checking randomness in time series
# https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html#visualization-autocorrelation


# In[ ]:


pd.plotting.autocorrelation_plot(df.Riders)
plt.show()


# In[ ]:


# the autocorrelation is positive for a lag of 30-40 days. this means there is a montly trend


# In[ ]:





# In[ ]:


# Lag plots are used to check if a data set or time series is random
# https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html#lag-plot


# In[ ]:


pd.plotting.lag_plot(df.Riders)
plt.show()


# In[ ]:


# we see there is a relationship between the time lags 


# In[ ]:





# ### moving average 

# In[ ]:


for window in [1,30,3*30]:
    df['Riders'].rolling(window).mean().plot()
    plt.title('window size: '+str(window))
    plt.show()


# In[ ]:


# the larger the window is, the more smoothing is done in order to remove noise and capture the patterns in data.
# there is a clear trend that the amount increases over the course of the year


# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # The 1000 Monkies traders
# ### visit algozi.com for the full tutorial for this notebook 
# ### this work is still in progress, stay tuned
# ### in this notebook we explore running 1000 random traders on a single stock
# 
# in this notebook we will:
# * Loading and visualizing stock data
# * Creating the RateOfChange metrics
# * Creating buy and hold strategy
# * Evaluate the strategy and show accumulated return on a graph 
# * Creating 1000 monkies traders 
# * Would you BE fooled to invest in a monkey 
# * Advanced metrics to evaluate the trader so we won't be fooled
# * Sampling by month
# * Calculating population mean and standard deviation(std)
# * Evaluating models using population sampling
# * Evaluating the best monkey  using population sampling
# * Creating a simple prediction model using Machine learning
# * Evaluating a basic ML model 
# * Summary

# In[ ]:





# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import matplotlib as mpl 
import plotly.offline as py
import plotly.graph_objs as go

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Selecting the Stock T Data

# In[ ]:


rawdata = pd.read_csv('../input/all_stocks_5yr.csv')
dataA = rawdata.loc[rawdata['Name']  == 'T']
dataA.head()


# **Data Cleaning Removing Null Values**

# In[ ]:


dataA[dataA.open.isnull()]
dataA.dropna(inplace=True)
dataA[dataA.open.isnull()].sum()
data = dataA.set_index('date')
data.head()


# Basic line Graph to see the trend of the companies stock for past 5 years 

# In[ ]:


fig,ax1 = plt.subplots(figsize=(20, 10))
plt.plot(data[['open','close','high','low']])
plt.show()


# In[ ]:




trace = go.Candlestick(
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'])
d = [trace]
py.iplot(d, filename='simple_candlestick')


# ## Creating the RateOfChange metrics
# now lets create a target coulem and plot it
# 

# In[ ]:


data["target"] = (data["close"] - data["close"].shift(-1))/data["close"]
data =data.dropna()
data = data.drop(["Name"],axis=1)
data.head()


# In[ ]:


data.target.plot()


# lets take the last 400 bars as our test piriod

# In[ ]:


data_train = data[:-400]
data_test = data[-400:]
test_size = data_test.shape[0]


data_test.shape


# ## Creating buy and hold strategy
# now lets create a trade stratgie for our test data
# we will define a stratgie as a list of 1,0,-1 
# * 1 go long
# * 0 cash
# * -1 go short

# In[ ]:


go_long_stratgie =  np.ones(test_size)
go_short_stratgie =  -1*np.ones(test_size)
go_hold_stratgie =  np.zeros(test_size)


# In[ ]:


target = data_test["target"].values


# In[ ]:


target.shape


# In[ ]:





# ## Evaluate the strategy and show accumulated return on a graph 

# In[ ]:


(go_long_stratgie*target).sum()


# In[ ]:


(go_short_stratgie*target).sum()


# In[ ]:


(go_hold_stratgie*target).sum()


# In[ ]:


plt.plot(range(test_size),(go_long_stratgie*target).cumsum()) 
plt.show()


# In[ ]:


def evaluate_and_plot(stratgie,target , show_graph=True):
    print(f"the sum profit is {(stratgie*target).sum()}")
    if show_graph:
        plt.plot(range(test_size),(stratgie*target).cumsum()) 
        plt.show()


# In[ ]:


evaluate_and_plot(go_long_stratgie,target)


# In[ ]:


random_monkie01 = np.random.randint(3, size=test_size)-1
random_monkie01[:10]


# In[ ]:


evaluate_and_plot(random_monkie01,target)


# More Monkieys 

# In[ ]:


random_monkie1000 = np.random.randint(3, size=(1000,test_size))-1


# In[ ]:


evaluate_and_plot(random_monkie1000[0],target)
evaluate_and_plot(random_monkie1000[1],target)


# In[ ]:


max_profit =-20
profits = np.zeros(1000)
for i in range(1000):
    profits[i] = (random_monkie1000[i]*target).sum()


# In[ ]:


profits[:5]


# In[ ]:


profits.max()


# In[ ]:


evaluate_and_plot(random_monkie1000[profits.argmax()],target)


# In[ ]:





# In[ ]:





# In[ ]:


data["bar_change"] = (data["close"] - data["open"])/data["open"]
data["bar_full_size"] = (data["close"] - data["open"])/data["open"]


#!/usr/bin/env python
# coding: utf-8

# ### This is my version of the algorithm taught in the chapter #1 of the book: "Learn Algorithmic Trading" by Donadio & Ghosh.

# #### This is a simple strategy, where we by GOOG stocks whenever it has dropped in the previous day, and sells in the first day it drops from the time of the buy.
# #### Although this strategy has shown profits in the 4 year period below, it is far from being fully validated and safe.

# In[ ]:


from pandas_datareader import data


# In[ ]:


start_date = '2014-01-01'
end_date = '2018-01-01'
goog_data = data.DataReader('GOOG', 'yahoo', start_date, end_date)
goog_data


# In[ ]:


import pandas as pd


# In[ ]:


goog_data_signal = pd.DataFrame(index=goog_data.index)
goog_data_signal


# In[ ]:


goog_data_signal['price'] = goog_data['Adj Close']
goog_data_signal


# In[ ]:


goog_data_signal['daily_difference'] = goog_data_signal['price'].diff()
goog_data_signal


# In[ ]:


import numpy as np


# In[ ]:


goog_data_signal['signal'] = 0.0
goog_data_signal['signal'] = np.where(goog_data_signal['daily_difference'] >= 0, 1.0, 0.0)
goog_data_signal


# In[ ]:


goog_data_signal['positions'] = goog_data_signal['signal'].diff()
goog_data_signal.head(10)


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


fig = plt.figure()
ax1 = fig.add_subplot(111, ylabel="Google Price in US$")
goog_data_signal['price'].plot(ax=ax1, color='r', lw=2.)
ax1.plot(goog_data_signal.loc[goog_data_signal.positions == 1.0].index, 
         goog_data_signal.price[goog_data_signal.positions == 1.0],
         '^', markersize=5, color='m')
ax1.plot(goog_data_signal.loc[goog_data_signal.positions == -1.0].index, 
         goog_data_signal.price[goog_data_signal.positions == -1.0],
         'v', markersize=5, color='k')
plt.show()


# In[ ]:


initial_capital = float(1000.00)


# In[ ]:


positions = pd.DataFrame(index=goog_data_signal.index).fillna(0.0)
portfolio = pd.DataFrame(index=goog_data_signal.index).fillna(0.0)


# In[ ]:


positions['GOOG'] = goog_data_signal['signal']
portfolio['positions'] = (positions.multiply(goog_data_signal['price'], axis=0))


# In[ ]:


positions.head(10)


# In[ ]:


portfolio.head(10)


# In[ ]:


portfolio['cash'] = initial_capital - (positions.diff().multiply(goog_data_signal['price'], axis=0)).cumsum()
portfolio.head(10)


# In[ ]:


portfolio['total'] = portfolio['positions'] + portfolio['cash']
portfolio


# In[ ]:


portfolio['total'].plot()
plt.show()


# So during this period of time that strategy has been profitable, as the amount of cash increased...

# In[ ]:





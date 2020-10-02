#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import datetime

#import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot
import seaborn as sns


# In[ ]:


tesla = pd.read_csv('../input/Tesla_Stock.csv',index_col='Date',parse_dates=True)
ford = pd.read_csv('../input/Ford_Stock.csv',index_col='Date',parse_dates=True)
gm = pd.read_csv('../input/GM_Stock.csv',index_col='Date',parse_dates=True)
tesla.head()


# In[ ]:


#Plot the opening price of stocks for each day.
fig, ax = plt.subplots()

tesla['Open'].plot(figsize=(16,6),color="blue",label='Tesla')
ford['Open'].plot(figsize=(16,6),color="green",label='Ford')
gm['Open'].plot(figsize=(16,6),color="orange",label='GM')

ax.legend()
ax.set_title('Open Price');


# In[ ]:


#Plot the Volume of stock traded each day.
fig, ax = plt.subplots()

tesla['Volume'].plot(figsize=(16,6),color="blue",label='Tesla')
ford['Volume'].plot(figsize=(16,6),color="green",label='Ford')
gm['Volume'].plot(figsize=(16,6),color="orange",label='GM')

ax.legend()
ax.set_title('Volume');


# In[ ]:


#Interesting, looks like Ford had a really big spike somewhere in late 2013.
#What was the date of this maximum trading volume for Ford?

ford['Volume'].idxmax()


# The Open Price Time Series Visualization makes Tesla look like its always been much more valuable as a company than GM and Ford. But to really understand this we would need to look at the total market cap of the company, not just the stock price. Unfortunately our current data doesn't have that information of total units of stock present. But what we can do as a simple calcualtion to try to represent total money traded would be to multply the Volume column by the Open price. Remember that this still isn't the actual Market Cap, its just a visual representation of the total amount of money being traded around using the time series. (e.g. 100 units of stock at $10 each versus 100000 units of stock at $1 each)

# In[ ]:


#Create a new column for each dataframe called "Total Traded" which is the Open Price multiplied by the Volume Traded.
fig, ax = plt.subplots()

tesla ['Total_Traded'] = tesla['Open'] * tesla['Volume']
ford ['Total_Traded'] = ford['Open'] * ford['Volume']
gm ['Total_Traded'] = gm['Open'] * gm['Volume']

tesla['Total_Traded'].plot(figsize=(16,6),color="blue",label='Tesla')
ford['Total_Traded'].plot(figsize=(16,6),color="green",label='Ford')
gm['Total_Traded'].plot(figsize=(16,6),color="orange",label='GM')

ax.legend()
ax.set_title('Total Trade');


# In[ ]:


#Interesting, looks like there was huge amount of money traded for Tesla somewhere in early 2014. 
#What date was that and what happened?
tesla['Total_Traded'].idxmax()


# In[ ]:


#Let's practice plotting out some MA (Moving Averages). Plot out the MA50 and MA200 for GM.
fig, ax = plt.subplots()

gm['MA50'] = gm['Open'].rolling(50).mean()
gm['MA200'] = gm['Open'].rolling(200).mean()

gm['Open'].plot(figsize=(16,6),color="blue",label='Open')
gm['MA50'].plot(figsize=(16,6),color="orange",label='MA50')
gm['MA200'].plot(figsize=(16,6),color="green",label='MA200')

ax.legend()
ax.set_title('Total Trade');


# Finally lets see if there is a relationship between these stocks, after all, they are all related to the car industry. We can see this easily through a scatter matrix plot. Import scatter_matrix from pandas.plotting and use it to create a scatter matrix plot of all the stocks'opening price. You may need to rearrange the columns into a new single dataframe.

# In[ ]:


df_open = pd.DataFrame()
df_open['Tesla'] = tesla['Open']
df_open['Ford'] = ford['Open']
df_open['GM'] = gm['Open']

pd.plotting.scatter_matrix(df_open,hist_kwds = {'bins' : 50})


# In[ ]:



start = datetime.date(2012, 1, 1)
end = datetime.date(2012, 1, 31)
df = ford.loc[start:end][['Open','High','Low','Close']]

trace = go.Candlestick(x=df.index,
                      open=df.Open,
                      high=df.High,
                      low=df.Low,
                      close=df.Close)
data = [trace]
fig = go.Figure(data=data)

iplot(fig, filename='simple_candlestick.html')


# In[ ]:


#Clalculating Daily Percentage Change
ford['returns'] = ford['Close'].pct_change()
gm['returns'] = gm['Close'].pct_change()
tesla['returns'] = tesla['Close'].pct_change()
returns = [ford['returns'].dropna(), gm['returns'].dropna(), tesla['returns'].dropna()]


# In[ ]:


plt.hist(returns , bins = 100 , alpha = 0.75)

plt.legend(['Ford', 'GM', 'Tesla'])
plt.show()


# In[ ]:


pd.DataFrame(np.array(returns).transpose()).plot(kind = 'kde')
plt.legend(['Ford', 'GM', 'Tesla'])
plt.show()


# In[ ]:


returns_df = pd.DataFrame()
returns_df['Ford'] = ford['returns']
returns_df['GM'] = gm['returns']
returns_df['Tesla'] = tesla['returns']
plt.boxplot(returns)
plt.xticks([1, 2, 3], ['Ford', 'GM', 'Tesla'])
plt.show()


# In[ ]:


#Create a scatter matrix plot to see the correlation between each of the stocks daily returns. 
#This helps answer the questions of how related the car companies are. 
#Is Tesla begin treated more as a technology company rather than a car company by the market?

pd.plotting.scatter_matrix(returns_df,hist_kwds = {'bins' : 50})


# In[ ]:


#It looks like Ford and GM do have some sort of possible relationship,
#let's plot just these two against eachother in scatter plot to view this more closely!
plt.scatter(returns_df['Ford'],returns_df['GM'], s = 5 , alpha = 0.5)


# In[ ]:


#Create a cumulative daily return column for each car company's dataframe
tesla['daily_cumulative_return'] = ( 1 + tesla['returns'] ).cumprod()
gm['daily_cumulative_return'] = ( 1 + gm['returns'] ).cumprod()
ford['daily_cumulative_return'] = ( 1 + ford['returns'] ).cumprod()


# In[ ]:


#Let's practice plotting out some MA (Moving Averages). Plot out the MA50 and MA200 for GM.
fig, ax = plt.subplots()

tesla['daily_cumulative_return'].plot(figsize=(16,6),color="blue",label='Tesla')
gm['daily_cumulative_return'].plot(figsize=(16,6),color="green",label='GM')
ford['daily_cumulative_return'].plot(figsize=(16,6),color="orange",label='Ford')

ax.legend()
ax.set_title('Daily Cumulative Return');


# In[ ]:





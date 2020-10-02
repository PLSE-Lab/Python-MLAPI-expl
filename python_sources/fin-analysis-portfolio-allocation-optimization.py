#!/usr/bin/env python
# coding: utf-8

# I hope you find this kernel useful and your UPVOTES would be highly appreciated

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import statsmodels.api as sm
import pandas_datareader
import datetime
import pandas_datareader.data as web


# In[ ]:


start= datetime.datetime(2015,1,1)
end= datetime.datetime.today()

#another way to set up dates:
#start = pd.to_datetime('2015-01-01')
#end = pd.to_datetime('2020-01-01')


# In[ ]:


aapl = web.DataReader('AAPL', 'yahoo', start, end)
ibm = web.DataReader('IBM', 'yahoo', start, end)
amzn = web.DataReader('AMZN', 'yahoo', start, end)

#Alternative way to download portfolio data:
#aapl = quandl.get('WIKI/AAPL.11',start_date=start,end_date=end)

aapl.head()


# In[ ]:


aapl.info()


# In[ ]:


aapl = aapl[['Adj Close']]
ibm = ibm[['Adj Close']]
amzn = amzn[['Adj Close']]


# In[ ]:


aapl.head()


# In[ ]:


for stock_df in (aapl, ibm, amzn):
    stock_df['Normed Return'] = stock_df['Adj Close']/ stock_df.iloc[0]['Adj Close']


# ## Allocation: 

# 20% to Apple, 30% to IBM and 50% to Amazon stocks. And assume that the initial investment was 500,000.

# In[ ]:


for stock_df, allo in zip((aapl, ibm,amzn),[0.2,0.3,0.5]):
    stock_df['Allocation'] = stock_df['Normed Return']*allo


# In[ ]:


for stock_df in (aapl, ibm, amzn):
    stock_df['Position Amount']= stock_df['Allocation']*500000


# ## Portfolio Value:

# In[ ]:


total_pos_vals = [aapl['Position Amount'], ibm['Position Amount'], amzn['Position Amount']]
portf_vals = pd.concat(total_pos_vals, axis = 1)
portf_vals.columns = ['Apple Pos', 'IBM Pos', 'Amazon Pos']
portf_vals['Total Pos'] = portf_vals.sum(axis=1)
portf_vals['Total Pos'].plot(figsize = (10,6))


# In[ ]:


portf_vals['2019-01-01':].drop('Total Pos', axis = 1).plot(figsize=(10,6));


# ## Portfolio Statistics

# In[ ]:


portf_vals['Daily Return'] = portf_vals['Total Pos'].pct_change(1)
portf_vals.dropna(inplace = True)
print('Daily Return Average: ',portf_vals['Daily Return'].mean())
print('Daily Return Standard Deviation: ',portf_vals['Daily Return'].std())


# To look at the distribution, let's take a look at a histogram and a KDE plot:

# In[ ]:


portf_vals['Daily Return'].plot(kind = 'hist', bins=100, figsize = (6,8), color = 'green')
portf_vals['Daily Return'].plot(kind = 'kde', figsize = (8,6), color = 'R');


# In[ ]:


cumulative_return = 100*(portf_vals['Total Pos'][-1]/portf_vals['Total Pos'][0]-1)
print('Cumulative return: ', cumulative_return)


# ## Sharpe Ration

# In[ ]:


SR = portf_vals['Daily Return'].mean()/portf_vals['Daily Return'].std()
print('Sharpe Ration = ', SR)


# In[ ]:


#Annual Sharpe Ratio:
ASR = (252**0.5) * SR
print('Annualized Sharpe Ratio = ', ASR)


# Now let's allocate our securities in a most optimal way:

# In[ ]:


stocks = pd.concat([aapl['Adj Close'], ibm['Adj Close'], amzn['Adj Close']], axis = 1)
stocks.columns = ['Apple', 'IBM', 'Amazon']
stocks.head()


# To calculate Mean Daily Return:

# In[ ]:


stocks.pct_change(1).mean()


# To calculate correlation between stocks:

# In[ ]:


stocks.pct_change(1).corr()


# In[ ]:


log_returns = np.log(stocks/stocks.shift(1))
log_returns.hist(bins = 100, figsize = (12,8), color = 'g')
plt.tight_layout()


# Covarience of the columns:

# In[ ]:


log_returns.cov()*252


# ## Sharpe Ration for One Portfolio

# In[ ]:


np.random.seed(101)
print(stocks.columns)

weights = np.array(np.random.random(3))

print('Random Weights: ')
print(weights)

#However, their sum must be equal to 100
print('Rebalance')
weights = weights/np.sum(weights)
print(weights)


# Expected Return:

# In[ ]:


exp_ret = np.sum((log_returns.mean() * weights) * 252)
print('Expected Portfolio Return: ',exp_ret)


# Expected Volatility or Expected 

# In[ ]:


exp_vol = np.sqrt(np.dot(weights.T,np.dot(log_returns.cov() * 252, weights)))
print('Expected Volatility: ', exp_vol)


# The Sharpe Ratio:

# In[ ]:


SR = exp_ret/exp_vol
print('Sharpe Ratio: ', SR)


# # Choosing an optimal portfolio allocation.

# In[ ]:


num_ports = 5000
all_weights = np.zeros((num_ports,len(stocks.columns)))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sharpe_arr = np.zeros(num_ports)

for ind in range(num_ports):
    weights = np.array(np.random.random(3))
    weights = weights / np.sum(weights)
    all_weights[ind,:] = weights
    ret_arr[ind] = np.sum((log_returns.mean() * weights) *252)
    vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 252, weights)))
    sharpe_arr[ind] = ret_arr[ind]/vol_arr[ind]


# In[ ]:


sharpe_arr.max()


# Index location for the best allocation:

# In[ ]:


sharpe_arr.argmax()


# Here is the optimal allocation for our portfolio

# In[ ]:


all_weights[sharpe_arr.argmax(),:]


# Let's plot the data:

# In[ ]:


plt.figure(figsize = (12,8))
plt.scatter(vol_arr,ret_arr,c=sharpe_arr,cmap='Spectral')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')

# Add red dot for max SR
max_sr_ret = ret_arr[sharpe_arr.argmax()]
max_sr_vol = vol_arr[sharpe_arr.argmax()]
plt.scatter(max_sr_vol,max_sr_ret,c='red',s=50,edgecolors='black');


# So our Optimal Sharpe Ration is 1.36 and Optimal Portfolio Allocation is 27.9% for Aplle, 0.2% for IBM, and 71.8% for Amazon. So, I would conclude to substitute IMB shares to something more profitable.

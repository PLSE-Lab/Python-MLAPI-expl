#!/usr/bin/env python
# coding: utf-8

# Here I will be covering how to work with stock prices?We will be considering the stock prices of four tech companies Apple,Google,Microsoft and Amazon.We will be be covering various concepts like moving average,correlation,risk analysis etc.This kernel is a work in process.If you like my work please do vote.

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


# **Import Python Modules **

# In[ ]:


import pandas as pd
import numpy as np 
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from datetime import datetime
#from __future__ import division


# **Importing the Stock Prices**

# In[ ]:


AAPL=pd.read_csv('../input/sandp500/individual_stocks_5yr/individual_stocks_5yr/AAPL_data.csv')
GOOG=pd.read_csv('../input/sandp500/individual_stocks_5yr/individual_stocks_5yr/GOOG_data.csv')
MSFT=pd.read_csv('../input/sandp500/individual_stocks_5yr/individual_stocks_5yr/MSFT_data.csv')
AMZN=pd.read_csv('../input/sandp500/individual_stocks_5yr/individual_stocks_5yr/AMZN_data.csv')


# In[ ]:


AAPL.head()


# In[ ]:


AAPL.index


# In[ ]:


#AAPL.index=pd.to_datetime(AAPL.index)
#AAPL.index


# In[ ]:


AAPL.describe().T


# We will be using the Adjusted close price to do historic price analysis of the stock

# In[ ]:


AAPL.info()


# **Display Historical Stock Price**

# In[ ]:


AAPL['date']=pd.to_datetime(AAPL['date'])
GOOG['date']=pd.to_datetime(AAPL['date'])
MSFT['date']=pd.to_datetime(AAPL['date'])
AMZN['date']=pd.to_datetime(AAPL['date'])
AAPL.info()


# We can see that the data column values have changed to datetime64 type

# **Close Price of Apple Stock**

# In[ ]:


AAPL.plot(x='date', y='close',legend=True,figsize=(10,4))
plt.ioff()


# We have stock Price for 5 years starting from 2013 to 2018 

# **Volume traded for Apple Stock**

# In[ ]:


title='VOLUME TRADED'
ylabel='Volume'
xlabel='Time'


# In[ ]:


ax=AAPL.plot(x='date', y='volume',legend=True,figsize=(10,4));
ax.autoscale(axis='x',tight=True)  # use both if want to scale both axis
ax.set(xlabel=xlabel,ylabel=ylabel)
plt.ioff()


# **Plotting between Specified time**

# In[ ]:


AAPL.plot(x='date', y='close',xlim=['2016-01-01','2017-12-31'],ylim=[80,180],legend=True,figsize=(10,4),ls='--',c='red')
plt.ioff()


# We have ploted the closing Price by specifying the range of dates xlim

# **Making X Axis equally Spaced **

# In[ ]:


"""from matplotlib import dates
AAPL.plot(x='date', y='close',xlim=['2016-01-01','2017-12-31'],ylim=[80,180],legend=True,figsize=(10,4),ls='--',c='red')
ax.set(xlabel='')
ax.xaxis.set_major_locator(dates.WeekdayLocator(byweekday=0))
ax.xaxis.set_major_formatter(dates.DateFormatter('%d'))
ax.xaxis.set_minor_locator(dates.MonthLocator())
ax.xaxis.set_minor_formatter(dates.DateFormatter('%b'))
plt.ioff()
"""


# **Moving Average for Apple Stock**

# In[ ]:


AAPL['close_10']=AAPL['close'].rolling(10).mean()
AAPL['close_50']=AAPL['close'].rolling(50).mean()


# In[ ]:


ax=AAPL.plot(x='date',y='close',title='AAPL Close Price',figsize=(10,4))
AAPL.plot(x='date',y='close_10',color='red',ax=ax)
AAPL.plot(x='date',y='close_50',color='k',ax=ax)
plt.ioff()


# **Daily Returns**

# In[ ]:


AAPL['Daily Return']=AAPL['close'].pct_change()
AAPL['Daily Return'].plot(figsize=(15,4),legend=True,linestyle='--',marker='o')
plt.ioff()


# We can See maximum daily fluctuation in ths stock is 8 % 

# **Average Daily return**

# In[ ]:


sns.distplot(AAPL['Daily Return'].dropna(),bins=2000,color='purple')
plt.ioff()


# In[ ]:


AAPL['Daily Return'].hist(bins=100)
plt.ioff()


# So the stock fluction follows a normal distribution betweem +2% and -2%

# **Checking if the Stock prices of tech companies are correlated**

# In[ ]:


df=AAPL['date'].copy()
df=pd.DataFrame(df)
df['AAPL']=AAPL['close']
df['GOOG']=GOOG['close']
df['MSFT']=MSFT['close']
df['AMZN']=AMZN['close']
df.drop(['date'], axis = 1, inplace = True, errors = 'ignore')


# In[ ]:


tech_rets=df.pct_change()
tech_rets=pd.DataFrame(tech_rets)
tech_rets['date']=AAPL['date']
tech_rets.shape


# In[ ]:


import scipy.stats as stats
sns.jointplot('GOOG','GOOG',tech_rets,kind='scatter',color='seagreen').annotate(stats.pearsonr)
plt.ioff()


# We just compared the daily percentage change of Google with itself.So the correlation is 1 as expected

# **Compare MICROSOFT & GOOGLE**

# In[ ]:


sns.jointplot('AMZN','AAPL',tech_rets,kind='scatter',color='seagreen').annotate(stats.pearsonr)
plt.ioff()


# We can see that p =0.29

# **Pair Plot**

# In[ ]:


sns.pairplot(tech_rets.dropna())
plt.ioff()


# We can see that Amazon and Google have good correlation on daily returns

# **Customised Pair Grid for Daily Return**

# In[ ]:


returns_fig=sns.PairGrid(tech_rets.dropna())
returns_fig.map_upper(plt.scatter,color='purple')
returns_fig.map_lower(sns.kdeplot,cmap='cool_d')
returns_fig.map_diag(plt.hist,bins=30)
plt.ioff()


# **Customized Pair Grid for Close price **

# In[ ]:


returns_fig=sns.PairGrid(df.dropna())
returns_fig.map_upper(plt.scatter,color='purple')
returns_fig.map_lower(sns.kdeplot,cmap='cool_d')
returns_fig.map_diag(plt.hist,bins=30)
plt.ioff()


# Closing price of Google and Microsoft are well correlated

# **Correlation plot for daily returns**

# **Heatmap Daily Return**

# In[ ]:


sns.heatmap(tech_rets.corr(),annot=True,cmap='summer',linewidths=1,linecolor='k',square=True,mask=False, vmin=-1, vmax=1,cbar_kws={"orientation": "vertical"},cbar=True)
plt.ioff()


# **Heatmap Closing Price**

# In[ ]:


sns.heatmap(df.corr(),annot=True,cmap='summer',linewidths=1,linecolor='k',square=True,mask=False, vmin=-1, vmax=1,cbar_kws={"orientation": "vertical"},cbar=True)
plt.ioff()


# All companies have Positive Correlations

# **Risk Analysis**

# In[ ]:


rets=tech_rets.dropna()


# In[ ]:


area=np.pi*20
plt.scatter(rets.mean(),rets.std(),s=area)
plt.xlabel('Expected Return')
plt.ylabel('Risk')

for label, x,y in zip(rets.columns,rets.mean(),rets.std()):
    plt.annotate(
        label,
        xy=(x,y),xytext=(50,50),
        textcoords='offset points',ha='right',va='bottom',
        arrowprops=dict(arrowstyle='-',connectionstyle='arc,rad=-0.3'))


# Tock with Low risk and high return are the best stock to buy

# **Value at Risk**

# In[ ]:


sns.distplot(AAPL['Daily Return'].dropna(),bins=2000,color='purple')
plt.ioff()


# In[ ]:


rets.head()


# In[ ]:


rets['AAPL'].quantile(0.05)


# -0.022 means that 95% of the times the worst daily Loss will not exceed 2.28%

# **Value at risk using Monte Carlo Method**

# In[ ]:


days=365
dt=1/365
mu=rets.mean()['GOOG']
sigma=rets.std()['GOOG']


# In[ ]:


def stock_monte_carlo(start_price,days,mu,sigma):
    price=np.zeros(days)
    price[0]=start_price
    shock=np.zeros(days)
    drift=np.zeros(days)
    
    for x in range(1,days):
        shock[x]=np.random.normal(loc=mu*dt,scale=sigma*np.sqrt(dt))
        drift[x]=mu*dt
        price[x]=price[x-1] + (price[x-1]*(drift[x]+shock[x]))
        
    return price


# In[ ]:


GOOG.head()


# In[ ]:


start_price=568
for run in range(100):
    plt.plot(stock_monte_carlo(start_price,days,mu,sigma))

plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Monte Carlo Analysis For Google')
plt.ioff()


# In[ ]:


runs=10000

simulations=np.zeros(runs)

for run in range(runs):
    simulations[run]=stock_monte_carlo(start_price,days,mu,sigma)[days-1]


# In[ ]:


q=np.percentile(simulations,1)

plt.hist(simulations,bins=200)


# Starting Price
plt.figtext(0.6, 0.8, s="Start price: $%.2f" %start_price)
# Mean ending price
plt.figtext(0.6, 0.7, "Mean final price: $%.2f" % simulations.mean())

# Variance of the price (within 99% confidence interval)
plt.figtext(0.6, 0.6, "VaR(0.99): $%.2f" % (start_price - q,))

# Display 1% quantile
plt.figtext(0.15, 0.6, "q(0.99): $%.2f" % q)

# Plot a line at the 1% quantile result
plt.axvline(x=q, linewidth=4, color='r')

# Title
plt.title(u"Final price distribution for Google Stock after %s days" % days, weight='bold');


# We can see from out simulation that the start price of the stock is 568 $ and the mean Final Price is very close at  568.69.This means that Google stock price is quite stable and doesnt fluctuate more. VaR(0.99) = 15.27 means that the maximm risk for insvestment in this stock is  15.27 which is small amount compared to the investement.

# **Forecasting Apple Stock Price **

# In[ ]:


AAPL.head()


# In[ ]:


df_prophet=AAPL[['date','close']]
df_prophet=df_prophet.sort_values('date')
df_prophet


# **Renaiming the Column names to Suite Prophet Algorithm**

# In[ ]:


df_prophet=df_prophet.rename(columns={'date':'ds','close':'y'})
df_prophet


# **Creating the Prophet Model**

# In[ ]:


import random 
import seaborn as sns
from fbprophet import Prophet
m=Prophet()
m.fit(df_prophet)
future=m.make_future_dataframe(periods=365)
forecast=m.predict(future)
forecast


# **Plotting the Apple Stock Forecast for Period of One years**

# In[ ]:


figure=m.plot(forecast,xlabel='Date',ylabel='Closing Price')


# So the model predicts that the Apple stock Price would increase from Mar 2018 to Mar 2019.

# **Plotting component of the Forecast**

# In[ ]:


figure=m.plot_components(forecast)


# 1.Historical Trend Show that the Price of Apple stock has been increasing.Ivestors must have made good money on it
# 
# 2.Weekly trend shows that the Stock price increase is highest on Tuesday then reduces as week proceeds.Please do note that Saturday and Sunday are off for the Stock Exchange.
# 
# 3.The annual trend shows the seasoniality of the stock.It can be figured out the stock price peaks in month of May.

#!/usr/bin/env python
# coding: utf-8

# WORK IN PROGRESS BEGINNER CODE

# In[ ]:


from pandas_datareader import data, wb
import pandas as pd
import numpy as np
import datetime
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from pandas import *
from numpy.random import randn


# In[ ]:


import os
print(os.listdir('../input'))
phstocks = pd.read_csv('../input/conso12282018.csv')


# phstocks = pd.read_csv('conso12282018.csv')

# In[ ]:


phstocks2 = phstocks


# #Top 30 Companies

# In[ ]:


tickersall = ['AEV', 'AP', 'AGI', 'AC', 'ALI', 'BPI', 'BDO', 'DMC', 'FGEN', 'GLO', 'GTCAP', 'ICT', 'JGS', 'JFC', 'LTG', 'MER', 'MEG', 'MPI', 'MBT', 'PCOR', 'TEL', 'PGOLD', 'RLC', 'RRHI', 'SMC', 'SECB', 'SM', 'SMPH', 'SCC', 'URC']


# In[ ]:


tickers= ['BDO', 'MBT', 'BPI','PNB','SECB','CHIB']


# In[ ]:


dfBDO = phstocks[phstocks.TICKER == 'BDO']
dfMBT = phstocks[phstocks.TICKER == 'MBT']
dfBPI = phstocks[phstocks.TICKER == 'BPI']
dfPNB = phstocks[phstocks.TICKER == 'PNB']
dfSECB = phstocks[phstocks.TICKER == 'SECB']
dfCHIB = phstocks[phstocks.TICKER == 'CHIB']


# In[ ]:


dfBDO = pd.pivot_table(dfBDO, index='DATE', columns='TICKER').swaplevel(axis=1)
dfMBT = pd.pivot_table(dfMBT, index='DATE', columns='TICKER').swaplevel(axis=1)
dfBPI = pd.pivot_table(dfBPI, index='DATE', columns='TICKER').swaplevel(axis=1)
dfPNB = pd.pivot_table(dfPNB, index='DATE', columns='TICKER').swaplevel(axis=1)
dfSECB = pd.pivot_table(dfSECB, index='DATE', columns='TICKER').swaplevel(axis=1)
dfCHIB = pd.pivot_table(dfCHIB, index='DATE', columns='TICKER').swaplevel(axis=1)


# In[ ]:


bank_stocks= pd.concat([dfBDO, dfMBT, dfBPI, dfPNB, dfSECB, dfCHIB], axis=1)


# In[ ]:


bank_stocks.head(3)


# In[ ]:


bank_stocks.columns.names= ['Bank Ticker', 'Stock Info']


# Max CLOSE price in time period
# #for tick in tickers:
#     print(tick, bank_stocks[tick]['CLOSE'].max())

# In[ ]:


bank_stocks.xs(key='CLOSE', axis=1, level='Stock Info').max()


# A rate of return (RoR) is the net gain or loss on an investment over a specified time period, expressed as a percentage of the investment's initial cost.

# In[ ]:


returns = pd.DataFrame()


# In[ ]:


for tick in tickers:
    returns[tick+ ' Return']= bank_stocks[tick]['CLOSE'].pct_change()


# In[ ]:


import seaborn as sns


# In[ ]:


sns.pairplot(returns[1:])


# Max single day loss and gain percentage

# In[ ]:


returns.idxmin()


# In[ ]:


returns.idxmax()


# Risk Analysis: standard deviation

# In[ ]:


returns.std()


# In[ ]:


returns.head(3)


# In[ ]:


returns.tail(3)


# In[ ]:


sns.distplot(returns['01/04/2018':'12/28/2018']['BDO Return'], color='green')


# In[ ]:


sns.distplot(returns['01/04/2018':'12/28/2018']['MBT Return'], color='red')


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly
import cufflinks as cf
cf.go_offline()


# In[ ]:


bank_stocks.xs(key='CLOSE', axis=1, level='Stock Info').plot()


# In[ ]:


bank_stocks.xs(key='CLOSE', axis=1, level='Stock Info').iplot()


# In[ ]:


dfAEV2.head(3)


# In[ ]:


plt.figure(figsize=(12, 4))
dfBDO['BDO']['CLOSE'].ix['01/04/2018':'12/28/2018'].rolling(window=30).mean().plot(label='30 day avg')
dfMBT['MBT']['CLOSE'].ix['01/04/2018':'12/28/2018'].plot(label='AEV')
dfPNB['PNB']['CLOSE'].ix['01/04/2018':'12/28/2018'].plot(label='AEV')
dfSECB['SECB']['CLOSE'].ix['01/04/2018':'12/28/2018'].plot(label='AEV')
dfCHIB['CHIB']['CLOSE'].ix['01/04/2018':'12/28/2018'].plot(label='AEV')
plt.legend()


# In[ ]:


sns.heatmap(bank_stocks.xs(key='CLOSE', axis=1, level='Stock Info').corr())


# In[ ]:


sns.clustermap(bank_stocks.xs(key='CLOSE', axis=1, level='Stock Info').corr())


# In[ ]:


close_corr= bank_stocks.xs(key='CLOSE', axis=1, level='Stock Info').corr()


# In[ ]:


close_corr.iplot(kind='heatmap', colorscale='rdylbu')


# In[ ]:


dfBDO.head(3)


# In[ ]:


dfBDO['BDO'][['OPEN', 'HIGH', 'LOW', 'CLOSE']].iplot(kind='candle')


# In[ ]:


dfMBT['MBT'][['OPEN', 'HIGH', 'LOW', 'CLOSE']].iplot(kind='candle')


# In[ ]:


dfBPI['BPI'][['OPEN', 'HIGH', 'LOW', 'CLOSE']].iplot(kind='candle')


# sma = 'simple moving average'

# In[ ]:


dfBDO['BDO']['CLOSE'].ta_plot(study='sma', periods=[13, 21, 55])


# boll = 'Bollinger Band Plot'

# In[ ]:


dfBDO['BDO']['CLOSE'].ta_plot(study='boll')


# In[ ]:





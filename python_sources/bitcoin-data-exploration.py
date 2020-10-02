#!/usr/bin/env python
# coding: utf-8

# ### Context 
# Bitcoin is the longest running and most well known cryptocurrency, first released as open source in 2009 by the anonymous Satoshi Nakamoto. Bitcoin serves as a decentralized medium of digital exchange, with transactions verified and recorded in a public distributed ledger (the blockchain) without the need for a trusted record keeping authority or central intermediary. Transaction blocks contain a SHA-256 cryptographic hash of previous transaction blocks, and are thus "chained" together, serving as an immutable record of all transactions that have ever occurred. As with any currency/commodity on the market, bitcoin trading and financial instruments soon followed public adoption of bitcoin and continue to grow. Included here is historical bitcoin market data at 1-min intervals for select bitcoin exchanges where trading takes place. Happy (data) mining! 
# 
# ### Content
# coincheckJPY_1-min_data_2014-10-31_to_2017-10-20.csv - 32% of all BTC Volume (past 30 days from last update of this data set)
# coinbaseUSD_1-min_data_2014-12-01_to_2017-10-20.csv - 8% of all BTC Volume  (past 30 days from last update of this data set)
#  bitstampUSD_1-min_data_2012-01-01_to_2017-10-20.csv - 9% of all BTC Volume  (past 30 days from last update of this data set)
# 
# 
#  Legacy/to be updated:  
#  krakenEUR_1-min_data_2014-01-08_to_2017-05-31.csv    
#  btceUSD_1-min_data_2012-01-01_to_2017-05-31.csv  
#  btcnCNY_1-min_data_2012-01-01_to_2017-05-31.csv  
#  krakenUSD_1-min_data_2014-01-07_to_2017-05-31.csv 
# 
# CSV files for select bitcoin exchanges for the time period of Jan 2012 to October 2017, with minute to minute updates of OHLC (Open, High, Low, Close), Volume in BTC and indicated currency, and weighted bitcoin price.  Timestamps are in Unix time.  Timestamps without any trades or activity have their data fields populated with NaNs. If a timestamp is missing, or if there are jumps, this may be because the exchange (or its API) was down, the exchange (or its API) did not exist, or some other unforseen technical error in data reporting or gathering. All effort has been made to deduplicate entries and verify the contents are correct and complete to the best of my ability, but obviously trust at your own risk. 
# 

# In[ ]:


import numpy as np 
import pandas as pd
import datetime
import os
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


#define a parsing function for the timestamps field in int 
def timeparse (time_in_secs):    
    return datetime.datetime.fromtimestamp(float(time_in_secs))


# In[ ]:


data = pd.read_csv("../input/bitstampUSD_1-min_data_2012-01-01_to_2017-10-20.csv", 
                   parse_dates = True, index_col = [0], date_parser = timeparse)


# In[ ]:


np.mean(data)


# In[ ]:


data.describe()


# In[ ]:


data.plot(subplots = True)


# In[ ]:


data["Diff"] = data["High"] - data["Low"]
data["Diff"].plot()


# In[ ]:


data[data['Diff'] > 200]


# In[ ]:


data[(data.index > '2016-06-23 12:30:00') & (data.index < '2016-06-23 13:00:00')]


# In[ ]:


open_series = data['Open']


# In[ ]:


open_series.head()


# In[ ]:


plt.plot(open_series)


# In[ ]:


from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12,center=False).mean()
    
    rolstd = timeseries.rolling(window=12,center=False).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    """
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    """
    
    


# In[ ]:


test_stationarity(open_series)


# In[ ]:





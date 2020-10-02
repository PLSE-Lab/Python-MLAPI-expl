#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import autocorrelation_plot
from pandas.plotting import lag_plot
from statsmodels.tsa import seasonal
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[2]:


def get_stock_data(symbol):
    df = pd.read_csv("../input/Data/Stocks/{}.us.txt".format(symbol), index_col='Date', parse_dates=True, 
                     na_values='nan', usecols=['Date', 'Close'])
    return df
ibm = get_stock_data('ibm')
ibm.plot(figsize=(15, 5))


# In[3]:


def plot_acf(df):
    plt.clf()
    fig, ax = plt.subplots(figsize=(15, 5))
    autocorrelation_plot(df, ax=ax)


# In[4]:


def calculate_acf(series, nlags=100):
    alpha = 0.05
    acf_value, confint, qstat, pvalues, *_ = acf(series,
                                             unbiased=True,
                                             nlags=nlags,
                                             qstat=True,
                                             alpha=alpha)
    for l, p_val in enumerate(pvalues):
        if p_val > alpha:
            print("Null hypothesis is accepted at lag = {} for p-val = {}".format(l, p_val))
        else:
            print("Null hypothesis is rejected at lag = {} for p-val = {}".format(l, p_val))


# #### Calculate the monthly mean to calculate it's ACF

# In[5]:


dm = ibm.resample('M').mean()
dm.plot(figsize=(15, 5))


# In[6]:


plot_acf(dm)


# #### Since the monthly mean drops above & below the lines, so it is non-stationary (with seasonal differences)

# #### Let's calculate the monthly seasonal differences to see if it's auto-correlated

# In[7]:


dmsd = dm.diff(12)[12:]


# In[8]:


dmsd.plot()


# In[9]:


plot_acf(dmsd)


# In[10]:


adf_result_dm = adfuller(dm['Close'])
print ("p-val of the ADF test for monthly data : ", adf_result_dm[1])


# #### The ADF test for the monthly values is too high showing that it's non-stationary

# In[11]:


adf_result_dmsd = adfuller(dmsd['Close'])
print ("p-val of the ADF test for monthly seasonal differences : ", adf_result_dmsd[1])


# #### The ADF test value for the seasonal differences is too low showing that it's stationary

# #### Let's now take a look at moving averages

# In[12]:


def moving_average(sf, window):
    dma = sf.rolling(window=window).mean()
    return dma.loc[~pd.isnull(dma)]
ibm.plot(figsize=(15, 5))


# In[13]:


SixXMA6 = moving_average(moving_average(ibm['Close'], window=6), window=6)
TenXMA10 = moving_average(moving_average(ibm['Close'], window=10), window=10)
TwentyXMA20 = moving_average(moving_average(ibm['Close'], window=50), window=50)


# In[14]:


f, ax = plt.subplots(4, sharex=True, figsize=(15, 10))

ibm['Close'].plot(color='b', linestyle='-', ax=ax[0])
ax[0].set_title('Raw data')

SixXMA6.plot(color='r', linestyle='-', ax=ax[1])
ax[1].set_title('6x6 day MA')

TenXMA10.plot(color='g', linestyle='-', ax=ax[2])
ax[2].set_title('10x10 day MA')

TwentyXMA20.plot(color='k', linestyle='-', ax=ax[3])
ax[3].set_title('20x20 day MA')


# #### You can see the smoothing effect as we increase the moving average window

# #### Let's remove the trend line to get the residuals (let's take trend-line as 10X10 MA)

# In[15]:


residuals = ibm['Close']-TenXMA10
residuals = residuals.loc[~pd.isnull(residuals)]


# In[16]:


residuals.plot(figsize=(15, 5))


# In[17]:


plot_acf(residuals)


# #### There isn't a huge auto-correlation for the residuals, but it is still statistically significant for a number of lags

# In[18]:


adf_result = adfuller(residuals)
print ("p-val of the ADF test for residuals : ", adf_result[1])


# #### The ADF test p-value is too low showing that this data is stationary

# #### Let's decompose using the additive model using statsmodel.tsa

# In[19]:


additive = seasonal.seasonal_decompose(ibm['Close'], freq=1000, model='additive')


# In[20]:


def plot_decomposed(decompose_result):
    fig, ax = plt.subplots(4, sharex=True)
    fig.set_size_inches(15, 10)
    
    ibm['Close'].plot(ax=ax[0], color='b', linestyle='-')
    ax[0].set_title('IBM Close price')
    
    pd.Series(data=decompose_result.trend, index=ibm.index).plot(ax=ax[1], color='r', linestyle='-')
    ax[1].set_title('Trend line')
    
    pd.Series(data=decompose_result.seasonal, index=ibm.index).plot(ax=ax[2], color='g', linestyle='-')
    ax[2].set_title('Seasonal component')
    
    pd.Series(data=decompose_result.resid, index=ibm.index).plot(ax=ax[3], color='k', linestyle='-')
    ax[3].set_title('Irregular variables')


# In[21]:


plot_decomposed(additive)


# In[22]:


adf_result = adfuller(additive.resid[np.where(np.isfinite(additive.resid))[0]])


# In[23]:


print("p-value for the irregular variations of the additive model : ", adf_result[1])


# #### Let's try the multiplicative model & compare it with the additive one

# In[26]:


multiplicative = seasonal.seasonal_decompose(ibm['Close'], freq=1000, model='multiplicative')


# In[27]:


plot_decomposed(multiplicative)


# In[28]:


adf_result_multiplicative = adfuller(multiplicative.resid[np.where(np.isfinite(multiplicative.resid))[0]])


# In[29]:


print("p-value for the irregular variations of the additive model : ", adf_result_multiplicative[1])


# #### The multiplicative model has a lower p-value for the irregular variables than the additive ones & hence, it's better at stationarizing the data

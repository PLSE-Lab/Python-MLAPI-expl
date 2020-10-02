#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import autocorrelation_plot
from pandas.plotting import lag_plot
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[6]:


def get_stock_data(symbol):
    df = pd.read_csv("../input/Data/Stocks/{}.us.txt".format(symbol), index_col='Date', parse_dates=True, 
                     na_values='nan', usecols=['Date', 'Close'])
    return df
ibm = get_stock_data('ibm')

ibm.plot(figsize=(15, 5))


# In[7]:


lag_plot(ibm['Close'])


# #### There's a very strong correlation between value at T & at T-1. Let's create the ACF plot for different lags

# In[8]:


plt.clf()
fig, ax = plt.subplots(figsize=(15, 5))
autocorrelation_plot(ibm['Close'], ax=ax)


# #### For a number of lag values (for eg. from 0-4000), the auto-correlation is significant (above or below the dotted lines).

# #### Let's try this using the statsmodels ACF plot to confirm it

# In[10]:


plt.clf()
fig, ax = plt.subplots(figsize=(15, 10))
plot_acf(ibm['Close'], lags=1200, use_vlines=False, ax=ax)


# #### Thus, the data is auto-correlated in a statistically significant way for the first 1000 lags

# In[11]:


fig, ax = plt.subplots(figsize=(15, 10))
plot_pacf(ibm['Close'], lags=100, use_vlines=False, ax=ax)


# #### Thus, only the 1st lag is auto-correlated in a statistically significant way for PACF

# In[12]:


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


# In[13]:


calculate_acf(ibm['Close'])


# #### So, this confirms that we have statistically significant ACF for at least the first 100 lags

# In[14]:


adf, p_value, usedlag, nobs, critical_values, *values = adfuller(ibm['Close'])
print ("ADF is ", adf)
print ("p value is ", p_value)
print ("lags used are ", usedlag)
print ("Number of observations are ", nobs)
print ("Critical Values are", critical_values)


# #### The high p-value confirms the auto-correlation & that the data is non-stationary

# #### Let's try this on the first-order difference time-series data

# In[15]:


ddiff = ibm.diff(1)[1:]


# In[16]:


ddiff.plot()


# In[17]:


fig.clf()
fig, ax = plt.subplots(figsize=(10, 10))
lag_plot(ddiff['Close'], ax=ax)


# In[18]:


fig.clf()
fig, ax = plt.subplots(figsize=(15, 8))
autocorrelation_plot(ddiff, ax=ax)


# #### There are some lag (appox. first 800) for which the ACF is statistically significant

# In[19]:


calculate_acf(ddiff['Close'])


# #### So, this confirms that we have statistically significant ACF for at least the first 100 lags for first-order difference

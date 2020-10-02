#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from pandas.tseries.offsets import DateOffset

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv('../input/brent-oil/BrentOilPrices.csv')
df.head()


# In[ ]:


df.columns = ['Date', 'Price']
df.head()


# In[ ]:


df.info()
df.isnull().sum()


# In[ ]:


df['Date'] = pd.to_datetime(df['Date'])


# In[ ]:


df.set_index ('Date', inplace = True)
df.index


# In[ ]:


df_new = df['1998-01-01':]
df_new.tail()


# In[ ]:


df_new.describe().transpose()


# In[ ]:


f, ax = plt.subplots(figsize = (16,10))
ax.plot(df_new, c = 'r');


# In[ ]:


df_new.boxplot('Price', rot = 80, fontsize = '12',grid = True);


# In[ ]:


time_series = df_new['Price']
type(time_series)


# In[ ]:


time_series.rolling(12).mean().plot(label = '12 Months Rolling Mean', figsize = (16,10))
time_series.rolling(12).std().plot(label = '12 Months Rolling Std')
time_series.plot()
plt.legend();


# In[ ]:


result = adfuller(df_new['Price'])


# In[ ]:


def adf_check(time_series):
    
    result = adfuller(time_series)
    print('Augmented Dickey-Fuller Test')
    labels = ['ADF Test Statistic', 'p-value', '# of lags', 'Num of Obs used']
    
    print('Critical values:')
    for key,value in result[4].items():
        print('\t{}: {}'.format(key, value) )
    
    for value, label in zip(result,labels):
        print(label+ ' : '+str(value))
    
    if ((result[1] <= 0.05 and  result[0] <= result[4]['1%']) or
    (result[1] <= 0.05 and  result[0] <= result[4]['5%']) or
        (result[1] <= 0.05 and  result[0] <= result[4]['10%'])):
        print('Reject null hypothesis')
        print ('Data has no unit root and is stationary')
   
    else:
        print('Fail to reject null hypothesis')
        print('Data has a unit root and it is non-stationary')


# In[ ]:


adf_check(df_new['Price'])


# In[ ]:


df_new['Dif_1'] = df_new['Price'] - df_new['Price'].shift(1)
df_new['Dif_1'].plot(rot = 80, figsize = (14,8));


# In[ ]:


adf_check(df_new['Dif_1'].dropna())


# In[ ]:


df_new['Dif_Season'] = df_new['Price'] - df_new['Price'].shift(12)
df_new['Dif_Season'].plot(rot = 80, figsize = (14,8));


# In[ ]:


adf_check(df_new['Dif_Season'].dropna())


# In[ ]:


df_new['Dif_Season_1'] = df_new['Dif_1'] - df_new['Dif_1'].shift(12)
df_new['Dif_Season_1'].plot(rot = 80, figsize = (14,8));


# In[ ]:


adf_check(df_new['Dif_Season_1'].dropna())


# In[ ]:


df_new['Dif_mean'] = df_new['Price'] - df_new['Price'].rolling(12).mean()
df_new['Dif_mean'].plot(rot = 80, figsize = (14,8));


# In[ ]:


adf_check(df_new['Dif_mean'].dropna())


# In[ ]:


decomp = seasonal_decompose(time_series, freq = 12)
fig = decomp.plot()
fig.set_size_inches(15,8)


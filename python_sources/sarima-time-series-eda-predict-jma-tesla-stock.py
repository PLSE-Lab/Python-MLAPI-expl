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


df = pd.read_csv('../input/tesla-stock-price/Tesla.csv')
df.head()


# In[ ]:


df.columns = ['Date', 'Val',"","","","",""]
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


df_new.boxplot('Val', rot = 80, fontsize = '12',grid = True);


# In[ ]:


time_series = df_new['Val']
type(time_series)


# In[ ]:


time_series.rolling(12).mean().plot(label = '12 Months Rolling Mean', figsize = (16,10))
time_series.rolling(12).std().plot(label = '12 Months Rolling Std')
time_series.plot()
plt.legend();

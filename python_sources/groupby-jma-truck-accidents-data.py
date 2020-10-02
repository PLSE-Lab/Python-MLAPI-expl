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


df = pd.read_csv('../input/truck-accidents-data/Trucks.csv')
df.head()


# In[ ]:


df.groupby(['light'])
df


# In[ ]:


df1=df.groupby('period').groups
df1


# In[ ]:


df2=df.groupby('collision').groups
df2


# In[ ]:


plt.clf()
df.groupby('period').size().plot(kind='bar')
plt.show()


# In[ ]:


plt.clf()
df.groupby('collision').size().plot(kind='bar')
plt.show()


# In[ ]:


plt.clf()
df.groupby('period').sum().plot(kind='bar')
plt.show()


# In[ ]:


plt.clf()
df.groupby('collision').sum().plot(kind='bar')
plt.show()


# In[ ]:


plt.clf()
df.groupby('parked').sum().plot(kind='bar')
plt.show()


# In[ ]:


plt.clf()
df.groupby('light').sum().plot(kind='bar')
plt.show()


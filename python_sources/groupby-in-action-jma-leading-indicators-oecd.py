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


df = pd.read_csv('../input/leading-indicators-oecd/MEI_20022020103548670.csv')
df.head()


# In[ ]:


df.groupby(['Country'])
df


# In[ ]:


df1=df.groupby('Country').groups


# In[ ]:


df1


# In[ ]:


plt.clf()
df.groupby('Country').size().plot(kind='bar')
plt.show()


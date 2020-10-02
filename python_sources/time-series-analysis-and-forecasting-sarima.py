#!/usr/bin/env python
# coding: utf-8

# **Importing all the required libraries**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
get_ipython().system('pip install pmdarima')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Reading the datafile with index column as observatiob date and setting parse date  and glancing over few values.**

# In[ ]:


data = pd.read_csv('../input/us-candy-production-by-month/candy_production.csv',index_col='observation_date',parse_dates=True)


# In[ ]:


data.head()


# **Plotting the data**

# In[ ]:


sns.set(rc={'figure.figsize':(20,5)})
data.plot()


# **Data analysis**
# 
# 

# In[ ]:


data.index


# As we can see in above output the frequency is set to none , for doing seasonal decompose we must need to set the frequency. We can do it by observing the data or by using a function given by pandas which is listed below

# In[ ]:


data.index.freq = pd.infer_freq(data.index)


# In[ ]:


data.index


# We can see in the output that now frequency(freq) is set to 'MS' that is monthly starting
# 

# Now we will perform the **Seasonal Decomposition** on the data so we can observe the seasonal,trend and residual component of the data. 

# In[ ]:


result = seasonal_decompose(data)
sns.set(rc={'figure.figsize': (20,5)})
result.plot();


# From the above graphs we can see that there is surely a seasonal component in the data, but to find out how many period we will plot an ACF plot

# In[ ]:


plot_acf(data);


# In[ ]:


plot_pacf(data);


# from the ACF plot we can clearly see that there is a trend repetation at every 12 months

# Now we will use auto_arima from pmdarima to get the optimal parameter of SARIMA

# In[ ]:


auto_arima(data['IPG3113N'],seasonal=True,m=12,suppress_warnings=True,information_criterion='aic',max_P=5,max_D=5,max_Q=5,max_p=5,max_d=5,max_q=5).summary()


# **Spltting data into test and train set**

# In[ ]:


train = data[:500]
test = data[500:]


# In[ ]:


model = SARIMAX(train['IPG3113N'],order=(3,1,3),seasonal_order=(1,0,2,12))


# In[ ]:


result_f = model.fit()


# In[ ]:


pred = result_f.predict(start=len(train),end=len(train)+len(test),type='levels')


# In[ ]:


fig, ax = plt.subplots()
ax=test.plot(color='red',ax=ax)
ax=pred.plot(color='green',ax=ax)
ax.legend(['test','pred'])


# The above chart displayes how close our model was with the actual data

# Now we will train over model on whole data for Forecasting

# In[ ]:


model_forecast = SARIMAX(data,order=(3,1,3),seasonal_order=(1,0,2,12))

model_forecast_fit = model_forecast.fit()

pred_forecast = model_forecast_fit.predict(len(data),len(data)+48,type='levels')


# In[ ]:


fig, ax = plt.subplots()
ax=data[400:].plot(color='red',ax=ax)
ax=pred_forecast.plot(color='green',ax=ax)
ax.legend(['Original_data','Forecast'])


# In[ ]:


len(data)


# In[ ]:





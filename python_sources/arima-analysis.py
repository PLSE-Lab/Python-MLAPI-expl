#!/usr/bin/env python
# coding: utf-8

# Seasonal ARIMA analysis of global Land and Sea temperature changes

# In[ ]:


import numpy as np 
import pandas as pd 
import statsmodels.api as sm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 6 #For better display


# In[ ]:


gt = pd.read_csv("../input/GlobalTemperatures.csv",index_col="dt",infer_datetime_format=True)
gt.head()


# In[ ]:


#Looks like we're just going back 150 years

avt = gt.LandAndOceanAverageTemperature
missing_dates = avt[avt.isnull() == True]
print(missing_dates.tail())


# In[ ]:


recent = gt.LandAndOceanAverageTemperature["1850":]
recent.isnull().sum()


# In[ ]:


var = recent.rolling(12).std()
mean = recent.rolling(12).mean()
mean.plot()
plt.title("Rolling Mean of Global Average Temperature post 1850")
plt.xlabel("Time")
plt.ylabel("Average Temperature")


# In[ ]:


var.plot()
plt.title("Rolling Std of Global Average Temperature post 1850")
plt.xlabel("Time")
plt.ylabel("Average Temperature")


# In[ ]:


#Clearly this is not stationary... lets difference it

diff = recent.diff().dropna()
mean_diff = diff.rolling(12).mean()
var_diff = diff.rolling(12).std()
diff.plot()
mean_diff.plot(c = "red")
var_diff.plot(c = "green")
plt.xlabel("Time")
dftest = sm.tsa.adfuller(diff, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)


# In[ ]:


###Check out the ACF and PACF
sm.tsa.graphics.plot_acf(diff,lags = np.arange(0,25,1))


# In[ ]:


sm.tsa.graphics.plot_pacf(diff,lags=np.arange(0,25,1))


# In[ ]:


mod = sm.tsa.SARIMAX(recent,order = (3,1,0), seasonal_order=(0,0,0,12)).fit()
mod.summary()


# In[ ]:


mod.plot_diagnostics()


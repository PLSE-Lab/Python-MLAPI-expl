#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Reading the complete.csv file that has confirmed covid19 cases and deaths

# In[ ]:


india_covid=pd.read_csv('../input/covid19-corona-virus-india-dataset/complete.csv')


# In[ ]:


india_covid.describe()


# In[ ]:


india_covid.columns
india_covid=india_covid.rename(columns={"Total Confirmed cases": "conf"})
india_conf=india_covid.groupby(['Date'],as_index=False).conf.sum()


# Coverting dataset for ready to be used for time series

# In[ ]:


ts = india_conf.set_index('Date')
ts.plot()


# The graph suggest that there is spike in covid cases from March 22nd onwards, suggesting Covid is increasing exponentially in India

# # Loading Statsmodel for using Arima and Dickey Fuller test for stationarity

# In[ ]:


from statsmodels.tsa.stattools import adfuller
values_tc_india=india_covid['conf'].values
dftest=adfuller(values_tc_india)
print('ADF Statistic: %f' % dftest[0])
print('p-value: %f' % dftest[1])
print('Critical Values:')
for key, value in dftest[4].items():
	print('\t%s: %.3f' % (key, value))


# The P value of dickey fuller test is not significant suggesting this is not a stationary time series which evident from graph before

# # Checking ACF plots and effect of differencing

# In[ ]:


import numpy as np   
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(ts); axes[0, 0].set_title('Original Series')
plot_acf(ts, ax=axes[0, 1])


# The ACF plot clearly shows correlation with pervious terms suggesting an AR model lets have a better plot autocorrelation_plot

# In[ ]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(ts)
pyplot.show()


# Here we get clear indication that upto lag 6 or 7 we have high correlation with previous values. For simplicty let us try AR(5) model firts
# 

# # Lets see if differencing 2 time makes thing stationary and somewhat removes trend

# In[ ]:


two_level_diff=np.diff(np.diff(values_tc_india))
axes[2, 0].plot(two_level_diff); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(two_level_diff, ax=axes[2, 1])


# So we can see in the last row that differencing has made an impact converted a trend type series to a white noise a random series, Now we have an idea to put AR(5) with differencing 2 in the model hence p=5 d=2 and q=0 will be tried

# # Buiding and ARIMA(5,2) model

# In[ ]:


from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(ts, order=(5,2,0))
model_fit = model.fit()
model_fit.summary()


# Apart from Lag 3 rest lags have significant p values suggesting that current value due have an impact from previous lag values(Table2)

# # Residual plots to see if they are white noise or not

# In[ ]:


residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()


# The density plot for residuals and residual plot suggest that randonmness of error terms , we have achieved somewhat a white noise

# 

# # Predicting next 5 days cases

# In[ ]:


next_5_day=pd.DataFrame(model_fit.forecast(5))
model_fit.forecast(5)


# In[ ]:


next_5_day


# The last row gives us the range according to 95% confidence interval , i will go for max suggesting 7158 cases in April 10 upto 10337 on april 14th

# # Please help in improving this model further

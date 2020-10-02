#!/usr/bin/env python
# coding: utf-8

# **The prediction of significant wave height (Hs) using ARIMA**
# 
# In physical oceanography, the significant wave height (SWH or Hs) is defined traditionally as the mean wave height (trough to crest) of the highest third of the waves (H1/3). It is scientifically represented as Hs or Hsig, is an important parameter for the statistical distribution of ocean waves. The most common waves are lower in height than Hs. This implies that encountering the significant wave is not too frequent. However, statistically, it is possible to encounter a wave that is much higher than the significant wave. 
# 
# 
# To predict the significant wave height, I  will analyze measured data collected by oceanographic wave measuring buoys anchored at Mooloolaba. Coverage period: 30 months collected at periods of 30 min intervals.
# The data comes from Queensland Government Data - https://data.qld.gov.au/dataset. I will use ARIMA method to predict the Hs values useing walk forward validation with a sliding window of 3000 time steps.

# We begin the analysis by importing the needed libraries and the data set.

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
from statsmodels.tsa.arima_model import ARIMA
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from math import sqrt


# In[ ]:


print(os.listdir('../input'))


# Now you're ready to read in the data and use the plotting functions to visualize the data.

# In[ ]:


# Coastal Data System - Waves (Mooloolaba) 01-2017 to 06 - 2019.csv has 43728 rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('../input/Coastal Data System - Waves (Mooloolaba) 01-2017 to 06 - 2019.csv',index_col=0, parse_dates=True,nrows=24000)
df1.dataframeName = 'Coastal Data System - Waves (Mooloolaba) 01-2017 to 06 - 2019.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')
df1.head()


# Lets prepare the data by replacing the mising values by interpolated data.

# In[ ]:


df1=df1.replace(-99.9,np.nan)
df1=df1.interpolate(limit_direction='both')


# In[ ]:


def plot_df(x, y, title, xlabel, ylabel, dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()


# In[ ]:


# Create Training and Test
train = df1.iloc[:3000,1]
test = df1.iloc[3000:3200,1]


# In[ ]:


plot_df( train.index, train.values, "significant wave height (SWH or Hs)", "date-time", "Hs(m)")


# ARIMA MODEL

# finding d

# In[ ]:


result=adfuller(train)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
result_diff = adfuller(train.diff().dropna())
print('ADF Statistic: %f' % result_diff[0])
print('p-value: %f' % result_diff[1])


# The series is stationary after one differencing, thus d=1

# Finding P

# In[ ]:


x=plot_pacf((train.diff()).dropna(),lags=10)


# A value of p=5 should be fine.

# In[ ]:


x=plot_acf((train.diff()).dropna(),lags=10)


# We found a value of q=2.

# In[ ]:


# Build Model
predictions=[]
history=[x for x in train]
for t in range(len(test)):
    model = ARIMA(history[t:], order=(5, 1, 2))
    model_fit = model.fit()
    predictions.append(model_fit.forecast(alpha=0.05)[0][0])
    history.append(test[t])
# Forecast
# Make as pandas series
predictions = pd.Series(predictions, index=test.index)
# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(predictions, label='forecast')
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# In[ ]:


plt.figure(figsize=(12,5), dpi=100)
plt.plot(test, label='actual')
plt.plot(predictions, label='forecast')
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# In[ ]:


mse = mean_squared_error(test, predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)
plt.scatter(predictions,test)


# In[ ]:


x=np.array(test)
y=np.array(predictions)
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(predictions,test.iloc[:])
print(r_value)


# ## Conclusion
# My model is simple. I use data from last two months to predict (Hs) for the next step. I use walk forward
# validation with a sliding window of 3000 time steps (62.5 days). The model is tested for successive 200 time steps (10 hours). On average the difference between the predicted and actual data is about 31cm. The predictions and the actual data are also tested with a linear regression and found R value of 0.94. The results are very promising and could be enhanced further by optimizing the training window size.
# 

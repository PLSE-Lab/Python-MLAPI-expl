#!/usr/bin/env python
# coding: utf-8

# #  In this notebook, closing price is explored and predicted
# 
# some of the code are simiar to this useful resource blow:
# 
# PythonDataScienceHandbook
# https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/03.11-Working-with-Time-Series.ipynb 

# In time-series data, there are some Important things to consider before
# fitting machine-learning models.
# 
# 1. A trend in the data
# 2. seasonality
# 3. outliers
# 4. long-run cycle
# 5. constant variance
# 6. abrupt changes
# 
# 
# resource:  STAT 510  https://onlinecourses.science.psu.edu/stat510/node/47 
# 

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn; seaborn.set()
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[2]:


train = pd.read_csv("../input/bitcoin_price_Training - Training.csv")
test = pd.read_csv("../input/bitcoin_price_1week_Test - Test.csv")


# In[3]:


print(train.shape)
print(test.shape)


# In[4]:


train.head()


# In[5]:


train.tail()


# In[6]:


test


#   #  Reverse the order of the date so that it becomes chronological order

# In[7]:


train = train[::-1] 
test = test[::-1]
train.head()


# # Change the date notation(string date) to numerical date
# ![](http://)-- define a function (which converts date to desired format) and apply each date
#  
#  Example -- 
#  
# f(x):= apply num
# 
# f(April 28, 2013) = 2013-04-28

# In[8]:


from dateutil.parser import parse
from datetime import datetime

def convert(date):
    holder = []
    for i in date:
        tp = parse(i).timestamp()
        dt = datetime.fromtimestamp(tp)
        holder.append(dt)
    return np.array(holder)


# In[9]:


date = train['Date'].values
date_n = convert(date)


# In[10]:


# sanity check
print(len(date_n) == train.shape[0])


# In[11]:


train['Date'] = date_n
train.head()


# # Set Index as Date

# In[12]:


train = train.set_index('Date')
train.head()


# In[13]:


train.describe()


# In[14]:


# check the missing values
train.isnull().any()


#  # Visualization of closing price (on row data & log-scale)
# 
# Idea: The reason why log-scale is used on y-axis is that it reveals the percentile change
#      
#      Ex: 
# 
#      Case 1) when price goes up from $ 10-$ 15: change(increase) is $5. Increase rate is 50%
# 
#      Case 2) when price goes up from $20-$25: change(increase) is  $5. Increase rate is 25%
#        
# In both cases , change is same but rate of change is different.
# 
# 
# 
# Refrence: What is the difference between a logarithmic price scale and a linear one?
# http://www.investopedia.com/ask/answers/05/logvslinear.asp#ixzz4pKMuY5HA 

# In[15]:


plt.figure(num=None, figsize=(20, 6))
plt.subplot(1,2,1)
ax = train['Close'].plot(style=['-'])
ax.lines[0].set_alpha(0.3)
ax.set_ylim(0, np.max(train['Close'] + 100))
plt.xticks(rotation=90)
plt.title("No scaling")
ax.legend()
plt.subplot(1,2,2)
ax = train['Close'].plot(style=['-'])
ax.lines[0].set_alpha(0.3)
ax.set_yscale('log')
ax.set_ylim(0, np.max(train['Close'] + 100))
plt.xticks(rotation=90)
plt.title("logarithmic scale")
ax.legend()


# Some features of the plot above:
# 
# 1. There is an uppward trend from 2016 for each graph
# 2.  There is no seasnality
# 3. There are no outliers
# 4. There are some vaiance in the logarithmic scaled data. This will be confirmed using rolling average and standard deviation.

#  # Resampling at  lower frequency
#  
#  plot the average of price in the previous year and price at the end of the year

# In[16]:


close = train['Close']
close.plot(alpha=0.5, style='-')
close.resample('BA').mean().plot(style=':')
close.asfreq('BA').plot(style='--')
plt.yscale('log')
plt.title("logarithmic scale")
plt.legend(['close-price', 'resample', 'asfreq'], 
           loc='upper left')
# 'resample'-- average of the previous year
# 'asfreq' -- value at the end of the year


#  # ROI

# In[17]:


ROI = 100 * (close.tshift(-365) / close - 1)
ROI.plot()
plt.ylabel('% Return on Investment');


# # Moving averages: SMA and EMA
#  
# moving averages are used to smooth out the data to see the underlying trend
# 
# SMA(simple mean average) calculates the mean of some span(N) while EMA (exponential mean average) does so putting
# more emphasis on recent points
# 
# reference: http://www.investopedia.com/university/movingaverage/movingaverages1.asp

# In[18]:


rolling = close.rolling(200, center=True)

data = pd.DataFrame({'input': close, 
                     '200days rolling_mean': rolling.mean(), 
                     '200days rolling_std': rolling.std()})

ax = data.plot(style=['-', '--', ':'])
ax.set_yscale('log')
ax.set_title("SMA on log scale")
rolling = close.rolling(365, center=True)
ax.lines[0].set_alpha(0.3)


# Important to note: standard deviation for (percentile)change of price  is not consitent over time or non-constant variance

# In[19]:


ax = data.plot(style=['-', '--', ':'])
ax.set_title("SMA on raw data")
ax.lines[0].set_alpha(0.3)


# In[20]:


rolling = pd.ewma(close, com=200)

data = pd.DataFrame({'input': close, 
                     '200days rolling_mean': rolling.mean(), 
                     '200days rolling_std': rolling.std()})

ax = data.plot(style=['-', '--', ':'])
ax.set_yscale('log')
ax.set_title("EMA on log scale")
ax.lines[0].set_alpha(0.3)


# In[21]:


ax = data.plot(style=['-', '--', ':'])
ax.set_title("EMA on raw data")
ax.lines[0].set_alpha(0.3)


# In[ ]:





# # Lag Plot ( check whether time series is random or not)
# 
# resource: http://www.itl.nist.gov/div898/handbook/eda/section3/lagplot.htm 
# 
# In the graph below, firt axis represents the  t(lag), seond axis represents t+1
# 
# Ex: if data is, [1,4,5,3,2], then y(t):= [1,4,5,3,2], y(t+1): = [4,5,3,2]
# 
# As we see the graph below, this suggests the non-random pattern (graph is poistively linear).
# 
# Non-randomness in the data reveals that we could use an autoregressive model

# In[22]:


from pandas.plotting import lag_plot
lag_plot(close)


#   # Autocorrelation
# resource: http://www.itl.nist.gov/div898/handbook/eda/section3/eda35c.htm 
# 
# A lag plot above shows the structure in the data. In order to quantify the correlation between the point at t and point at t+1
# withe respect to expectation, autocrrelation is used.
# 
# A  black line in the graph below shows the expectation for random data(thus 0 correlation) and two dash lines above and below it represent the confidence interval with each 95% and 99%.
# 
# The graph shows a strong correlation for lags of < 100 days. (lag 0 is always 1 corrleation)

# In[23]:


from pandas.plotting import autocorrelation_plot
autocorrelation_plot(close)


# In[ ]:





# # Partial autocorrelation
# 
# Idea: Partical autocorrelation only descirbes the direct effect of a value at t-k(k is a lag) on value at t, ignoring the values between them (values from t-k+1 to t-1). Autocrrelaiton take the in-between values into account. 
# Partial autocrrelation will us detemine the order of an autoregressive model, p. (AR(p))
# reference: Difference between autocrrelation and partial autocorrelation
# https://stats.stackexchange.com/questions/129052/acf-and-pacf-formula
# 

# In[33]:


from pandas import Series
from statsmodels.graphics.tsaplots import plot_pacf


# In[34]:


plot_pacf(close, lags=50)


# # Autoregression
# lag plot and autocorrelation revealed that we could use autoregression for fitting data
# we will predict test data (closing price for 7 days) using training data
# 
# resource: 
# Autoregression Models for Time Series Forecasting With Python
# http://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/

# In[36]:


from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error

test = test['Close'].values


# In[37]:


train_pr = train['Close'].values


# In[38]:


# train and fit autoregression
model = AR(train_pr)
model_fit = model.fit()

print("Lag: %s" % model_fit.k_ar)
print("Coefficients: %s" % model_fit.params)

pred = model_fit.predict(start=len(train), end=len(train_pr)+len(test)-1, dynamic=False)
mse = mean_squared_error(test, pred)
print("Test MSE {0:.3f}".format(mse))


# Important to note:
# 
# 1. 24 lags are used to train the model (24 previous points are used to predict a next point)
# 
# 2. MSE is high

# In[39]:


plt.plot(test, label='true value')
plt.plot(pred, color='red', label='prediction')
plt.title("Autoregressive model")
plt.legend()


# 

# In[ ]:





# In[ ]:





# In[ ]:





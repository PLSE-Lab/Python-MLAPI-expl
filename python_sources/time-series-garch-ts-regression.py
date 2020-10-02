#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import time
import datetime
import gc
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from scipy import stats
from tqdm import tqdm
from tqdm import tqdm_notebook
#timeseries libraries
from statsmodels import tsa
from statsmodels.graphics import tsaplots
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.api import qqplot
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.arima_model import ARMA 
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error


# In[ ]:


get_ipython().run_cell_magic('time', '', "train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})")


# In[ ]:


#test = pd.read_csv('../input/test/.csv')
#test.shape


# In[ ]:


train_acoustic_data_small = train['acoustic_data'].values[::50]
train_time_to_failure_small = train['time_to_failure'].values[::50]

fig, ax1 = plt.subplots(figsize=(16, 8))
plt.title("Trends of acoustic_data and time_to_failure. 2% of data (sampled)")
plt.plot(train_acoustic_data_small, color='b')
ax1.set_ylabel('acoustic_data', color='b')
plt.legend(['acoustic_data'])
ax2 = ax1.twinx()
plt.plot(train_time_to_failure_small, color='g')
ax2.set_ylabel('time_to_failure', color='g')
plt.legend(['time_to_failure'], loc=(0.875, 0.9))
plt.grid(False)

del train_acoustic_data_small
del train_time_to_failure_small


# In[ ]:


train_time_to_failure_small = train['time_to_failure'].values[::50]

fig, ax1 = plt.subplots(figsize=(16, 8))
ax2 = ax1.twinx()
plt.plot(train_time_to_failure_small, color='g')
ax2.set_ylabel('time_to_failure', color='g')
plt.legend(['time_to_failure'], loc=(0.875, 0.9))
plt.grid(False)
del train_time_to_failure_small


# # Preparing and exploring data using average and std devation.

# In[ ]:


#let's prepare our data
seg_length = 60000
total_samples = int(np.floor((train.shape[0]) / seg_length))

#we will be using a total of nine different features as given below for making our predictions
cols = ['average', 'std'] #our features used for the prediction
x_train = pd.DataFrame(index = range(total_samples), columns = cols, dtype = np.float64) #an empty dataframe holding our feature values
y_train = pd.DataFrame(index = range(total_samples), columns = ['time_to_failure'], dtype = np.float64) #an empty dataframe holding our target labels


# In[ ]:


for value in tqdm(range(total_samples)):
    sample = train.iloc[value*seg_length : value*seg_length + seg_length]
    x = sample['acoustic_data'].values
    y = sample['time_to_failure'].values[-1]
    
    y_train.loc[value, 'time_to_failure'] = y
    
    x_train.loc[value, 'average'] = x.mean()
    x_train.loc[value, 'std'] = x.std()


# In[ ]:


train_acoustic_data_small = x_train['average']
train_time_to_failure_small = y_train['time_to_failure']

fig, ax1 = plt.subplots(figsize=(16, 8))
plt.title("Trends of acoustic_data and time_to_failure. 2% of data (sampled)")
plt.plot(train_acoustic_data_small, color='b')
ax1.set_ylabel('acoustic_data', color='b')
plt.legend(['acoustic_data'])
ax2 = ax1.twinx()
plt.plot(train_time_to_failure_small, color='g')
ax2.set_ylabel('time_to_failure', color='g')
plt.legend(['time_to_failure'], loc=(0.875, 0.9))
plt.grid(False)

del train_acoustic_data_small
del train_time_to_failure_small


# In[ ]:


x_train['seismic'] = x_train.average.diff()
train_seismic = x_train['seismic']


# In[ ]:


nlags=13  #define number of lags to plot on ACF/PACF plots
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = tsaplots.plot_acf(train_seismic.dropna(), lags=nlags, ax=ax1,title='Autocorrelation of Seismic Data')
plt.xticks(list(range(0,nlags)))

ax2 = fig.add_subplot(212)
fig = tsaplots.plot_pacf(train_seismic.dropna(), lags=nlags, ax=ax2,title='Partial Autocorrelation of Seismic Data')
plt.xticks(list(range(0,nlags)))
plt.xlabel('Lag-k')
plt.show()


# In[ ]:


train_seismic.dropna(inplace=True)


# In[ ]:


fig=plt.figure(figsize=(16,6))

ax1=plt.subplot(121)
fig=sns.distplot(train_seismic,bins=200,ax=ax1)
plt.title('Distribution of Seismic Data')

ax2=plt.subplot(122)
fig=  qqplot(train_seismic,fit=True,line='45',ax=ax2)
plt.title('Initial Q-Q plot for Seismic Data')

plt.show()


# In[ ]:


#Jarque-Bera test for normality
val,p=stats.jarque_bera(train_seismic)

print('Jarque-Bera Test Results:\nStatistics = {}\np-value = {}'.format(val,p))


# In[ ]:


# Ljung-box test for auto-correlation
_,p=acorr_ljungbox(train_seismic,lags=10)
print('Ljung-Box test p-values for 10-lags:\n',p)


# In[ ]:


test_stat,pval,usedlag,_,CI=adfuller(train_seismic,regression='c',autolag='BIC')[:5]
print('Augmented Dickey Fuller test results:')
print('Test Statistics:',test_stat)
print('p-value:',pval)
print('Used Lags:',usedlag)
print('Critical Values:')
for key, value in CI.items():
    print('\t%s: %.3f' % (key, value))


# Null Hypothesis (H0): If accepted, it suggests the time series has a unit root, meaning it is non-stationary. It has some time dependent structure.   
# Alternate Hypothesis (H1): The null hypothesis is rejected; it suggests the time series does not have a unit root, meaning it is stationary. It does not have time-dependent structure. We interpret this result using the p-value from the test.    
# A p-value below a threshold (such as 5% or 1%) suggests we reject the null hypothesis (stationary), otherwise a p-value above the threshold suggests we accept the null hypothesis (non-stationary).  
# p-value > 0.05: Accept the null hypothesis (H0), the data has a unit root and is non-stationary.     
# p-value <= 0.05: Reject the null hypothesis (H0), the data does not have a unit root and is stationary.    
# 
# 
# Running the test statistic value of -26.753. As per ADF statistic we can see that our statistic value of -26.753 is less than the value of -3.432 at 1%.
# The p-value is < 0.05.
# Comparing the p-value and test statistic to the critical values, it looks like we would have to Reject the null hypothesis (H0),  the data does not have a unit root and is stationary.

# In[ ]:


x_train.to_csv("x_train_105.csv")
y_train.to_csv("y_train_105.csv")


# In[ ]:





# # Taking difference of the dataset and the preparing the data.

# In[ ]:


#let's prepare our data
seg_length = 50000
total_samples = int(np.floor((train.shape[0]) / seg_length))

#we will be using a total of nine different features as given below for making our predictions
cols = ['average', 'std'] #our features used for the prediction
x_train = pd.DataFrame(index = range(total_samples), columns = cols, dtype = np.float64) #an empty dataframe holding our feature values
y_train = pd.DataFrame(index = range(total_samples), columns = ['time_to_failure'], dtype = np.float64) #an empty dataframe holding our target labels


# In[ ]:


for value in tqdm(range(total_samples)):
    sample = train.iloc[value*seg_length : value*seg_length + seg_length]
    x = sample['acoustic_data'].values
    y = sample['time_to_failure'].values[-1]
    
    y_train.loc[value, 'time_to_failure'] = y
    
    x_train.loc[value, 'average'] = x.mean()
    x_train.loc[value, 'std'] = x.std()


# In[ ]:


train_acoustic_data_small = x_train['average']

fig, ax1 = plt.subplots(figsize=(16, 8))
plt.title("Trends of acoustic_data and time_to_failure. 2% of data (sampled)")
plt.plot(train_acoustic_data_small, color='b')
ax1.set_ylabel('acoustic_data', color='b')
plt.legend(['acoustic_data'])
plt.grid(False)

del train_acoustic_data_small


# In[ ]:


y_train.shape


# In[ ]:


nlags=60  #define number of lags to plot on ACF/PACF plots
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = tsaplots.plot_acf(y_train, lags=nlags, ax=ax1,title='Autocorrelation of Seismic Data')
plt.xticks(list(range(0,nlags)))

ax2 = fig.add_subplot(212)
fig = tsaplots.plot_pacf(y_train, lags=nlags, ax=ax2,title='Partial Autocorrelation of Seismic Data')
plt.xticks(list(range(0,nlags)))
plt.xlabel('Lag-k')
plt.show()


# # Building a Time Series Regression Model

# In[ ]:


train_main = train.values[::6000]


# In[ ]:


train_main = pd.DataFrame(train_main, columns=['seismic','ttf'])
train_main.head()


# In[ ]:


train_acoustic_data_small = train_main['seismic']
train_time_to_failure_small = train_main['ttf']

fig, ax1 = plt.subplots(figsize=(16, 8))
plt.title("Trends of acoustic_data and time_to_failure. 2% of data (sampled)")
plt.plot(train_acoustic_data_small, color='b')
ax1.set_ylabel('acoustic_data', color='b')
plt.legend(['acoustic_data'])
ax2 = ax1.twinx()
plt.plot(train_time_to_failure_small, color='g')
ax2.set_ylabel('time_to_failure', color='g')
plt.legend(['time_to_failure'], loc=(0.875, 0.9))
plt.grid(False)

del train_acoustic_data_small
del train_time_to_failure_small


# In[ ]:


train_lr = train_main[['seismic']]
target_lr = train_main[['ttf']]
print(train_lr.shape)
print(target_lr.shape)


# In[ ]:


plt.plot(train_main.ttf.diff())


# In[ ]:


lr = LinearRegression()
model = lr.fit(train_lr,target_lr)
predict = model.predict(train_lr)
residuals = model.predict(train_lr) - target_lr


# In[ ]:


mean_absolute_error(target_lr, predict)


# In[ ]:


'''fig = plt.figure(figsize=(16,10))
nlags = 15
ax1 = plt.subplot(2, 2, 1)
fig = tsaplots.plot_acf(residuals, lags=nlags, ax=ax1,title='ACF for residuals')
plt.xticks(list(range(0,nlags)))

ax2 = plt.subplot(2, 2, 2)
fig = tsaplots.plot_pacf(residuals, lags=nlags, ax=ax2,title='PACF for residuals')
plt.xticks(list(range(0,nlags)))
plt.xlabel('Lag-k')
plt.show() '''


# In[ ]:


model = SARIMAX(train_main.ttf, order=(1,0,0), exog=train_main[['seismic']])
result = model.fit()
result.summary()


# In[ ]:


sample_submission = pd.read_csv("../input/sample_submission.csv")
sample_submission.shape


# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
X_test = pd.DataFrame(columns=train_main.columns, dtype=np.float64, index=submission.index)


# In[ ]:





# In[ ]:


test_result = []


# In[ ]:


for seg_id in tqdm(X_test.index):
    test = pd.read_csv('../input/test/' + seg_id + '.csv')
    pred_uc = result.get_forecast(steps = len(test), exog=test[['acoustic_data']])
    test_result.append(max(pred_uc.predicted_mean))


# In[ ]:


test_result_submission = pd.DataFrame(test_result).T


# In[ ]:


test_result_submission = test_result_submission.T


# In[ ]:


test_result_submission.to_csv("submission")


# In[ ]:


print(test_result_submission[:50])


# In[ ]:


print(test_result_submission[50:100])


# In[ ]:


print(test_result_submission[100:150])


# In[ ]:


print(test_result_submission[150:200])


# In[ ]:


print(test_result_submission[200:250])


# In[ ]:


print(test_result_submission[250:300])


# In[ ]:


print(test_result_submission[300:350])


# In[ ]:





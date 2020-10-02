#!/usr/bin/env python
# coding: utf-8

# This is a tutorial on **Time series forecasting using multi-layer LSTM.** 
# Data: Household electric power consumption 
# The description of data can be found here: http://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption   
# Task: Given some parameters in time steps (t-1, t-2 ...t-x), can we predict Global_active_power at the current time t.
# ***More to come***

# **1. Importing libraries and reading data**

# In[ ]:


#import all essential libraries
import sys 
import numpy as np # linear algebra
from scipy.stats import randint
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph. 
from sklearn.metrics import mean_squared_error,r2_score
## Deep-learing:
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import itertools
from keras.layers import LSTM
from keras.layers import Dropout


# Read the txt file. Raw Data has the datetime field split into two different columns so we will merge it into single column  

# In[ ]:


data = pd.read_csv('../input/household_power_consumption.txt', sep=';', 
                parse_dates={'dt' : ['Date', 'Time']}, infer_datetime_format=True, 
                low_memory=False, na_values=['nan','?'], index_col='dt')


# In[ ]:


data.head()


# In[ ]:


data.describe()


# **2. Data Preprocessing **
# 
# Do we have any missing values? Do we need all the columns?

# In[ ]:


## finding all columns that have nan:
data.columns[data.isnull().any()]


# Its important to clean the data and treat missing values. But filling in  with mean or zeros can skew the model a lot espeically for fields with high variance. 
# So First step before filling in the missing values is to check linearity and variation in data. This problem is more tricky than it sounds as ideally for any time series, 
# we would like to decompose it to see the trend and seasonality. But this tutorial is limited to providing a general workflow for LSTM based forecasting.
# 
# Below, we get a plot of how autocorelation of different signals change with lag to better model the imputation

# In[ ]:


#To fill in missing values, first check for linearity in each column using autocorelation plots
def get_acf(data,lags): 
    frame = []
    for i in range(lags+1):
        frame.append(data.apply(lambda col: col.autocorr(i), axis=0))
    return pd.DataFrame(frame).plot.line()
get_acf(data,10)


# Seems like acfs start dropping after 10 lags for majority of signals. So lets do linear interpolation but with a limit of 10 

# In[ ]:


#fill in missing values using linear interpolation with limit to 10 using autocorelation plot
clean_data = data.interpolate(method = 'linear', axis = 0, limit = 10)


# Now, the data is aggregated every minute. Do we need that? Maybe not, we can do corelation plots of different aggregations to see how data change. But first lets do quick charting per day.

# In[ ]:


#Its imperative from autocorealtion plots that we dont need each minute data, perhaps we can decide roll up the aggregate
i = 1
# plot each column
plt.figure(figsize=(20, 15))
for counter in range(1,len(clean_data.columns)):
    plt.subplot(len(clean_data.columns), 1, i)
    plt.plot(clean_data.resample('D').mean().values[:, counter], color = 'blue')
    plt.title(clean_data.columns[counter], y=0.8, loc='right')
    i = i+1
plt.show()


# Looks reasonable? But how do the corelation changes when we roll up the aggregation? Lets check it out

# In[ ]:


#check corelation matrices for minute, hour and day
#minute
corr_min = clean_data.corr()
#hour
corr_hour = pd.DataFrame(clean_data.resample('H').mean().values).corr()
#day
corr_day = pd.DataFrame(clean_data.resample('D').mean().values).corr()
cmap=sns.diverging_palette(5, 250, as_cmap=True)

def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
]
#Plot Minute
corr_min.style.background_gradient(cmap, axis=1)    .set_properties(**{'max-width': '80px', 'font-size': '10pt'})    .set_caption("Hover to magify")    .set_precision(2)    .set_table_styles(magnify())


# By Hour

# In[ ]:


corr_hour.style.background_gradient(cmap, axis=1)    .set_properties(**{'max-width': '80px', 'font-size': '10pt'})    .set_caption("Hover to magify")    .set_precision(2)    .set_table_styles(magnify())


# By Day

# In[ ]:


corr_day.style.background_gradient(cmap, axis=1)    .set_properties(**{'max-width': '80px', 'font-size': '10pt'})    .set_caption("Hover to magify")    .set_precision(2)    .set_table_styles(magnify())


# Grouping by day, minute, hour is really a business problem. If the business needs a strategy for short term forecasting then by hour seems a good option. Lets assume this is the case. 
# Corelation plot shows a very high corelation between global active power and global intensity for all 3 aggregations. It is safe to drop itl 

# In[ ]:


#drop highly corelated columns 

clean_data = clean_data.drop(columns = ['Global_intensity'])


# Now comes the problem formulation.  Lets assume the task is to predict global active power at current hour given previous readings. We have to convert this to a supervised problem. 
# Number of lags can be a hyperparameter for the model. it specifies number of time steps to go back for feature engineering. Its easier to understand by playing with different lags in below function.

# In[ ]:


#Credit: Adopted from https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    dff = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(dff.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(dff.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# **3. Problem formulation and training LSTM**

# In[ ]:


#framing the problem
#resample over day

## resampling of data over hour
resampled_data = clean_data.resample('H').mean() 
resampled_data.shape


# Now before moving forward, I would like to point out that you could potentially normalise the data in the data preprocessign phase and later decode it, i have left it out intentionally here as it made no difference in results.

# In[ ]:


#normalised_data=(resampled_data-resampled_data.min())/(resampled_data.max()-resampled_data.min())
#print(normalised_data.head())


# In[ ]:


# Feature engineering for LSTM
reframed_data = series_to_supervised(resampled_data, 1, 1)
print(reframed_data.head())


# In[ ]:


# drop columns we don't want
reframed_data.drop(reframed_data.columns[[7,8,9,10,11]], axis=1, inplace=True)
print(reframed_data.columns)


# In[ ]:


# split into train and test sets
values = reframed_data.values
train_index = 500*24 #The logic is to have 500 days worth of training data. this could also be a hyperparameter that can be tuned.
train = values[:train_index, :]
test = values[train_index:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape) 
# We reshaped the input into the 3D format as expected by LSTMs, namely [samples, timesteps, features].


# This problem is mapped as many to one with stacked LSTMs with dropouts to control fitting. Loss is measured in MSE and optimised using adam optimizer.
# Hyperparameter tuning: Technically I should be doing a grid/bayesian/random search on parameters like number of nuerons, dropouts, etc and get the parameters with least MSE. Beyond scope of this notebook.

# In[ ]:


model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=10, batch_size=20, validation_data=(test_X, test_y), verbose=1, shuffle=False)
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# In[ ]:


# make a prediction
yhat = model.predict(test_X, verbose=0)
rmse = np.sqrt(mean_squared_error(test_y, yhat))
print('Test RMSE: %.3f' % rmse)


# Plot Actual vs Predicted

# In[ ]:


## time steps, every step is one hour (you can easily convert the time step to the actual time index)
## for a demonstration purpose, I only compare the predictions in 200 hours. 

aa=[x for x in range(200)]
plt.plot(aa, test_y[:200], marker='.', label="actual")
plt.plot(aa, yhat[:200], 'r', label="prediction")
plt.ylabel('Global_active_power', size=15)
plt.xlabel('Time', size=15)
plt.legend(fontsize=13)
plt.show()


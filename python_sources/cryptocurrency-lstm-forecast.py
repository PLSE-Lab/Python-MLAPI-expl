#!/usr/bin/env python
# coding: utf-8

# This notebook forecasts cryptocurrency prices and volumes using a basic LSTM (long-short term memory) RNN architecture. We experiment with a few different approaches and plot the results.

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
from numpy.lib.stride_tricks import as_strided
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import Series, DataFrame
from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = [12, 8]
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten
from keras.optimizers import Adam
import keras.backend as K
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Below are a series of helper functions to prep the data for the LSTM. 

# In[2]:


def gen_raw_data(input_path, cryptos_to_pred):
    
    # prepare data
    all_currencies_raw = pd.read_csv(input_path)
    mask = all_currencies_raw['Symbol'].apply(lambda x: x in cryptos_to_pred)
    prediction_currencies = all_currencies_raw.loc[mask].drop('Unnamed: 0', axis=1)[['Date', 'Symbol', 'Volume', 'Close']]
    wide_close = prediction_currencies.pivot(index='Date', columns='Symbol', values='Close')
    wide_close.index = pd.to_datetime(wide_close.index)
    wide_close.rename(index=str, columns={'%s' % crypto: '%s_closing_price' % crypto for crypto in cryptos_to_pred}, inplace=True)
    wide_volume = prediction_currencies.pivot(index='Date', columns='Symbol', values='Volume')
    wide_volume.index = pd.to_datetime(wide_volume.index)
    wide_volume.rename(index=str, columns={'%s' % crypto: '%s_volume' % crypto for crypto in cryptos_to_pred}, inplace=True)
    prepped_data = wide_close.join(wide_volume)
    return(prepped_data)


# In[3]:


def gen_train_test_data(input_df, start, cutoff):
    
    train_mask = ((input_df.index >= start) & (input_df.index <= cutoff))
    test_mask = input_df.index > cutoff
    X_train_raw = input_df.loc[train_mask]
    X_test_raw = input_df.loc[test_mask]
    Y_train_raw = X_train_raw.shift(-1).dropna()
    Y_test_raw = X_test_raw.shift(-1).dropna()
    X_train_raw = X_train_raw[:-1]
    X_test_raw = X_test_raw[:-1]
    X_train_raw = normalize_data(X_train_raw)
    X_test_raw = normalize_data(X_test_raw)
    Y_train_raw = normalize_data(Y_train_raw)
    Y_test_raw = normalize_data(Y_test_raw)
    return X_train_raw, X_test_raw, Y_train_raw, Y_test_raw


# In[4]:


def normalize_data(input_df):
    
    sc = MinMaxScaler(feature_range=(0,1))
    for column in input_df.columns:
        input_df[column] = sc.fit_transform(input_df[column].values.reshape(-1, 1))
    return input_df


# In[5]:


def shape_for_lstm(input_df, seq_length=50):

    raw_values = input_df.values
    raw_shape = raw_values.shape
    new_values = np.empty([raw_shape[0] - seq_length, seq_length, raw_shape[1]])
    cur_pos = 0
    for i in range(new_values.shape[0]):
        next_pos = cur_pos + seq_length
        new_values[i,:,:] = raw_values[cur_pos:next_pos,:]
        cur_pos = cur_pos + 1
    return new_values


# In[6]:


def prep_data(input_path, cryptos_to_predict, start_date, train_test_cutoff, seq_length=50):
    
    input_data = gen_raw_data('../input/all_currencies.csv', cryptos_to_predict)
    X_train_raw, X_test_raw, Y_train_raw, Y_test_raw = gen_train_test_data(input_data, start_date, train_test_cutoff)
    names = X_train_raw.columns.tolist()
    X_train = shape_for_lstm(X_train_raw, seq_length)
    X_test = shape_for_lstm(X_test_raw, seq_length)
    Y_train = shape_for_lstm(Y_train_raw, seq_length)
    Y_test = shape_for_lstm(Y_test_raw, seq_length)
    return X_train, X_test, Y_train, Y_test, names


# In[7]:


def run_lstm_model(X_train, X_test, Y_train, Y_test):
    
    K.clear_session()
    model = Sequential()
    model.add(LSTM(X_train.shape[2], input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error', metrics=['mse'])
    model.fit(X_train, Y_train, batch_size=1, epochs=10, verbose=1)
    Y_pred = model.predict(X_test, batch_size=1)
    return(Y_pred, model)


# In[8]:


def gen_test_pred_frame(Ytest, Ypred, slice_number, pred_names, test_names):
    Y_pred_frame = DataFrame(Ytest[slice_number], columns=pred_names)
    Y_test_frame = DataFrame(Ypred[slice_number], columns=test_names)
    joined_frame = Y_pred_frame.join(Y_test_frame)
    joined_frame.index = pd.date_range(pd.to_datetime('2017-04-02') + timedelta(days=slice_number), periods=30)
    return(joined_frame)


# In[9]:


input_data_path = '../input/all_currencies.csv'
cryptos_to_predict = {'LTC', 'BTC', 'NMC', 'NVC', 'PPC'}
start_date = '2013-12-27'
train_test_cutoff = '2017-03-31'
seq_length = 30


# In[10]:


X_train, X_test, Y_train, Y_test, names = prep_data(
    input_data_path, 
    cryptos_to_predict, 
    start_date, 
    train_test_cutoff,
    seq_length
)
pred_names = ['%s_pred' % name for name in names]
test_names = ['%s_true' % name for name in names]


# Our first LSTM model architecture is quite simple. There is one single layer with 30 neurons. Each input value is a 30 day sequence of $x_{t}$, a 10-dimensional vector of closing prices and volumes for five cryptocurrencies: bitcoin, litecoin, namecoin, novacoin, and peercoin. The corresponding $y_{t}$ the same 10-dimensional vector shifted forward a day (in other words, $x_{t+1} = y_{t}$. In other words, the LSTM predicts the next day's prices and volumes based on the current day's prices and volumes as well as all historical prices and volumes and their predictions within a 30 day window. We train the model on data from 2013-12-27 through 2017-03-31 (which gives us 1159 overlapping 30 day sequencies). Next, we run the model and look at some plots. 

# In[11]:


Y_pred, model = run_lstm_model(X_train, X_test, Y_train, Y_test)


# In[12]:


test_scores = model.evaluate(Y_pred, Y_test, batch_size=1)
Y_train_pred = model.predict(X_train, batch_size=1)
train_scores = model.evaluate(Y_train_pred, Y_train, batch_size=1)
print('Test Set MSE: %f' % test_scores[0])
print('Training Set MSE: %f' % train_scores[0])


# Uh oh - looks like our model does a lot better on the training set than it does on the test set. This isn't too surprising: we've started with just a single layer with 30 nodes and have no regularization. Let's take a quick look at how well this forecasting approach does with some plots. We'll do this by grabbing a few random 30 day segments from our test set (post 2017-04-01 data) and plotting the predicted and true values for one of our 10 time series. For now I have left the data in normalized form, so predicted and true values will largely lie between 0 and 1. 

# In[13]:


testframe1 = gen_test_pred_frame(Y_test, Y_pred, 127, pred_names, test_names)
ax = testframe1[['BTC_closing_price_pred', 'BTC_closing_price_true']].plot()
ax.xaxis.grid(True, which='minor', linestyle='-', linewidth=0.5)
testframe2 = gen_test_pred_frame(Y_test, Y_pred, 127, pred_names, test_names)
ax = testframe2[['BTC_volume_pred', 'BTC_volume_true']].plot()
ax.xaxis.grid(True, which='minor', linestyle='-', linewidth=0.5)


# In[14]:


testframe1 = gen_test_pred_frame(Y_test, Y_pred, 18, pred_names, test_names)
ax = testframe1[['LTC_closing_price_pred', 'LTC_closing_price_true']].plot()
ax.xaxis.grid(True, which='minor', linestyle='-', linewidth=0.5)
testframe2 = gen_test_pred_frame(Y_test, Y_pred, 18, pred_names, test_names)
ax = testframe2[['LTC_volume_pred', 'LTC_volume_true']].plot()
ax.xaxis.grid(True, which='minor', linestyle='-', linewidth=0.5)


# In[15]:


testframe1 = gen_test_pred_frame(Y_test, Y_pred, 175, pred_names, test_names)
ax = testframe1[['NMC_closing_price_pred', 'NMC_closing_price_true']].plot()
ax.xaxis.grid(True, which='minor', linestyle='-', linewidth=0.5)
testframe2 = gen_test_pred_frame(Y_test, Y_pred, 175, pred_names, test_names)
ax = testframe2[['NMC_volume_pred', 'NMC_volume_true']].plot()
ax.xaxis.grid(True, which='minor', linestyle='-', linewidth=0.5)


# In[16]:


testframe1 = gen_test_pred_frame(Y_test, Y_pred, 222, pred_names, test_names)
ax = testframe1[['NVC_closing_price_pred', 'NVC_closing_price_true']].plot()
ax.xaxis.grid(True, which='minor', linestyle='-', linewidth=0.5)
testframe2 = gen_test_pred_frame(Y_test, Y_pred, 222, pred_names, test_names)
ax = testframe2[['NVC_volume_pred', 'NVC_volume_true']].plot()
ax.xaxis.grid(True, which='minor', linestyle='-', linewidth=0.5)


# In[17]:


testframe1 = gen_test_pred_frame(Y_test, Y_pred, 330, pred_names, test_names)
ax = testframe1[['PPC_closing_price_pred', 'PPC_closing_price_true']].plot()
ax.xaxis.grid(True, which='minor', linestyle='-', linewidth=0.5)
testframe2 = gen_test_pred_frame(Y_test, Y_pred, 330, pred_names, test_names)
ax = testframe2[['PPC_volume_pred', 'PPC_volume_true']].plot()
ax.xaxis.grid(True, which='minor', linestyle='-', linewidth=0.5)


# Long story short, some of these predictions look better than others. There are a few cases where the LSTM systematically under or overestimates the true value (the LTC volume prediction for example). There are others where it more or less bounces around very close to the true value (the PPC price prediction). Before we get too excited about this, we should address the elephant in the room: these are only next-day forecasts, so we should in general expect them to do quite well. Let's make things more challenging: we'll reuse our already trained network to make predictions k steps into the future beyond the 30 day window. 

# In[18]:


def gen_k_step_forecast(slice_number, X_test, model, k=30):
    new_predictions = np.empty((k, 10))
    sliding_X = X_test[slice_number,:,:].reshape(1, X_test.shape[1], X_test.shape[2]).copy()
    for i in range(k):
        this_prediction = model.predict(sliding_X)
        next_step = this_prediction[0,seq_length-1,:].reshape(1,1,10)
        new_predictions[i] = next_step
        sliding_X = np.concatenate((sliding_X, next_step), axis=1)[0,1:,:].reshape(1,k,10)
    return new_predictions


# In[19]:


def gen_k_step_joined_frame(new_predictions, Y_test, slice_number, pred_names):
    Y_pred_frame = DataFrame(new_predictions, columns=pred_names)
    Y_test_frame = DataFrame(Y_test[slice_number+29], columns=test_names)
    joined_frame = Y_pred_frame.join(Y_test_frame)
    joined_frame.index = pd.date_range(pd.to_datetime('2017-04-02') + timedelta(days=slice_number+29), periods=30)
    return joined_frame


# In[20]:


def gen_plots(slice_number, currency, X_test, Y_test, model, pred_names):
    new_preds = gen_k_step_forecast(slice_number, X_test, model)
    joined_frame = gen_k_step_joined_frame(new_preds, Y_test, slice_number, pred_names)
    ax1 = joined_frame[['%s_closing_price_pred' % currency, '%s_closing_price_true' % currency]].plot()
    ax1.xaxis.grid(True, which='minor', linestyle='-', linewidth=0.5)
    ax2 = joined_frame[['%s_volume_pred' % currency, '%s_volume_true' % currency]].plot()
    ax2.xaxis.grid(True, which='minor', linestyle='-', linewidth=0.5)


# In[21]:


gen_plots(180, 'BTC', X_test, Y_test, model, pred_names)


# In[22]:


gen_plots(175, 'LTC', X_test, Y_test, model, pred_names)


# In[23]:


gen_plots(145, 'NMC', X_test, Y_test, model, pred_names)


# In[24]:


gen_plots(285, 'NVC', X_test, Y_test, model, pred_names)


# In[25]:


gen_plots(315, 'PPC', X_test, Y_test, model, pred_names)


# The first thing that's notable about these forecasts is that they are much less capable of self-correction. The bitcoin price forecast in particular gets going the wrong direction and never turns around. Meanwhile, the PPC price forecast misses day-to-day variation but is closer to the trend of the true data. Still, none of these forecasts does a great job. 

# **Further Steps**
# 
# There are a variety of ways we can improve these forecasts. I'll save these for another time:
# * Gather and analyze the cases where there appears to be systematic, persistent under or overestimation. 
# * Improve the model by adding more layers, tuning hyperparameters, adding regularization, and perhaps generating some new features by hand.
# * Compare model performance to a simple baseline model like a moving average model.

# In[ ]:





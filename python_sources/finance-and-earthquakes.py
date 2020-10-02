#!/usr/bin/env python
# coding: utf-8

# 
# **Hi, how are you?**
# I'm a big advocate of using algorithms from other fields (in this case finnace) for uncanny applications (in this case to predict earthquakes). 
# I would like to put a big shout out to this git hub : https://github.com/borisbanushev/stockpredictionai
# Which is amazing and from which I took out most of the code to adapt it to this problem, and you should definitely check that repo out !

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from utils import *

import os
import time
import numpy as np

from mxnet import nd, autograd, gluon
from mxnet.gluon import nn, rnn
import mxnet as mx
import datetime
import seaborn as sns

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.decomposition import PCA

import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from tqdm import tqdm_notebook as tqdm
import os
import gc
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# The input data

# In[ ]:


#This method is widely used by other kernels to implement data because the data is too large

train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})


# In[ ]:


#This method is widely used by other kernels to seperate the data.

rows = 150_000
segments = int(np.floor(train.shape[0] / rows))

x_train = pd.DataFrame(index=range(segments), dtype=np.int16,
                       columns=['acoustic_data'])
                       
y_train = pd.DataFrame(index=range(segments), dtype=np.float32, columns=['time_to_failure'])
                       
for segment in tqdm(range(segments)):
    seg = train.iloc[segment*rows:segment*rows+rows]
    x = seg['acoustic_data'].values
    y = seg['time_to_failure'].values[-1]
    y_train.loc[segment, 'time_to_failure'] = y
    x_train.loc[segment, 'acoustic_data']=x.mean()                   


# **Make some space**

# In[ ]:


#My god this is 10 GB guys !! What the hell, delete it
del train
#Collect garbage
gc.collect()


# Small plot to see what it looks like

# In[ ]:


plt.plot(x_train['acoustic_data'][0:35])


# ### Technical Analysis
# **Technical indicators** are a big part of financial analysis, this computes them.

# In[ ]:


def get_technical_indicators(dataset):
    # Create 7 and 21 days Moving Average
    dataset['ma7'] = dataset['acoustic_data'].rolling(window=7).mean()
    dataset['ma21'] = dataset['acoustic_data'].rolling(window=21).mean()
    
    # Create MACD
    dataset['26ema'] = dataset['acoustic_data'].ewm(span=26).mean() #pd.ewma(dataset['Close'], span=26)
    dataset['12ema'] = dataset['acoustic_data'].ewm(span=12).mean() #pd.ewma(dataset['Close'], span=12)
    dataset['MACD'] = (dataset['12ema']-dataset['26ema'])

    # Create Bollinger Bands
    dataset['20sd'] = dataset['acoustic_data'].rolling(20).std() #pd.stats.moments.rolling_std(dataset['Close'],20)
    dataset['upper_band'] = dataset['ma21'] + (dataset['20sd']*2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20sd']*2)
    
    # Create Exponential moving average
    dataset['ema'] = dataset['acoustic_data'].ewm(com=0.5).mean()
    
    # Create Momentum
    dataset['momentum'] = dataset['acoustic_data']-1
    
    return dataset


# In[ ]:




#Lets get these technical indicators
technical_train = get_technical_indicators(x_train)


# In[ ]:


#Drop the NA values got from the rolling averages
technical_train_na = technical_train.dropna()


# In[ ]:


def plot_technical_indicators(dataset, last_days):
    plt.figure(figsize=(16, 10), dpi=100)
    shape_0 = dataset.shape[0]
    xmacd_ = shape_0-last_days
    
    #dataset = dataset.iloc[-last_days:, :]
    x_ = range(3, dataset.shape[0])
    x_ =list(dataset.index)
    
    # Plot first subplot
    #plt.subplot(2, 1, 1)
    plt.plot(dataset['ma7'],label='MA 7', color='g',linestyle='--')
    plt.plot(dataset['acoustic_data'],label='Real Data', color='b')
    plt.plot(dataset['ma21'],label='MA 21', color='r',linestyle='--')
    plt.plot(dataset['upper_band'],label='Upper Band', color='c')
    plt.plot(dataset['lower_band'],label='Lower Band', color='c')
    plt.plot((y_train['time_to_failure']/10)+4,label='Time to failure',color = 'y')
    plt.fill_between(x_, dataset['lower_band'], dataset['upper_band'], alpha=0.35)
    plt.title('Technical indicators')
    plt.ylabel('Signal')
    plt.legend()

    
    plt.show()


# **Man does it look good ! **

# In[ ]:


plot_technical_indicators(technical_train_na,1000)


# In[ ]:


#Let's use the FFT to get more features.
data_FT = x_train[['acoustic_data']]

close_fft = np.fft.fft(np.asarray(data_FT['acoustic_data'].tolist()))
fft_df = pd.DataFrame({'fft':close_fft})
fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))


# In[ ]:


plt.figure(figsize=(14, 7), dpi=100)
fft_list = np.asarray(fft_df['fft'].tolist())
for num_ in [3, 6, 9, 25, 100]:
    fft_list_m10= np.copy(fft_list); fft_list_m10[num_:-num_]=0
    plt.plot(np.fft.ifft(fft_list_m10), label='Fourier transform with {} components'.format(num_))
    
#plt.plot(data_FT['acoustic_data'],  label='Real')
plt.xlabel('rows')
plt.ylabel('amplitude')
plt.title('Fourier transform')
plt.legend()
plt.show()


# **Here we use the Fourrier transform to get more features**

# In[ ]:


from collections import deque
items = deque(np.asarray(fft_df['absolute'].tolist()))
items.rotate(int(np.floor(len(fft_df)/2)))
plt.figure(figsize=(10, 7), dpi=80)
plt.stem(items)
plt.title('Components of Fourier transforms')
plt.show()


# **The ARIMA model** is big for stock prediction, let's see if it can also predict earthquake amplitude.

# In[ ]:


from statsmodels.tsa.arima_model import ARIMA
from pandas import DataFrame
from pandas import datetime


# In[ ]:


series = data_FT['acoustic_data']
model = ARIMA(series, order=(5, 1, 0))
model_fit = model.fit(disp=0)
print(model_fit.summary())


# In[ ]:


from pandas.tools.plotting import autocorrelation_plot
autocorrelation_plot(series)
plt.figure(figsize=(10, 7), dpi=80)
plt.show() 


# In[ ]:


from pandas import read_csv
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

X = series.values
size = int(len(X) * 0.01)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)


# In[ ]:


error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)


# In[ ]:


plt.figure(figsize=(12, 6), dpi=100)
plt.plot(test, label='Real')
plt.plot(predictions, color='red', label='Predicted')
plt.xlabel('Rows')
plt.ylabel('Amplitude')
plt.title('ARIMA model on Amplitude')
plt.legend()
plt.show()


# In[ ]:


arima = [np.NaN for i in range(41)] + list(test) #complete list for total database concatenation
ARIMA = pd.DataFrame(data=arima, columns=['ARIMA'])


# **Wow looks good ! We'll try to use it in our model**

# In[ ]:


df_total = pd.concat([technical_train,ARIMA,fft_df.drop(['fft'],axis=1),y_train],ignore_index=False,sort=False,axis=1)


# In[ ]:


df_total.head()


# In[ ]:


df_total_withoutna=df_total.dropna()
df_total_withoutna.iloc[:,:14].head()


# In[ ]:


def get_feature_importance_data(data_income):
    data = data_income.copy()
    y = data['time_to_failure']
    X = data.iloc[:, :14]
    
    train_samples = int(X.shape[0] * 0.65)
 
    X_train = X.iloc[:train_samples]
    X_test = X.iloc[train_samples:]

    y_train = y.iloc[:train_samples]
    y_test = y.iloc[train_samples:]
    
    return (X_train, y_train), (X_test, y_test)


# In[ ]:


# Get training and test data
(X_train_FI, y_train_FI), (X_test_FI, y_test_FI) = get_feature_importance_data(df_total_withoutna)
regressor = xgb.XGBRegressor(gamma=0.0,n_estimators=150,base_score=0.7,colsample_bytree=1,learning_rate=0.05)
xgbModel = regressor.fit(X_train_FI,y_train_FI,                          eval_set = [(X_train_FI, y_train_FI), (X_test_FI, y_test_FI)],                          verbose=False)
eval_result = regressor.evals_result()
training_rounds = range(len(eval_result['validation_0']['rmse']))


# In[ ]:


plt.scatter(x=training_rounds,y=eval_result['validation_0']['rmse'],label='Training Error')
plt.scatter(x=training_rounds,y=eval_result['validation_1']['rmse'],label='Validation Error')
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.title('Training Vs Validation Error')
plt.legend()
plt.show()


# In[ ]:


fig = plt.figure(figsize=(8,8))
plt.xticks(rotation='vertical')
plt.bar([i for i in range(len(xgbModel.feature_importances_))], xgbModel.feature_importances_.tolist(), tick_label=X_test_FI.columns)
plt.title('Feature importance of indicators.')
plt.show()


# **We drop** the features which are not useful

# In[ ]:


df_final = df_total_withoutna.drop(['acoustic_data','12ema','ema','momentum','ARIMA'],axis=1)


# **Let's build the network now that we have some features**

# In[ ]:


import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model
from keras.layers import Input
from tensorflow.python.keras import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


# In[ ]:


def get_feature_importance_data(data_income):
    data = data_income.copy()
    y = data['time_to_failure']
    X = data.iloc[:, :9]
    
    train_samples = int(X.shape[0] * 0.65)
 
    X_train = X.iloc[:train_samples]
    X_test = X.iloc[train_samples:]

    y_train = y.iloc[:train_samples]
    y_test = y.iloc[train_samples:]
    
    return (X_train, y_train), (X_test, y_test)


# In[ ]:


(X_train_FI, y_train_FI), (X_test_FI, y_test_FI) = get_feature_importance_data(df_final)
# Get training and test data
regressor = xgb.XGBRegressor(gamma=0.0,n_estimators=150,base_score=0.7,colsample_bytree=1,learning_rate=0.05)
xgbModel = regressor.fit(X_train_FI,y_train_FI,                          eval_set = [(X_train_FI, y_train_FI), (X_test_FI, y_test_FI)],                          verbose=False)
eval_result = regressor.evals_result()
training_rounds = range(len(eval_result['validation_0']['rmse']))


# In[ ]:


lol = regressor.predict(X_test_FI)
plt.plot(lol-5)
plt.plot(y_test_FI.values)


# That's what XGBoost can do

# In[ ]:



Xt=X_train_FI.values
Yt=y_train_FI.values

Xt.shape


# In[ ]:


model = Sequential()
model.add(Dense(32,input_shape = (9,),activation = 'relu'))
model.add(Dense(32,activation = 'relu'))
model.add(Dense(32,activation = 'relu'))
model.add(Dense(1))
model.compile(loss = 'mae',optimizer = 'adam')
model.fit(Xt, Yt, epochs=300, verbose=0)



# In[ ]:


y_pred = model.predict(X_test_FI.values)


# In[ ]:


plt.plot((y_pred-5)*2.5,color='g')
plt.plot(y_test_FI.values,color='r')


# **That's** what a simple NN can do, hopefully we can find something better because it just looks like noise...

# **SO**:
# Basically, this doesn't give good results, but anyway, I hope it was still useful to you and give you ideas to solve this problem, I just finished this and will update as time goes on, thanks for the read, please please please coment if you have ideas on how to use this and maybe we'll be able to work something out. Much luck to you on this project !

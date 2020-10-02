#!/usr/bin/env python
# coding: utf-8

# # **How to use the Python programming Language for Time Series Analysis!**
# 
# This work was prepared together with [Gul Bulut](https://www.kaggle.com/gulyvz) and [Bulent Siyah](https://www.kaggle.com/bulentsiyah/). **The whole study consists of two parties**
# * [Time Series Forecasting and Analysis- Part 1](https://www.kaggle.com/gulyvz/time-series-forecasting-and-analysis-part-1)
# * [Time Series Forecasting and Analysis- Part 2](https://www.kaggle.com/bulentsiyah/time-series-forecasting-and-analysis-part-2)
# 
# This kernel will teach you everything you need to know to use Python for forecasting time series data to predict new future data points.
# 
# ![](https://iili.io/JaZxFS.png)
# 
# we'll learn about state of the art Deep Learning techniques with Recurrent Neural Networks that use deep learning to forecast future data points.
# 
# ![](https://iili.io/JaZCMl.png)
# 
# 
# This kernel even covers Facebook's Prophet library, a simple to use, yet powerful Python library developed to forecast into the future with time series data.
# 
# ![](https://iili.io/JaZnP2.png)
# 
# # **Content Part 1** 
# 
# 1. [How to Work with Time Series Data with Pandas](https://www.kaggle.com/gulyvz/time-series-forecasting-and-analysis-part-1#1.)
# 1. [Use Statsmodels to Analyze Time Series Data](https://www.kaggle.com/gulyvz/time-series-forecasting-and-analysis-part-1#2.)
# 1. [General Forecasting Models - ARIMA(Autoregressive Integrated Moving Average)](https://www.kaggle.com/gulyvz/time-series-forecasting-and-analysis-part-1#3.)
# 1. [General Forecasting Models - SARIMA(Seasonal Autoregressive Integrated Moving Average)](https://www.kaggle.com/gulyvz/time-series-forecasting-and-analysis-part-1#4.)
# 1. [General Forecasting Models - SARIMAX](https://www.kaggle.com/gulyvz/time-series-forecasting-and-analysis-part-1#5.)
# 
# # **Content Part 2**
# 
# 1. [Deep Learning for Time Series Forecasting - (RNN)](#1.)
# 1. [Multivariate Time Series with RNN](#2.)
# 1. [Use Facebook's Prophet Library for forecasting](#3.)
# 

# <a class="anchor" id="1."></a> 
# # 1.Deep Learning for Time Series Forecasting - (RNN)

# In[ ]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('/kaggle/input/for-simple-exercises-time-series-forecasting/Alcohol_Sales.csv',index_col='DATE',parse_dates=True)
df.index.freq = 'MS'
df.head()


# In[ ]:


df.columns = ['Sales']
df.plot(figsize=(12,8))


# In[ ]:


from statsmodels.tsa.seasonal import seasonal_decompose

results = seasonal_decompose(df['Sales'])
results.observed.plot(figsize=(12,2))


# In[ ]:


results.trend.plot(figsize=(12,2))


# In[ ]:


results.seasonal.plot(figsize=(12,2))


# In[ ]:


results.resid.plot(figsize=(12,2))


# ## Train Test Split

# In[ ]:


print("len(df)", len(df))

train = df.iloc[:313]
test = df.iloc[313:]


print("len(train)", len(train))
print("len(test)", len(test))


# ## Scale Data

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# IGNORE WARNING ITS JUST CONVERTING TO FLOATS
# WE ONLY FIT TO TRAININ DATA, OTHERWISE WE ARE CHEATING ASSUMING INFO ABOUT TEST SET
scaler.fit(train)


# In[ ]:


scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)


# ## Time Series Generator
# 
# This class takes in a sequence of data-points gathered at
# equal intervals, along with time series parameters such as
# stride, length of history, etc., to produce batches for
# training/validation.

# In[ ]:


from keras.preprocessing.sequence import TimeseriesGenerator
scaled_train[0]


# In[ ]:


# define generator
n_input = 2
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

print('len(scaled_train)',len(scaled_train))
print('len(generator)',len(generator))  # n_input = 2


# In[ ]:


# What does the first batch look like?
X,y = generator[0]

print(f'Given the Array: \n{X.flatten()}')
print(f'Predict this y: \n {y}')


# In[ ]:


# Let's redefine to get 12 months back and then predict the next month out
n_input = 12
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

# What does the first batch look like?
X,y = generator[0]

print(f'Given the Array: \n{X.flatten()}')
print(f'Predict this y: \n {y}')


# ## Create the Model

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()


# In[ ]:


# fit model
model.fit_generator(generator,epochs=50)


# In[ ]:


model.history.history.keys()
loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)),loss_per_epoch)


# ## Evaluate on Test Data

# In[ ]:


first_eval_batch = scaled_train[-12:]
first_eval_batch


# In[ ]:


first_eval_batch = first_eval_batch.reshape((1, n_input, n_features))
model.predict(first_eval_batch)


# In[ ]:


scaled_test[0]


# In[ ]:


test_predictions = []

first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):
    
    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]
    
    # store prediction
    test_predictions.append(current_pred) 
    
    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
    
test_predictions


# In[ ]:


scaled_test


# ## Inverse Transformations and Compare

# In[ ]:


true_predictions = scaler.inverse_transform(test_predictions)
true_predictions


# In[ ]:


test['Predictions'] = true_predictions
test


# In[ ]:


test.plot(figsize=(12,8))


# ## Saving and Loading Models

# In[ ]:


model.save('my_rnn_model.h5')
'''from keras.models import load_model
new_model = load_model('my_rnn_model.h5')'''


# <a class="anchor" id="2."></a> 
# # 2.Multivariate Time Series with RNN

# Experimental data used to create regression models of appliances energy use in a low energy building. Data Set Information: The data set is at 10 min for about 4.5 months. The house temperature and humidity conditions were monitored with a ZigBee wireless sensor network. Each wireless node transmitted the temperature and humidity conditions around 3.3 min. Then, the wireless data was averaged for 10 minutes periods. The energy data was logged every 10 minutes with m-bus energy meters. Weather from the nearest airport weather station (Chievres Airport, Belgium) was downloaded from a public data set from Reliable Prognosis (rp5.ru), and merged together with the experimental data sets using the date and time column. Two random variables have been included in the data set for testing the regression models and to filter out non predictive attributes (parameters). 

# ## Data
# 
# Let's read in the data set:

# In[ ]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

df = pd.read_csv('../input/for-simple-exercises-time-series-forecasting/energydata_complete.csv',index_col='date', infer_datetime_format=True)
df.head()


# In[ ]:


df.info()


# In[ ]:


df['Windspeed'].plot(figsize=(12,8))


# In[ ]:


df['Appliances'].plot(figsize=(12,8))


# ## Train Test Split

# In[ ]:


df = df.loc['2016-05-01':]
df = df.round(2)

print('len(df)',len(df))
test_days = 2
test_ind = test_days*144 # 24*60/10 = 144
test_ind


# In[ ]:


train = df.iloc[:-test_ind]
test = df.iloc[-test_ind:]


# ## Scale Data

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# IGNORE WARNING ITS JUST CONVERTING TO FLOATS
# WE ONLY FIT TO TRAININ DATA, OTHERWISE WE ARE CHEATING ASSUMING INFO ABOUT TEST SET
scaler.fit(train)


# In[ ]:


scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)


# ## Time Series Generator
# 
# This class takes in a sequence of data-points gathered at equal intervals, along with time series parameters such as stride, length of history, etc., to produce batches for training/validation.

# In[ ]:


from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# define generator
length = 144 # Length of the output sequences (in number of timesteps)
batch_size = 1 #Number of timeseries samples in each batch
generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=batch_size)


# In[ ]:


print('len(scaled_train)',len(scaled_train))
print('len(generator) ',len(generator))

X,y = generator[0]

print(f'Given the Array: \n{X.flatten()}')
print(f'Predict this y: \n {y}')


# ## Create the Model

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM

scaled_train.shape


# In[ ]:


# define model
model = Sequential()

# Simple RNN layer
model.add(LSTM(100,input_shape=(length,scaled_train.shape[1])))

# Final Prediction (one neuron per feature)
model.add(Dense(scaled_train.shape[1]))

model.compile(optimizer='adam', loss='mse')

model.summary()


# ## EarlyStopping

# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',patience=1)
validation_generator = TimeseriesGenerator(scaled_test,scaled_test, 
                                           length=length, batch_size=batch_size)

model.fit_generator(generator,epochs=10,
                    validation_data=validation_generator,
                   callbacks=[early_stop])


# In[ ]:


model.history.history.keys()

losses = pd.DataFrame(model.history.history)
losses.plot()


# ## Evaluate on Test Data

# In[ ]:


first_eval_batch = scaled_train[-length:]
first_eval_batch


# In[ ]:


first_eval_batch = first_eval_batch.reshape((1, length, scaled_train.shape[1]))
model.predict(first_eval_batch)


# In[ ]:


scaled_test[0]


# In[ ]:


n_features = scaled_train.shape[1]
test_predictions = []

first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(len(test)):
    
    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]
    
    # store prediction
    test_predictions.append(current_pred) 
    
    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
    


# ## Inverse Transformations and Compare

# In[ ]:


true_predictions = scaler.inverse_transform(test_predictions)

true_predictions = pd.DataFrame(data=true_predictions,columns=test.columns)
true_predictions


# <a class="anchor" id="3."></a> 
# # 3.Use Facebook's Prophet Library for forecasting

# In[ ]:


import pandas as pd
from fbprophet import Prophet


# ## Load Data
# 
# The input to Prophet is always a dataframe with two columns: ds and y. The ds (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. The y column must be numeric, and represents the measurement we wish to forecast.

# In[ ]:


df = pd.read_csv('../input/for-simple-exercises-time-series-forecasting/Miles_Traveled.csv')
df.head()


# In[ ]:


df.columns = ['ds','y']
df['ds'] = pd.to_datetime(df['ds'])
df.info()


# In[ ]:


pd.plotting.register_matplotlib_converters()

try:
    df.plot(x='ds',y='y',figsize=(18,6))
except TypeError as e:
    figure_or_exception = str("TypeError: " + str(e))
else:
    figure_or_exception = df.set_index('ds').y.plot().get_figure()


# In[ ]:


print('len(df)',len(df))
print('len(df) - 12 = ',len(df) - 12)


# In[ ]:


train = df.iloc[:576]
test = df.iloc[576:]


# ## Create and Fit Model

# In[ ]:


# This is fitting on all the data (no train test split in this example)
m = Prophet()
m.fit(train)


# ## Forecasting
# 
# **NOTE: Prophet by default is for daily data. You need to pass a frequency for sub-daily or monthly data. Info: https://facebook.github.io/prophet/docs/non-daily_data.html**

# In[ ]:


future = m.make_future_dataframe(periods=12,freq='MS')
forecast = m.predict(future)


# In[ ]:


forecast.tail()


# In[ ]:


test.tail()


# In[ ]:


forecast.columns


# In[ ]:


forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12)


# ### Plotting Forecast
# 
# We can use Prophet's own built in plotting tools

# In[ ]:


m.plot(forecast);


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
m.plot(forecast)
plt.xlim(pd.to_datetime('2003-01-01'),pd.to_datetime('2007-01-01'))


# In[ ]:


m.plot_components(forecast);


# In[ ]:


from statsmodels.tools.eval_measures import rmse
predictions = forecast.iloc[-12:]['yhat']
predictions


# In[ ]:


test['y']


# In[ ]:


rmse(predictions,test['y'])


# In[ ]:


test.mean()


# ## Prophet Diagnostics
# 
# Prophet includes functionality for time series cross validation to measure forecast error using historical data. This is done by selecting cutoff points in the history, and for each of them fitting the model using data only up to that cutoff point. We can then compare the forecasted values to the actual values.

# In[ ]:


from fbprophet.diagnostics import cross_validation,performance_metrics
from fbprophet.plot import plot_cross_validation_metric

len(df)
len(df)/12

# Initial 5 years training period
initial = 5 * 365
initial = str(initial) + ' days'
# Fold every 5 years
period = 5 * 365
period = str(period) + ' days'
# Forecast 1 year into the future
horizon = 365
horizon = str(horizon) + ' days'

df_cv = cross_validation(m, initial=initial, period=period, horizon = horizon)

df_cv.head()


# In[ ]:


df_cv.tail()


# In[ ]:


performance_metrics(df_cv)


# In[ ]:


plot_cross_validation_metric(df_cv, metric='rmse');


# In[ ]:


plot_cross_validation_metric(df_cv, metric='mape');


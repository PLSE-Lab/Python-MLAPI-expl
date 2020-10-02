#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style('whitegrid')


# # Preprocessing

# In[ ]:


df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',index_col='ObservationDate',parse_dates=True)
covidIndia = df[df['Country/Region'] == 'India']
covidIndia.drop(['SNo','Last Update','Province/State'],axis=1,inplace = True)
covidIndia.head()


# In[ ]:


covidIndia.tail()


# In[ ]:


print("Shape of Data is ==> ",covidIndia.shape)


# In[ ]:


covidIndia.index[:10]


# In[ ]:


covidIndia.index.freq = 'D'
covidIndia.index[:10]


# > I have checked Index to set frequency of Time Series data. It is important to set frequency of Time Series data to avoid unnecessary errors in statsmodels.

# # ETS Decomposition
# 
# The <a href='https://en.wikipedia.org/wiki/Decomposition_of_time_series'>decomposition</a> of a time series attempts to isolate individual components such as <em>error</em>, <em>trend</em>, and <em>seasonality</em> (ETS).

# In[ ]:


from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(covidIndia['Confirmed'], model='mul')

from pylab import rcParams
rcParams['figure.figsize'] = 12,8
result.plot();


# > I will be using Multiplicative model because trend is not linear , It is exponential. Also i will not be using seasonality because it looks like there is no seasonal pattern.

# # Holt Winters Method
# 
# ### <font color=blue>Simple Exponential Smoothing / Simple Moving Average</font>
# This is the simplest to forecast. $\hat{y}$ is equal to the most recent value in the dataset, and the forecast plot is simply a horizontal line extending from the most recent value.
# ### <font color=blue>Double Exponential Smoothing / Holt's Method</font>
# This model takes trend into account. Here the forecast plot is still a straight line extending from the most recent value, but it has slope.
# ### <font color=blue>Triple Exponential Smoothing / Holt-Winters Method</font>
# This model has (so far) the "best" looking forecast plot, as it takes seasonality into account. When we expect regular fluctuations in the future, this model attempts to map the seasonal behavior.

# In[ ]:


train_data = covidIndia.iloc[:78]
test_data = covidIndia.iloc[78:]

from statsmodels.tsa.holtwinters import ExponentialSmoothing

fitted_model = ExponentialSmoothing(train_data['Confirmed'],trend='mul').fit()

test_predictions = fitted_model.forecast(15).rename('Confirmed Forecast')


# In[ ]:


print("Prediction ==> \n",test_predictions[:5])
print("\n","Actual Data ==> \n",test_data[:5]['Confirmed'])


# In[ ]:


fig = plt.figure(dpi = 120)
ax = plt.axes()
ax.set(xlabel = 'Date',ylabel = 'Count of Cases',title = 'Comparision : Test VS Prediction')
train_data['Confirmed'].plot(legend=True,label='TRAIN',lw = 2)
test_data['Confirmed'].plot(legend=True,label='TEST',figsize=(8,4),lw = 2)
test_predictions.plot(legend=True,label='PREDICTION',lw = 2);


# In[ ]:


fig = plt.figure(dpi = 120)
ax = plt.axes()
ax.set(xlabel = 'Date',ylabel = 'Count of Cases',title = 'Comparision : Test VS Prediction (Zoon In)')
test_data['Confirmed'].plot(legend=True,label='TEST DATA',figsize=(8,4),lw = 2)
test_predictions.plot(legend=True,label='PREDICTION',xlim=['2020-04-23','2020-04-27'],lw = 2);


# In[ ]:


from sklearn.metrics import mean_squared_error,mean_absolute_error

print("MAE ==> ",mean_absolute_error(test_data['Confirmed'],test_predictions))
print("MSE ==> ",mean_squared_error(test_data['Confirmed'],test_predictions))
print("RMSE ==> ",np.sqrt(mean_squared_error(test_data['Confirmed'],test_predictions)))


# In[ ]:


test_data.describe()['Confirmed']['std']


# ## Forecasting using Holt Method

# In[ ]:


final_model = ExponentialSmoothing(train_data['Confirmed'],trend='mul').fit()
forecast_predictions = final_model.forecast(30)


# In[ ]:


fig = plt.figure(dpi = 120)
ax = plt.axes()
ax.set(xlabel = 'Date',ylabel = 'Count of Cases',title = 'Forecast : (May 1, 2020) to (May 15, 2020)')
covidIndia['Confirmed'].plot(figsize=(8,4),lw = 2,legend = True,label = 'Actual Confirmed')
forecast_predictions.plot(lw=2,legend = True,label = 'Forecast Confirmed',xlim = ['2020-04-20','2020-05-15']);


# > Let me explain the things now :-
# 
# 1. I have splitted the data in train and test set, Size of test is 15 because i wanted to forecast for next 15 days.
# 2. We can clearly see , How good this simple model is explaining the trend and able to forecast result.
# 3. Looking at MSE, you might think the model is doing worst, But think again!! One should not judge model performance by just looking at MSE,RMSE. Compare these values with given data. If it is close to our data then model is doing good. In this case we can see there are little difference between RMSE and STD of actual data. So this simple is not doing that much bad :)

# # RNN
# 
# ![](https://miro.medium.com/fit/c/1838/551/1*HgAY1lLMYSANqtgTgwWeXQ.png)
# 
# A recurrent neural network (RNN) is a class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence. This allows it to exhibit temporal dynamic behavior. Derived from feedforward neural networks, RNNs can use their internal state (memory) to process variable length sequences of inputs.This makes them applicable to tasks such as unsegmented, connected handwriting recognition or speech recognition.
# 
# You can read more on RNN [here.](https://en.wikipedia.org/wiki/Recurrent_neural_network)
# 
# ### RNN for time series =>
# 
# ![](https://upload.wikimedia.org/wikipedia/commons/3/3b/The_LSTM_cell.png)
# 
# A powerful type of neural network designed to handle sequence dependence is called recurrent neural networks. The Long Short-Term Memory network or LSTM network is a type of recurrent neural network used in deep learning because very large architectures can be successfully trained.
# 
# You can read more on LSTM RNN [here](https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)
# 

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

train = pd.DataFrame(covidIndia.iloc[:78,1])
test = pd.DataFrame(covidIndia.iloc[78:,1])

scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)


# In[ ]:


print("Scaled Train Set ==> \n", scaled_train[:5],"\n")
print("Scaled Test Set==> \n", scaled_test[:5])


# ## Time Series Generator
# 
# This class takes in a sequence of data-points gathered at
# equal intervals, along with time series parameters such as
# stride, length of history, etc., to produce batches for
# training/validation.
# 
# #### Arguments
#     data: Indexable generator (such as list or Numpy array)
#         containing consecutive data points (timesteps).
#         The data should be at 2D, and axis 0 is expected
#         to be the time dimension.
#     
#     targets: Targets corresponding to timesteps in `data`.
#         It should have same length as `data`.
#     
#     length: Length of the output sequences (in number of timesteps).
#     
#     batch_size: Number of timeseries samples in each batch
#         (except maybe the last one).

# In[ ]:


from keras.preprocessing.sequence import TimeseriesGenerator
n_input = 15
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)


# ## RNN LSTM Model

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# define model
model = Sequential()
model.add(LSTM(150, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# In[ ]:


model.summary()


# In[ ]:


# fit model
model.fit_generator(generator,epochs=25)


# In[ ]:


loss_per_epoch = model.history.history['loss']
fig = plt.figure(dpi = 120,figsize = (8,4))
ax = plt.axes()
ax.set(xlabel = 'Number of Epochs',ylabel = 'MSE Loss',title = 'Loss Curve of RNN LSTM')
plt.plot(range(len(loss_per_epoch)),loss_per_epoch,lw = 2);


# > We can see how good our model is able to converge !!

# ## Evaluate on Test Data
# 
# This part is little bit trickt to understand. RNN LSTM actually uses past data to predict next one data point. Here our input length is 15, So in each iteration model will use past 15 data to predict next one.

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


# In[ ]:


true_predictions = scaler.inverse_transform(test_predictions)
test['Predictions'] = true_predictions
test.head()


# In[ ]:


fig = plt.figure(dpi = 120)
ax=plt.axes()
test.plot(legend=True,figsize=(14,6),lw = 2,ax=ax)
plt.xlabel('Date')
plt.ylabel('Count of Cases')
plt.title('Comparision B/W Test and Prediction')
plt.show();


# > We can see our model is caapturing trend, But India is beating Corona !! :)

# ## Forecasting
# 
# To forecast, we need to refit model again with whole data points.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

train = pd.DataFrame(covidIndia.iloc[:,1])


scaler.fit(train)
scaled_train = scaler.transform(train)

n_input = 15
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

# define model
model = Sequential()
model.add(LSTM(150, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit_generator(generator,epochs=25)


# In[ ]:


loss_per_epoch = model.history.history['loss']
fig = plt.figure(dpi = 120,figsize = (8,4))
ax = plt.axes()
ax.set(xlabel = 'Number of Epochs',ylabel = 'MSE Loss',title = 'Loss Curve of RNN LSTM')
plt.plot(range(len(loss_per_epoch)),loss_per_epoch,lw = 2);


# In[ ]:


forecast = []

first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(15):
    current_pred = model.predict(current_batch)[0]
    forecast.append(current_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

forecast= scaler.inverse_transform(forecast)


# In[ ]:


forecast = pd.DataFrame({'Forecast':forecast.flatten()})
forecast.index = np.arange('2020-05-01',15,dtype='datetime64[D]')
forecast.head()


# In[ ]:


fig = plt.figure(dpi=120,figsize = (14,6))
ax = plt.axes()
ax.set(xlabel = 'Date',ylabel = 'Count of Cases (Lacs)',title = 'Forecast : (May 1, 2020) to (May 15, 2020)')
forecast.plot(label = 'Forecast',ax=ax,color='red',lw=2);


# > So , Here is forecast for 15 days using RNN.

# # Facebook Prophet
# 
# ![](https://www.kdnuggets.com/wp-content/uploads/prophet-facebook.jpg)
# 
# Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.
# 
# ## IMPORTANT NOTE ONE:
# 
# **You should really read the papaer for Prophet! It is relatively straightforward and has a lot of insight on their techniques on how Prophet works internally!**
# 
# Link to paper: https://peerj.com/preprints/3190.pdf

# ## Preprocessing
# 
# Prophet needs a specific format at input. So we need little bit preprocessing here.

# In[ ]:


df = pd.DataFrame(covidIndia.iloc[:,1])
df.reset_index(inplace = True)
df.head()


# In[ ]:


df.columns = ['ds','y']
df.head()


# In[ ]:


fig = plt.figure(dpi = 120)
axes = plt.axes()
axes.set(xlabel = 'Date',ylabel = 'Count of Cases',title = 'Trend')
df.plot(x='ds',y='y',figsize=(8,4),lw=2,color = 'blue',ax=axes);


# ## Model Validation

# In[ ]:


train = df.iloc[:78]
test = df.iloc[78:]


# In[ ]:


from fbprophet import Prophet
m = Prophet()
m.fit(train)
future = m.make_future_dataframe(periods=15)
forecast = m.predict(future)


# In[ ]:


forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5)


# In[ ]:


test.tail(5)


# In[ ]:


fig = plt.figure(dpi = 120)
ax = plt.axes()
ax.set(xlabel = 'Date',ylabel = 'Count of Cases',title = 'Comparision B/W Test & Prediction')
forecast.plot(x='ds',y='yhat',label='Predictions',legend=True,figsize=(8,4),ax=ax,lw=2)
test.plot(x='ds',y='y',label='True Miles',legend=True,ax=ax,xlim=('2020-04-17','2020-05-01'),lw=2);


# > Well according to Prophet prediction, Number of Cases should have been decreased as compared to real data!! But our model is doing good.

# In[ ]:


from statsmodels.tools.eval_measures import rmse
predictions = forecast.iloc[-15:]['yhat']
print("RMSE ==> ",rmse(predictions,test['y']))


# In[ ]:


print("Test Mean ==> ",test.mean())


# ## Forecast
# 
# Let us forecast for May 1, 2020 to May 15, 2020. Note we have to fit again on whole data...

# In[ ]:


from fbprophet import Prophet
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=15)
forecast = m.predict(future)


# In[ ]:


df.tail()


# In[ ]:


forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5)


# In[ ]:


fig = plt.figure(dpi = 120 )
axes = plt.axes()
m.plot(forecast, figsize = (8,4),ax=axes)
plt.xlabel('Date')
plt.ylabel('Count of Cases')
plt.title('Forecast')
plt.xticks(rotation = 90);


# ### How good Prophet is doing, Just see black dots are actual confirmed cases while line are prediction with uncertainity...

# In[ ]:


plt.figure(dpi = 120)
axes = plt.axes()
m.plot(forecast,ax=axes,figsize = (8,4))
start = pd.to_datetime(['2020-04-25'])
end = pd.to_datetime(['2020-05-15'])
plt.xlabel('Date')
plt.ylabel('Count of Cases')
plt.title('Forecast : (May 1, 2020) - (May 15, 2020)')
plt.xlim(start,end)
plt.ylim(20000,60000)
plt.xticks(rotation = 90);


# ### Now we have forecast for May 1, 2020 to May 15, 2020.
# 
# ## Now i will end up my notebook here , Thank you for reading!! I hope you guys have learned something.
# 
# ## Please do upvote, if you like !!

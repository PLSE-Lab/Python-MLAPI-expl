#!/usr/bin/env python
# coding: utf-8

# # Particle levels prediction using stateful LSTM networks
# 
# Long Short Term Memory networks (LSTM) are a type of Recurrent Neural Networks (RNN) that were specially designed to detect long-term dependencies in the data, making the training usually succesful for different time series problems. Finding these dependencies usually imply that they are able to find different trends in the data and predict which the following outcomes will be. To do so, the LSTM's (and in general, RNN networks) output will depend not only on its input, but also on its current state. To ensure that this state finds the long term dependencies (across training batches) we need to implement a stateful LSTM.
# 
# These properties seem specially important for predicting different pollution levels. This task, which is basically time series forecasting, is a good fit for LSTM and they are probably able to find different trends to predict from one day to the other. Using LSTMs will make it possible not to only use the recent training data, but also explore how different long-term trends may affect the forecast. Therefore, the main dataset used in this kernel will be the historical data of air pollution levels in Madrid, which has hourly data for the period 2001-2018.

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed
from keras.optimizers import RMSprop

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# Also, this notebook will not enter into the theorical details of how an LSTM works; it will instead focus on the implementation details of the network with Keras. If you are not familiar with RNN/LSTM architectures or want a crash course on how they work, these are the typical resources recommended to get some basic insight:
# - [*Understanding LSTM Networks*](https://colah.github.io/posts/2015-08-Understanding-LSTMs/), in Colah's Blog.
# - [*The Unreasonable Effectiveness of Recurrent Neural Networks*](http://karpathy.github.io/2015/05/21/rnn-effectiveness/), by Andrej Karpathy.

# ## Initial model: time-series forecasting
# 
# ### Reading the data
# 
# For this task, we will use the records of a single station, `28079016`, which is one of the few that were active the whole 18 years period we are using to tune the model. However, not every particle was measured during 18 years, so we have to be careful when choosing which one will be used for the task.

# In[ ]:


with pd.HDFStore('../input/air-quality-madrid/madrid.h5') as data:
    df = data['28079016']
    
df = df.sort_index()
df.info()


# As we can see in the `missingno` plot, `CO`, `NO_2` and `O_3` are the most promising candidates, since there is a lot of data available.

# In[ ]:


msno.matrix(df, freq='Y')


# A quick check on the interpolated series for each particle reveals that all of them are quite similar: there is a noticeable yearly seasonality in all of them, and only the `CO` has a clear tendency to lower. Since `NO_2` is the one where the least amount of data had to be interpolated, let's stick to that one.

# In[ ]:


fig, ax = plt.subplots(figsize=(20, 5))

candidates = df[['CO', 'NO_2', 'O_3']] 

candidates /= candidates.max(axis=0)

(candidates.interpolate(method='time')
           .rolling(window=24*15).mean()
           .plot(ax=ax))


# ### Data preparation
# 
# To prepare the data for the task, we will apply several steps to it. To smooth the variation in the curve to be predicted and be able to properly reconstruct it, we are going to apply the natural logarithm to the data. Also, our target is to input 24 timesteps to the network (a complete day, containing 24 hourly records) and make the network predict the next 24 timesteps (the next day). To do so, we want to pivot the next 48 hours for each sample into columns. We will apply this to the whole dataset and after it select the training examples with a 50% overlap: we will make a prediction not only at midnight but also at noon, which will force the data to be more general and generate twice as many examples.

# In[ ]:


def pivot_with_offset(series, offset):
    pivot = pd.DataFrame(index=df.index)

    for t in range(offset * 2):
        pivot['t_{}'.format(t)] = series.shift(-t)

    pivot = pivot.dropna(how='any')
    return pivot


offset = 24

series = (df.NO_2.interpolate(method='time')
                 .pipe(pivot_with_offset, offset)
                 .apply(np.log, axis=1)
                 .replace(-np.inf))

# Get only timestamps at 00:00 and 12:00
series = series[(series.index.hour % 12) == 0]

# Make it a multiple of the chosen batch_size
if series.shape[0] % 32 != 0:
    series = series.iloc[:-(series.shape[0]%32)]


# The next step is to split the series into features and labels: the first 24 hours of each row will be used to predict (`X`) and the next 24 hours will be the levels that we aim to predict (`y`). Also we need to split before starting to do more data processing into train and test data: 20% of the data will not be used for learning. Instead, it will be used to check that the model is able to generalize to new data properly. Take into account that these are time-stamped data and we are training a stateful network: make sure that the data is kept in its original order!

# In[ ]:


test_ratio = 0.2

split_point = int(series.shape[0] * (1 - test_ratio))
split_point -= split_point % 32

np_series = series.values

X_train = series.values[:split_point , :offset]
y_train = series.values[:split_point, offset:]
X_test = series.values[split_point:, :offset]
y_test = series.values[split_point:, offset:]


# Also, it is important to notice that Keras requires a certain shape for the input data: a three dimensional input array with the shape (*batch-size*, *timesteps*, *features*). If we are going predict 24 hours based solely on the previous 24 hours records, we have 24 timesteps and 1 single feature (the `NO_2` levels). We also want the data to be scaled in the interval $[0, 1]$ - most machine learning techniques perform better with normalized features. However, make sure to scale the data only to the training data (you technically do not have the test one, that is what you want to predict): doing so is a form of look-ahead bias and we should keep it legit.

# In[ ]:


# Scale only to train data to prevent look-ahead bias
lift = X_train.min()
scale = X_train.max()

def scale_array(arr, lift, scale):
    return (arr - lift) / scale

X_train = np.expand_dims(scale_array(X_train, lift, scale), axis=2)
y_train = np.expand_dims(scale_array(y_train, lift, scale), axis=2)
X_test = np.expand_dims(scale_array(X_test, lift, scale), axis=2)
y_test = np.expand_dims(scale_array(y_test, lift, scale), axis=2)


# ### Network architecture
# 
# We will use Keras to implement the network, and our best option is to write a function to do it (since we might be doing this quite a few times). Building on top of the Sequential model, we want to add a first `LSTM` layer of a certain size, maybe a second `LSTM` layer and a `TimeDistributed` - `Dense` layer to predict each of the 24 target timestamps. Make sure to check that `stateful=True` for the LSTM layers and that the `batch_input_shape` is correctly configured as described previously. The optimizer will be `RMSprop` and the loss function will be a simple mean squared error.

# In[ ]:


def create_lstm(offset, neurons=(2,1), batch_size=32, lr=0.005, n_features=1):
    model = Sequential()
    
    # First LSTM layer
    model.add(LSTM(
        neurons[0], return_sequences=True, stateful=True, 
        batch_input_shape=(batch_size, offset, n_features))
    )
    
    
    # Second LSTM layer if defined
    if neurons[1]:
        model.add(LSTM(
            neurons[1], return_sequences=True, stateful=True, 
            batch_input_shape=(batch_size, offset, n_features))
        )
    
    # TimeDistributed layer to generate all the timesteps
    model.add(TimeDistributed(Dense(1)))
    
    optimizer = RMSprop(lr=lr)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    
    return model


# Since the model is stateful, we want to reset the state of the network after each epoch. This does not erase the weights learned, instead it just resets the state to be fed via the context neurons. To do so, we can define a function to perform this workaround.

# In[ ]:


def train_model(model, X_train, y_train, batch_size=32, epochs=20):
    mse = list()

    for i in range(epochs):
        if i % 1 == 0:
            print('Epoch {:02d}/{}...'.format(i + 1, epochs), end=' ')

        log = model.fit(
            X_train, y_train, 
            epochs=1, batch_size=32, 
            verbose=0, shuffle=False
        )
    
        mse.append(log.history['loss'][-1])
        print('loss: {:.4f}'.format(mse[-1]))
    
        model.reset_states()
        
    return model, mse


# ### Evaluation of the model
# 
# A function to automatically compute the mean squared error to the data and reset the state of the machine may come in handy when looking for a good architecture or hyperparameters.

# In[ ]:


def validate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    model.reset_states()
    return np.mean((y_test - preds) ** 2)


# After my experimentation, I came to the realization that a few number of neurons and favor a lot of epochs rather than higher learning rates is the best way to approach the problem. This is still a craft, so the best bet is still to just imitate other architectures that have been proved to work or to perform a grid search on the hyperparameters (in this case, the number of hidden neurons, the learning rate and the training epochs).

# In[ ]:


model = create_lstm(offset)
model, _ = train_model(model, X_train, y_train)

model.fit(
    X_train, y_train, 
    epochs=1, batch_size=32, 
    verbose=1, shuffle=False
)
model.reset_states()

preds = model.predict(X_test)


# A quick validation of the model reveals that the MSE is slightly larger than the one for the training data. Still, the model seems to generalize good enough, so let us check how does the network predict future trends in the data.

# In[ ]:


'MSE: {:.5f}'.format(validate_model(model, X_test, y_test))


# Doing a quick plot of a slice of two weeks in the testing data and plotting several predictions show that, even though the network shows potential, the model can be still improved: the main problem is that the data seems to be failing in scale and the predictions are quite linear. However, it is possible to see that some peaks are correctly predicted.

# In[ ]:


fig, ax = plt.subplots(figsize=(20, 5))

start = X_test.shape[0] - 42
interval = X_test[start+3:start+33:2, :, 0].reshape(-1, 1)

truth, = plt.plot(np.arange(24*15), interval, alpha=0.6)

old_preds = list()

for point in range(1, 15, 2):
    prediction = np.squeeze(preds[start + point*2])
    pred, = plt.plot(point * offset + np.arange(offset) - 12, prediction,
                     linestyle='--', color='red')
    old_preds.append(prediction)

plt.legend(
    [truth, pred],
    ['Observation', 'Prediction']
)
ax.set_ylim([-.1, 1.1])
ax.set_xticks(12 + np.arange(15) * offset)
_ = ax.set_xticklabels([])


# ## Upping the ante: using traffic records as a feature
# 
# Both of the problems described above may be tried to solve by using more data. Since it is not possible to use more years (and that may not be what we are looking for), we can also feed the network with auxiliary information as different features. Without expert knowledge, the levels of nitrogen dioxide are probably related with two other factors: the weather and polluting emissions. Finding hourly weather records of Madrid is hard or expensive, but if possible the wind and rain levels for each observation may give some great insight to the network. Regarding emissions, it is actually possible to find traffic intensity records of Madrid each 15 minutes in the Open Data website. As a proof of concept, we can try to feed this data to the model and see the impact on the forecast.
# 
# **Note**: traffic records are heavy files, so to simplify the task `traffic.csv` is a homebrewed file contains the sum of the traffic load all stations have recorded during a single hour. I am not sure if this measure is actually valid to other tasks or simply misleading, which is the reason why these data remains private.

# In[ ]:


traffic = (pd.read_csv('../input/traffic-in-madrid/traffic.csv', parse_dates=['date'])
             .rename({'intensidad': 'traffic'}, axis=1)
             .set_index('date'))

_ = (traffic.rolling(window=24*7).mean()
            .plot(figsize=(20,5)))


# It is possible to see that there are no traffic records before 2013. This is the first tradeoff that we are facing with this modification: instead of eighteen years of data, we will just have 5 of them available.

# In[ ]:


df = (df.reset_index()
        .merge(traffic.reset_index(), on='date')
        .set_index('date'))


# We can also help the network to expect a change of trend using *one-hot* columns that mark special sections. That way, the network may find different ways to predict depending on the features. Let us add a column called `weekend`, that will be 1 if the observation happened on Saturday or Sunday, and else it will be zero. We can also trivially foresee when the prediction will be on a weekend, so both the observation and prediction entries can be available for the network prediction

# In[ ]:


df['weekend'] = ((df.index.weekday == 5) | (df.index.weekday == 6)).astype(np.int32)


# The data preparation is similar to the previous one, simply notice that now the shape we need to take into account that we need four features: `NO_2` levels, traffic intensity and both observation and prediction weekend indicators.

# In[ ]:


air = (df.NO_2.interpolate(method='time')
              .pipe(pivot_with_offset, offset)
              .apply(np.log, axis=1)
              .replace(-np.inf))

traf = pivot_with_offset(df.traffic, offset)
week = pivot_with_offset(df.weekend, offset)

# Get only timestamps at 00:00 and 12:00
air = air[(air.index.hour % 12) == 0]
traf = traf[(traf.index.hour % 12) == 0]
week = week[(week.index.hour % 12) == 0]

# Substitute nans with zeros
air = air.fillna(0)
traf = traf.fillna(0)
week = week.fillna(0)

all_data = np.dstack([air.values, traf.values, week.values, week.shift(-1)])

if all_data.shape[0] % 32 != 0:
    all_data = all_data[:-(all_data.shape[0]%32)]
    
all_data.shape


# Just as before, split the training and testing data and scale them. I was running into problems with the `MinMaxScaler` with this data, so I went for the artisan way to scale it.

# In[ ]:


test_ratio = 0.2

split_point = int(all_data.shape[0] * (1 - test_ratio))
split_point -= split_point % 32

X_train = all_data[:split_point , :offset, :]
y_train = all_data[:split_point, offset:, :1]
X_test = all_data[split_point:, :offset, :]
y_test = all_data[split_point:, offset:, :1]


# In[ ]:


X_train[:,:,0] = scale_array(X_train[:,:,0], lift, scale)
y_train[:,:,0] = scale_array(y_train[:,:,0], lift, scale)
X_test[:,:,0] = scale_array(X_test[:,:,0], lift, scale)
y_test[:,:,0] = scale_array(y_test[:,:,0], lift, scale)

# Scale traffic differently to prevent errors by assuming columns
lift = X_train[:,:,1].min()
scale = X_train[:,:,1].max()

X_train[:,:,1] = scale_array(X_train[:,:,1], lift, scale)
X_test[:,:,1] = scale_array(X_test[:,:,1], lift, scale)


# Since we lack all the data before 2013, it is a good idea to increase the number of epochs so that the network can be properly trained to fit the data.

# In[ ]:


model = create_lstm(offset, lr=0.025, n_features=4)
model, _ = train_model(model, X_train, y_train, epochs=50)

model.fit(
    X_train, y_train, 
    epochs=1, batch_size=32, 
    verbose=1, shuffle=False
)

preds = model.predict(X_test)
model.reset_states()


# We can see that the training loss is higher than the previous one. For that reason, let's take a look at the predictions on the graph, and see if the trend predictions make more sense in the new model

# In[ ]:


'MSE: {:.5f}'.format(validate_model(model, X_test, y_test))


# Comparing both models, it is not as clear as looking at the MSE which model performs better: it is possible to see that the new model fits better some days, while underperforming the previous model, especially in sudden increase of the particle measures. In the short plot below, the new model seems to provide a little more reliability over the previous one (although the hardest prediction are still failed by both of them).

# In[ ]:


fig, ax = plt.subplots(figsize=(20, 5))

start = X_test.shape[0] - 42
interval = X_test[start+3:start+33:2, :, :]

truth, = plt.plot(np.arange(24*15), interval[:, :, 0].reshape(-1, 1), alpha=0.6)

for point, old_pred in zip(range(2, 15, 2), old_preds):
    prediction = np.squeeze(preds[start + point*2])
    old, = plt.plot(point * offset + np.arange(offset) - 12, old_pred,
                    linestyle='--', color='red', alpha=0.4)
    new, = plt.plot(point * offset + np.arange(offset) - 12, prediction,
                    linestyle='--', color='green')

plt.legend(
    [truth, old, new],
    ['Observation', 'Original prediction', 'New model']
)
    
ax.set_ylim([-.1, 1.1])
ax.set_xticks(12 + np.arange(15) * offset)
_ = ax.set_xticklabels([])


# Also please bear in mind that this experiment should not be directly compared to the previous one: we are not only changing the features but also the amount of historical data. However, as a playing ground the results are interesting enough to acknowledge that more relevant features seem to be more promising than more historical data.

# ## Conclusions and future work
# 
# There is no doubt that the results are promising, but there are still some issues to correct. The most notable one is the difficulty that the model has to predict a sudden change of trends. It does not seem unclear, since the *one-hot* column strategy had limited results. 
# 
# Another possibility is to include not only data like wind or rain levels measured, but also add the forecast: add as a new feature the forecast for the next day. That new feature would ease the learning for the network and should be able to detect a direct relation with the forecast if the data is actually relevant. Although reliable forecast are not common for traffic, wind and rain are actively being predicted. The problem is, once again, the difficulty to find suitable data: a good history of the last 18 years of hourly weather forecasts for the next day in Madrid sounds like an unicorn to me.
# 
# Despite these problems, the model still presents rather interesting results. The way the predictions fit with the general shape of the real day, despite the scale being off some times can be already interesting to predict when the higher levels of pollution are expected to be encountered. Possible interesting experiments include testing other recurrent architectures to check if they actually present better results.

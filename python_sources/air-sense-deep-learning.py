#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip install keras-self-attention


# In[1]:


import glob
import pandas as pd
from math import sqrt
from numpy import array2string
from numpy import concatenate
from matplotlib import pyplot
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU, Flatten
from keras.callbacks import EarlyStopping
from keras.backend import clear_session
# from keras_self_attention import SeqSelfAttention
from keras import regularizers


# In[2]:


df = pd.read_csv('../input/avg-v2/AVG_v2.csv')
df.head()


# In[8]:


ESP_00983468 = df[df.node_id == 'ESP_00983468']
ESP_00983468.head()


# In[9]:


ESP_00983468 = ESP_00983468.drop(columns=['node_id', 'DateTime', 'temperature', 'humidity', 'SO2data', 'COdata', 'O3data','PM10data', 'pressure', 'PM1data', 'sound','NO2data','NOdata','verhicle','pollutant','CO2data', 'AQI'])
data_mt = ESP_00983468.reset_index(drop=True)


# In[10]:


data_mt.head()


# In[87]:


class trainModel(object):
    def __init__(self, n_hours=1, n_time_predicts=1, config_train=None, units=None, activation='tanh'):
        self.n_hours = n_hours
        self.n_time_predicts = n_time_predicts
        self.config_train = config_train
        self.units = units
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.activation = activation

    # convert series to supervised learning
    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def build_model(self, units, train_X, loss='mse', optimizer='SGD'):
        model = Sequential()
        model.add(LSTM(units,input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
        model.add(LSTM(units))
#         model.add(SeqSelfAttention(attention_width=24,
#                         history_only=True,
#                        attention_activation='sigmoid',
#                        name='Attention'))
#         model.add(Flatten())
        model.add(Dense(1, activation = self.activation))
        model.compile(loss=loss, optimizer=optimizer)
        return model

    def save_img_predict_test(self, inv_yhat, inv_y):
        pyplot.plot(inv_yhat, label='predict')
        pyplot.plot(inv_y, label='test')
        pyplot.legend()
        pyplot.show()
        pyplot.close()

    def make_predict(self, model, test_X, n_features = 1):
        yhat = model.predict(test_X)
        test_X = test_X.reshape((test_X.shape[0], self.n_hours*n_features))
        inv_yhat = self.invert_scaling(yhat, test_X, n_features)
        return inv_yhat
    
    def make_actual(self, test_X, test_y, n_features = 1):
        test_X = test_X.reshape((test_X.shape[0], self.n_hours*n_features))
        test_y = test_y.reshape((len(test_y), 1))
        inv_y = self.invert_scaling(test_y, test_X, n_features)
        return inv_y

    def invert_scaling(self, test_y, test_X_reshape, n_features):
        inv_y = concatenate((test_y, test_X_reshape[:, -(n_features - 1):]), axis=1)
        inv_y = self.scaler.inverse_transform(inv_y)
        inv_y = inv_y[:,0]
        return inv_y

    def normalize_data(self, dataset, dropnan=True):
        values = dataset.values
        values = values.astype('float32')
        scaled = self.scaler.fit_transform(values)
        reframed = self.series_to_supervised(scaled, self.n_hours, 1, dropnan)
        values = reframed.values
        return values
    
    def split_train_test(self, values, n_time_predicts):
        n_train_hours = len(values) - n_time_predicts
        train = values[:n_train_hours, :]
        test = values[n_train_hours:, :]
        return train, test

    def split_into_inputs_and_outputs(self, values, n_features = 10):
        n_time_predicts = len(values)
        n_obs = self.n_hours * n_features
        test_X, test_y = values[:, :n_obs], values[:, -n_features]
        test_X = test_X.reshape((n_time_predicts, self.n_hours, n_features))
        return test_X, test_y

    def fit_model(self, model, train_X, train_y, test_X, test_y, config):
        epochs, batch_size, verbose, min_delta, patience, monitor = config
        model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=False, validation_data=(test_X, test_y),
            callbacks = [EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience)])
        return model

    def train_model(self, dataset):
        n_features = len(dataset.columns)
        values = self.normalize_data(dataset)
        train, test = self.split_train_test(values, self.n_time_predicts)
#         train, validate = self.split_train_test(train, self.n_time_predicts)
        train_X, train_y = self.split_into_inputs_and_outputs(train, n_features=n_features)
#         validate_X, validate_y = self.split_into_inputs_and_outputs(validate, n_features=n_features)        
        test_X, test_y = self.split_into_inputs_and_outputs(test, n_features=n_features)
        model = self.build_model(units=self.units, train_X = train_X)
        model = self.fit_model(model, train_X, train_y, test_X, test_y, self.config_train)
#         inv_yhat = self.make_predict(model, test_X, n_features)
#         inv_y = self.make_actual(model, test_X, test_y, n_features)
#         inv_yhat = inv_yhat[:-1]
#         inv_y = inv_y[1:]
#         self.save_img_predict_test(inv_yhat, inv_y)
        return model


# In[88]:


units = 128*2
n_hours = 24
n_time_predicts = 24*7
epochs = 200
batch_size = 512
verbose = 2
min_delta = 1e-10
patience = 30
monitor = 'val_loss'
# actions = ['sigmoid','elu','selu', 'tanh', 'relu', 'hard_sigmoid','linear','exponential'] 
actions = ['linear'] 


# In[89]:


def save_img_predict_test(inv_yhat, inv_y):
    pyplot.plot(inv_yhat, label='forecast')
    pyplot.plot(inv_y, label='test')
    pyplot.legend()
    pyplot.show()
    pyplot.close()
    
def evaluate_model(inv_y, inv_yhat):
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    save_img_predict_test(inv_yhat,inv_y)
    return 'rmse %s  trend_error %s'%(rmse, trend_error(inv_y, inv_yhat))
def sign(x):
    return int(x>0)
def trend_error(inv_y, inv_yhat):
    err = 0
    for i in range(len(inv_y)-1):
        if sign(inv_y[i+1] - inv_y[i]) != sign(inv_yhat[i+1] - inv_yhat[i]):
            err+=1
    return err/(len(inv_y)-1)*100


# In[90]:


split_train = int(len(data_mt)-24*7)
train, test_o = data_mt[0:split_train], data_mt[split_train:]


# In[91]:


config_train = (epochs, batch_size, verbose, min_delta, patience, monitor)
train_model = trainModel(n_hours, n_time_predicts, config_train, units, 'linear')
model = train_model.train_model(train)


# In[ ]:


predict = []
history = []
for i in range(len(test_o)):
    test_tr = data_mt[0:split_train+i]
    test = train_model.normalize_data(test_tr,dropnan=False)
    test_X, test_y = train_model.split_into_inputs_and_outputs(test, n_features=1)
    inv_yhat = train_model.make_predict(model, test_X, 1)[-1]
    predict.append(inv_yhat)
    test_tr = data_mt[0:split_train+i+1]
    history.append(test_tr.values[-1][0])


# In[ ]:


evaluate_model(history[:-2], predict[2:])


# In[ ]:


len(history)


# In[ ]:





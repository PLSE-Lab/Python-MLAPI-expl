#!/usr/bin/env python
# coding: utf-8

# # Models

# Top models in the leadboard right now ( 7 April 2020 )
# out of 15 models
# 1. LightGBM - 12 models ( 80 % )
# 2. NN
#     * Dense (MLP) - 2 models( 13.33 % )
# 3. Catboost - 1 model ( 6.66 % )

# <h1 style='color: red'> Please upvote this notebook if you found it helpful </h1>

# <h1 style='text-align:center'>Neural Networks Models</h1>

# <div style="text-align:center">
# <hr>
#     <h1> Content of this notebook</h1>
#     <h3>1. Models to be evaluated</h3>
#     <h3>2. Plots and scores</h3>
#     <h3>3. Model diagrams</h3>
# <hr>
# </div>

# <div style="text-align:center">
# <hr>
#     <h1> Types of Models for evaluation</h1>
#     <h3>1. CNN + LSTM model with filter cells</h3>
#     <h3>2. Vanilla LSTM model</h3>
#     <h3>3. Vanilla CNN model</h3>
#     <h3>4. CNN model with filter cells</h3>
#     <h3>5. Dense model with filter cells</h3>
#     <h3>6. Encoder - Decoder Dense Model</h3>
# <hr>
# </div>

# In[ ]:


import pandas as pd
import numpy as np
from numpy import mean
from numpy import std

from keras.models import Model
from keras.layers import Input, Dense, LSTM, Conv1D, BatchNormalization, Dropout, multiply, MaxPooling1D, Flatten
from keras.utils.vis_utils import plot_model

# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedKFold
# from sklearn.datasets import make_regression

from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error


# In[ ]:


train = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
submission = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')


# In[ ]:


def measure_mse(actual, predicted):
    return mean_squared_error(actual, predicted)


# In[ ]:


def series_to_supervised(data, n_in = 10, n_out=1):
    df = pd.DataFrame(data)
    cols = list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))

    for i in range(0, n_out):
        cols.append(df.shift(-i))

    agg = pd.concat(cols, axis=1)

    agg.dropna(inplace=True)
    return agg.values


# # CNN + LSTM model

# In[ ]:


def train_model_cnn_lstm(train_x, train_y, n_nodes):
    layer_in = Input(shape=(n_nodes,1))
    layer_regr = Conv1D(filters=500, kernel_size=3, activation='relu', padding='same', kernel_initializer='truncated_normal')(layer_in)
    
    layer_class = LSTM(128, activation='softmax', kernel_initializer='truncated_normal')(layer_regr)
    
    layer_regr = LSTM(128, activation='relu', kernel_initializer='truncated_normal')(layer_regr)
    layer_regr = BatchNormalization()(layer_regr)
    
    layer_regr = multiply([layer_regr, layer_class])
    
    layer_regr = Dropout(0.3)(layer_regr)
    layer_regr = Dense(32, activation='relu', kernel_initializer='truncated_normal')(layer_regr)
    layer_regr = Dropout(0.2)(layer_regr)
    layer_out = Dense(1,)(layer_regr)

    model = Model(inputs=layer_in, outputs=layer_out)
    model.compile(loss='mse', optimizer='adam')
    
    for i in range(300):
        model.fit(train_x, train_y, batch_size=len(train_x), epochs = 1, verbose = 0)
        model.reset_states()
    return model


# # Vanilla LSTM

# In[ ]:


def train_model_lstm(train_x, train_y, n_nodes):
    layer_in = Input(shape=(n_nodes,1))
    layer_regr = LSTM(128, activation='relu', kernel_initializer='truncated_normal', return_sequences=True)(layer_in)
    layer_regr = BatchNormalization()(layer_regr)
    layer_regr = Dropout(0.3)(layer_regr)
    layer_regr = LSTM(n_nodes, activation='relu', kernel_initializer='truncated_normal', return_sequences=True)(layer_regr)
    layer_regr = LSTM(128, activation='relu', kernel_initializer='truncated_normal')(layer_regr)
    layer_regr = Dropout(0.3)(layer_regr)
    layer_regr = Dense(32, activation='relu', kernel_initializer='truncated_normal')(layer_regr)
    layer_regr = Dropout(0.2)(layer_regr)
    layer_out = Dense(1,)(layer_regr)

    model = Model(inputs=layer_in, outputs=layer_out)
    model.compile(loss='mse', optimizer='adam')
    
    for i in range(300):
        model.fit(train_x, train_y, batch_size=len(train_x), epochs = 1, verbose = 0)
        model.reset_states()
    return model


# # Vanilla CNN

# In[ ]:


def train_model_cnn(train_x, train_y, n_nodes):
    layer_in = Input(shape=(n_nodes,1))
    layer_regr = Conv1D(filters=250, kernel_size=3, activation='relu', padding='same', kernel_initializer='truncated_normal')(layer_in)
    layer_regr = BatchNormalization()(layer_regr)
    layer_regr = Conv1D(125, kernel_size=3, activation='relu', kernel_initializer='truncated_normal')(layer_regr)
    layer_regr = MaxPooling1D(pool_size=2)(layer_regr)
    layer_regr = Flatten()(layer_regr)
    layer_regr = Dense(32, activation='relu', kernel_initializer='truncated_normal')(layer_regr)
    layer_regr = Dropout(0.2)(layer_regr)
    layer_out = Dense(1,)(layer_regr)

    model = Model(inputs=layer_in, outputs=layer_out)
    model.compile(loss='mse', optimizer='adam')
    
    for i in range(300):
        model.fit(train_x, train_y, batch_size=len(train_x), epochs = 1, verbose = 0)
        model.reset_states()
    return model


# # CNN model with filter cells

# In[ ]:


def train_model_cnn_filter(train_x, train_y, n_nodes):
    layer_in = Input(shape=(n_nodes,1))
    layer_regr = Conv1D(filters=250, kernel_size=3, activation='relu', padding='same', kernel_initializer='truncated_normal')(layer_in)
    layer_regr = BatchNormalization()(layer_regr)
    layer_class = Conv1D(125, kernel_size=3, activation='softmax', kernel_initializer='truncated_normal')(layer_regr)
    layer_regr = Conv1D(125, kernel_size=3, activation='relu', kernel_initializer='truncated_normal')(layer_regr)
    layer_regr = multiply([layer_regr, layer_class])
    layer_regr = MaxPooling1D(pool_size=2)(layer_regr)
    layer_regr = Flatten()(layer_regr)
    layer_regr = Dense(32, activation='relu', kernel_initializer='truncated_normal')(layer_regr)
    layer_regr = Dropout(0.2)(layer_regr)
    layer_out = Dense(1,)(layer_regr)

    model = Model(inputs=layer_in, outputs=layer_out)
    model.compile(loss='mse', optimizer='adam')
    
    for i in range(300):
        model.fit(train_x, train_y, batch_size=len(train_x), epochs = 1, verbose = 0)
        model.reset_states()
    return model


# # Dense model with filter cells

# In[ ]:


def train_model_dense(train_x, train_y, n_nodes):
    layer_in = Input(shape=(n_nodes,))
    layer_regr = Dense(64, activation='relu', kernel_initializer='truncated_normal')(layer_in)
    layer_regr = BatchNormalization()(layer_regr)
    layer_class = Dense(32, activation='softmax', kernel_initializer='truncated_normal')(layer_regr)
    layer_regr = Dense(32, activation='relu', kernel_initializer='truncated_normal')(layer_regr)
    layer_regr = multiply([layer_regr, layer_class])
    layer_out = Dense(1,)(layer_regr)

    model = Model(inputs=layer_in, outputs=layer_out)
    model.compile(loss='mse', optimizer='adam')
    
    for i in range(300):
        model.fit(train_x, train_y, epochs = 1, verbose = 0)
        model.reset_states()
    return model


# # Encoder -  Decoder Dense model

# In[ ]:


def train_model_dense_enc(train_x, train_y, n_nodes):
    layer_in = Input(shape=(n_nodes,))
    layer_regr = Dense(64, activation='relu', kernel_initializer='truncated_normal')(layer_in)
    layer_regr = BatchNormalization()(layer_regr)
    layer_regr = Dense(n_nodes, activation='relu', kernel_initializer='truncated_normal')(layer_regr)
    layer_regr = Dense(32, activation='relu', kernel_initializer='truncated_normal')(layer_regr)
    layer_out = Dense(1,)(layer_regr)

    model = Model(inputs=layer_in, outputs=layer_out)
    model.compile(loss='mse', optimizer='adam')
    
    for i in range(300):
        model.fit(train_x, train_y, epochs = 1, verbose = 0)
        model.reset_states()
    return model


# In[ ]:


n_nodes = 21


# In[ ]:


train_set = train.iloc[1][6:-200].values
train_set = series_to_supervised(train_set, n_in = n_nodes, n_out=1)
train_x, train_y = train_set[:, :-1], train_set[:, -1]


# In[ ]:


model_dense = train_model_dense(train_x, train_y, n_nodes)
model_dense_enc = train_model_dense_enc(train_x, train_y, n_nodes)

train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))

model_cnn_lstm = train_model_cnn_lstm(train_x, train_y, n_nodes)
model_lstm = train_model_lstm(train_x, train_y, n_nodes)
model_cnn = train_model_cnn(train_x, train_y, n_nodes)
model_cnn_filter = train_model_cnn_filter(train_x, train_y, n_nodes)


# In[ ]:


series = train.iloc[1][6:-200].values
actual_set = series_to_supervised(series, n_in = (n_nodes-1), n_out=1)

pred_d = model_dense.predict(actual_set)
pred_d_enc = model_dense_enc.predict(actual_set)

actual_set = actual_set.reshape((actual_set.shape[0], actual_set.shape[1], 1))

pred_cnn_lstm= model_cnn_lstm.predict(actual_set)
pred_lstm = model_lstm.predict(actual_set)
pred_cnn = model_cnn.predict(actual_set)
pred_cnn_f = model_cnn_filter.predict(actual_set)


# In[ ]:


test = train.iloc[1][-200:].values
test_set = series_to_supervised(test, n_in = (n_nodes-1), n_out=1)

test_d = model_dense.predict(test_set)
test_d_enc = model_dense_enc.predict(test_set)

test_set = test_set.reshape((test_set.shape[0], test_set.shape[1], 1))

test_cnn_lstm= model_cnn_lstm.predict(test_set)
test_lstm = model_lstm.predict(test_set)
test_cnn = model_cnn.predict(test_set)
test_cnn_f = model_cnn_filter.predict(test_set)

print('cnn + lstm score: ', measure_mse(test[20:], test_cnn_lstm))
print('lstm score: ', measure_mse(test[20:], test_lstm))
print('cnn score: ', measure_mse(test[20:], test_cnn))
print('cnn with filter cells score: ', measure_mse(test[20:], test_cnn_f))
print('dense with filter cells score: ', measure_mse(test[20:], test_d))
print('dens autoencoder score: ', measure_mse(test[20:], test_d_enc))


# In[ ]:


series = train.iloc[1][6:].values

fig, ax = plt.subplots(6, figsize=(15,30))

ax[0].plot( series, 'tab:red')
ax[0].plot( np.vstack((np.array([0 for x in range(n_nodes)]).reshape(-1,1),pred_cnn_lstm.reshape(-1,1), test_cnn_lstm.reshape(-1,1))), 'tab:green')
ax[0].axvspan(len(series)-200, len(series), color='red', alpha=0.2)
ax[0].set_title('CNN + LSTM plot')
ax[0].set(xlabel='days', ylabel='sales')

ax[1].plot( series, 'tab:red')
ax[1].plot( np.vstack((np.array([0 for x in range(n_nodes)]).reshape(-1,1),pred_lstm.reshape(-1,1), test_lstm.reshape(-1,1))), 'tab:green')
ax[1].axvspan(len(series)-200, len(series), color='red', alpha=0.2)
ax[1].set_title('LSTM plot')
ax[1].set(xlabel='days', ylabel='sales')

ax[2].plot( series, 'tab:red')
ax[2].plot( np.vstack((np.array([0 for x in range(n_nodes)]).reshape(-1,1),pred_cnn.reshape(-1,1), test_cnn.reshape(-1,1))), 'tab:green')
ax[2].axvspan(len(series)-200, len(series), color='red', alpha=0.2)
ax[2].set_title('CNN plot')
ax[2].set(xlabel='days', ylabel='sales')

ax[3].plot( series, 'tab:red')
ax[3].plot( np.vstack((np.array([0 for x in range(n_nodes)]).reshape(-1,1),pred_cnn_f.reshape(-1,1), test_cnn_f.reshape(-1,1))), 'tab:green')
ax[3].axvspan(len(series)-200, len(series), color='red', alpha=0.2)
ax[3].set_title('CNN with filter cells plot')
ax[3].set(xlabel='days', ylabel='sales')

ax[4].plot( series, 'tab:red')
ax[4].plot( np.vstack((np.array([0 for x in range(n_nodes)]).reshape(-1,1),pred_d.reshape(-1,1), test_d.reshape(-1,1))), 'tab:green')
ax[4].axvspan(len(series)-200, len(series), color='red', alpha=0.2)
ax[4].set_title('Dense with filter cells plot')
ax[4].set(xlabel='days', ylabel='sales')

ax[5].plot( series, 'tab:red')
ax[5].plot( np.vstack((np.array([0 for x in range(n_nodes)]).reshape(-1,1),pred_d_enc.reshape(-1,1), test_d_enc.reshape(-1,1))), 'tab:green')
ax[5].axvspan(len(series)-200, len(series), color='red', alpha=0.2)
ax[5].set_title('Dense encoder - decoder plot')
ax[5].set(xlabel='days', ylabel='sales')

fig.show()


# In[ ]:


print('Dense model with filter cells')
plot_model(model_dense, show_shapes=True, show_layer_names=True)


# In[ ]:


print('Dense model autoencoder')
plot_model(model_dense_enc, show_shapes=True, show_layer_names=True)


# In[ ]:


print('CNN + LSTM model')
plot_model(model_cnn_lstm, show_shapes=True, show_layer_names=True)


# In[ ]:


print('CNN filter model')
plot_model(model_cnn_filter, show_shapes=True, show_layer_names=True)


# In[ ]:


print('CNN model')
plot_model(model_cnn, show_shapes=True, show_layer_names=True)


# In[ ]:


print('LSTM model')
plot_model(model_lstm, show_shapes=True, show_layer_names=True)


# In[ ]:





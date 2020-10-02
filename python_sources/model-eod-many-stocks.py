#!/usr/bin/env python
# coding: utf-8

# # 20200218: Use stock data from many stocks to train RNN model - Regression
# Using data of many stocks, The hope is to extract common patterns of stock movement

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from IPython.display import display
import pkg_resources 

pkg_resources.get_distribution('pandas').version


# ## Import and look at data
# * PSA and prices have the same content. Use PSA hereafter
# * Not all symbols have the same time lenght of data, but most have data from 2010 to 2016. 
# * The fundamentals has at most 4 data points for any company and do not seem useful. Ignore.
# * data in securities are not useful for us. 

# In[ ]:


PSA = pd.read_csv('../input/nyse/prices-split-adjusted.csv')  
PSA.head()


# ## Feature engineering
# * How long is the look back? for intraday data, use 6 month (120 trading days) 
# * All features should be normalized so they are nondimensional and the fitted model can be applied to any stock, but not necessarily normalized by min and max
# * Normalize each sequence with information within this sequence
#     - OHLC devided by some value from each sequence (no more minus and then normalize with prior days value)
#     - distance between OHLC.mean and EMA20 (or EMA5) normalzied by EMA20 (or EMA5)
#     - volume normalized by some value from each sequence
#     - other technical indicators 

# In[ ]:


# use volume weighted averaged OHLC.mean to represent market average
# comparison with SPY500 shows that market_average calculated this way is representative of overall market
tickers_with_all_dates = (PSA
                          .groupby('symbol')
                          .size()
                          .loc[lambda s: s.values==s.values.max()]
                          .index
                          .to_list())
tickers_with_invalid_values = (PSA
                               .loc[PSA[['open','close','high','low','volume']]
                               .le(0.)
                               .any(axis=1), 'symbol']
                               .unique()
                               .tolist())
tickers_with_all_dates = list(set(tickers_with_all_dates).difference(set(tickers_with_invalid_values)))
print(f"there are {len(tickers_with_all_dates)} tickers with all dates")


# In[ ]:


market = (PSA
          .loc[PSA['symbol'].isin(tickers_with_all_dates)]
          .assign(**{'average': lambda df: df.loc[:,['open','high','low','close']].mean(axis=1), 
                     'price x volume': lambda df: df['average']*df['volume']})
          .groupby('date')
          .agg(**{'price x volume sum': pd.NamedAgg(column='price x volume', aggfunc=np.sum), 
                  'volume sum': pd.NamedAgg(column='volume', aggfunc=np.sum)})
          .assign(**{'market_average': lambda df: df['price x volume sum']/df['volume sum']})
          .sort_index(ascending=True))
# plt.plot(market['market_average'])


# In[ ]:


lookback = 120
lookahead = 5
test_size = 0.2
X_train = []
X_test = []
y_train = []
y_test = []
features = ['open', 'close', 'low', 'high', 'volume', 'dist_EMA20', 'dist_EMA5', 'market']
for k, ticker in enumerate(tickers_with_all_dates):
    # data of a ticker
    ticker_df = (PSA
                 .loc[PSA['symbol']==ticker]
                 .drop(columns='symbol')
                 .sort_values(by='date',ascending=True)
                 .reset_index(drop=True)
                 .assign(**{'average': lambda df: df.loc[:,['open','high','low','close']].mean(axis=1), 
                            'EMA20': lambda df: df['average'].ewm(span=20, adjust=False).mean(), 
                            'EMA5': lambda df: df['average'].ewm(span=5, adjust=False).mean(), 
                            'dist_EMA20': lambda df: (df['average'] - df['EMA20'])/df['EMA20'], 
                            'dist_EMA5': lambda df: (df['average'] - df['EMA5'])/df['EMA5'], 
                            'market': market['market_average'].values}))

    # Create sequence samples
    # note: normalization is done with information within a sequence
    data_array = ticker_df[features].values
    sample_num = len(data_array)-lookback-lookahead
    idx_C = lookback-1
    idx_F = lookback+lookahead-1
    
    X = np.array([data_array[i:i+lookback] for i in range(sample_num)])
    X[:, :, 0:4] = X[:, :, 0:4] / X[:, 0:1, 0:1]
    X[:, :, 4:5] = X[:, :, 4:5] / X[:, 0:1, 4:5]
    X[:, :, -1] = X[:,:,-1] / X[:, 0:1, -1]
    y = np.array([(data_array[i+idx_F,1] - data_array[i+idx_C,1]) / data_array[i+idx_C,1] for i in range(sample_num)])
    
    if k==0:
        n = np.round(len(X)*(1-test_size)).astype(int)
        print(f"for each ticker, {n} train samples, {len(X)-n} test samples")    

    X_train.append(X[:n,:,:])
    X_test.append(X[n:,:,:])
    y_train.append(y[:n])
    y_test.append(y[n:])
        
    # show progress
    if (k>0) & (k%50 == 0):
        print(f"ticker {k}/{len(tickers_with_all_dates)}")


# In[ ]:


X_train = np.concatenate(X_train, axis=0)
X_test = np.concatenate(X_test, axis=0)
y_train = np.concatenate(y_train)
y_test = np.concatenate(y_test)
print(f"for all tickers, {X_train.shape[0]} train samples, {X_test.shape[0]} test samples")


# In[ ]:


__ = plt.hist(y_train, range=(-0.2, 0.2), bins=40)


# ## RNN model

# In[ ]:


import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
from keras import backend as K
from keras.utils import plot_model

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

# instantiating the model in the strategy scope creates the model on the TPU
with tpu_strategy.scope():
    lstm_input = Input(shape=(lookback, len(features)), name='lstm_input')
    x = LSTM(units=64, return_sequences=False, return_state=False)(lstm_input)  # ??????????? return_sequence, return_state
    x = Dropout(0.2)(x)
    x = Dense(units=32, activation='relu')(x)
    output = Dense(1, activation='linear')(x)
    model = Model(inputs=lstm_input, outputs=output)

    adam = optimizers.Adam(lr=0.0005)
    model.compile(optimizer=adam, loss='mse', metrics=[r2_keras])
    
    
model.summary()
plot_model(model) 


# ## Train model

# In[ ]:


History = model.fit(X_train, y_train, 
                    epochs=10, 
                    shuffle=True,
                    batch_size=128*8,
                    validation_data=(X_test, y_test))


# In[ ]:


# Plot training & validation accuracy values
plt.plot(History.history['r2_keras'], label='Train')
plt.plot(History.history['val_r2_keras'], label='validation')
plt.title('R2 score')
plt.ylabel('R2 score')
plt.xlabel('Epoch')
plt.legend(loc='best')
plt.show()


# R2 score is lower than but very close to zero --> the model is no better than a average guess which is close to zero. 

# ## Test model

# In[ ]:


# y_pred = model.predict(X_test, batch_size=128*8, verbose=1)

# fig, ax = plt.subplots(figsize=(10, 5))
# ax.plot(y_test, label='y_test')  
# ax.plot(y_pred, label='y_pred')  
# ax.legend(loc='best')


# In[ ]:


# r2_keras(y_test, y_pred)


# In[ ]:


# num_right = (np.sign(y_test) == np.sign(y_pred).squeeze()).sum()
# num_right/len(y_test)


# In[ ]:





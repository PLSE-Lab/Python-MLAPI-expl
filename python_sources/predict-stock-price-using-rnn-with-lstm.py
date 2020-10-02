#!/usr/bin/env python
# coding: utf-8

# # This is to show some of the issues trying to use machine learning for stock price prediction
# 
# Conclusion: It is NOT as easy as you may think.

# In[ ]:


import numpy as np 
import pandas as pd 

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


# # Import and look at data

# In[ ]:


prices = pd.read_csv('../input/nyse/prices.csv')
PSA = pd.read_csv('../input/nyse/prices-split-adjusted.csv')  # prices-split-adjusted
securities = pd.read_csv('../input/nyse/securities.csv')
fundamentals = pd.read_csv('../input/nyse/fundamentals.csv')


# In[ ]:


display([prices.shape, PSA.shape, securities.shape, fundamentals.shape])
display(prices.head(), PSA.head(), securities.head(), fundamentals.head())


# In[ ]:


display(PSA.groupby('symbol').size().sort_values(ascending=True),
        fundamentals.groupby('Ticker Symbol').size().sort_values(ascending=True))


# * PSA and prices have the same content. Use PSA hereafter
# * Not all symbols have the same time lenght of data, but most have data from 2010 to 2016. 
# * The fundamentals has at most 4 data points for any company and do not seem useful. Ignore.
# * data in securities are not useful for us. 
# 
# Use data of APPLE for further analysis

# # Feature engineering

# In[ ]:


apple = (PSA.loc[PSA['symbol']=='AAPL']
         .drop(columns='symbol')
         .sort_values(by='date',ascending=True)
         .reset_index(drop=True)
         .assign(**{'average': lambda df: df.loc[:,['open','high','low','close']].mean(axis=1), 
                    'EMA20': lambda df: df['average'].ewm(span=20, adjust=False).mean(), 
                    'EMA5': lambda df: df['average'].ewm(span=5, adjust=False).mean(), 
                    'dist_EMA20': lambda df: (df['average'] - df['EMA20'])/df['EMA20']*100, 
                    'dist_EMA5': lambda df: (df['average'] - df['EMA5'])/df['EMA5']*100}))
apple.head()


# In[ ]:


# use volume weighted averaged OHLC.mean to represent market average
# comparison with SPY500 shows that market_average calculated this way is representative of overall market

tickers_with_all_dates = PSA.groupby('symbol').size().loc[lambda s: s.values==s.values.max()].index.to_list()
market = (PSA.loc[PSA['symbol'].isin(tickers_with_all_dates)]
          .assign(**{'average': lambda df: df.loc[:,['open','high','low','close']].mean(axis=1), 
                     'price x volume': lambda df: df['average']*df['volume']})
          .groupby('date')
          .agg(**{'price x volume sum': pd.NamedAgg(column='price x volume', aggfunc=np.sum), 
                  'volume sum': pd.NamedAgg(column='volume', aggfunc=np.sum)})
          .assign(**{'market_average': lambda df: df['price x volume sum']/df['volume sum']})
          .sort_index(ascending=True))

apple['market'] = market['market_average'].values
apple.head()


# In[ ]:


fig, ax = plt.subplots(figsize=(10,5))
ax.plot(apple.index[:120], apple['dist_EMA20'][:120], label='dist_EMA20')
ax.plot(apple.index[:120], apple['dist_EMA5'][:120], label='dist_EMA5')
ax.set_xlabel('day')
ax.set_ylabel('distance from moving mean')
ax.legend(loc='best')

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(apple.index[:120], apple['open'][:120], label='open')
ax.plot(apple.index[:120], apple['close'][:120], label='close')
ax.set_xlabel('day')
ax.set_ylabel('apple')
ax.legend(loc='best')


# # Create sample of sequences and normalize within each sample
# - will this make each sample independent from each other? -- No, you will see why later.

# In[ ]:


# use previous 120 days' data to predict price in 5 days
lookback = 120  
lookahead = 5

# Create sequence samples
# note: normalization is done with information within a sequence
features = ['open', 'close', 'low', 'high', 'volume', 'dist_EMA20', 'dist_EMA5', 'market']
data = apple[features].values
X = np.array([data[i:i+lookback].copy() for i in range(len(data) - lookback - lookahead)])
X[:,:,0:4] = X[:,:,0:4]/X[:,0,0, None, None]
X[:,:,4] = X[:,:,4]/X[:,0,None,4]
X[:,:,-1] = X[:,:,-1]/X[:,0,None,-1]
y = np.array([(data[i+lookback+lookahead-1,1] - data[i+lookback-1,1])/data[i+lookback-1,1]*100 for i in range(len(data) - lookback - lookahead)])


# In[ ]:


fig, ax = plt.subplots(figsize=(10,5))
ax.plot(range(X[0].shape[0]), X[0,:,0], label='open')
ax.plot(range(X[0].shape[0]), X[0,:,-1], label='market')
ax.set_xlabel('day')
ax.set_ylabel('apple')
ax.legend(loc='best')


# # RNN model

# In[ ]:


import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers

np.random.seed(4)

lstm_input = Input(shape=(lookback, len(features)), name='lstm_input')
x = LSTM(units=64, return_sequences=False, return_state=False)(lstm_input)  
x = Dropout(0.2)(x)
x = Dense(units=32, activation='relu')(x)
output = Dense(1, activation='linear')(x)
model = Model(inputs=lstm_input, outputs=output)

adam = optimizers.Adam(lr=0.0005)

model.compile(optimizer=adam, loss='mse')

weights = model.get_weights()  # needed to reset the model 

model.summary()
from keras.utils import plot_model
plot_model(model) 


# # Train model

# In[ ]:


# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
print("{} train samples, {} test samples".format(X_train.shape[0], X_test.shape[0]))

model.fit(X_train, y_train, epochs=80, verbose=0)


# # test model

# In[ ]:


y_pred = model.predict(X_test)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(len(y_test)), y_test, label='y_test')
ax.plot(range(len(y_test)), y_pred, label='y_pred')


# In[ ]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[ ]:


num_right = (np.sign(y_test) == np.sign(y_pred).squeeze()).sum()
num_right/len(y_test)


# The accuracy is too high. You have to question if the sample are truly independent.
# The fact is that they are not because we create the samples by moving forward one step at a time. 
# So for each sample, the samples before and after it will be very similar to it. 
# When we split train and test samples randomly, for each test sample, there may be several train sample similar to it, which gives the fake high accuracy. 
# <font color=red>Do NOT split samples randomly for time-series problem</font>

# # Split train and test without randomness and retrain model

# In[ ]:


test_size = 0.2
k = np.round(len(X)*(1-test_size)).astype(int)
X_train, X_test, y_train, y_test = X[:k,:,:], X[k:,:,:], y[:k], y[k:]
print("{} train samples, {} test samples".format(X_train.shape[0], X_test.shape[0]))


# In[ ]:


model.set_weights(weights)

train_history = model.fit(X_train, y_train, epochs=50, verbose=0)


# In[ ]:


y_pred = model.predict(X_test)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(len(y_test)), y_test, label='y_test')
ax.plot(range(len(y_test)), y_pred, label='y_pred')


# In[ ]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[ ]:


num_right = (np.sign(y_test) == np.sign(y_pred).squeeze()).sum()
num_right/len(y_test)


# After removing randomness in spliting train and test samples, the model accuracy becomes <font color=red>drastically worse</font>. Close to random guess!

# # Predict into unknown future (use predicted output as input) 
# 
# For this to work, the output must have same shape as the input and lookahead must be 1 -- so the output can be used as input. Need to modify model.

# ## Still train model with 1 day lookahead

# In[ ]:


lookback = 120
lookahead = 1

# Create sequence samples
features = ['open', 'close', 'low', 'high', 'volume', 'dist_EMA20', 'dist_EMA5', 'market']
data = apple[features].values
X = np.array([data[i:i+lookback].copy() for i in range(len(data) - lookback - lookahead)])
y = np.array([data[i+lookback+lookahead-1] for i in range(len(data) - lookback - lookahead)])

# must normalize y first
y[:,0:4] = y[:,0:4]/X[:,0,0, None]
y[:,4] = y[:,4]/X[:,0,4]
y[:,-1] = y[:,-1]/X[:,0,-1]

X[:,:,0:4] = X[:,:,0:4]/X[:,0,0, None, None]
X[:,:,4] = X[:,:,4]/X[:,0,None,4]
X[:,:,-1] = X[:,:,-1]/X[:,0,None,-1]


# In[ ]:


lstm_input = Input(shape=(lookback, len(features)), name='lstm_input')
x = LSTM(units=64, return_sequences=False, return_state=False)(lstm_input)  
x = Dropout(0.2)(x)
x = Dense(units=32, activation='relu')(x)
output = Dense(8, activation='linear')(x)
model_1 = Model(inputs=lstm_input, outputs=output)

adam = optimizers.Adam(lr=0.0005)

model_1.compile(optimizer=adam, loss='mse')

model_1.summary()
from keras.utils import plot_model
plot_model(model_1) #, to_file='model.png')


# In[ ]:


test_size = 0.2
k = np.round(len(X)*(1-test_size)).astype(int)
X_train, X_test, y_train, y_test = X[:k,:,:], X[k:,:,:], y[:k], y[k:]
print("{} train samples, {} test samples".format(X_train.shape[0], X_test.shape[0]))


# In[ ]:


train_history = model_1.fit(X_train, y_train, epochs=40, verbose=0, validation_data=(X_test, y_test))


# ## apply model to predict 328 days' price movement in a row

# In[ ]:


# function to put the new output at the end of input
def insert_end(Xin,new_input):
    for i in range(lookback-1):
        Xin[:,i,:] = Xin[:,i+1,:]
    Xin[:,lookback-1,:] = new_input
    return Xin


# In[ ]:


first =0   
future=X_test.shape[0]
forcast = []
Xin = X_test[first:first+1,:,:]
for i in range(future):
    out = model_1.predict(Xin, batch_size=1)    
    forcast.append(out) 
    Xin = insert_end(Xin,out) 

forcast_output = np.array(forcast).squeeze()


# In[ ]:


plt.figure(figsize=(10,5))
plt.plot(y_test[:,1] , 'black', linewidth=4)
plt.plot(forcast_output[:,1],'r' , linewidth=4)
plt.legend(('close, test','close, Forcasted'))
plt.show()


# Apparently, predicting many days of price movement does NOT work well. 

# ## apply model to predict 1 day at a time

# In[ ]:


y_pred = model_1.predict(X_test)

for i in range(y_test.shape[1]):
    print(r2_score(y_test[:, i], y_pred[:, i]))


# In[ ]:


plt.figure(figsize=(10,5))
plt.plot(y_test[:,1] , 'black', linewidth=4)
plt.plot(y_pred[:,1],'r' , linewidth=4)
plt.legend(('close, test','close, Forcasted'))
plt.show()


# The R2 scores are not too bade, and two curves seem to follow the same overall trend. But don't let them fool you. 

# In[ ]:


# calculate the changes relative the prior day
y_test_change = y_test - X_test[:, -1, :]
y_pred_change = y_pred - X_test[:, -1, :]

plt.figure(figsize=(10,5))
plt.plot(y_test_change[:,1] , 'black', linewidth=4)
plt.plot(y_pred_change[:,1],'r' , linewidth=4)
plt.legend(('test','Forcasted'))
plt.show()


# In[ ]:


y_test_close_up = y_test_change[:,1] > 0
y_pred_close_up = y_pred_change[:,1] > 0
print("number of close up days, actural = {}, number of close up days, predicted = {}".format(y_test_close_up.sum(), y_pred_close_up.sum()))
accuracy = (y_pred_close_up == y_test_close_up).sum() / len(y_pred_close_up)
print("percentage of actual and predict have same direction = {:2.2%}".format(accuracy))


# The accuray is close to 50% --> No better than random guess! 

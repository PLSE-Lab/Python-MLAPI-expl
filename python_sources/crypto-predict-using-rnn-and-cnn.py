#!/usr/bin/env python
# coding: utf-8

# # Cryptocurrency Predict using Recurrent Neural Network(RNN) and Convolutional Neural Network(CNN)
# This code is from [pythonprogramming.net](https://pythonprogramming.net/cryptocurrency-recurrent-neural-network-deep-learning-python-tensorflow-keras/) tutorial.
# 
# Using Recurrent Neural Network (RNN) and Convolutional Neural Network (CNN), this code will predict the price movement of  Bitcoin, Ethereum, Litecoin and Bitcoin Cash cryptocurrency data. The target output will be a binary of **0** for price increase and **1** for price decrease. Also, this code will show the performace of RNN vs CNN.

# In[ ]:


import numpy as np
import pandas as pd
import random
import time
from collections import deque
from sklearn import preprocessing


# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras.layers import Activation, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint


# ## Data

# In[ ]:


df = pd.read_csv("../input/LTC-USD.csv", names = ['time','low','high','open','close','volume'])

df.head()


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.columns


# ## Parameters
# 
# A sequence lenght of 60 minutes data will be collected to be feed into the model and will predict the futere in 3 minute time.

# In[ ]:


SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = 'LTC-USD'


# ## Classify Target

# In[ ]:


def classify (current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0


# ## Preprocessing Data

# In[ ]:


def preprocess_df(df):
    df = df.drop('future', 1)
    
    for col in df.columns:
        if col != 'target':
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
    df.dropna(inplace=True)
    
    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)
    
    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])
    random.shuffle(sequential_data)
    
    buys = []
    sells = []
    
    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])
    random.shuffle(buys)
    random.shuffle(sells)
    
    lower = min(len(buys), len(sells))
    buys = buys[:lower]
    sells = sells[:lower]
    
    sequential_data = buys + sells
    random.shuffle(sequential_data)
    
    X = []
    y = []
    
    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)
        
    return np.array(X), y


# ## Build the Dataframe for data training and validation

# In[ ]:


main_df = pd.DataFrame()
ratios = ['LTC-USD', 'BCH-USD', 'BTC-USD', 'ETH-USD'] 

for ratio in ratios:
    ratio = ratio.split('.csv')[0]
    dataset = f'../input/{ratio}.csv'
    df = pd.read_csv(dataset, names=['time','low','high','open','close','volume'])
    df.rename(columns={'close':f'{ratio}_close','volume':f'{ratio}_volume'}, inplace=True)
    df.set_index('time', inplace=True)
    df = df[[f'{ratio}_close',f'{ratio}_volume']]
    
    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)
        


# In[ ]:


main_df.fillna(method='ffill', inplace=True)
main_df.dropna(inplace=True)
print(main_df.head())


# In[ ]:


main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'],main_df['future']))
main_df.dropna(inplace=True)
print(main_df.head())


# In[ ]:


times = sorted(main_df.index.values)
last_5pct = sorted(main_df.index.values)[-int(0.05*len(times))]

print(time)
print(last_5pct)


# In[ ]:


validation_main_df = main_df[(main_df.index >= last_5pct)]
train_main_df = main_df[(main_df.index < last_5pct)]

print(validation_main_df.head())
print(train_main_df.head())


# In[ ]:


train_x, train_y = preprocess_df(train_main_df)
validation_x, validation_y = preprocess_df(validation_main_df)


# In[ ]:


print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")


# In[ ]:


print(train_x.shape[1:])


# ## RNN Model

# In[ ]:


EPOCHS = 10
BATCH_SIZE = 64
NAME = f'{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}'


# In[ ]:


rnn_model = Sequential()

rnn_model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]),return_sequences=True))
rnn_model.add(Dropout(0.2))
rnn_model.add(BatchNormalization())

rnn_model.add(CuDNNLSTM(128,return_sequences=True))
rnn_model.add(Dropout(0.1))
rnn_model.add(BatchNormalization())

rnn_model.add(CuDNNLSTM(128))
rnn_model.add(Dropout(0.2))
rnn_model.add(BatchNormalization())

rnn_model.add(Dense(32, activation='relu'))
rnn_model.add(Dropout(0.2))

rnn_model.add(Dense(2, activation='softmax'))


# In[ ]:


opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)


# In[ ]:


rnn_model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


# In[ ]:


tensorboard = TensorBoard(log_dir='../{}'.format(NAME))

filepath = 'RNN_Final-{epoch:02d}-{val_acc:.3f}'
checkpoint = ModelCheckpoint('../{}.model'.format(filepath, monitor='val-acc', verbose=1, save_best_only=True, mode='max'))


# In[ ]:


history = rnn_model.fit(train_x, train_y,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(validation_x, validation_y),
                    callbacks = [tensorboard, checkpoint])


# In[ ]:


rnn_score = rnn_model.evaluate(validation_x, validation_y, verbose=0)
print('Test loss:', rnn_score[0])
print('Test accuracy:', rnn_score[1])


# ## CNN Model

# In[ ]:


cnn_model = Sequential()

cnn_model.add(Conv1D(128,3,input_shape=(train_x.shape[1:])))
cnn_model.add(Activation('relu'))
cnn_model.add(Dropout(0.2))
cnn_model.add(MaxPooling1D(pool_size=2))

cnn_model.add(Conv1D(128,3))
cnn_model.add(Activation('relu'))
cnn_model.add(Dropout(0.2))
cnn_model.add(MaxPooling1D(pool_size=2))

cnn_model.add(Conv1D(128,3))
cnn_model.add(Activation('relu'))
cnn_model.add(Dropout(0.2))
cnn_model.add(MaxPooling1D(pool_size=2))

cnn_model.add(Flatten())
cnn_model.add(Dense(32))

cnn_model.add(Dense(2, activation='softmax'))

cnn_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

cnn_history = cnn_model.fit(train_x, train_y,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(validation_x, validation_y))


# In[ ]:


cnn_score = cnn_model.evaluate(validation_x, validation_y, verbose=0)
print('Test loss:', cnn_score[0])
print('Test accuracy:', cnn_score[1])


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# ![](https://i.imgur.com/gLH0p3a.jpg)

# I'm going to use the New York Stock Exchange database in order to predict General Eletric's stock price in 30 days (buy or sell) from the 60 previous days worth of data from companies in the S&P500. 
# 
#  1. [LSTM]()
#  2. [LSTM + GRU]()
#  3. [Attention]()
#  4. [Multi-Head Attention]()

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


df = pd.read_csv("../input/prices.csv")


# In[ ]:


df['date'] = pd.to_datetime(df['date'])
df.info()


# In[ ]:


df.head(20)


# In[ ]:


df_pivot = df.pivot('date','symbol','close').reset_index()
df_pivot.head()


# In[ ]:


# set the index
df_pivot.set_index('date', inplace=True)


# In[ ]:


df_pivot.head()


# In[ ]:


df_pivot.dropna(axis=1, how='any', inplace=True)


# In[ ]:


df_pivot.head()


# In[ ]:


df_pivot.shape


# In[ ]:


SEQ_LEN = 60  # how long of a preceeding sequence to collect for RNN
FUTURE_PERIOD_PREDICT = 30  # how far into the future are we trying to predict?
STOCK_TO_PREDICT = 'GE'


# In[ ]:


def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0


# In[ ]:


df_pivot['future'] = df_pivot[STOCK_TO_PREDICT].shift(-FUTURE_PERIOD_PREDICT)


# In[ ]:


df_pivot['target'] = list(map(classify, df_pivot[STOCK_TO_PREDICT], df_pivot['future']))


# In[ ]:


times = sorted(df_pivot.index.values)  # get the times
last_10pct = sorted(df_pivot.index.values)[-int(0.1*len(times))]  # get the last 10% of the times
last_20pct = sorted(df_pivot.index.values)[-int(0.2*len(times))]  # get the last 20% of the times

test_df = df_pivot[(df_pivot.index >= last_10pct)]
validation_df = df_pivot[(df_pivot.index >= last_20pct) & (df_pivot.index < last_10pct)]  
train_df = df_pivot[(df_pivot.index < last_20pct)]  # now the train_df is all the data up to the last 20%


# In[ ]:


from collections import deque
import numpy as np
import random


# In[ ]:


from sklearn import preprocessing  

def preprocess_df(df):
    df = df.drop(columns=["future"])  # don't need this anymore.

    
    df.dropna(inplace=True)  # remove the nas created by pct_change
    df[STOCK_TO_PREDICT] = preprocessing.scale(df[STOCK_TO_PREDICT].values)  # scale between 0 and 1.

    df.dropna(inplace=True)  # cleanup again... jic.


    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

    for i in df.values:  # iterate over the values
        prev_days.append([n for n in i[:-1]])  # store all but the target
        if len(prev_days) == SEQ_LEN:  # make sure we have 60 sequences!
            sequential_data.append([np.array(prev_days), i[-1]])  # append those bad boys!

    random.shuffle(sequential_data)  # shuffle for good measure.

    buys = []  # list that will store our buy sequences and targets
    sells = []  # list that will store our sell sequences and targets

    for seq, target in sequential_data:  # iterate over the sequential data
        if target == 0:  # if it's a "not buy"
            sells.append([seq, target])  # append to sells list
        elif target == 1:  # otherwise if the target is a 1...
            buys.append([seq, target])  # it's a buy!

    random.shuffle(buys)  # shuffle the buys
    random.shuffle(sells)  # shuffle the sells!

    lower = min(len(buys), len(sells))  # what's the shorter length?

    buys = buys[:lower]  # make sure both lists are only up to the shortest length.
    sells = sells[:lower]  # make sure both lists are only up to the shortest length.

    sequential_data = buys+sells  # add them together
    random.shuffle(sequential_data)  # another shuffle, so the model doesn't get confused with all 1 class then the other.

    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

    return np.array(X), y  # return X and y...and make X a numpy array!


# In[ ]:


X_train, y_train = preprocess_df(train_df)
X_val, y_val = preprocess_df(validation_df)
X_test, y_test = preprocess_df(test_df)


# In[ ]:


print(f"train data: {len(X_train)} validation: {len(X_val)}, test: {len(X_test)}")
print(f"Train Dont buys: {y_train.count(0)}, buys: {y_train.count(1)}")
print(f"Validation Dont buys: {y_val.count(0)}, buys: {y_val.count(1)}")
print(f"Test Dont buys: {y_test.count(0)}, buys: {y_test.count(1)}")


# # 1. [LSTM]()

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import GRU
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam 


# In[ ]:


EPOCHS = 100  # how many passes through our data
BATCH_SIZE = 32  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
import time

NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"  # a unique name for the model


# In[ ]:


lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, epsilon=0.0001, patience=3, verbose=1)
checkpoint = ModelCheckpoint(NAME, monitor='val_loss', verbose=1, save_best_only=True, mode='min')


# In[ ]:


baseline = Sequential()
baseline.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False, activation='relu'))
baseline.add(Dropout(0.2))
baseline.add(BatchNormalization())  #normalizes activation outputs, same reason you want to normalize your input data.

baseline.add(Dense(2, activation='softmax'))


# In[ ]:


baseline.compile(loss='sparse_categorical_crossentropy', optimizer=Adam() , metrics = ['accuracy'])


# In[ ]:


baseline.fit(X_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(X_val, y_val),
                    callbacks = [checkpoint, lr_reduce])


# In[ ]:


y_prob = baseline.predict(X_test) 
predicted_stock_price_baseline = y_prob.argmax(axis=-1)


# In[ ]:


plt.figure(figsize = (18,9))
plt.plot(y_test, color = 'black', label = 'GE Stock Price')
plt.plot(predicted_stock_price_baseline, color = 'green', label = 'Predicted GE Mid Price')
plt.title('GE Close Price Prediction', fontsize=30)
#plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
plt.xlabel('Date')
plt.ylabel('GE Close Price')
plt.legend(fontsize=18)
plt.show()


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(y_test, predicted_stock_price_baseline)


# # 2. [LSTM + GRU]()

# In[ ]:


filepath="LSTM_GRU.hdf5"


# In[ ]:


model = Sequential()
model.add(GRU(256 , input_shape = (X_train.shape[1], X_train.shape[2]) , return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(256))
model.add(Dropout(0.4))
model.add(Dense(64 ,  activation = 'relu'))
model.add(Dense(2, activation='softmax'))
model.summary()


# In[ ]:


model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam() , metrics = ['accuracy'])


# In[ ]:


history_lstm = model.fit(X_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(X_val, y_val), 
                    callbacks = [checkpoint , lr_reduce])


# In[ ]:


y_prob = model.predict(X_test) 
predicted_stock_price_gru = y_prob.argmax(axis=-1)


# In[ ]:


plt.figure(figsize = (18,9))
plt.plot(y_test, color = 'black', label = 'GE Stock Price')
plt.plot(predicted_stock_price_gru, color = 'green', label = 'Predicted GE Close Price')
plt.title('GE Close Price Prediction', fontsize=30)
#plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
plt.xlabel('Date')
plt.ylabel('GE Close Price')
plt.legend(fontsize=18)
plt.show()


# In[ ]:


accuracy_score(y_test, predicted_stock_price_gru)


# # 3. [Attention]()

# In[ ]:


from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.engine.input_layer import Input
from keras import backend as K
from keras.models import Model


# In[ ]:


# https://www.kaggle.com/qqgeogor/keras-lstm-attention-glove840b-lb-0-043
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


# In[ ]:


inp = Input(shape = (X_train.shape[1], X_train.shape[2]))
x = LSTM(128, return_sequences=True)(inp)
x = Dropout(0.2)(x)
x = BatchNormalization()(x)
x = Attention(SEQ_LEN)(x)
x = Dense(32, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(2, activation="softmax")(x)
model_lstm_attention = Model(inputs=inp, outputs=x)


# In[ ]:


model_lstm_attention.compile(loss='sparse_categorical_crossentropy', optimizer=Adam() , metrics = ['accuracy'])

model_lstm_attention.summary()


# In[ ]:


model_lstm_attention.fit(X_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(X_val, y_val), 
                    callbacks = [checkpoint , lr_reduce]
             )


# In[ ]:


y_prob = model_lstm_attention.predict(X_test) 
predicted_stock_price_attention = y_prob.argmax(axis=-1)


# In[ ]:


plt.figure(figsize = (18,9))
plt.plot(y_test, color = 'black', label = 'GE Stock Price')
plt.plot(predicted_stock_price_attention, color = 'green', label = 'Predicted GE Close Price')
plt.title('GE Close Price Prediction', fontsize=30)
#plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
plt.xlabel('Date')
plt.ylabel('GE Close Price')
plt.legend(fontsize=18)
plt.show()


# In[ ]:


accuracy_score(y_test, predicted_stock_price_attention)


#  # [Multi-Head Attention]()

# In[ ]:


# https://www.kaggle.com/shujian/transformer-with-lstm

import random, os, sys
import numpy as np
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *
import tensorflow as tf
from keras.engine.topology import Layer

try:
    from dataloader import TokenList, pad_to_longest
    # for transformer
except: pass



embed_size = 60

class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)
    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    def compute_output_shape(self, input_shape):
        return input_shape

class ScaledDotProductAttention():
    def __init__(self, d_model, attn_dropout=0.1):
        self.temper = np.sqrt(d_model)
        self.dropout = Dropout(attn_dropout)
    def __call__(self, q, k, v, mask):
        attn = Lambda(lambda x:K.batch_dot(x[0],x[1],axes=[2,2])/self.temper)([q, k])
        if mask is not None:
            mmask = Lambda(lambda x:(-1e+10)*(1-x))(mask)
            attn = Add()([attn, mmask])
        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x:K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn

class MultiHeadAttention():
    # mode 0 - big martixes, faster; mode 1 - more clear implementation
    def __init__(self, n_head, d_model, d_k, d_v, dropout, mode=0, use_norm=True):
        self.mode = mode
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        if mode == 0:
            self.qs_layer = Dense(n_head*d_k, use_bias=False)
            self.ks_layer = Dense(n_head*d_k, use_bias=False)
            self.vs_layer = Dense(n_head*d_v, use_bias=False)
        elif mode == 1:
            self.qs_layers = []
            self.ks_layers = []
            self.vs_layers = []
            for _ in range(n_head):
                self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization() if use_norm else None
        self.w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v, mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        if self.mode == 0:
            qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
            ks = self.ks_layer(k)
            vs = self.vs_layer(v)

            def reshape1(x):
                s = tf.shape(x)   # [batch_size, len_q, n_head * d_k]
                x = tf.reshape(x, [s[0], s[1], n_head, d_k])
                x = tf.transpose(x, [2, 0, 1, 3])  
                x = tf.reshape(x, [-1, s[1], d_k])  # [n_head * batch_size, len_q, d_k]
                return x
            qs = Lambda(reshape1)(qs)
            ks = Lambda(reshape1)(ks)
            vs = Lambda(reshape1)(vs)

            if mask is not None:
                mask = Lambda(lambda x:K.repeat_elements(x, n_head, 0))(mask)
            head, attn = self.attention(qs, ks, vs, mask=mask)  
                
            def reshape2(x):
                s = tf.shape(x)   # [n_head * batch_size, len_v, d_v]
                x = tf.reshape(x, [n_head, -1, s[1], s[2]]) 
                x = tf.transpose(x, [1, 2, 0, 3])
                x = tf.reshape(x, [-1, s[1], n_head*d_v])  # [batch_size, len_v, n_head * d_v]
                return x
            head = Lambda(reshape2)(head)
        elif self.mode == 1:
            heads = []; attns = []
            for i in range(n_head):
                qs = self.qs_layers[i](q)   
                ks = self.ks_layers[i](k) 
                vs = self.vs_layers[i](v) 
                head, attn = self.attention(qs, ks, vs, mask)
                heads.append(head); attns.append(attn)
            head = Concatenate()(heads) if n_head > 1 else heads[0]
            attn = Concatenate()(attns) if n_head > 1 else attns[0]

        outputs = self.w_o(head)
        outputs = Dropout(self.dropout)(outputs)
        if not self.layer_norm: return outputs, attn
        # outputs = Add()([outputs, q]) # sl: fix
        return self.layer_norm(outputs), attn

class PositionwiseFeedForward():
    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
        self.w_2 = Conv1D(d_hid, 1)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)
    def __call__(self, x):
        output = self.w_1(x) 
        output = self.w_2(output)
        output = self.dropout(output)
        output = Add()([output, x])
        return self.layer_norm(output)

class EncoderLayer():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn_layer  = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
    def __call__(self, enc_input, mask=None):
        output, slf_attn = self.self_att_layer(enc_input, enc_input, enc_input, mask=mask)
        output = self.pos_ffn_layer(output)
        return output, slf_attn


def GetPosEncodingMatrix(max_len, d_emb):
    pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] 
        if pos != 0 else np.zeros(d_emb) 
            for pos in range(max_len)
            ])
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2]) # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2]) # dim 2i+1
    return pos_enc

def GetPadMask(q, k):
    ones = K.expand_dims(K.ones_like(q, 'float32'), -1)
    mask = K.cast(K.expand_dims(K.not_equal(k, 0), 1), 'float32')
    mask = K.batch_dot(ones, mask, axes=[2,1])
    return mask

def GetSubMask(s):
    len_s = tf.shape(s)[1]
    bs = tf.shape(s)[:1]
    mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
    return mask

class Transformer():
    def __init__(self, len_limit, embedding_matrix, d_model=embed_size,               d_inner_hid=512, n_head=10, d_k=64, d_v=64, layers=2, dropout=0.1,               share_word_emb=False, **kwargs):
        self.name = 'Transformer'
        self.len_limit = len_limit
        self.src_loc_info = False # True # sl: fix later
        self.d_model = d_model
        self.decode_model = None
        d_emb = d_model

        pos_emb = Embedding(len_limit, d_emb, trainable=False,                             weights=[GetPosEncodingMatrix(len_limit, d_emb)])

        i_word_emb = Embedding(max_features, d_emb, weights=[embedding_matrix]) # Add Kaggle provided embedding here

        self.encoder = Encoder(d_model, d_inner_hid, n_head, d_k, d_v, layers, dropout,                                word_emb=i_word_emb, pos_emb=pos_emb)

        
    def get_pos_seq(self, x):
        mask = K.cast(K.not_equal(x, 0), 'int32')
        pos = K.cumsum(K.ones_like(x, 'int32'), 1)
        return pos * mask

    def compile(self, active_layers=999):
        src_seq_input = Input(shape=(None, ))
        x = Embedding(max_features, embed_size, weights=[embedding_matrix])(src_seq_input)
        
        # LSTM before attention layers
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = Bidirectional(LSTM(64, return_sequences=True))(x) 
        
        x, slf_attn = MultiHeadAttention(n_head=3, d_model=300, d_k=64, d_v=64, dropout=0.1)(x, x, x)
        
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        conc = concatenate([avg_pool, max_pool])
        conc = Dense(64, activation="relu")(conc)
        x = Dense(2, activation="softmax")(conc)   
        
        
        self.model = Model(inputs=src_seq_input, outputs=x)
        self.model.compile(optimizer = 'Adamax', loss = 'binary_crossentropy', metrics=['accuracy'])


# In[ ]:


def build_model():
    inp = Input(shape = (X_train.shape[1], X_train.shape[2]))
    
    # LSTM before attention layers
    x = Bidirectional(LSTM(128, return_sequences=True))(inp)
    x = Bidirectional(LSTM(64, return_sequences=True))(x) 
        
    x, slf_attn = MultiHeadAttention(n_head=3, d_model=300, d_k=64, d_v=64, dropout=0.1)(x, x, x)
        
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    conc = Dense(64, activation="sigmoid")(conc)
    x = Dense(2, activation="softmax")(conc)      

    model = Model(inputs = inp, outputs = x)
    model.compile(
        loss = "binary_crossentropy", 
        #optimizer = Adam(lr = config["lr"], decay = config["lr_d"]), 
        optimizer = "RMSprop",
        metrics=['accuracy'])
    
    return model


# In[ ]:


from keras.utils import to_categorical

y_train_cat = to_categorical(y_train, num_classes=None)
y_val_cat = to_categorical(y_val, num_classes=None)
y_test_cat = to_categorical(y_test, num_classes=None)


# In[ ]:


multi_head = build_model()


# In[ ]:


multi_head.summary()


# In[ ]:


multi_head.fit(X_train, y_train_cat,
                    epochs=EPOCHS,
                    validation_data=(X_val, y_val_cat), 
                    callbacks = [checkpoint , lr_reduce]
             )


# In[ ]:


y_prob = multi_head.predict(X_test) 
predicted_stock_price_transformer = y_prob.argmax(axis=-1)


# In[ ]:


plt.figure(figsize = (18,9))
plt.plot(y_test, color = 'black', label = 'GE Stock Price')
plt.plot(predicted_stock_price_transformer, color = 'green', label = 'Predicted GE Close Price')
plt.title('GE Close Price Prediction', fontsize=30)
#plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
plt.xlabel('Date')
plt.ylabel('GE Close Price')
plt.legend(fontsize=18)
plt.show()


# In[ ]:


accuracy_score(y_test, predicted_stock_price_transformer)


# # Work in progress, any suggestion for improvements is welcomed. 

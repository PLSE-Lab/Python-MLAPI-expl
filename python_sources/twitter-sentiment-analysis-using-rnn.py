#!/usr/bin/env python
# coding: utf-8

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
print(os.listdir("../input/twitter-senti"))
print(os.listdir("../input/glove-twitter"))


# In[ ]:



train = pd.read_csv('../input/twitter-senti/train.tsv',sep='\t')
test = pd.read_csv('../input/twitter-senti/test.tsv',sep='\t')
train.head()


# In[ ]:


test.head()


# In[ ]:


embed_matrix = {}
file= open('../input/glove-twitter/glove.twitter.27B.25d.txt')
for line in file:
    values = line.split()
    word = values[0]
    coefs = np.array(values[1:],dtype='float32')
    embed_matrix[word]=coefs
file.close()


# In[ ]:


len(embed_matrix)


# In[ ]:


train.info()


# In[ ]:


train.label.unique()


# In[ ]:


from  keras.preprocessing.text import Tokenizer
token = Tokenizer(num_words=2000, split=' ')
token.fit_on_texts(list(train['tweet']))
X = token.texts_to_sequences(list(train['tweet']))


# In[ ]:


length =[len(x) for x in X]
print (max(length))


# In[ ]:


from keras.preprocessing.sequence import pad_sequences
X_pad = pad_sequences(X)   # Pad sequences with maximum length


# In[ ]:


X_pad.shape[1]


# In[ ]:


Y = pd.get_dummies(train['label'])
Y.shape


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout,BatchNormalization


# In[ ]:


model = Sequential()
model.add(Dense(128,input_dim=X_pad.shape[1],activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(3,activation='softmax'))
model.summary()


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X_pad,Y, test_size=0.2,random_state = 100)


# In[ ]:


x_train.shape


# In[ ]:


x_test.shape


# In[ ]:


model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(x_train,y_train,epochs= 50,batch_size=128)


# In[ ]:


model.evaluate(x_test,y_test)


# In[ ]:


diction = token.word_index
len(diction)


# In[ ]:


EMBEDDING_DIM = 25
embedding_matrix = np.zeros((len(token.word_index) + 1, EMBEDDING_DIM))
for word, i in token.word_index.items():
    embedding_vector = embed_matrix.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# In[ ]:


embedding_matrix.shape


# In[ ]:


VOC_SIZE = len(token.word_index)+1
EMB_DIM = 25
INPUT_DIM = X_pad.shape[1]
from keras.layers import Embedding, SimpleRNN,SpatialDropout1D,LSTM,GRU
from keras import regularizers
from keras.optimizers import *
rnn_model = Sequential()
rnn_model.add(Embedding(VOC_SIZE, EMB_DIM, input_length=INPUT_DIM,weights=[embedding_matrix],trainable=False))
rnn_model.add(SpatialDropout1D(0.4))
rnn_model.add(SimpleRNN(64,activation='relu',dropout=0.5, recurrent_dropout=0.5))
rnn_model.add(Dropout(0.5))
rnn_model.add(Dense(3,activation='softmax',kernel_regularizer=regularizers.l2(0.01)))
rnn_model.compile(loss='categorical_crossentropy', optimizer=Adam(0.01),metrics=['accuracy'])
rnn_model.summary()


# In[ ]:


rnn_model.fit(x_train,y_train,epochs=5,batch_size=128)


# In[ ]:


rnn_model.evaluate(x_test,y_test)


# In[ ]:


lstm_model = Sequential()
lstm_model.add(Embedding(VOC_SIZE, EMB_DIM, input_length=INPUT_DIM,weights=[embedding_matrix],trainable=False))
lstm_model.add(LSTM(64,activation='relu'))
lstm_model.add(Dense(3,activation='softmax'))
lstm_model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001),metrics=['accuracy'])
lstm_model.summary()


# In[ ]:


lstm_model.fit(x_train,y_train,epochs=5,batch_size=128)


# In[ ]:


lstm_model.evaluate(x_test,y_test)


# In[ ]:


gru_model = Sequential()
gru_model.add(Embedding(VOC_SIZE, EMB_DIM, input_length=INPUT_DIM,weights=[embedding_matrix],trainable=False))
gru_model.add(GRU(64,activation='relu'))
gru_model.add(Dense(3,activation='softmax'))
gru_model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001),metrics=['accuracy'])
gru_model.summary()


# In[ ]:


gru_model.fit(x_train,y_train,epochs=10,batch_size=128)


# In[ ]:


gru_model.evaluate(x_test,y_test)


# In[ ]:





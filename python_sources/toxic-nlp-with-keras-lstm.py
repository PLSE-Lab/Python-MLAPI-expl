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

import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Model
from keras.layers import Dense, Embedding, Input, LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Any results you write to the current directory are saved as output.


# load datas!

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
subm = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


train.head()


# In[ ]:


train.shape


# ** check data with simple EDA **

# In[ ]:


train_length = train.comment_text.apply(len)
train_length.head()


# In[ ]:


plt.figure(figsize = (12, 5))
plt.hist(train_length, bins = 60, alpha = 0.5, color = 'r')
plt.show()


# In[ ]:


print("max length : ", np.max(train_length))
print("min length : ", np.min(train_length))
print("mean length : ", np.mean(train_length))
print("75 % percentile : ", np.percentile(train_length, 75))
print("85 % percentile : ", np.percentile(train_length, 85))
print("std length : ", np.std(train_length))


# In[ ]:


print(train.comment_text.isna().sum())
print(test.comment_text.isna().sum())


# ** preprocssing with keras Tokenizer **

# In[ ]:


X = train.comment_text
y = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
test = test.comment_text


# In[ ]:


y[:5]


# In[ ]:


num_words = 20000
max_len = 150
emb_size = 128


# In[ ]:


tok = Tokenizer(num_words = num_words)
tok.fit_on_texts(list(X))


# In[ ]:


X = tok.texts_to_sequences(X)
test = tok.texts_to_sequences(test)


# In[ ]:


X[0]


# In[ ]:


X = sequence.pad_sequences(X, maxlen = max_len)
X_test = sequence.pad_sequences(test, maxlen = max_len)


# In[ ]:


X[0]


# ** now, make model! **

# In[ ]:


def model():
    inp = Input(shape = (max_len, ))
    layer = Embedding(num_words, emb_size)(inp)
    layer = Bidirectional(LSTM(50, return_sequences = True, recurrent_dropout = 0.15))(layer)
    layer = GlobalMaxPool1D()(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(50, activation = 'relu')(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(6, activation = 'sigmoid')(layer)
    model = Model(inputs = inp, outputs = layer)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    return model


# In[ ]:


model = model()
model.summary()


# ** create callbacks **

# In[ ]:


file_path = 'save_best'
checkpoint = ModelCheckpoint(file_path, monitor = 'val_loss', verbose = 1, save_best_only=True)
early_stop = EarlyStopping(monitor = 'val_loss', patience = 1)


# In[ ]:


hist = model.fit(X, y, batch_size = 32, epochs = 2, validation_split = 0.2, callbacks = [checkpoint, early_stop])


# In[ ]:


vloss = hist.history['val_loss']
loss = hist.history['loss']

x_len = np.arange(len(loss))

plt.plot(x_len, vloss, marker='.', c='red', label='vloss')
plt.plot(x_len, loss, marker='.', c='blue', label='loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()


# In[ ]:


vacc = hist.history['val_acc']
acc = hist.history['acc']

x_len = np.arange(len(vacc))

plt.plot(x_len, vacc, marker='.', c='red', label='vacc')
plt.plot(x_len, acc, marker='.', c='blue', label='acc')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()


# In[ ]:


y_test = model.predict(X_test)


# In[ ]:


subm[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_test

subm.to_csv("sub.csv", index=False)


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
import seaborn as sns
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from sklearn.model_selection import train_test_split
from keras.models import Model
from sklearn.preprocessing import LabelEncoder
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping


# load data using pandas

# In[ ]:


data = pd.read_csv('../input/spam.csv', delimiter=",", encoding="latin-1")
data.head()


# drop unnecessary columns

# In[ ]:


data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)
data.head()


# In[ ]:


data.columns = ['category', 'content']


# In[ ]:


data['category'].value_counts().plot(kind = 'bar')


# In[ ]:


sns.countplot(data['category'])


# ** create input, output vector and split data **

# In[ ]:


X = data.content
y = data.category


# In[ ]:


le = LabelEncoder()
y = le.fit_transform(y)
print(y)


# In[ ]:


y = y.reshape(-1,1)


# In[ ]:


print(y[:5])


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)


# ** preprocessing data **
# 
# - tokenize the data and convert text to sequence
# - add padding
# - set max length and max words

# In[ ]:


train_length = X.apply(len)
train_length.head()


# In[ ]:


plt.figure(figsize = (12, 5))
plt.hist(train_length, bins = 50, alpha = 0.5, color = 'r')
plt.show()


# In[ ]:


print(np.mean(train_length))
print(np.std(train_length))
print(np.percentile(train_length, 75))
print(np.percentile(train_length, 80))


# In[ ]:


max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)


# In[ ]:


sequences = tok.texts_to_sequences(X_train)
print(sequences[:1])


# In[ ]:


sequences_matrix = sequence.pad_sequences(sequences, maxlen = max_len)


# In[ ]:


sequences_matrix[0]


# In[ ]:


for key, value in tok.word_index.items():
    if value > 10:break
    print(value, " : ", key)


# In[ ]:


def model():
    inputs = Input(name = 'input', shape = [max_len])
    layer = Embedding(max_words, 50, input_length = max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256, name = 'hidden')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1, name='output')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs = inputs, outputs = layer)
    return model


# In[ ]:


model = model()
model.summary()


# In[ ]:


model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics =['accuracy'])


# In[ ]:


hist = model.fit(sequences_matrix, y_train, batch_size = 128, epochs = 10, validation_split = 0.2, callbacks = [EarlyStopping(monitor = 'val_loss', patience = 2)])


# In[ ]:


test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)


# In[ ]:


acc = model.evaluate(test_sequences_matrix, y_test)


# In[ ]:


print("loss : %.2f,  acc : %.2f" %(acc[0], acc[1]))


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





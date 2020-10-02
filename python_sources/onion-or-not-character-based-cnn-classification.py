#!/usr/bin/env python
# coding: utf-8

# After reviewing a few of the other onion-or-not kernels, I figured I'd try a character based approach. Instead of cleaning the text and tokenizing each word, I tried tokenizing each character, then fed the integer sequences to a CNN to see what results I got. I consistently achieved over 80% accuracy, which is not bad for such a simple approach.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow import keras
from keras.models import Sequential
from keras.layers import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/onion-or-not/OnionOrNot.csv')


# In[ ]:


df.head()


# In[ ]:


# Here we tokenize each character (rather than each word).
tokenize = Tokenizer(char_level=True)
tokenize.fit_on_texts(df.text)


# In[ ]:


X = pad_sequences(tokenize.texts_to_sequences(df.text), maxlen=250, padding="post")
Y = df.label


# In[ ]:



model = Sequential([
                   Embedding(len(tokenize.word_index) + 1, 64),
                   Conv1D(64, 5, activation="relu"),
                   Conv1D(64, 5, activation="relu"),
                   GlobalMaxPooling1D(),
                   Dense(64, activation="relu"),
                   Dropout(.25),
                   Dense(16, activation="relu"),
                   Dropout(.25),
                   Dense(2, activation="softmax"),
])


# In[ ]:


model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])


# In[ ]:


# split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.1, random_state=5)


# In[ ]:


history = model.fit(x_train, y_train, validation_data=([x_test, y_test]), epochs=5, verbose=1)


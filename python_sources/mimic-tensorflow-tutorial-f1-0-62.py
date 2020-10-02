#!/usr/bin/env python
# coding: utf-8

# Simple text classification, following example https://www.tensorflow.org/tutorials/keras/basic_text_classification
# The idea is to train an embedding layer, and use average1D to average the latent embedding vectors of the words in each sentence. 
# 
# Also borrowed code for checking f1 score from https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings
# 
# Potential improving point.
# 
# 1. Dropout
# 2. Regularization
# 3. More complex structure
# 4. GRU
# 5. Capsule
# 6. Attention
# 7. Stacking

# | | f1 (validation) | f1 (public LB) |
# |---|---|---|
# |without any RNN layer |0.626|0.613|
# |one Bidirectional LSTM |0.633|0.61|
# |one Bi-LSTM w dropout |0.62|0.61|
# |one Bi-LSTM w dropout, regularization  |0.62|0.61|

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import itertools
import tensorflow as tf
from tensorflow import keras
import numpy as np

print(tf.__version__)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


STRING_LENGTH_MAX =256


# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Train shape : ",train_df.shape)
print("Test shape : ",test_df.shape)


# In[ ]:


train_df['question_text'].apply(len).max()


# In[ ]:


train_df.head(3)


# In[ ]:


# train = train_df.sample(10000)
train = train_df.sample(1000000)
x_train = train['question_text']
y_train = train['target']


# In[ ]:


import re

x_list_words = [re.findall(r'\w+',x) for x in x_train.values]
vocab = set( itertools.chain.from_iterable(x_list_words) )
word_to_index = {w:(i+3) for i,w in enumerate(vocab)}
word_to_index["<PAD>"] = 0
word_to_index["<START>"] = 1
word_to_index["<UNK>"] = 2  # unknown
word_to_index["<UNUSED>"] = 3
index_to_word = {v:k for k,v in word_to_index.items()}
len(index_to_word)


# In[ ]:


x_train_int = [
    [word_to_index[w] for w in words]
    for words in x_list_words
]


# In[ ]:


validate = train_df.loc[~train_df.index.isin(train.index)]
# validate = train_df.loc[~train_df.index.isin(train.index)].sample(10000)
x_val = validate['question_text']
y_val = validate['target']

def encode_x(x):
    x_list_words = [re.findall(r'\w+',x) for x in x.values]
    x_list_int = [
        [word_to_index.get(w,2) for w in words]
        for words in x_list_words
    ]
    return x_list_int

x_val_int = encode_x(x_val)


# In[ ]:


def decode_ints(list_ints):
    return ' '.join([index_to_word.get(i, '?') for i in list_ints])


# In[ ]:


decode_ints(x_val_int[0])


# In[ ]:


x_train.apply(len).max()


# In[ ]:


x_val.apply(len).max()


# Simple text classification, following example https://www.tensorflow.org/tutorials/keras/basic_text_classification
# The idea is to train an embedding layer, and use average1D

# In[ ]:


train_data = keras.preprocessing.sequence.pad_sequences(x_train_int,
                                                        value=word_to_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=STRING_LENGTH_MAX)

test_data = keras.preprocessing.sequence.pad_sequences(x_val_int,
                                                       value=word_to_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=STRING_LENGTH_MAX)


# In[ ]:


test_data.shape


# In[ ]:


# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = len(word_to_index)+1

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 64, input_length=STRING_LENGTH_MAX))
model.add(keras.layers.Bidirectional(keras.layers.CuDNNLSTM(64)))
model.add(keras.layers.Dropout(0.3))
# model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()


# In[ ]:


model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# To save time on commit, only took a small sample. If using all training data, the best f1 score is 0.626. 

# In[ ]:


early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=0,
                              verbose=0, mode='auto')

history = model.fit(train_data,
                y_train,
                epochs=100,
                batch_size=1024,
                validation_data=(test_data, y_val),
                verbose=1,
                callbacks=[early_stop,])


# In[ ]:


from sklearn import metrics
y_pred = model.predict(test_data, batch_size=2048, verbose=1)
for thresh in np.arange(0.1, 0.9, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(y_val, (y_pred>thresh).astype(int))))


# Make predictions

# In[ ]:


x_test = test_df['question_text']
x_test_int = encode_x(x_test)
sub_data = keras.preprocessing.sequence.pad_sequences(x_test_int,
                                                       value=word_to_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=STRING_LENGTH_MAX)
pred_val_y = model.predict([sub_data], batch_size=1024, verbose=0)


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub.prediction = pred_val_y > 0.245
sub.to_csv("submission.csv", index=False)


# In[ ]:





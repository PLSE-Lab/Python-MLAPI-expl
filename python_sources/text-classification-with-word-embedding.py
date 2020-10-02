#!/usr/bin/env python
# coding: utf-8

# ## Text Classification with Word Embedding  :
# - befor see this notebook you can see basic text classification : 
# https://www.kaggle.com/salahuddinemr/basic-text-classification
# 
# ### in this notebook we will see :
# - model without word embedding  vs model with word embedding 
# - for word embedding larg size and small size for perceptrons and vocabulary 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
get_ipython().system('pip install tensorflow==2.0.0-rc1')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf
from tensorflow import keras



print(tf.__version__)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


imdb = keras.datasets.imdb


# # 1-Model without Embedding Layer

# ### Prepare Data with zeros

# In[ ]:


# Prepare Data 
NUM_WORDS = 10000

(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)

def multi_hot_sequences(sequences, dimension):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0  # set specific indices of results[i] to 1s
    return results


train_zeros = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_zeros = multi_hot_sequences(test_data, dimension=NUM_WORDS)


# In[ ]:


# input shape is the vocabulary count used for the movie reviews (10000 words)

without_embedding= keras.Sequential([
    # `input_shape` is only required here so that `.summary` works.
    keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(NUM_WORDS,)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

without_embedding.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy', 'binary_crossentropy'])

without_embedding.summary()
history_without_embedding = without_embedding.fit(train_zeros,
                                      train_labels,
                                      epochs=20,
                                      batch_size=512,
                                      validation_data=(test_zeros, test_labels),
                                      verbose=2)
without_embedding_result = without_embedding.evaluate(test_zeros, test_labels)


# In[ ]:


print(without_embedding_result)


# # 2- Model with Embedding Layer

# ### Prepare Data with pad sequence

# In[ ]:


word_index = imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0         # PAD words  int = 0   
word_index["<START>"] = 1       # the start of text  =  int =1 
word_index["<UNK>"] = 2         # unknown words  int =2 
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)


# In[ ]:


decode_review(train_data[0])


# ## Build Model

# In[ ]:


embedding_model = keras.Sequential()
embedding_model.add(keras.layers.Embedding(10000, 64))
embedding_model.add(keras.layers.GlobalAveragePooling1D())
embedding_model.add(keras.layers.Dense(64, activation=tf.nn.relu))
embedding_model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

embedding_model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy', 'binary_crossentropy'])

embedding_model.summary()
history_embedding_model = embedding_model.fit(train_data,
                                      train_labels,
                                      epochs=20,
                                      batch_size=512,
                                      validation_data=(test_data, test_labels),
                                      verbose=2)
embedding_model_result = embedding_model.evaluate(test_data, test_labels)


# In[ ]:


print(embedding_model_result)


# # Now let change the word embedding size

# ## Small word embedding size

# In[ ]:


# embe
embedding_small = keras.Sequential()
embedding_small.add(keras.layers.Embedding(10000, 16))
embedding_small.add(keras.layers.GlobalAveragePooling1D())
embedding_small.add(keras.layers.Dense(16, activation=tf.nn.relu))
embedding_small.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

embedding_small.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy', 'binary_crossentropy'])

embedding_small.summary()
history_embedding_small = embedding_small.fit(train_data,
                                      train_labels,
                                      epochs=20,
                                      batch_size=512,
                                      validation_data=(test_data, test_labels),
                                      verbose=2)
embedding_small_result = embedding_small.evaluate(test_data, test_labels)


# In[ ]:


print(embedding_small_result)


# # Small Vocabulary size

# In[ ]:


(sm_train_data, sm_train_labels), (sm_test_data, sm_test_labels) = keras.datasets.imdb.load_data(num_words=1000)
sm_train_data = keras.preprocessing.sequence.pad_sequences(sm_train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

sm_test_data = keras.preprocessing.sequence.pad_sequences(sm_test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)


# In[ ]:



small_vocab = keras.Sequential()
small_vocab.add(keras.layers.Embedding(1000, 16))              # put vocabulary size 1000 instead of 10000
small_vocab.add(keras.layers.GlobalAveragePooling1D())
small_vocab.add(keras.layers.Dense(16, activation=tf.nn.relu))
small_vocab.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

small_vocab.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy', 'binary_crossentropy'])

small_vocab.summary()
history_small_vocab = small_vocab.fit(sm_train_data,
                                      sm_train_labels,
                                      epochs=20,
                                      batch_size=512,
                                      validation_data=(sm_test_data, sm_test_labels),
                                      verbose=2)
embedding_small_vocab = small_vocab.evaluate(sm_test_data, sm_test_labels)


# In[ ]:


print(embedding_small_vocab)


# # Plot Results 

# In[ ]:


import matplotlib.pyplot as plt

def plot_history(histories, key='binary_crossentropy'):
  plt.figure(figsize=(16,10))

  for name, history in histories:
    val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  plt.legend()

  plt.xlim([0,max(history.epoch)])


# In[ ]:


def plot_history_acc(histories, key='accuracy'):
  plt.figure(figsize=(16,10))

  for name, history in histories:
    val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  plt.legend()

  plt.xlim([0,max(history.epoch)])


# # Model without word Embedding vs word Embedding model

# ## binary cross entropy

# In[ ]:


plot_history([('Without word Embedding ', history_without_embedding),
              ('with word Embedding ', history_embedding_model)])


# ### the model with word embedding is more effictive than model without embedding

# ## accuracy 

# In[ ]:


plot_history_acc([('Without word Embedding ', history_without_embedding),
              ('with word Embedding ', history_embedding_model)])


# # Word Embedding small size VS Big Size

# ## binary cross entropy

# In[ ]:


plot_history([('word Embedding small = 16', history_embedding_small),
              ('word Embedding big = 64 ', history_embedding_model)])


# ### word embedding with small size better than big size 

# ## accuracy

# In[ ]:


plot_history_acc([('word Embedding small = 16', history_embedding_small),
              ('word Embedding big = 64 ', history_embedding_model)])


# ### model with small size embedding is more accurate

# # Vocabulary Size : 10000 VS 1000

# ## binary cross entropy

# In[ ]:


plot_history([('Vocabulary size = 10000', history_embedding_small),
              ('Vocabulary size = 1000 ', history_small_vocab)])


# ## accuracy

# In[ ]:


plot_history_acc([('Vocabulary size = 10000', history_embedding_small),
              ('Vocabulary size = 1000 ', history_small_vocab)])


# ### the model with bigest vocabulary size is more accurate 

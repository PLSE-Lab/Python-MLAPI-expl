#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


pip install tensorflow==2.0.0-rc1


# In[ ]:


import tensorflow as tf

from tensorflow import keras


# In[ ]:


import numpy as np
import pandas as pd


print(tf.__version__)


# In[ ]:


imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


# In[ ]:


print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))


# In[ ]:


print(train_data[0])


# In[ ]:


len(train_data[0]), len(train_data[1])


# In[ ]:


# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# In[ ]:


decode_review(train_data[0])


# In[ ]:


train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)


# In[ ]:


len(train_data[0]), len(train_data[1])


# In[ ]:


print(train_data[0])


# # Original Model

# In[ ]:


# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()


# In[ ]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])


# In[ ]:


x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]


# In[ ]:


history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)


# In[ ]:


results = model.evaluate(test_data, test_labels)

print(results)


# In[ ]:


history_dict = history.history
history_dict.keys()


# In[ ]:


import matplotlib.pyplot as plt

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[ ]:


plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# # Model without Embedding Layer

# In[ ]:


# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model1 = keras.Sequential()
#model1.add(keras.layers.Embedding(vocab_size, 16))
model1.add(keras.layers.GlobalAveragePooling1D())
model1.add(keras.layers.Dense(16, activation=tf.nn.relu))
model1.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model1.summary()


# # As we show, the embedding layer support model to be bulit and without it ,  model will not work

# # without Embedding word

# In[ ]:


# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model1 = keras.Sequential()
model1.add(keras.layers(vocab_size, 16))
model1.add(keras.layers.GlobalAveragePooling1D())
model1.add(keras.layers.Dense(16, activation=tf.nn.relu))
model1.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model1.summary()


# without embadding word , the TFModuleWrapper object which using for make gragh not callable

# # with different word embedding size
# # different heddin unit

# **with 4 size**

# In[ ]:


# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model2 = keras.Sequential()
model2.add(keras.layers.Embedding(vocab_size, 4))
model2.add(keras.layers.GlobalAveragePooling1D())
model2.add(keras.layers.Dense(4, activation=tf.nn.relu))
model2.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model2.summary()


# In[ ]:


model2.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])


# In[ ]:


history2 = model2.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)


# In[ ]:


results2 = model2.evaluate(test_data, test_labels)

print(results)


# In[ ]:


history_dict2 = history2.history
history_dict2.keys()


# In[ ]:


import matplotlib.pyplot as plt

acc = history_dict2['acc']
val_acc = history_dict2['val_acc']
loss = history_dict2['loss']
val_loss = history_dict2['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[ ]:


plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# **with 512 size**

# In[ ]:


# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model3 = keras.Sequential()
model3.add(keras.layers.Embedding(vocab_size, 512))
model3.add(keras.layers.GlobalAveragePooling1D())
model3.add(keras.layers.Dense(512, activation=tf.nn.relu))
model3.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model3.summary()


# In[ ]:


model3.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])


# In[ ]:


history3 = model3.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)


# In[ ]:


results3 = model3.evaluate(test_data, test_labels)

print(results)


# In[ ]:


history_dict3 = history3.history
history_dict3.keys()


# In[ ]:


import matplotlib.pyplot as plt

acc = history_dict3['acc']
val_acc = history_dict3['val_acc']
loss = history_dict3['loss']
val_loss = history_dict3['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[ ]:


plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# 
# # with different vocabulary size

# In[ ]:


vocab_size = 15000

modelV1 = keras.Sequential()
modelV1.add(keras.layers.Embedding(vocab_size, 16))
modelV1.add(keras.layers.GlobalAveragePooling1D())
modelV1.add(keras.layers.Dense(16, activation=tf.nn.relu))
modelV1.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

modelV1.summary()


# In[ ]:


modelV1.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])


# In[ ]:


x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]


# In[ ]:


historyV1 = modelV1.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)


# In[ ]:


resultsV1 = modelV1.evaluate(test_data, test_labels)

print(results)


# In[ ]:


history_dictV1 = historyV1.history
history_dictV1.keys()


# In[ ]:


import matplotlib.pyplot as plt

acc = history_dictV1['acc']
val_acc = history_dictV1['val_acc']
loss = history_dictV1['loss']
val_loss = history_dictV1['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[ ]:


plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# In[ ]:


imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=17000)


# In[ ]:


vocab_size = 17000

modelV2 = keras.Sequential()
modelV2.add(keras.layers.Embedding(vocab_size, 16))
modelV2.add(keras.layers.GlobalAveragePooling1D())
modelV2.add(keras.layers.Dense(16, activation=tf.nn.relu))
modelV2.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

modelV2.summary()


# In[ ]:


modelV2.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])


# In[ ]:


historyV2 = modelV2.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)


# In[ ]:


#resultsV2 = modelV2.evaluate(test_data, test_labels)

#print(results)


# In[ ]:


history_dictV2 = historyV2.history
history_dictV2.keys()


# In[ ]:


import matplotlib.pyplot as plt

acc = history_dictV2['acc']
val_acc = history_dictV2['val_acc']
loss = history_dictV2['loss']
val_loss = history_dictV2['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[ ]:


plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# In[ ]:





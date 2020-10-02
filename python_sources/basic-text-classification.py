#!/usr/bin/env python
# coding: utf-8

# # Basic Text Classification 
# In this notebook we will use the tensorflow , keras to classify text data set from IMDB reviews.
# we will use machine learning on reviews  to determine if it's negative or positive ? (Binary Classification).
# 

# - for more information about text classification 
# https://monkeylearn.com/text-classification/

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



get_ipython().system('pip install tensorflow==2.0.0-rc1')
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


# The **IMDB** dataset that contains the text of **50,000** movie reviews from the Internet Movie Database. 
# 
# These are split into **25,000** reviews for training and **25,000** reviews for testing. The training and testing sets are balanced, meaning they contain an equal number of positive and negative reviews.
# The **IMDB** dataset comes **packaged with TensorFlow**. **It has already been preprocessed** such that the reviews (**sequences of words**) have been converted to sequences of integers, where each integer represents a specific word in a dictionary.

# # Step 1 : Download and Split data set 

# In[ ]:


imdb = keras.datasets.imdb 

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000) 


# In[ ]:


print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))


# ### As you can see below the review is sequence of words have been converted to sequences of integers. 

# # Step 2: Explore Data 

# In[ ]:


print(train_data[0])


# In[ ]:


len(train_data[0]), len(train_data[1]) 


# In[ ]:


word_index = imdb.get_word_index()


# ### using get_word_index() we can see dictinary from word(key) and its value(integer) 

# In[ ]:


word_index


# # Step 3: Prepare the Data

# In[ ]:


# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0         # PAD words  int = 0   
word_index["<START>"] = 1       # the start of text  =  int =1 
word_index["<UNK>"] = 2         # unknown words  int =2 
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


# # Step 4: Build the Model

# In[ ]:


# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()


# # Step 5: Compile 

# In[ ]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc','binary_crossentropy'])


# In[ ]:


x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]


# # Step 6: Fit 

# In[ ]:



history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=2)


# # Step 7: Evaluate

# In[ ]:


results = model.evaluate(test_data, test_labels)


# In[ ]:


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


#  ##  from accuracy plot you can see that the training accuracy stay increase with epochs but the validation accuracy become constant, This is what we call the Overfitting

# # Early Stopping: 
# ## to avoid overfitting caused by epochs increasing we use the Early Stopping to stop  learning model at overfitting area

# In[ ]:


model_Estop = keras.Sequential()
model_Estop.add(keras.layers.Embedding(vocab_size, 16))
model_Estop.add(keras.layers.GlobalAveragePooling1D())
model_Estop.add(keras.layers.Dense(16, activation=tf.nn.relu))
model_Estop.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model_Estop.summary()


# In[ ]:


model_Estop.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc','binary_crossentropy'])


# ## At fit model we activate the early stopping in callback  
# ### patience : No of epochs after overfitting 

# In[ ]:


callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3) ## activate early stopping
history_Estop = model_Estop.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,callbacks=[callback],                   ## activate the call back                  
                    validation_data=(x_val, y_val),
                    verbose=2)


# - you can see that the model stoped at 30 epochs 

# In[ ]:


results_Estop = model_Estop.evaluate(test_data, test_labels)


# In[ ]:


print(results_Estop)


# In[ ]:


history_Estop = history_Estop.history
history_Estop.keys()


# In[ ]:


plt.clf()   # clear figure
acc = history_Estop['acc']
val_acc = history_Estop['val_acc']
loss = history_Estop['loss']
val_loss = history_Estop['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# ##   the model stopped its training when validation accuracy became constant at 30 epochs

# ## You can see word embedding in text classification in this notebook:
# https://www.kaggle.com/salahuddinemr/text-classification-with-word-embedding
# 

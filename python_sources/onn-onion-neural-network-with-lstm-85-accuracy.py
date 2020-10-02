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
for dirname, _, filenames in os.walk('/kaggle/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # WHY?
# 
# First of all, I want to introduce what's so called 'onion'. Of course, a kind of cooking ingredient (although one of my friend treat it like a fruit). A story could be so utopiad, then crowds don't believe it, in spite of the fact that it happened. Later, I will show you about the story. 
# 
# The onion articles labeled as 1, and the r/NotTheOnion articles are labeled 0.

# In[ ]:


data = pd.read_csv('/kaggle/input/onion-or-not/OnionOrNot.csv')


# In[ ]:


data.shape


# ## Let's have a look

# In[ ]:


data.head(10)


# ### Lets see this onion

# In[ ]:


display(data['text'][0])
display(data['text'][2])
display(data['text'][5])
display(data['text'][9])


# ### And this is r/NotTheOnion

# In[ ]:


display(data['text'][1])
display(data['text'][6])
display(data['text'][7])


# ### See the differences? they were awful, right? 

# In[ ]:


import matplotlib.pyplot as plt
labels = [1,0]
plt.pie(data['label'].value_counts() )
plt.legend(labels)
plt.title('Onions or not')


# ## Preparing the library and Preprocessing
# 
# The ONN was based on NLP with DNN approach.
# 
# 1. First, we will remove punctuation.
# 2. Second, decapitalize letters
# 3. Tokenization

# ### 1. Remove Puctuation

# In[ ]:


import re


# In[ ]:


data_process = data.copy()


# In[ ]:


data_process.head()


# In[ ]:


data_process["text"] = data_process["text"].str.replace('[^a-zA-Z]', ' ', regex=True)


# In[ ]:


display(data['text'][2])
display(data_process['text'][2])


# ### 2. Turn to lowercase

# In[ ]:


data_process["text"] = data_process["text"].str.lower()


# In[ ]:


display(data['text'][2])
display(data_process['text'][2])


# ### 3. Tokenization

# In[ ]:


from keras.preprocessing.text import Tokenizer


# In[ ]:


num_words = 20000
max_len = 150
emb_size = 128
X = data_process["text"]


# In[ ]:


X


# In[ ]:


token = Tokenizer(num_words = num_words)
token.fit_on_texts(list(X))


# In[ ]:


X = token.texts_to_sequences(X)


# In[ ]:


display(data['text'][2])
display(data_process['text'][2])
display(X[2])


# In[ ]:


plt.plot(X[2], label = 'sample text spectra')
plt.title(str(data['text'][2]))


# In[ ]:


from keras.preprocessing import sequence

X = sequence.pad_sequences(X, maxlen = 150)
y = pd.get_dummies(data_process['label'])


# In[ ]:


y = y.values


# I got the feature extraction notes on https://www.kaggle.com/nihalbey/spam-detection-and-deep-nlp . I was stuck before wondering how to be done with those dataframe

# ## NLP
# 
# Finally this part came, lets start!
# 
# The traditional LSTM was time consuming, therefore the bidirectional used and combined with Dense layer for time saving, thanks to this kernel https://www.kaggle.com/lsjsj92/toxic-nlp-with-keras-lstm#648911 . 

# ### Separating train and test data

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.1, random_state = 42)


# In[ ]:


import nltk
from keras.models import Sequential
from keras.layers import Input,Dense, LSTM, Dropout, Flatten, Embedding, Bidirectional, GlobalMaxPool1D


# In[ ]:


def model():
    
    inp = Input(shape = (max_len, ))
    layer = Embedding(num_words, emb_size)(inp)
    layer = Bidirectional(LSTM(50, return_sequences = True, recurrent_dropout = 0.1))(layer)
    
    layer = GlobalMaxPool1D()(layer)
    layer = Dropout(0.2)(layer)
    
    
    layer = Dense(50, activation = 'relu')(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(50, activation = 'relu')(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(50, activation = 'relu')(layer)
    layer = Dropout(0.2)(layer)
    
    
    layer = Dense(2, activation = 'softmax')(layer)
    model = Model(inputs = inp, outputs = layer)
    
    
    model.compile(loss = 'binary_crossentropy', optimizer = 'nadam', metrics=['accuracy'])
    return model


# In[ ]:


from keras.models import Model
from keras.utils import plot_model
import matplotlib.image as mpimg

model = model()
model.summary()

plot_model(model, to_file='onion.png',show_shapes=True, show_layer_names=True)
plt.figure(figsize = (30,20))
img = mpimg.imread('/kaggle/working/onion.png')
imgplot = plt.imshow(img)


# In[ ]:


from keras.callbacks import EarlyStopping, ModelCheckpoint

file_path = 'save.hd5'
checkpoint = ModelCheckpoint(file_path, monitor = 'val_loss', save_best_only=True)
early_stop = EarlyStopping(monitor = 'loss', patience = 1)


# ### Fitting model

# In[ ]:


history = model.fit(x_train, y_train, batch_size = 32, epochs = 3, validation_split = 0.1, callbacks = [checkpoint,early_stop])


# Limit the time, and save the model. 
# 
# How to save and load if interrupted: https://www.kaggle.com/danmoller/make-best-use-of-a-kernel-s-limited-uptime-keras

# #### Prepare yourself, it will be long, so its better to save the model

# ## Lets see the prediction result!

# In[ ]:


val_loss = history.history['val_loss']
loss = history.history['loss']


# In[ ]:


print('validation loss: ', val_loss[-1])
print('training loss: ', loss[-1])


# In[ ]:


score = model.evaluate(x_test, y_test)
print(model.metrics_names)
print(score)
print('test loss: ', score[0])
print('test accuracy: ', score[1])


# In[ ]:


history.history


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568


# Model selection is the task of selecting a statistical model from a set of candidate models, given data. In the simplest cases, a pre-existing set of data is considered. Given candidate models of similar predictive or explanatory power, the simplest model is most likely to be the best choice.

# The data is available in Google BigQuery that can be downloaded from here. The data is also publicly available at this Cloud Storage URL: https://storage.googleapis.com/tensorflow-workshop-examples/stack-overflow-data.csv.

# In[ ]:


import logging
import pandas as pd
import numpy as np
from numpy import random
import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
# from bs4 import BeautifulSoup

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_train = pd.read_csv('../input/train.csv')
# df = df[pd.notnull(df['tags'])]
df_train.head(10)


# In[ ]:


# df['post'].apply(lambda x: len(x.split(' '))).sum()


# We have over 10 million words in the data.

# In[ ]:


plt.figure(figsize=(10,4))
df_train.Category.value_counts().plot(kind='bar');


# The classes are NOT very well balanced.

# ### BOW with keras

# In[ ]:


import itertools
import os

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils


# In[ ]:


from keras import backend as K
K.tensorflow_backend._get_available_gpus()


# In[ ]:


train_posts, test_posts, train_tags, test_tags = train_test_split(df_train['title'], 
                                                                  df_train['Category'], 
                                                                  test_size=0.001, 
                                                                  random_state=42)


# In[ ]:


max_words = 120
tokenize = text.Tokenizer(num_words=max_words, char_level=False)


# In[ ]:


tokenize.fit_on_texts(train_posts) # only fit on train
x_train = tokenize.texts_to_matrix(train_posts)
x_test = tokenize.texts_to_matrix(test_posts)


# In[ ]:


encoder = LabelEncoder()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)


# In[ ]:


num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)


# In[ ]:


print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)


# In[ ]:


batch_size = 32
epochs = 2


# In[ ]:


# Build the model
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[ ]:


history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)


# In[ ]:


score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test accuracy:', score[1])


# In[ ]:


df_public = pd.read_csv('../input/test.csv')
df_public.head(10)


# In[ ]:


x_public = tokenize.texts_to_matrix(df_public["title"])


# In[ ]:


preds = model.predict(x_public, verbose=1)


# In[ ]:


df_public['Category'] = [np.argmax(pred) for pred in preds]
df_submit = df_public[['itemid', 'Category']].copy()
df_submit.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # Fake News Classifier with Deep Learning with Embedding Layer

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


# # Imports

# In[ ]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# # Load Data

# In[ ]:


df = pd.read_csv('../input/textdb3/fake_or_real_news.csv') # Load data into DataFrame


# # Pre-processing

# In[ ]:


# Pre-Processing
df['text'] = df['text'].apply(lambda x: x.lower())


# # Tokenization

# In[ ]:



max_features = 2000 # Vocabulary Size

tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(df['text'].values)
X = tokenizer.texts_to_sequences(df['text'].values)


# # Padding

# In[ ]:


max_length = 1000
# Padding
X = pad_sequences(X,maxlen = max_length, padding = 'post')


# # Processing Target

# In[ ]:


#y = df.label
y = pd.get_dummies(df['label']).values


# # Train-Test Split

# In[ ]:



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=53)

print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)


# # Design Deep Neural Network with Embedding Layer

# In[ ]:


# define the model
model = Sequential()
model.add(Embedding(max_features, 24, input_length=max_length))
model.add(Flatten())
model.add(Dense(2, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model.summary())


# # Training

# In[ ]:


# fit the model
model.fit(X_train, y_train, epochs=50, verbose=0)


# # Evaluation

# In[ ]:


# evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %f' % (accuracy*100))


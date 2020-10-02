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
from keras.utils.np_utils import to_categorical

# load data
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
Y_train = train["label"]

# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1) 


# In[ ]:


X_train = X_train / 255.0
test = test / 255.0


# In[ ]:


timesteps = 28
data_dim = 28

X_train = X_train.values.reshape(-1, timesteps, data_dim)
test = test.values.reshape(-1, timesteps, data_dim)


# In[ ]:


Y_train = to_categorical(Y_train, num_classes = 10)


# In[ ]:


from keras.layers import LSTM, Dropout, Dense, Flatten, Embedding
from keras.models import Sequential

# the number of neurons
n_neurons = 150

model = Sequential()
# LSTM layer1
model.add(LSTM(n_neurons, return_sequences=True, input_shape=(timesteps, data_dim)))
model.add(Dropout(0.25))

# LSTM layer2
model.add(LSTM(n_neurons, return_sequences=True))
model.add(Dropout(0.25))

# LSTM layer3
model.add(LSTM(n_neurons))
model.add(Dropout(0.25))

# FC layer
model.add(Dense(784, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=10, batch_size=150, validation_data=(X_train, Y_train))


# In[ ]:


# predict results
results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")


# In[ ]:


submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("lstm_mnist.csv",index=False)


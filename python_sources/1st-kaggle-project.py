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


from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils


# In[ ]:


np.random.seed(100) #for reproducibility..,
batch_size = 128 # number of images that we use in each epoch
nb_classes = 10 # one class for each digit(0 -- 9)
nb_epoch = 20 # number of times we train the whole data


# # load data

# In[ ]:


(X_train, Y_train),(X_test, Y_test) = mnist.load_data()


# # we flatten data , because mlp doesen't use the 2d strusture of data

# In[ ]:


X_train = X_train.reshape(60000, 784) #60,000 digit images
X_test = X_test.reshape(10000, 784)


# we convert train and test data to float32

# In[ ]:


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# # now we have to normalized the data (by Z- SCORE)
# 
# 

# In[ ]:


X_train = (X_train- np.mean(X_train))/np.std(X_train)
X_test = (X_test- np.mean(X_test))/np.std(X_test)


# # display the number of train and test samples present in the data

# In[ ]:


print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# # now we convert class vectors to binary class metrics 

# In[ ]:


Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)


# # now we define our model()

# In[ ]:


model = Sequential()
model.add(Dense(512, input_shape=(784, )))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(120))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))


# # now its time to use an optimizer function to optimize loss function

# In[ ]:


rms = RMSprop()


# # now we compile our network and add metrics for knowing accuracy

# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer = rms, metrics = ["accuracy"])


# In[ ]:


# now we train our model for real!!!


# In[ ]:


model.fit(X_train, Y_train,
batch_size = batch_size, nb_epoch = nb_epoch,
verbose=2,
validation_data = (X_test, Y_test))


# In[ ]:





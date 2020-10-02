#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.optimizers import SGD,RMSprop,Adam
from keras.utils import np_utils
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(1671) # for reproducibility

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


NB_EPOCH=20
BATCH_SIZE=128
VERBOSE=1
NB_CLASSES=10 # number of outputs = number of digits
#OPTIMIZER=SGD() # SGD Optimizer
#OPTIMIZER=RMSprop()
OPTIMIZER=Adam()
N_HIDDEN=128
VALIDATION_SPLIT=0.2 # how much TRAIN is reserved for VALIDATION
DROPOUT=0.3


# In[ ]:


def load_data(path):
  with np.load(path) as f:
      x_train, y_train = f['x_train'], f['y_train']
      x_test, y_test = f['x_test'], f['y_test']
      return (x_train, y_train), (x_test, y_test)


# In[ ]:


#data Shuffled and split between train and test sets
(X_train,y_train),(X_test,y_test)=load_data("../input/mnist.npz")


# In[ ]:


RESHAPED=784
X_train=X_train.reshape(60000,RESHAPED)
X_test=X_test.reshape(10000,RESHAPED)
X_train=X_train.astype('float32')
X_test=X_test.astype('float32')


# In[ ]:


#normalize
X_train/=255
X_test/=255

print(X_train.shape[0],'train samples')
print(X_test.shape[0],'test samples')

#Comvert class Vectors to binary class matrices

Y_train=np_utils.to_categorical(y_train,NB_CLASSES)
Y_test=np_utils.to_categorical(y_test,NB_CLASSES)


# In[ ]:


model= Sequential()
model.add(Dense(N_HIDDEN,input_shape=(RESHAPED,)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()


# In[ ]:


example_index=221
plt.figure()
_=plt.imshow(np.reshape(X_train[example_index,:],(28,28)),"gray")


# In[ ]:


model.compile(loss='categorical_crossentropy',optimizer=OPTIMIZER,metrics=['accuracy'])


# In[ ]:


history=model.fit(X_train,Y_train,batch_size=BATCH_SIZE,epochs=NB_EPOCH,verbose=VERBOSE,validation_split=VALIDATION_SPLIT)


# In[ ]:


score =model.evaluate(X_test,Y_test,verbose=VERBOSE)
print("Test Score",score[0])
print('Test Accuracy',score[1])


# In[ ]:





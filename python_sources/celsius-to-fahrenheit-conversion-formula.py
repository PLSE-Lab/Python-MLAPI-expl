#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import python packages
import keras
from keras.models import Sequential
from keras.layers import Dense
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# Any results you write to the current directory are saved as output.


# In[3]:


celsius_q    = [-40, -10,  0,  8, 15, 22,  38]
fahrenheit_a = [-40,  14, 32, 46, 59, 72, 100]


# In[4]:


#data set x for celsius,y for fehrenheit
X=np.array(celsius_q ,dtype=float)
Y=np.array(fahrenheit_a,dtype=float)


# In[5]:


#building model
model=Sequential([Dense(input_shape=([1]),units=8)])
model.add(Dense(units=4))
model.add(Dense(units=2))
model.add(Dense(units=1))


# In[7]:


model.compile(loss=keras.losses.mean_squared_error,
             optimizer=keras.optimizers.Adam(0.01),
             metrics=['accuracy'])


# In[8]:


history=model.fit(X,Y,
                 epochs=10)


# In[9]:


history.history.keys()


# In[10]:


import matplotlib.pyplot as plt
plt.xlabel("Epoch number")
plt.ylabel("loss")
plt.title("loss vs number of Epochs")
plt.plot(history.history['loss']);


# In[11]:


#prediction
model.predict([100])


# In[ ]:





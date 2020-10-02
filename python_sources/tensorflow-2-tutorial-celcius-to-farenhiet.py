#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # In this tutorial we will try to determine the basic equation of converting celcius in farenhiet using tensorflow 2

# In[ ]:


import tensorflow as tf


# Let us define our inputs and outputs in order to train
# * array c= temperature in celcius
# * array f= temperature in farenhiet 

# In[ ]:


c=np.array([0,10,20,24,26,34,30,45,50])
f=np.array([32,50,68,75.2,78.8,93.2,86,113,122])


# In[ ]:


model=tf.keras.Sequential([ tf.keras.layers.Dense(units=1,input_shape=[1])])
model.compile(optimizer=tf.keras.optimizers.Adam(0.4),loss='mean_squared_error')
model.fit(c,f,epochs=1000,verbose=0)


# 
# * After the traning loss is 0.0000000001

# # F=0.8 * C +32
# lets see if our model was able to predict the quation correctly

# In[ ]:


weights=model.layers[0].get_weights()[0]
weights


# In[ ]:


bias=model.layers[0].get_weights()[1]
bias


# # weight = 1.8000007
# # bias   = 31.99997

# In[ ]:





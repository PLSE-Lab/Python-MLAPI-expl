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


# # Importing the library and dataset 

# In[ ]:


import numpy as np
import tensorflow as tf
import datetime
from tensorflow.keras.datasets import fashion_mnist


# In[ ]:


tf.__version__


# # 1.Data Preprocessing

# ### Loading the data

# In[ ]:


(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


# ### 2.Normalising the data

# In[ ]:


X_train = X_train / 255.0
X_test = X_test / 255.0


# In[ ]:


print(X_train.shape)


# In[ ]:


X_train[:5]


# ### 3. Change the Shape of this array 

# In[ ]:


X_train = X_train.reshape(-1, 28*28)


# In[ ]:


print(X_test.shape)


# In[ ]:


#Reshape the testing subset in the same way
X_test = X_test.reshape(-1, 28*28)


# ### Stage 4: Building an Artificial Neural network
# 

# In[ ]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

model = Sequential()
model.add(Dense(units=128, activation='relu', input_shape= (784,)))
model.add(Dropout(0.25))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units = 10, activation = 'softmax'))


# ### Stage 4: Compiling an Artificial Neural network
# 

# In[ ]:


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.summary()


# ### Stage5: fitting the model to data

# In[ ]:


model.fit(X_train, y_train, epochs=100)


# ### Stage5: Evaluate the model to data

# In[ ]:


model.evaluate(X_test, y_test)


# In[ ]:





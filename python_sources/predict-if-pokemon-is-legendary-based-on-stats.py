#!/usr/bin/env python
# coding: utf-8

# **Function for preping the data**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(2)
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv1D, Dropout
from keras.optimizers import RMSprop

num_inputs = 6
num_classes = 2

def data_prep(raw):
    out_y = keras.utils.to_categorical(raw.Legendary, num_classes)
    batch_size = raw.shape[0]
    out_x = raw.values[...,5:11,]
    return out_x, out_y


# **Reading and preping the data**

# In[ ]:


train_file = "../input/Pokemon.csv"
raw_data = pd.read_csv(train_file)

x, y = data_prep(raw_data)


# **Building the model**

# In[ ]:


model = Sequential()
model.add(Flatten())
model.add(Dense(20, activation='relu', input_shape=(800, 6)))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))


# **Compiling the model**

# In[ ]:


model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])


# **Fitting the model**

# In[ ]:


model.fit(x, y, 
        batch_size=80, 
        epochs=5, 
        validation_split=0.2)


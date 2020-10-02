#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import ast


# In[4]:


sensor = pd.read_csv('../input/touch_events.csv').drop('doc_created_utc_milli', axis=1)


# In[5]:


y = sensor.drop(['event'], axis=1)
y = pd.get_dummies(y)
y = np.array(y)
y = y.tolist()
x = sensor['event']
x = np.array(x)


# In[6]:


for i in range(x.shape[0]):
    x[i] = ast.literal_eval(x[i])
xn = []
for arr in x:
    xn.append(arr)
x = xn


# In[7]:


xtemp = x 
ytem = y


# In[8]:


#augment
import random
for i in range(len(xtemp)):
    y.append(y[i])
    for j in range(len(xtemp[i])):
        if random.random() > .95:
            xtemp[i][j] = xtemp[i][int(random.random()*len(xtemp[i]))]
    x.append(xtemp[i])


# In[9]:


x = np.array(x)
y = np.array(y)
x.shape, y.shape


# In[10]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)


# In[11]:


from keras.models import Sequential
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU


# In[12]:


alpha=0.1
model = Sequential()
model.add(LSTM(40, input_shape=(100,8),return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(40, input_shape=(100,8),return_sequences=True))
model.add(Dropout(.2))
model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(.2))
model.add(LeakyReLU(alpha=alpha))
model.add(Dense(32))
model.add(Dropout(.2))
model.add(LeakyReLU(alpha=alpha))
model.add(Dense(8))
model.add(Dropout(.2))
model.add(LeakyReLU(alpha=alpha))
model.add(Dense(3))
model.add(Activation('softmax'))
model.summary()


# In[13]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model.fit(x_train, y_train, validation_split=.2, epochs=100, batch_size=32, verbose=1)


# In[14]:


model.evaluate(x_test, y_test)


# In[15]:


model.save('model.h5')


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score
import graphviz
from sklearn import tree
import random
from sklearn.metrics import confusion_matrix


# In[ ]:


test = pd.read_csv('../input/liverpool-ion-switching/test.csv')
train = pd.read_csv('../input/liverpool-ion-switching/train.csv')
train.head()


# In[ ]:


pd.value_counts(train['open_channels'])


# In[ ]:


train.shape


# In[ ]:


import tensorflow as tf

import numpy as np


# # Load the data 

# In[ ]:


train_val_signal = np.array(train['signal'])
train_val_y = np.array(train['open_channels'])


# # Split to train and validation

# In[ ]:


train_signal = train_val_signal[:4800000]
train_y = train_val_y[:4800000]


# In[ ]:


val_signal = train_val_signal[4800000:]
val_y = train_val_y[4800000:]


# # Scale the data

# In[ ]:


from sklearn.preprocessing import scale
train_signal = scale(train_signal, axis=0, with_mean=True, with_std=True, copy=True )


# In[ ]:


val_signal = scale(val_signal, axis=0, with_mean=True, with_std=True, copy=True )


# # Create a data generator
# This is important to load the data in patches otherwise, loading data will take long time
# 
# Here, I am using 50 signals before and 50 signals after the point we want to predict its channels. That's why I am using signal size to be 101
# 

# In[ ]:


def data_generator(data_signal, data_y, batch_size, signal_size):
    def g():
        
        start_index = random.randint(0,len(data_signal) - (signal_size + 1))
         
        x = data_signal[start_index:(start_index+signal_size)]
        y = data_y[start_index + (signal_size // 2)]
 
        return x,y
            
    while True:
        x_batch = np.zeros(shape = (batch_size,signal_size))
        y_batch = np.zeros(shape = (batch_size,1))
        for k in range(batch_size):
            x_batch[k],y_batch[k] = g()
            
        yield x_batch,y_batch
        


# In[ ]:


train_gen = data_generator(train_signal, train_y, batch_size = 200, signal_size = 101)
val_gen = data_generator(val_signal, val_y, batch_size = 200, signal_size = 101)


# In[ ]:


for x,y in val_gen:
    print(y.shape)
    break;
    


# # Build the model

# In[ ]:


from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout,BatchNormalization

inputs = Input(shape=(101,))
xx = Dense(101, activation= 'softmax')(inputs)
xx = BatchNormalization()(xx)
xx = Dense(101, activation= 'softmax')(xx)
xx = BatchNormalization()(xx)
xx = Dense(101, activation= 'softmax')(xx)


outputs = Dense(11, activation= 'softmax')(xx)

model = keras.Model(inputs=inputs, outputs=outputs)


# In[ ]:



model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
             metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


model.fit(train_gen,
            steps_per_epoch=1000,
            epochs=10)


# In[ ]:





# In[ ]:


y_hat = []
y_true = []


for x,y in train_gen:
    y_true = y_true +  list(y)
    y_hat = y_hat + list(np.argmax(model.predict(x), -1))

    break;
        
print(confusion_matrix(y_hat,y_true))


# # Reshape test in batches
# The new test should have the following shape
# 
# (10000, 200, 101)
# 
# where 10000 prediction are going to happend
# Each batch contains 200 lists for each we have 101 signals, output y will be a list of shape 200
# 
# 

# In[ ]:


test_signal = test['signal']


# In[ ]:


new_test = []
batch_size = 200
input_size = 101
start = 0
test_signal_list = list(test_signal)+ [0] * input_size
while (start+101) < len(test_signal_list):
    batch = []
    for x in range(batch_size):
        batch.append(test_signal_list[start:(start + 101)])
        #batch.append([0]*101)
        start = start + 1
    new_test.append(batch)
    if len(new_test) % 1000 == 0:
        print(len(new_test))


# In[ ]:


arrp = []
for k in range(len(new_test)):
    arrp.append(np.argmax(model.predict(np.array(new_test[k])), -1))
    if k % 200 == 0:
        print(k)


# In[ ]:


h = [i for sublist in arrp for i in sublist]
h = [0]* 50 + h
print(len(h))
for x in range((101 // 2) ):
    h.pop()


# In[ ]:





# In[ ]:


pd.value_counts(h)


# # Submition

# In[ ]:


sub = pd.read_csv('../input/liverpool-ion-switching/sample_submission.csv')


# In[ ]:


sub['open_channels'] = h


# In[ ]:


sub.to_csv('submission.csv',index=False,float_format='%.4f')


# The current score is .55 but this means the method works to some extent

# In[ ]:





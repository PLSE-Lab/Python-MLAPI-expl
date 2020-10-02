#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout # new!
from keras.layers.normalization import BatchNormalization # new!
from keras import regularizers # new! 
from keras.optimizers import SGD

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("../input/train.csv",header=0,dtype=np.float32)
test = pd.read_csv("../input/test.csv",header=0,dtype=np.float32)


# In[ ]:


train_data = train.values
x_train=train_data[1:,1:]
#x_train[0]


# In[ ]:


y_train = train_data[1:,0]
print(train_data[1:,0])  


# In[ ]:


x_train.shape


# In[ ]:


x_temp = x_train.reshape(41999,28,28) 
x_temp.shape


# In[ ]:


test_data=test.values
x_test = test_data[1:,...]
x_test.shape


# In[ ]:


x_train = tf.keras.utils.normalize(x_train,axis =1)
x_test = tf.keras.utils.normalize(x_test,axis =1)


# In[ ]:


plt.imshow(x_temp[10],cmap = plt.cm.binary)
plt.show()


# In[ ]:


model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(784,)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


# In[ ]:


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=100)


# In[ ]:


y_test=model.predict(x_test,
       batch_size=128,        
       verbose=1
       )


# In[ ]:


x_temp = x_test.reshape(27999,28,28)


# In[ ]:


#*************************** Verify the images with less than 1 probability about lable. Currently, i am displaying first 10 digits but  you can extend as needed. 
gcount = 0
for i in y_test:
    if np.float64(max(i))!=1.0:
        print(i.tolist().index((np.float64(max(i)))),np.float64(max(i)),gcount)        
        plt.imshow(x_temp[gcount],cmap = plt.cm.binary)
        plt.show()
        gcount = gcount + 1      
    if(gcount>10):
        break
    


# If you like the kernel. Please **fork** and **upvote** it. In upcoming days, I would add comments/description to explaing each command.

# I am also upload the python code under my [Git repository](https://github.com/Muhammads786/Deep-Learning). You can download the code and run at your machine as well. 

# In[ ]:





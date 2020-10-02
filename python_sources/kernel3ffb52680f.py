#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
train = pd.read_csv('../input/train.csv')
train.head()


# In[ ]:


test = pd.read_csv('../input/test.csv')
test.head()


# In[ ]:


X_train = train.drop(["label"],axis = 1) 
Y_train = train["label"]
import numpy as np 
np.array(X_train)
np.array(Y_train)


# In[ ]:


X_train.head()


# In[ ]:


Y_train.head()


# In[ ]:


Y_train[100]


# In[ ]:


X_train = X_train / 255.0
test = test / 255


# In[ ]:


from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import adam
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
model = keras.Sequential()
model.add(tf.keras.layers.Flatten())  
model.add(Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) 
model.add(Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) 


# In[ ]:


model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


model.fit(np.array(X_train),np.array(Y_train), epochs=5,validation_split=0.01,shuffle=True,batch_size=40)


# In[ ]:


test_loss, test_acc = model.evaluate(X_train,Y_train)
print('Test accuracy:', test_acc)


# In[ ]:


predictions = model.predict(test)
predictions[0]
np.argmax(predictions[0])


# In[ ]:


predictions = np.argmax(predictions,axis = 1)
predictions = pd.Series(predictions,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predictions],axis = 1)
submission.to_csv("submission_hansdah.csv",index=False)


# In[ ]:


submission.head()


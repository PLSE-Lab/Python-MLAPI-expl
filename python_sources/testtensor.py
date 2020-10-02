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


# In[3]:


#Shallow-net demo


# ### Shallownet in Keras

# Building a Shallow net in keras for digit recognizer for MNIST

# #### Set seed for reproducibility

# In[ ]:


import numpy as np
np.random.seed(42)


# #### Load Dependencies

# In[ ]:


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD


# #### Load data

# In[ ]:


import pandas as pd
#(X_train, y_train), (X_test, y_test) = mnist.load_data() Not working due to some issues


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


test.shape


# In[ ]:


train.shape


# In[ ]:


X_train = train.drop('label', axis = 1)
#X_test = test.drop('label', axis = 1)


# In[ ]:


y_train = train['label']


# In[ ]:


X_train /= 255
X_test = test/255


# In[ ]:


n_classes = 10
y_train = keras.utils.to_categorical(y_train, n_classes)


# In[ ]:





# ####  Design neural network architecture

# In[ ]:


model = Sequential()
model.add(Dense(64, activation='relu', input_dim=784))
#model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


# In[ ]:


model.summary()


# ####  Configure Model

# In[ ]:


model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


# Train the model
model.fit(X_train, y_train, epochs=50)


# In[ ]:


y_test = model.predict(X_test)


# In[ ]:


y_test


# In[ ]:


y_test[0]


# In[ ]:


y_test = y_test.argmax(axis =1 )


# In[ ]:


y_test


# In[ ]:


y_test.shape


# In[ ]:


test.head()


# In[ ]:


out = pd.DataFrame({'ImageId':np.arange(1,28001), 'Label': y_test})


# In[ ]:


out.to_csv('submit.csv', index = False)


# In[ ]:


ls


# In[ ]:





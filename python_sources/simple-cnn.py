#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


data = pd.read_csv('../input/train.csv')
data.head()


# In[3]:


data.shape


# In[4]:


img_data = data.iloc[:,1:].values
lables = data['label']


# In[5]:


img_data = np.array(img_data)
img_data.shape


# In[6]:


img_data = img_data.reshape(-1,28,28,1)


# In[7]:


img_data.shape


# In[8]:


plt.imshow(img_data[1].reshape(28,28))


# In[9]:


from keras.utils import to_categorical
Y = to_categorical(lables)


# In[10]:


Y.shape


# In[15]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


# In[16]:


model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])


# In[17]:


model.fit(img_data, Y,
          batch_size=128,
          epochs=20,
          verbose=1)


# In[30]:


submission = pd.read_csv('../input/sample_submission.csv')
submission.shape
submission.head()


# In[33]:


img_id = submission['ImageId']
img_id = np.array(img_id)


# In[19]:


test_data = pd.read_csv('../input/test.csv')
test_data.head()


# In[20]:


test_data.shape


# In[21]:


test_data = np.array(test_data).reshape(-1,28,28,1)
test_data.shape


# In[24]:


pred = model.predict_classes(test_data)


# In[25]:


pred.shape


# In[26]:


pred


# In[28]:


plt.imshow(test_data[0].reshape(28,28))


# In[36]:


img_id = img_id.reshape(-1,1)
pred = pred.reshape(-1,1)


# In[37]:


output = np.array(np.concatenate((img_id, pred), 1))


# In[38]:


output = pd.DataFrame(output,columns = ["ImageId","Label"])


# In[39]:


output.head()


# In[40]:


output.to_csv('out.csv',index = False)


# In[ ]:





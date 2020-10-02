#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import cv2
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


get_ipython().run_line_magic('ls', "'../input/'")
labels=pd.read_csv('../input/labels.csv')
test=pd.read_csv('../input/sample_submission.csv')


# In[3]:


path=labels['id']
target1=labels['breed']
testpath=test['id']


# In[4]:


target=target1.append(target1)
target=target.append(target)


# In[5]:


from keras.preprocessing.image import load_img,img_to_array
image_train1=[]
for x in path:
    y=cv2.imread('../input/train/'+x+'.jpg')
    y=cv2.resize(y,(64,64))
    y=np.array(y).flatten()
    image_train1.append(y)


# In[6]:


image_test=[]
for x in testpath:
    y=cv2.imread('../input/test/'+x+'.jpg')
    y=cv2.resize(y,(64,64))
    y=np.array(y).flatten()
    image_test.append(y)


# In[7]:


image_train=np.array(image_train1)
image_test=np.array(image_test)


# In[8]:


image_train=np.append(image_train,image_train,0)
image_train=np.append(image_train,image_train,0)


# In[9]:


target_all=pd.get_dummies(target,sparse=True)
target=np.asarray(target_all)


# In[10]:


from keras.layers import Dense,Dropout,Flatten
from sklearn.model_selection import train_test_split as tts
xtrain,xtest,ztrain,ztest=tts(image_train,target,train_size=0.8)


# In[11]:


from keras.layers import Conv2D
from keras.models import Sequential
model=Sequential()
model.add(Dense(1000, activation = "tanh"))
model.add(Dropout(0.1))
model.add(Dense(500, activation = "relu"))
model.add(Dropout(0.1))
model.add(Dense(120, activation = "softmax"))


# In[12]:


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[13]:


history=model.fit(xtrain,ztrain, batch_size=5000, epochs=10)


# In[ ]:





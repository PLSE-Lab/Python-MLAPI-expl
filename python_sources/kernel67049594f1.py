#!/usr/bin/env python
# coding: utf-8

# In[28]:


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


# In[29]:


mnist=pd.read_csv("../input/train.csv")


# In[30]:


mnist.columns


# In[31]:


X_train=mnist.iloc[:,1:].values


# In[32]:


y_train=mnist.iloc[:,0].values


# In[33]:


X_train


# In[34]:


X_train=X_train/255.0


# In[35]:


X_train.shape


# In[36]:


y_train.shape


# In[37]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# In[38]:


classifier=Sequential()
classifier.add(Dense(output_dim=128,init='uniform',activation='relu',input_dim=784))


# In[39]:


classifier.add(Dense(output_dim=64,init='uniform',activation='relu'))


# In[40]:


classifier.add(Dense(output_dim=10,init='uniform',activation='softmax'))


# In[41]:


classifier.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[42]:


classifier.fit(X_train,y_train,epochs=10,batch_size=50)


# In[43]:


mnist_test=pd.read_csv("../input/test.csv")


# In[44]:


X_test=mnist_test.values


# In[45]:


X_test


# In[46]:


X_test=X_test/255.0


# In[47]:


pred=classifier.predict(X_test)


# In[48]:


type(pred)


# In[49]:


pred=np.argmax(pred,axis=1)


# In[50]:


pred.shape


# In[51]:


mnist_sub=pd.read_csv('../input/sample_submission.csv')


# In[52]:


mnist_sub.head()


# In[53]:


mnist_sub['Label']=pred


# In[55]:


mnist_sub.to_csv('sample_submission.csv',index=False)


# In[ ]:





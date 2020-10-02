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


# In[65]:


import os
print(os.listdir("../input"))


# In[66]:


import pandas as pd
from pandas import DataFrame,Series
import numpy as np


# In[67]:


train=pd.read_csv("../input/train.csv",sep=',')


# In[68]:


test=pd.read_csv("../input/test.csv",sep=',')


# In[69]:


train.head()


# In[70]:


test.head()


# In[71]:


train.shape


# In[72]:


test.shape


# In[73]:


x_train=train.drop('label',axis=1)


# In[74]:


y_train=train['label']


# In[75]:


x_train=x_train/255.0


# In[76]:


test=test/255.0


# In[77]:


y_train.unique()


# In[78]:


import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[79]:


plt.imshow(np.array(x_train.iloc[0]).reshape((28,28)),cmap='gray')


# In[80]:


import sklearn.model_selection as model_selection
x_train,x_test,y_train,y_test=model_selection.train_test_split(x_train,y_train,test_size=0.20,random_state=200)


# In[81]:


from keras.models import Sequential
from keras.layers.convolutional import Conv2D


# In[82]:


from keras.layers.core import Dense,Flatten
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras.layers import Dropout


# In[83]:


model=Sequential()
model.add(Conv2D(filters=6,kernel_size=(3,3),padding='same',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=16,kernel_size=(3,3),padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(0.2,seed=100))
model.add(Dense(120,activation='relu'))
model.add(Dense(84,activation='relu'))
model.add(Dense(10,activation='softmax'))


# In[84]:


model.summary()


# In[85]:


sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])


# In[86]:


x=np.array(x_train)
x_train=x.reshape(x.shape[0],28,28,1)
y=keras.utils.to_categorical(np.array(y_train),10)
m=model.fit(x_train,y,epochs=10,batch_size=1000,validation_split=0.20)


# In[87]:


x_test=np.array(x_test)
x_test=x_test.reshape(x_test.shape[0],28,28,1)


# In[88]:


plt.imshow(x_test[8,:,:].reshape(28,28),cmap='gray')


# In[89]:


p=model.predict_proba(x_test[8,:,:].reshape(-1,28,28,1))


# In[90]:


np.argmax(p)


# In[91]:


results=model.predict(x_test)


# In[92]:


results=np.argmax(results,axis=1)


# In[93]:


results=pd.Series(results,name='label')


# In[94]:


submission=pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)


# In[95]:


submission.to_csv("kaggle_digitrecognizer_datasolved.csv",index=False)


# In[ ]:





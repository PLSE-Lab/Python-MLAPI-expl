#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from keras.layers import Input,Dense,Dropout
from keras.models import Model
from keras.utils import to_categorical

import numpy as np
import pandas as pd


# In[2]:


df_train=pd.read_csv('../input/train.csv')
df_test=pd.read_csv('../input/test.csv')
print(df_train.shape)
print(df_test.shape)


# In[3]:


X_train=df_train.iloc[:,1:].values
y_train=df_train.iloc[:,0].values
X_test=df_test.values


# In[4]:


X_train=X_train.astype(float)
X_test=X_test.astype(float)


# In[5]:


X_train /=255
X_test /=255


# In[6]:


y_train=to_categorical(y_train,10)


# In[7]:


input1=Input(shape=(784,),name='input_layers')


# In[8]:


d1=Dense(512,activation='tanh',name='hidden_dense_layer1')(input1)
d2=Dropout(0.3)(d1)
d3=Dense(512,activation='tanh',name='hidden_dense_layer2')(d2)
d4=Dropout(0.3)(d3)
output=Dense(10,activation='softmax',name='output_layer')(d4)


# In[9]:


model=Model(inputs=input1,outputs=output)


# In[10]:


model.summary()


# In[11]:


model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(lr=0.05,momentum=0.9,decay=1e-6),metrics=['accuracy'])


# In[12]:


model.fit(X_train,y_train,batch_size=64,epochs=20)


# In[ ]:





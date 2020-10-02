#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


X = np.load('../input/Sign-language-digits-dataset/X.npy')
y = np.load('../input/Sign-language-digits-dataset/Y.npy')


# In[17]:


i=np.random.randint(X_train.shape[0])
fig,ax=plt.subplots()
ax.imshow(X_train[i].reshape(64,64),cmap='viridis')
ax.annotate('Label:'+str(10-np.argmax(y_train[i])),xy=(30,0))


# In[4]:


X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.75,random_state=43)


# In[19]:


X_train=X_train.reshape(X_train.shape[0],64,64,1)
X_test=X_test.reshape(X_test.shape[0],64,64,1)


# In[20]:


X_train.shape,y_train.shape


# In[21]:


from keras.layers import Convolution2D,Dense,Flatten,MaxPooling2D,Dropout,Input,BatchNormalization
from keras.optimizers import adam
from keras.losses import categorical_crossentropy
from keras.models import Sequential


# In[22]:


def createKerasModel():
    model=Sequential()
    
    model.add(Convolution2D(64,(3, 3), activation='relu', input_shape=(64,64,1)))
    model.add(BatchNormalization(axis=-1))
    
    model.add(Convolution2D(64,(3, 3), activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(padding='same',pool_size=(2,2)))
    
    model.add(Convolution2D(64,(5, 5), activation='relu', input_shape=(64,64,1)))
    model.add(BatchNormalization(axis=-1))
    
    model.add(Convolution2D(64,(5, 5), activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(padding='same',pool_size=(2,2)))
    
    model.add(Flatten())
    
    model.add(Dense(1000,activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.2))
    
    model.add(Dense(500,activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.2))
    
    model.add(Dense(10,activation='softmax'))
    return model


# In[23]:


kerasModel=createKerasModel()
kerasModel.compile(optimizer=adam(),loss=categorical_crossentropy,metrics=['accuracy'])
kerasModel.summary()


# In[24]:


kerasModel.fit(batch_size=128,epochs=15,verbose=1,x=X_train,y=y_train)


# In[25]:


print("Accuracy:",kerasModel.evaluate(x=X_test,y=y_test)[1]*100)


# **TEST Results**

# In[26]:


j=10
plt.imshow(X_test[j].reshape(64,64))
print('Actual result:',10-np.argmax(y_test[j]))
print('Predicted Result:',10-kerasModel.predict_classes(X_test[j].reshape(1,64,64,1))[0])


# In[38]:


i=np.random.randint(len(X_test))
plt.imshow(X_test[i].reshape(64,64))
print('Actual result:',10-np.argmax(y_test[i]))
print('Predicted Result:',10-kerasModel.predict_classes(X_test[i].reshape(1,64,64,1))[0])


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


# In[1]:


import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
import pandas as pd
import numpy as np


# In[2]:


dir="../input"
os.chdir(dir)


# In[3]:


train_x=pd.read_csv("fashion_train.csv",header=0)
train_y=pd.read_csv("fashion_train_labels.csv",header=0)
test_x=pd.read_csv("fashion_test.csv",header=0)
test_y=pd.read_csv("fashion_test_labels.csv",header=0)


# In[4]:


labels=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
train_x=train_x/255
train_y=train_y/255
test_x=test_x/255
test_y=test_y/255


# In[5]:


print(train_x.shape)


# In[6]:


print(train_y.shape)


# In[7]:


print(test_x.shape)


# In[8]:


print(test_y.shape)


# In[9]:


print(train_x.head())


# In[10]:


print(test_x.head())


# In[11]:


import matplotlib.pyplot as plt


# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


plt.imshow(np.array(train_x.iloc[0]).reshape((28,28)),cmap='gray')


# In[14]:


plt.imshow(np.array(train_x.iloc[1]).reshape((28,28)),cmap='gray')


# In[15]:


plt.imshow(np.array(train_x.iloc[2]).reshape((28,28)),cmap='gray')


# In[16]:


plt.imshow(np.array(train_x.iloc[9]).reshape((28,28)),cmap='gray')


# In[17]:


from keras.wrappers.scikit_learn import KerasClassifier
from keras.regularizers import l1,l2


# In[18]:


model=tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=30,input_dim=784,kernel_regularizer=l2(0.001)),
    tf.keras.layers.Dense(units=1000,activation=tf.nn.relu),
    tf.keras.layers.Dense(units=1000,activation=tf.nn.relu),
    tf.keras.layers.Dense(units=500,activation=tf.nn.relu),
    tf.keras.layers.Dense(units=200,activation=tf.nn.relu),
    tf.keras.layers.Dense(units=10,activation=tf.nn.softmax)
])


# In[19]:


model.summary()


# In[20]:


model.compile(loss='sparse_categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])


# In[21]:


m=model.fit(train_x,train_y,epochs=10,validation_split=0.20)


# In[22]:


test_loss,test_acc=model.evaluate(test_x,test_y)
print("Test Accuracy:",(test_acc*100))


# In[23]:


p=model.predict(np.array(test_x.loc[0]).reshape(1,784))
p


# In[24]:


np.argmax(p)


# In[25]:


plt.imshow(np.array(test_x.loc[0]).reshape((28,28)),cmap='gray')


# In[26]:


predict=model.predict(test_x)
predict[1]


# In[27]:


np.argmax(predict[1])


# In[28]:


labels[0]


# In[29]:


plt.plot(m.history['acc'],color='green')
plt.plot(m.history['val_acc'],color='blue')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train','test'],loc='upper_left')
plt.show()


# In[30]:


plt.plot(m.history['loss'],color='green')
plt.plot(m.history['val_loss'],color='blue')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train','test'],loc='upper_left')
plt.show()


# In[31]:


models=Sequential()
models.add(Dense(units=30,input_dim=784,kernel_regularizer=l2(0.001)))
models.add(Activation('relu'))
models.add(Dense(units=1000))
models.add(Activation('relu'))
models.add(Dense(units=1000))
models.add(Activation('relu'))
models.add(Dense(units=500))
models.add(Activation('relu'))
models.add(Dense(units=200))
models.add(Activation('relu'))
models.add(Dense(units=10))
models.add(Activation('softmax'))


# In[32]:


models.summary()
models.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])


# In[33]:


x=np.array(train_x)
import keras
y=keras.utils.to_categorical(np.array(train_y),10)
m2=models.fit(x,y,epochs=10,validation_split=0.20)


# In[34]:


p2=models.predict(np.array(test_x.loc[0]).reshape(1,784))
p2


# In[35]:


np.argmax(p2)


# In[36]:


plt.imshow(np.array(test_x.loc[5]).reshape((28,28)),cmap='gray')


# In[37]:


labels[0]


# In[38]:


p3=models.predict_proba(np.array(test_x.loc[8]).reshape(1,784))


# In[39]:


np.argmax(p3)


# In[40]:


labels[0]


# In[42]:


plt.plot(m2.history['acc'])
plt.plot(m2.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train','test'],loc='upper_left')
plt.show()


# In[43]:


plt.plot(m2.history['loss'])
plt.plot(m2.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train','test'],loc='upper_left')
plt.show()


# In[ ]:





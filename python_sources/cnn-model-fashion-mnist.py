#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


from keras.models import Sequential
from keras.layers import Input, Convolution2D,MaxPooling2D,Flatten,Dense,Dropout
from keras.utils import np_utils
import tensorflow


# In[ ]:


dataframe = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_train.csv")


# In[ ]:


data = np.array(dataframe)


# In[ ]:


X = data[:,1:]
y = data[:,0]


# In[ ]:


X = X/255.0


# In[ ]:


X.shape


# In[ ]:


y.shape


# In[ ]:


np.unique(y,return_counts=True)


# In[ ]:


X_train = X.reshape((-1,28,28,1))


# In[ ]:


# One-hot encoded
Y_train = np_utils.to_categorical(y)
Y_train.shape


# In[ ]:


for i in range(10):
    plt.imshow(X_train[i].reshape(28,28),cmap = 'gray')
    plt.show()


# # CNN model

# In[ ]:


model = Sequential()
model.add(Convolution2D(32,(3,3),activation = 'relu',input_shape = (28,28,1)))
model.add(Convolution2D(64,(3,3),activation = 'relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(2,2))
model.add(Convolution2D(32,(5,5),activation = 'relu'))
model.add(Convolution2D(8,(5,5),activation= 'relu'))
model.add(Flatten())
model.add(Dense(10,activation = 'softmax'))
model.summary()


# In[ ]:


model.compile(loss="categorical_crossentropy",optimizer = 'adam',metrics = ["accuracy"])


# In[ ]:


hist = model.fit(X_train,Y_train,epochs = 20,shuffle = True,batch_size=256,validation_split=0.20)


# In[ ]:


plt.figure()
plt.plot(hist.history['loss'],'g')
plt.plot(hist.history['val_loss'],'b')
plt.plot(hist.history['accuracy'],'r')
plt.plot(hist.history['val_accuracy'],'black')
plt.show()


# In[ ]:


testDataframe = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_test.csv")
data = np.array(testDataframe)
X = data[:,1:]
y = data[:,0]
X = X/255.0


# In[ ]:


X_test = X.reshape((-1,28,28,1))
y_test = np_utils.to_categorical(y)


# In[ ]:


print("X_test: %d, %d  and y_test : %d, %d" %(X_test.shape[0],X_test.shape[1],y_test.shape[0],y_test.shape[1]))


# In[ ]:


model.evaluate(X_test,y_test)


# In[ ]:





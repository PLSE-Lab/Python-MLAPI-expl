#!/usr/bin/env python
# coding: utf-8

# In[39]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
os.listdir("../input/")

data=pd.read_csv('../input/train.csv')
# Any results you write to the current directory are saved as output.


# importing the libraries required

# In[40]:


import keras
from keras.models import Model, Sequential, model_from_json
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, concatenate,Activation
from keras import regularizers
import numpy as np
import glob
import cv2
import numpy as np
import csv 
import os
from numpy import genfromtxt
from sklearn import preprocessing
from PIL import Image
import matplotlib.pyplot as plt


# function for inception module using keras 

# In[41]:


def inception():
    inputs = Input(shape=(28, 28,1))
    input_img = Conv2D(32, (5,5), strides = (2,2), activation='relu')(inputs)
    input_img = Conv2D(16, (3,3), activation='relu')(input_img)
    
    tower_4 = MaxPooling2D((3,3), strides=(2,2), padding='same')(input_img)
    
    tower_5 = Conv2D(16, (4,4), strides=(2,2), padding='same', activation='relu')(input_img)
    
    input_img = concatenate([tower_4, tower_5], axis = 3)
    
    tower_1 = AveragePooling2D((3,3), strides=(1,1), padding='same')(input_img)
    tower_1 = Conv2D(4, (1,1), padding='same', activation='relu')(tower_1)
    
    tower_2 = Conv2D(4, (1,1), padding='same', activation='relu')(input_img)
    tower_2 = Conv2D(4, (2,2), padding='same', activation='relu')(tower_2)
    
    tower_3 = Conv2D(4, (1,1), padding='same', activation='relu')(input_img)
    tower_3 = Conv2D(4, (2,2), padding='same', activation='relu')(tower_3)
    tower_3 = Conv2D(4, (2,2), padding='same', activation='relu')(tower_3)
    
    outputs = concatenate([tower_1, tower_2, tower_3], axis = 3)
    
    dense = MaxPooling2D((2, 2), strides=(2,2))(outputs)
    dense = Flatten(name='flatten')(dense)
    dense = Dense(128, activation='relu', name='dense_1')(dense)
    dense = Dropout(0.5)(dense)
    dense = Dense(10, name='dense_2')(dense)
    
    prediction = Activation('softmax', name='softmax')(dense)
    
    model = Model(input=inputs, output=prediction)
    
    return model
    


# function to remove nans from input data

# In[42]:


def line_check(line):
    line=np.array(line)
    tran=np.nan_to_num(line)
    if tran.all()!=line.all():
        print(1)
    return tran
    


# data preprocessing

# In[43]:


X_train=list()
Y_train=list()
i=0
with open('../input/train.csv') as csvfile: 
    mpg_data = csv.reader(csvfile)
    for line in mpg_data:
      if i==0:
        i+=1
      else:
        r=list()
        Y_train.append(int(line[0]))
        for j in range(0,784,28):
          a=line_check(np.array(list(map(int, line[1+j:29+j]))))
          r.append(a)
        X_train.append(r)


# In[44]:


x_train=list()
for i in range(42000):
  t=list()
  t.append(X_train[i])
  x_train.append(np.dstack(t))

x_train=np.array(x_train)/255.0


# In[45]:


print(np.shape(x_train))
plt.imshow(x_train[3].reshape(28,28))


# In[46]:


y_train=list()
for i in range(len(Y_train)):
  r=list()
  r.append(Y_train[i])
  y_train.append(r)
  
enc = preprocessing.OneHotEncoder()
enc.fit(y_train)
y_transform= enc.transform(y_train).toarray()
y_transform.shape
  


# In[47]:


y_transform=np.array(y_transform)
y_transform=line_check(y_transform)


# starting the inception module

# In[48]:


model=inception()
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])


# training the inception module on input images

# In[49]:


model.fit(x_train, y_transform,
          batch_size=2000,
          epochs=50,validation_data=(x_train, y_transform))


# preprocessing test data

# In[50]:


X_test=list()
i=0
with open('../input/test.csv') as csvfile: 
    mpg_data = csv.reader(csvfile)
    for line in mpg_data:
      if i==0:
        i+=1
      else:
        r=list()
        for j in range(0,784,28):
          r.append(list(map(int, line[0+j:28+j])))
        X_test.append(r)


# In[51]:


x_test=list()
for i in range(28000):
  t=list()
  t.append(X_test[i])
  x_test.append(np.dstack(t))

x_test=np.array(x_test)/255.0


# In[52]:


print(np.shape(x_test))
plt.imshow(x_test[3].reshape(28,28))


# submitting the results as a csv file

# In[53]:


out = open('result.csv', "w")
out.write("ImageId,Label\n")
Y_test=model.predict(x_test)
rows =['']*len(Y_test)
for i in range(len(Y_test)):
  rows[i]='%s,%s\n' % (i+1,int(np.argmax(Y_test[i])))
out.writelines(rows)
out.close()


# In[ ]:





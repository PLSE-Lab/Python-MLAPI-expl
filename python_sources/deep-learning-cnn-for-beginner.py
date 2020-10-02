#!/usr/bin/env python
# coding: utf-8

# In this kernel i will be providing a beginners guide to convolutinal neural networks.This is a work in process.I will be updating this kernel in coming days.If you like my work please vote

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


# **Importing the module **

# In[ ]:


import keras 
from keras.models import Sequential 
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# **Importing and displaying the dataset **

# In[ ]:


data = pd.read_csv("../input/train.csv")
data.head()


# So every digit is represented by 784 pixels and the column of label indicated the digit.

# In[ ]:


#data.iloc[3,0:].values


# So we can see that all the pixels are arranged as rows.To feed the data to CNN we need to convert the data into matrix of 28x28 using grey scale.

# In[ ]:


# Reshaping into 28 x 28 array
#data.iloc[3,1:].values.reshape(28,28).astype('uint8')


# **Preprocessing the data **

# In[ ]:


# Storing Pixel array in form length width and channel in df_x
df_x=data.iloc[:,1:].values.reshape(len(data),28,28,1)

# Storing the label in y
y=data.iloc[:,0].values


# **Converting the labels into Catogerical features **

# In[ ]:


df_y=keras.utils.to_categorical(y,num_classes=10)


# As the digits in label are  numerical we convert them into catogerical value so that they dont affect our result prediction 

# In[ ]:


df_x=np.array(df_x)
df_y=np.array(df_y)


# In[ ]:


# Labels 
y


# In[ ]:


# Catogerical Labels
df_y


# In[ ]:


df_x.shape


# **Splitting into to test and train data**

# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.2,random_state=4)


# **Building the CNN Model **

# In[ ]:


# We will be using a sequential with Convolution layer 
# 32 3x3 filters(+Relu) Normalization not needed as Relu used 
# Max Pooling layer Window Size -2x2
# Flattened nodes 
# Layer of 100 Nodes 
# Final layer with 10 nodes for ten digits 
model=Sequential()
model.add(Convolution2D(32,3,data_format='channels_last',activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
# Dropout is added to prevent overfitting 
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])


# In[ ]:


model.summary()


# **Fitting the model **

# In[ ]:


# Testing the model with 100 images 
#model.fit(x_train[1:1000],y_train[1:1000],validation_data=(x_test[1:20],y_test[1:20]))

# Testing the model on the whole dataset 
model.fit(x_train,y_train,validation_data=(x_test,y_test))


# In[ ]:


model.evaluate(x_test,y_test)


# Improve the accuracy by more epochs till the loss is almost same 

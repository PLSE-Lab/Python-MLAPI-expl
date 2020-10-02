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


# **Importing the tensorflow and keras library to build a classification model.**

# In[ ]:


import tensorflow
print(tensorflow.__version__)
from tensorflow import keras


# **Reading the data using pandas library**

# In[ ]:


data_train = pd.read_csv("../input/train.csv")
data_test = pd.read_csv("../input/test.csv")


# **Cleaning data**
# 1. Getting the label values as a numpy array from train.csv
# 2. Removing the label column from the dataframe
# 3. Extracting the pixels from the dataframe in a numpy nd array type
# 4. Printing the shapes

# In[ ]:


train_labels = data_train.label.values
data_train = data_train.drop("label",axis =1)


# In[ ]:


train_images = data_train.values
test_images = data_test.values
print(train_images.shape,'\n',test_images.shape,'\n',train_labels.shape)


# In[ ]:


plt.figure(figsize = (10,10))
for i in range(25):
    plt.subplot(5,5,1+i)
    plt.xticks([])
    plt.imshow(train_images[i].reshape(28,28))
    plt.xlabel(prediction[i])
plt.show()


# **Model**
# > Making the model using keras having a Flatten layer (Flatens the array data in the shape (784,) and two densely connected layes the first one with relu activation and the second one with softmax.
# > The output is a probability matrix with shape (n,10)

# In[ ]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28*28,)),
    keras.layers.Dense(512,activation = 'relu'),
    keras.layers.Dense(10,activation = 'softmax')
])


# In[ ]:


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics= ['accuracy'])


# In[ ]:


model.fit(train_images,train_labels,epochs = 10)


# In[ ]:


pred = model.predict(test_images)
prediction = np.argmax(pred,axis=1)
print(pred.shape,'\n',prediction.shape)


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(test_images[i].reshape(28,28))
    plt.xlabel("pred:{}  ".format(prediction[i]))
    plt.xticks([])


# In[ ]:


get_ipython().system('cd ../input')
get_ipython().system('ls -a')


# In[ ]:


data = {"ImageId" : np.arange(1,prediction.shape[0]+1), "Label" : prediction}
submission = pd.DataFrame(data)


# In[ ]:





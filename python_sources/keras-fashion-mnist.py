#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from subprocess import check_output
print(check_output(["ls","../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
print(tf.VERSION)


# In[ ]:


from tensorflow import keras


# In[ ]:


data_train_file="../input/fashion-mnist_train.csv"
data_test_file="../input/fashion-mnist_test.csv"
df_train=pd.read_csv(data_train_file)
df_test=pd.read_csv(data_test_file)


# In[ ]:


df_train.head()


# In[ ]:


def get_features_labels(df):
    features=df.values[:,1:]/255
    labels=df['label'].values
    return features,labels


# In[ ]:


train_features,train_labels=get_features_labels(df_train)
test_features,test_labels=get_features_labels(df_test)


# In[ ]:


print (train_features.shape)
print(train_labels.shape)


# In[ ]:


train_features[20,300:320]


# In[ ]:


example_index=221
plt.figure()
_=plt.imshow(np.reshape(train_features[example_index,:],(28,28)),"gray")


# In[ ]:


train_labels.shape


# In[ ]:


train_labels[example_index]


# In[ ]:


train_labels=keras.utils.to_categorical(train_labels)
test_labels=keras.utils.to_categorical(test_labels)


# In[ ]:


train_labels.shape


# In[ ]:


train_labels[0]


# In[ ]:


train_labels[example_index]


# In[ ]:


model=tf.keras.Sequential()
model.add(tf.keras.layers.Dense(30,activation=tf.nn.relu,input_shape=(784,)))
model.add(tf.keras.layers.Dense(30,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

model.compile(loss="categorical_crossentropy",
              optimizer="rmsprop",
              metrics=['accuracy'])
model.summary()



# In[ ]:


EPOCHS=2
BATCH_SIZE=128
model.fit(train_features,train_labels,epochs=EPOCHS,batch_size=BATCH_SIZE)


# In[ ]:


test_loss,test_acc=model.evaluate(test_features,test_labels)


# In[ ]:


print('test_acc:',test_acc)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# Load useful libraries

# In[3]:


from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


# Declare variables

# In[4]:


batch_size = 128
num_classes = 10
epochs = 100


# Load train data

# In[35]:


df = pd.read_csv('../input/train.csv')
df.head()

msk = np.random.rand(len(df)) < 0.8
train_odata = df[msk]
test_odata = df[~msk]

print(train_odata.shape)
print(test_odata.shape)


# In[36]:


train_data = train_odata.loc[:, 'pixel0':'pixel783']
train_data.head()


# In[37]:


train_labels = train_odata.loc[:, 'label':'label']
train_labels.head()


# In[38]:


test_data = test_odata.loc[:,'pixel0':'pixel783']
test_labels = test_odata.loc[:, 'label':'label']


# In[39]:


# check shape
print(train_data.shape)
print(test_data.shape)


# In[10]:


# reshape data
#train_data = train_data.reshape(60000, 28*28)
#test_data = test_data.reshape(10000, 28*28)


# In[40]:


# Train and test data value lies between 0.0 to 255.0
# So, let's Normalize data
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')
train_data = train_data/255.0
test_data = test_data/255.0


# In[41]:


# check shape
print(train_data.shape)
print(test_data.shape)


# In[ ]:


# check number of categories in label data
#train_labels
#temp = pd.DataFrame({'labels': train_labels[:]})
#temp['labels'].value_counts()


# In[42]:


# let's convert class to categorical variable
train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)


# In[43]:


# now let's check with Dense architecture
denseModel = Sequential()
denseModel.add(Dense(512, activation='relu', input_shape=(784,)))
denseModel.add(Dropout(0.2))
denseModel.add(Dense(256, activation='relu'))
denseModel.add(Dropout(0.2))
denseModel.add(Dense(10, activation='softmax'))

denseModel.summary()


# In[44]:


denseModel.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])


# In[45]:


history = denseModel.fit(train_data, train_labels, 
                    batch_size=batch_size, 
                    epochs=epochs, 
                    verbose=1, 
                    validation_data=(test_data, test_labels))


# In[46]:


score = denseModel.evaluate(test_data, test_labels)
score


# In[47]:


tdf = pd.read_csv('../input/test.csv')
tdf.head()


# In[ ]:





# In[48]:


# predict results
results = denseModel.predict(tdf)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
results.shape


# In[50]:


submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("submission.csv",index=False)


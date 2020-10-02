#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ### Load dataset

# In[17]:


train_data = pd.read_csv("../input/fashion-mnist_train.csv")
test_data = pd.read_csv("../input/fashion-mnist_test.csv")


# In[3]:


train_data.head()


# In[6]:


test_data.head()


# ### Separate labels from train and test data

# In[18]:


train_labels = train_data.iloc[:,0:1]
test_labels = test_data.iloc[:,0:1]

train_data = train_data.drop("label", axis=1)
test_data = test_data.drop("label", axis=1)


# In[19]:


print(train_data.shape, test_labels.shape)


# ### Class names for labels

# In[35]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# ### View an item from training data

# In[44]:


plt.figure()
plt.imshow(train_data.iloc[1:2,:].values.reshape(28,28))
plt.grid(False)
plt.xlabel(class_names[train_labels.iloc[1][0]])


# ### Scale the values in 0 to 1 range

# In[45]:


# Scaling
train_data = train_data / 255.0
test_data = test_data / 255.0


# ### Reshape into 28 x 28 images

# In[47]:


train_data = train_data.values.reshape(60000,28,28)
test_data = test_data.values.reshape(10000,28,28)


# ### Define model

# In[48]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


# ### Compile model

# In[49]:


model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# ### Fit training data

# In[50]:


model.fit(train_data, train_labels, epochs=5)


# ### Evaluate model on test data

# In[52]:


test_loss, test_acc = model.evaluate(test_data, test_labels)

print('Test accuracy:', test_acc)


# ### Make predictions

# In[53]:


predictions = model.predict(test_data)
print(predictions[0])     # Prints the confidence level for each class


# In[54]:


print(class_names[np.argmax(predictions[0])])


# In[57]:


print(class_names[test_labels.iloc[0][0]])


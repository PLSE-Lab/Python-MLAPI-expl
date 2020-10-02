#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist


# In[ ]:


#preprocessing
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


# In[ ]:


#Normalizing

X_train = X_train/255.0
X_test = X_test/255.0


# In[ ]:


#Reshaping
X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)


# In[ ]:


#Building a Neural Network
model = tf.keras.models.Sequential()


# In[ ]:


#Adding first Layer
model.add(tf.keras.layers.Dense(units = 128, activation = 'relu', input_shape = (784,)))
#dropout
model.add(tf.keras.layers.Dropout(0.2))
#Adding Second Layer(output)
model.add(tf.keras.layers.Dense(units = 10, activation = 'softmax'))


# In[ ]:


#Compiling
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['sparse_categorical_accuracy'])


# In[ ]:


#training
model.fit(X_train, y_train, epochs =5)


# In[ ]:


#Model Evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test)


# In[ ]:


#saving model
model_json = model.to_json()
with open("fashion_model.json", "w") as json_file:
    json_file.write(model_json)
    


# In[ ]:


model.save_weights("fashion_model.h5")


# In[ ]:





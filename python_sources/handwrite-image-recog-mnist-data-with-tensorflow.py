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


# implication of what i learn from tutorial: https://www.youtube.com/watch?v=wQ8BIBpya2k

# In[ ]:


import tensorflow as tf


# In[ ]:


print(tf.__version__)


# In[ ]:


mnist = tf.keras.datasets.mnist


# In[ ]:


(x_train, y_train),(x_test, y_test) = mnist.load_data()


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


#how it looks like
plt.imshow(x_train[2])


# In[ ]:


#normalization is needed in some cases, so it is better to do it now too
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


# In[ ]:


plt.imshow(x_train[2]) #values are between 0 and 1 now


# In[ ]:


classifier = tf.keras.models.Sequential() #creating the model


# In[ ]:


classifier.add(tf.keras.layers.Flatten())


# In[ ]:


classifier.add(tf.keras.layers.Dense(8, activation=tf.nn.relu))#adding layer 1


# In[ ]:


classifier.add(tf.keras.layers.Dense(8, activation=tf.nn.relu))#adding layer 2


# In[ ]:


classifier.add(tf.keras.layers.Dense(10, activation=tf.nn.sigmoid))#output layer


# In[ ]:


classifier.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) #compile with basic settings


# In[ ]:


#now the model is ready
classifier.fit(x_train,y_train,epochs=3)


# not a very good model now, but it is just to learn

# In[ ]:


val_loss, val_acc = classifier.evaluate(x_test, y_test)  # evaluate the out of sample data with model
print(val_loss)  # model's loss (error)
print(val_acc)  # model's accuracy


# In[ ]:


prediction=classifier.predict(x_test)


# In[ ]:


prediction[0]


# In[ ]:


np.argmax(prediction[5])


# In[ ]:


plt.imshow(x_test[5])


# It looks like working!

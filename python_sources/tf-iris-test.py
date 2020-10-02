#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install tensorflow==2.0.0-alpha0')


# In[119]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from time import time


# In[110]:


print(tf.__version__)


# In[122]:


#!pip install tensorflow==2.0.0-alpha0


# In[35]:


df = pd.read_csv('../input/Iris.csv')


# In[36]:


del df['Id']


# In[37]:


le = preprocessing.LabelEncoder()
le.fit(df['Species'])
list(le.classes_), le.transform(list(le.classes_))


# In[38]:


df['Species'] = le.transform(df['Species'])


# In[39]:


df.head()


# In[40]:


x_train, x_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'Species'], 
                                                    df['Species'], 
                                                    test_size=0.33, 
                                                    random_state=0)


# In[45]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[47]:


x_train.values.shape #to convert from dataframe to np.array


# In[ ]:





# In[85]:


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(4, )),
  tf.keras.layers.Dense(15, activation=tf.nn.relu),
  tf.keras.layers.Dense(15, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(3, activation=tf.nn.softmax)
])


# In[86]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[87]:


history = model.fit(x_train.values, y_train.values, 
                    epochs=500, 
                    verbose = 0)
None


# In[81]:


model.evaluate(x_test.values, y_test.values)


# In[103]:


fig, axes = plt.subplots(figsize = (12,4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'])
plt.xlabel('epoch')
plt.ylabel('acc')
None


# In[95]:


history.history.keys()


# In[ ]:





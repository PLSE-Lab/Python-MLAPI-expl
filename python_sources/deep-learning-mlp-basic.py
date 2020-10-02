#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf


# In[ ]:


import pandas as pd


# In[ ]:


data = pd.read_csv('../input/Iris.csv')


# In[ ]:


data


# In[ ]:


cols = data.columns


# In[ ]:


features = cols[1:5]


# In[ ]:


labels = cols[5]


# In[ ]:





# In[ ]:


import numpy as np
from pandas import get_dummies


# In[ ]:


indices = data.index.tolist()


# In[ ]:


indices


# In[ ]:


indices = np.array(indices)


# In[ ]:


np.random.shuffle(indices)


# In[ ]:


X = data.reindex(indices)[features]


# In[ ]:


y = data.reindex(indices)[labels]
y


# In[ ]:


y = get_dummies(y)
y


# In[ ]:





# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.3)


# In[ ]:


X_train = np.array(X_train).astype(np.float32)
X_test  = np.array(X_test)
y_train = np.array(y_train).astype(np.float32)
y_test  = np.array(y_test)


# In[ ]:


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[ ]:


X_train[3]


# In[ ]:


from tensorflow.keras import layers


# In[ ]:


model = tf.keras.Sequential([
  layers.Dense(32, input_shape=(4,)),
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train,y_train,
          validation_data=(X_test,y_test),
          epochs=50)


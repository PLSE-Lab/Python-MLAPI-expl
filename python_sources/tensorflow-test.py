#!/usr/bin/env python
# coding: utf-8

# In[11]:


import tensorflow as tf


# In[13]:


mnist = tf.keras.datasets.mnist


# In[14]:


(x_train, y_train),(x_test, y_test) = mnist.load_data()


# In[ ]:


x_train, x_test = x_train / 255.0, x_test / 255.0


# In[ ]:


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])


# In[ ]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


model.fit(x_train, y_train, epochs=5)


# In[ ]:


model.evaluate(x_test, y_test)


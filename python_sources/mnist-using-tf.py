#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow import keras


# In[ ]:


mnist = tf.keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()


# In[ ]:


train_images=train_images/255
test_images=test_images/255


# In[ ]:


model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])


# In[ ]:


model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
             )


# In[ ]:


model.fit(train_images,train_labels, epochs=19)


# In[ ]:


model.evaluate(test_images, test_labels)


# In[ ]:





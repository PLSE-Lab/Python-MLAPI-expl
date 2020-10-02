#!/usr/bin/env python
# coding: utf-8

# ## Install TensorFlow 2.0

# In[ ]:


get_ipython().system('pip install tensorflow-gpu==2.0.0.alpha0')


# ## Import libs

# In[ ]:


import tensorflow as tf
import numpy as np
import pandas as pd


# ## Load training data

# In[ ]:


training = pd.read_csv('../input/digit-recognizer/train.csv')


# ## Prepare traing data

# In[ ]:


x_train, y_train = training.iloc[:, 1:], training.iloc[:, 0:1]


# In[ ]:


x_train = x_train / 255


# In[ ]:


x_train = x_train.values.reshape(-1, 28, 28, 1)


# ## Configure the model

# In[ ]:


model = tf.keras.models.Sequential()


# In[ ]:


model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding="same", activation="relu", input_shape=[28, 28, 1]))


# In[ ]:


model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding="same", activation="relu"))


# In[ ]:


model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))


# In[ ]:


model.add(tf.keras.layers.Dropout(0.2))


# In[ ]:


model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))


# In[ ]:


model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))


# In[ ]:


model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))


# In[ ]:


model.add(tf.keras.layers.Dropout(0.2))


# In[ ]:


model.add(tf.keras.layers.Flatten())


# In[ ]:


model.add(tf.keras.layers.Dense(units=256, activation='relu'))


# In[ ]:


model.add(tf.keras.layers.Dropout(0.4))


# In[ ]:


model.add(tf.keras.layers.Dense(units=10, activation='softmax'))


# In[ ]:


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])


# In[ ]:


model.summary()


# In[ ]:


model.fit(x_train, y_train, epochs=15)


# ## Load & prepare test data

# In[ ]:


test = pd.read_csv('../input/digit-recognizer/test.csv')


# In[ ]:


test = test / 255


# In[ ]:


test = test.values.reshape(-1, 28, 28, 1)


# ## Predict numbers

# In[ ]:


predictions = model.predict(test)


# ## Export predictions

# In[ ]:


export = pd.DataFrame([np.argmax(prediction) for prediction in predictions])


# In[ ]:


export.index += 1 


# In[ ]:


export = export.reset_index()


# In[ ]:


export.columns = ['ImageId', 'Label']


# In[ ]:


export.to_csv('export.csv', index=False)


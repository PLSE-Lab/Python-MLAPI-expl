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


# ## Configure the model

# In[ ]:


model = tf.keras.models.Sequential()


# In[ ]:


model.add(tf.keras.layers.Dense(units=256, activation='relu', input_shape=(784, )))


# In[ ]:


model.add(tf.keras.layers.Dropout(0.2))


# In[ ]:


model.add(tf.keras.layers.Dense(units=10, activation='softmax'))


# In[ ]:


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])


# In[ ]:


model.summary()


# ## Train the model

# In[ ]:


model.fit(x_train, y_train, epochs=10)


# ## Load & prepare test data

# In[ ]:


test = pd.read_csv('../input/digit-recognizer/test.csv')


# In[ ]:


test = test / 255


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


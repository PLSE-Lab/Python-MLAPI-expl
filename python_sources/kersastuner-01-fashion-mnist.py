#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install keras-tuner --upgrade --quiet')


# In[ ]:


import tensorflow as tf
from tensorflow import keras
import numpy as np


# In[ ]:


print(tf.__version__)


# In[ ]:


fashion_mnist = keras.datasets.fashion_mnist


# In[ ]:


(train_images,train_labels),(test_images,test_lables)=fashion_mnist.load_data()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
print(train_labels[0])


# In[ ]:


print(train_images[0])
plt.imshow(train_images[0])
print(train_labels[0])
print('\n',train_images[0].shape,'\n')


# Scaling down images between 0 to 1

# In[ ]:


train_images=train_images/255.0
test_images=test_images/255.0


# In[ ]:


print(train_images[0])


# In[ ]:


train_images=train_images.reshape(len(train_images),28,28,1)
test_images=test_images.reshape(len(test_images),28,28,1)


# In[ ]:


def build_model(hp):
    model = keras.Sequential([
      keras.layers.Conv2D(
      filters=hp.Int('conv_1_filter', min_value=32,max_value=128, step=16),
      kernel_size=hp.Choice('conv_1_kernal', values = [3,5]),
      activation='relu',
      input_shape=(28,28,1)
      ),
      keras.layers.Conv2D(
      filters=hp.Int('conv_2_filter', min_value=32, max_value=64, step=16),
      kernel_size=hp.Choice('conv_2_kernel', values = [3,5]),
      activation='relu'
      ),
      keras.layers.Flatten(),
      keras.layers.Dense(
          units=hp.Int('Dense_1_units', min_value=32, max_value=128, step=16),
          activation='relu'
      ),
      keras.layers.Dense(10,activation='softmax')
  ])


    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2,1e-3])),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model


# In[ ]:


def build_model(hp):
    
    model = keras.Sequential([
    keras.layers.Conv2D(
    filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
    kernel_size=hp.Choice('conv_1_kernel', values = [3,5]),
    activation='relu',
    input_shape=(28,28,1)
    ),
    keras.layers.Conv2D(
    filters=hp.Int('conv_2_filter', min_value=32, max_value=64, step=16),
    kernel_size=hp.Choice('conv_2_kernel', values = [3,5]),
    activation='relu'
    ),
    keras.layers.Flatten(),
    keras.layers.Dense(
    units=hp.Int('dense_1_units', min_value=32, max_value=128, step=16),
    activation='relu'
    ),
    keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

    return model


# In[ ]:


from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters


# In[ ]:


tuner_search=RandomSearch(build_model,
                          objective='val_accuracy',
                          max_trials=5,directory='output',
                          project_name="Mnist Fasshion")


# In[ ]:


tuner_search.search(train_images,train_labels,epochs=3,validation_split=0.1)


# In[ ]:


model=tuner_search.get_best_models(num_models=1)[0]


# In[ ]:


model.summary()


# In[ ]:


get_ipython().system('pip install jovian --upgrade --quiet')


# In[ ]:


import jovian


# In[ ]:


jovian.commit(project='kersastuner')


# In[ ]:





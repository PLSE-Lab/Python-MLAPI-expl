#!/usr/bin/env python
# coding: utf-8

# ### This kernel shows how to easy use TPU (tensor processing unit)

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

print("Tensorflow version " + tf.__version__)


# ### Enable TPU

# In[ ]:


# Detect hardware, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)


# ### Load train data

# In[ ]:


train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
train.head()


# In[ ]:


Y_train = train['label'].values.astype('float32')
Y_train


# In[ ]:


X_train = train.drop(labels=['label'], axis=1)
X_train.shape


# #### Normalize data

# In[ ]:


X_train = X_train.astype('float32')
X_train = X_train / 255


# #### Reshape data

# In[ ]:


X_train = X_train.values.reshape(42000,28,28,1)
X_train.shape


# #### Show some image

# In[ ]:


plt.imshow(X_train[1][:,:,0])


# ### Model

# In[ ]:


def create_model():        
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), input_shape = (28,28,1)))
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(10, activation="softmax"))
  model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=['accuracy'])   
    
  return model


# ### Train

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nwith strategy.scope():\n  model = create_model()\n\nmodel.fit(\n    X_train, Y_train,\n    epochs=10,\n    steps_per_epoch=40,\n    verbose = 2\n)')


# ### Load test data

# In[ ]:


test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")


# In[ ]:


test = test.astype('float32')
test = test / 255
test = test.values.reshape(len(test),28,28,1)
test.shape


# ### Prediction

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ntest_predictions = model.predict(test)')


# In[ ]:


# select the index with the maximum probability

results = np.argmax(test_predictions,axis = 1)
results = pd.Series(results,name="Label")


# ### Submit

# In[ ]:


submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("submission.csv",index=False)

submission.head()


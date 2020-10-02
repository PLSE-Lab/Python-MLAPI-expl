#!/usr/bin/env python
# coding: utf-8

# ### This kernel shows how to easy use TPU (tensor processing unit) with tf.keras

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

print("Tensorflow version " + tf.__version__)


# In[ ]:


get_ipython().system('pip install -q efficientnet')

import efficientnet.tfkeras as efn


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
Y_train = tf.keras.utils.to_categorical(Y_train, 10)
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


# In[ ]:


X_train = np.pad(X_train, ((0,0), (2,2), (2,2), (0,0)), mode='constant')
X_train.shape


# In[ ]:


X_train = np.squeeze(X_train, axis=-1)
X_train = stacked_img = np.stack((X_train,)*3, axis=-1)
X_train.shape


# In[ ]:


plt.imshow(X_train[1][:,:,0])


# ### Model

# In[ ]:


def create_model():
    enet = efn.EfficientNetB3(
    input_shape=(32, 32, 3),
    weights='imagenet',
    include_top=False,
    )        
    
    model = tf.keras.Sequential([
        enet,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    return model


# ### Train

# In[ ]:


with strategy.scope():
  model = create_model()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nmodel.fit(\n    X_train, Y_train,\n    epochs=10,\n    batch_size = 210,\n    shuffle=True,\n    verbose = 1\n)')


# ### Load test data

# In[ ]:


test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
test.head()


# In[ ]:


test = test.astype('float32')
test = test / 255
test = test.values.reshape(len(test),28,28,1)
test = np.pad(test, ((0,0), (2,2), (2,2), (0,0)), mode='constant')
test = np.squeeze(test, axis=-1)
test = stacked_img = np.stack((test,)*3, axis=-1)
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


# In[ ]:





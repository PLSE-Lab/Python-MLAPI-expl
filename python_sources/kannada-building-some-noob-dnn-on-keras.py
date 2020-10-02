#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


validation = pd.read_csv('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv')
validation.head()


# In[ ]:


validation.shape


# In[ ]:


train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
train.head()


# In[ ]:


train.shape


# In[ ]:


test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
test.head()


# In[ ]:


test.shape


# In[ ]:


# the data seems sequential, lets do some shuffling
np.random.shuffle(train.values)
np.random.shuffle(validation.values)

train.head()


# In[ ]:


# spilt into x_train and y_train
x_train = train.iloc[:,1:].values
y_train = train.iloc[:,0].values

x_val = validation.iloc[:,1:].values
y_val = validation.iloc[:,0].values


# In[ ]:


# build some neural network on keras :)
# do we need convolution???
# idk man, lets try some dense layers and check the score first

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile

# class myCallback(tf.keras.callbacks.Callback):
#       def on_epoch_end(self, epoch, logs={}):
#         if(logs.get('acc')>0.90):
#             print("\nReached 90% accuracy so cancelling training!")
#             self.model.stop_training = True
# callbacks = myCallback()

model = tf.keras.models.Sequential([
                        tf.keras.layers.Flatten(input_shape=(784,)), 
                        tf.keras.layers.Dense(256, activation=tf.nn.relu), 
                        tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                        tf.keras.layers.Dense(64, activation=tf.nn.relu), 
                        tf.keras.layers.Dense(32, activation=tf.nn.relu), 
                        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
                        ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs =16, validation_data=(x_val, y_val)
                    #, callbacks=[callbacks]
                   )


# ## Models tried
# 
# ### Simple model
# model = tf.keras.models.Sequential([
#                         tf.keras.layers.Flatten(input_shape=(784,)), 
#                         tf.keras.layers.Dense(128, activation=tf.nn.relu), 
#                         tf.keras.layers.Dense(10, activation=tf.nn.softmax)
#                         ])
#                         
# val_accuracy 0.6455
# 
# ### Slightly complex model
# model = tf.keras.models.Sequential([
#                         tf.keras.layers.Flatten(input_shape=(784,)), 
#                         tf.keras.layers.Dense(128, activation=tf.nn.relu), 
#                         tf.keras.layers.Dense(64, activation=tf.nn.relu), 
#                         tf.keras.layers.Dense(32, activation=tf.nn.relu), 
#                         tf.keras.layers.Dense(10, activation=tf.nn.softmax)
#                         ])
#                         
# val_accuracy 0.6590
# 
# ### More more complex model
# model = tf.keras.models.Sequential([
#                         tf.keras.layers.Flatten(input_shape=(784,)), 
#                         tf.keras.layers.Dense(256, activation=tf.nn.relu), 
#                         tf.keras.layers.Dense(128, activation=tf.nn.relu), 
#                         tf.keras.layers.Dense(64, activation=tf.nn.relu), 
#                         tf.keras.layers.Dense(32, activation=tf.nn.relu), 
#                         tf.keras.layers.Dense(10, activation=tf.nn.softmax)
#                         ])
#                         
# val_accuracy 0.6808****

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt


acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) 
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.figure()


# In[ ]:


# score looks pretty well, but that's probably over-fitting a bit too much
# shall cross check validation data next time
# time to do some prediction!
test_predict = model.predict(test.iloc[:,1:].values)

prediction = test.iloc[:,:1]
prediction['label'] = np.argmax(test_predict, axis=1)
prediction.head()


# In[ ]:


prediction.to_csv('submission.csv', index=False)


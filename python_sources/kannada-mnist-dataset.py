#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
import tensorflow as tf


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


def get_data(file):
    data = pd.read_csv(file)
    data = data.to_numpy()
    labels = []
    images = []   
    for row_ind in range(data.shape[0]):
        labels.append(data[row_ind, 0])
        images.append(data[row_ind, 1:])
    images = np.reshape(images, newshape=(-1, 28, 28,1))
    labels = np.array(labels)
    return images, labels

training_images, training_labels = get_data('../input/Kannada-MNIST/train.csv')
val_images, val_labels = get_data('../input/Kannada-MNIST/Dig-MNIST.csv')
test_images, test_id = get_data('../input/Kannada-MNIST/test.csv')

print(training_images.shape)
print(training_labels.shape)
print(test_images.shape)


# In[ ]:


sample_sub=pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')


# In[ ]:


from keras.utils.np_utils import to_categorical


# In[ ]:


training_labels = to_categorical(training_labels, num_classes = 10)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_val, Y_train, Y_val = train_test_split(training_images,training_labels, test_size = 0.2, random_state=10)


# In[ ]:


from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop


# In[ ]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1),padding='same'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1),padding='same'),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(28, 28, 1),padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu',padding='same'),
    tf.keras.layers.Conv2D(192, (3, 3), activation='relu',padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(192, (3, 3), activation='relu',padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])


# In[ ]:


model.compile(optimizer= keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9) , loss= "categorical_crossentropy", metrics=['accuracy'])


# In[ ]:


model.fit(X_train, Y_train, batch_size = 120, epochs =40, validation_data = (X_val, Y_val), verbose = 2)


# In[ ]:


results = model.predict(test_images)
results = np.argmax(results,axis = 1)

sample_sub['label']=results
sample_sub.to_csv('submission.csv',index=False)


# In[ ]:





# In[ ]:





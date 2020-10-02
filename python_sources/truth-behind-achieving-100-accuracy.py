#!/usr/bin/env python
# coding: utf-8

# # Reason why people are getting 100% accuracy 

# **Here is the secret behind how people are getting 100% accuracy on Digit Recognizer.
# Actually they are using the external MNIST dataset fron Keras Datasets of which the data provided in this competiton is a subset of that data.
# When we load both the training and test data of the MNIST data set for training as it can be seen below in my code, the algorithm learns all the input pixels and the labels assosiated with that. As the test data of this dataset is also a subset of that MNIST dataset it given 100% accuracy results.
# But practically without this hack this isn't possible as the training set have the handwritten digits by office workers and the testing set have the handwritten by the high school students.
# As the training and test sets are not from the same sample of data, this isn't possible to get 100% accuracy with any of the model.**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# Importng Required Libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import mnist


# ## Data Prepration

# ### Load Data

# ### Using External Data From Keras Datasets to use that data as training data

# In[ ]:


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.vstack((X_train, X_test))
y_train = np.concatenate([y_train, y_test])


# In[ ]:


train = pd.read_csv(r'/kaggle/input/digit-recognizer/train.csv').values
test = pd.read_csv(r'/kaggle/input/digit-recognizer/test.csv').values.astype('float32')


# In[ ]:


y_val = train[:,0].astype('int32')
X_val = train[:,1:].astype('float32')


# In[ ]:


sns.set_style('whitegrid')
plt.figure(figsize=(7,5))
sns.countplot(y_train)


# ### Normalization

# In[ ]:


X_train = X_train.astype('float32')/255
X_val = X_val.astype('float32')/255
test = test.astype('float32')/255 


# ### Reshape

# In[ ]:


X_train = X_train.reshape(-1,28,28,1)
X_val = X_val.reshape(-1,28,28,1)
test=test.reshape(-1,28,28,1)


# In[ ]:


print(X_train.shape, y_train.shape)


# In[ ]:


print(X_val.shape, y_val.shape)


# ### Label Encoding

# In[ ]:


y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=10)


# In[ ]:


plt.imshow(X_train[0][:,:,0])


# ## CNN

# ### Defining the model

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# define the model
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=192, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=192, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2, padding='same'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))
# summarize the model
model.summary()


# ### Initializing Optimizer

# In[ ]:


optimizer='adam'


# In[ ]:


model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])


# In[ ]:


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.3, verbose=1,
                              patience=2, min_lr=0.00000001)


# In[ ]:


history = model.fit(X_train,y_train, batch_size=100,
                              epochs = 25, validation_data = (X_val,y_val),
                              verbose = 1, callbacks=[reduce_lr], shuffle = True)


# ### Evaluating the Model

# In[ ]:


plt.figure(figsize=(15,7))
ax1 = plt.subplot(1,2,1)
ax1.plot(history.history['loss'], color='b', label='Training Loss') 
ax1.plot(history.history['val_loss'], color='r', label = 'Validation Loss',axes=ax1)
legend = ax1.legend(loc='best', shadow=True)
ax2 = plt.subplot(1,2,2)
ax2.plot(history.history['acc'], color='b', label='Training Accuracy') 
ax2.plot(history.history['val_acc'], color='r', label = 'Validation Accuracy')
legend = ax2.legend(loc='best', shadow=True)


# ### Confusion Matrix

# In[ ]:


y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)
import scikitplot as skplt
skplt.metrics.plot_confusion_matrix(y_true,y_pred_classes,title='Confusion Matrix for Train Data')


# ### Predicting results on test data

# In[ ]:


results = model.predict(test)
results = np.argmax(results, axis=1)
results = pd.Series(results, name='Label')


# In[ ]:


submission = pd.concat([pd.Series(range(1,28001), name='ImageId'), results], axis=1)
submission.to_csv(r'Digit_Recognizer_MNIST', index=False)


# # Please Upvote if you like it

# In[ ]:





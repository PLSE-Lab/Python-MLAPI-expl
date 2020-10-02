#!/usr/bin/env python
# coding: utf-8

# # How to build an Image Classification Model ?
# In this notebook you will learn to build and train an image classificationmodel with Convolutional Neural Network using Keras api.
# ### Steps:
# 1. Data Preparation
# 2. Define CNN model
# 3. Evaluate the model
# 4. Prediction

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


# ### Importing Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Lambda, Flatten, Dense
from keras.utils.np_utils import to_categorical 
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


# ## Data Preparation

# In[ ]:


# Loading Train and Test Data
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')


# In[ ]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# In[ ]:


train.head()


# In[ ]:


X = train.drop('label', axis=1)
y = train['label']


# In[ ]:


print('Shape of independent features: ', X.shape)
print('Shape of target label: ', y.shape)


# In[ ]:


sns.countplot(y)
y.value_counts()


# ### Check Missing Values(if any)

# In[ ]:


X.isnull().any().describe()


# In[ ]:


test.isnull().any().describe()


# Hence, there is no missing value present in the data

# ### Normalizing the data

# In[ ]:


X = X/255.0
test = test/255.0


# ### Reshape image into 3-dimensions (height=28px, width=28px, channels=1)

# In[ ]:


X = X.values.reshape(X.shape[0], 28,28,1)
test = test.values.reshape(test.shape[0], 28,28,1)


# In[ ]:


X.shape, test.shape


# ### One-hot-encoding

# In[ ]:


# Encode labels to one-hot-vectors (For ex '3' is represented as: [0,0,0,1,0,0,0,0,0,0])
y = to_categorical(y, num_classes=10)


# In[ ]:


# Let's look the output
y


# In[ ]:


# Train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# ### Note:
# Be carefull with some unbalanced dataset, a simple random split could cause inaccurate evaluation during the validation. To avoid that, you could use stratify = True option in train_test_split function.

# In[ ]:


# Let's check one of the training image
plt.imshow(X_train[15][:,:,0], cmap='gray')


# ### Define the model

# In[ ]:


model = Sequential()

model.add(Conv2D(32, (5,5), padding='Same', activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, (5,5),padding = 'Same', activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), padding='Same', activation='relu'))
model.add(Conv2D(64, (3,3), padding='Same',activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25)),

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))


# In[ ]:


model.summary()


# In[ ]:


# Define Optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0.001)


# In[ ]:


# Compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


# Set learning rate reduction
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                           factor=0.5,
                                           verbose=2,
                                           min_lr=0.00001)


# ### Data Augmentation
# To generate more images for training the model 

# In[ ]:


datagen = ImageDataGenerator(rotation_range=20,
                            zoom_range=0.2,
                            height_shift_range=0.2,
                            width_shift_range=0.2,
                            horizontal_flip=False,
                            vertical_flip=False)
datagen.fit(X_train)


# In[ ]:


num_epoch =35
batch_size=86


# In[ ]:


# Fit the model
history = model.fit_generator(datagen.flow(X_train, y_train,batch_size=batch_size),
                             epochs=num_epoch, steps_per_epoch=X_train.shape[0] // batch_size,
                             validation_data=(X_test, y_test))


# In[ ]:


# Plot the loss and accuracy curves for training and validation 
plt.figure(figsize=(14,6))
ax1 = plt.subplot(1,2,1)
ax1.plot(history.history['loss'], color='b', label='Training Loss') 
ax1.plot(history.history['val_loss'], color='r', label = 'Validation Loss',axes=ax1)
legend = ax1.legend(loc='best', shadow=True)
ax2 = plt.subplot(1,2,2)
ax2.plot(history.history['accuracy'], color='b', label='Training Accuracy') 
ax2.plot(history.history['val_accuracy'], color='r', label = 'Validation Accuracy')
legend = ax2.legend(loc='best', shadow=True)


# ### Confusion Matrix

# In[ ]:


y_prediction = model.predict(X_test)
y_prediction = np.argmax(y_prediction, axis=1)
y_true = np.argmax(y_test, axis=1)

import scikitplot as skplt
skplt.metrics.plot_confusion_matrix(y_true, y_prediction, title='Confusion Matrix for Validation Data')


# ## Predict the result

# In[ ]:


pred = model.predict(test)
pred


# In[ ]:


# select the index with the maximum probability
prediction = np.argmax(pred, axis = 1)
prediction


# In[ ]:


prediction = pd.Series(data=prediction, name='Label')


# In[ ]:


# Save predictions
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),prediction],axis = 1)

submission.to_csv("Digit_Recognizer_submission09.csv",index=False)


# ### Give it a try:)
# ### Upvote if you like this notebook

#!/usr/bin/env python
# coding: utf-8

# # Problem Statement

# In this competition, goal is to correctly identify digits from a dataset of tens of thousands of handwritten images.

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


# ## Data Prepration

# ### Load Data

# In[ ]:


train = pd.read_csv(r'/kaggle/input/digit-recognizer/train.csv')
test = pd.read_csv(r'/kaggle/input/digit-recognizer/test.csv')


# In[ ]:


y_train = train['label']
X_train=train.drop('label',axis=1)


# In[ ]:


sns.set_style('whitegrid')
plt.figure(figsize=(7,5))
sns.countplot(y_train)


# ### Check for Missing Values

# In[ ]:


X_train.isnull().any().describe()


# In[ ]:


test.isnull().any().describe()


# ### Normalization

# In[ ]:


X_train /= 255.0
test /= 255.0


# ### Reshape

# In[ ]:


X_train=X_train.values.reshape(-1,28,28,1)
test=test.values.reshape(-1,28,28,1)


# ### Label Encoding

# In[ ]:


y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)


# ### Split Training and Validation Set

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=2)


# In[ ]:


plt.imshow(X_train[0][:,:,0])


# In[ ]:


y_train.shape


# ## CNN

# ### Defining the model

# In[ ]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(5,5),activation='relu',padding='same',input_shape=(28,28,1)),
    tf.keras.layers.Conv2D(32,(5,5),activation='relu',padding='same'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same'),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same'),
    tf.keras.layers.MaxPooling2D((2,2), strides=(2,2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10,activation='softmax')
])
model.summary()


# ### Initializing Optimizer

# In[ ]:


optimizer= 'adam'


# In[ ]:


model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])


# ### Data Augmentation

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
datagen=ImageDataGenerator(rotation_range=10,
                          width_shift_range=0.1,
                          height_shift_range=0.1,
                          shear_range=0.1,
                          zoom_range=0.1,
                          horizontal_flip=False,
                          fill_mode='nearest')
datagen.fit(X_train)


# In[ ]:


learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# In[ ]:


history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=86),
                              epochs = 50 , validation_data = (X_val,y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // 86,
                              callbacks=[learning_rate_reduction])


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
submission.to_csv(r'Digit_Recognizer', index=False)


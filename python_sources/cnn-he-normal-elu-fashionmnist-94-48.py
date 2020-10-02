#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, BatchNormalization, MaxPooling2D, Lambda
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.regularizers import L1L2
dense_regularizer = L1L2(l2=0.0001)
from tensorflow.keras.metrics import categorical_crossentropy
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data_train_file = "../input/fashionmnist/fashion-mnist_train.csv"
data_test_file = "../input/fashionmnist/fashion-mnist_test.csv"

data_train = pd.read_csv(data_train_file)
data_test = pd.read_csv(data_test_file)

X = np.array(data_train.iloc[:, 1:])
y = to_categorical(np.array(data_train.iloc[:, 0]))

#Here we split validation data to optimiza classifier during training
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=13)

#Test data
X_test = np.array(data_test.iloc[:, 1:])
y_test = to_categorical(np.array(data_test.iloc[:, 0]))



X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_val = X_val.astype('float32')
X_train /= 255
X_test /= 255
X_val /= 255


# In[ ]:


BATCH_SIZE = 128

def Model_1(x=None):
    # we initialize the model
    model = Sequential()

    # Conv Block 1
    model.add(Conv2D(64, (5, 5), input_shape=(28,28,1),  padding='same', kernel_regularizer=dense_regularizer,kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Conv2D(64, (5, 5),   padding='same', kernel_regularizer=dense_regularizer,kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Conv2D(64, (5, 5),  padding='same', kernel_regularizer=dense_regularizer,kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    # Conv Block 2
    model.add(Conv2D(128, (3, 3),  padding='same', kernel_regularizer=dense_regularizer,kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Conv2D(128, (3, 3),  padding='same', kernel_regularizer=dense_regularizer,kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Conv2D(128, (3, 3),  padding='same', kernel_regularizer=dense_regularizer,kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # Conv Block 3
    model.add(Conv2D(256, (3, 3),  padding='same', kernel_regularizer=dense_regularizer,kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Conv2D(256, (3, 3),  padding='same', kernel_regularizer=dense_regularizer,kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Conv2D(256, (3, 3),  padding='same', kernel_regularizer=dense_regularizer,kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # Conv Block 4
    model.add(Conv2D(512, (3, 3),  padding='same', kernel_regularizer=dense_regularizer,kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Conv2D(512, (3, 3),  padding='same', kernel_regularizer=dense_regularizer,kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))

    # FC layers
    model.add(Flatten())
    model.add(Dense(10, activation='softmax', kernel_regularizer=dense_regularizer,kernel_initializer="he_normal"))

    return model

model = Model_1()
model.summary()


# In[ ]:


model.compile(Adam(lr=0.001),loss="categorical_crossentropy",metrics=["accuracy"])


# In[ ]:


import tensorflow.keras
callbacks_list = [
tensorflow.keras.callbacks.EarlyStopping(
monitor='val_accuracy', min_delta=0.0001, 
patience=20, verbose=1, mode='auto',
baseline=None, restore_best_weights=True),
tensorflow.keras.callbacks.ReduceLROnPlateau(
monitor='val_accuracy',
factor=0.5,
patience=10,
verbose=1,
mode='auto'),
tensorflow.keras.callbacks.ModelCheckpoint(
filepath='./my_model.h5',
monitor='val_accuracy',
save_best_only=True,
)
]


# In[ ]:


history = model.fit(X_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=200,
          verbose=1,
          validation_data=(X_val, y_val), callbacks=callbacks_list)


# In[ ]:


score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:


import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import seaborn as sns
import random

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("whitegrid")

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='val_Loss')
plt.legend()
plt.title('Loss evolution')

plt.subplot(2, 2, 2)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.title('Accuracy evolution')


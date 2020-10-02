#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Network achieving 93.28% Accuracy on Fashion MNIST using strong Regulariziation and Data Augmentation 
# 
# This notebook is applying a CNN model that I built for the CIFAR10 dataset to the Fashion MNIST dataset to test its architecture. The model performs pretty well achieving a 93.28% accuracy on unseen test data. If you wanna checkout the notebook where I built this model and explained how I chose the hyperparameters, click on this link: https://www.kaggle.com/sid2412/cifar10-cnn-model-85-97-accuracy
# 
# 

# ### Import the required libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import utils

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, GlobalAveragePooling2D, Activation, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ### Read the Train and Test data

# In[ ]:


train_df = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')
test_df = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')


# In[ ]:


print(train_df.shape)
train_df.head()


# In[ ]:


print(test_df.shape)
test_df.head()


# ### Convert train and test df to array and normalize it by dividing it by 255.0 so that all our values are between 0 and 1

# In[ ]:


train_df = np.array(train_df, dtype='float32')
test_df = np.array(test_df, dtype='float32')


# In[ ]:


X_train = train_df[:,1:] / 255.0
X_test = test_df[:,1:] / 255.0

y_train = train_df[:,0]
y_test = test_df[:, 0]


# ### Further splitting the training data to create a validation set to test model performance

# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=36)


# In[ ]:


class_names = ['T_shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# ### Visualize a few images to get a feel of the dataset

# In[ ]:


plt.figure(figsize=(12,12))
for i in range(36):
    plt.subplot(6, 6, i+1)
    plt.imshow(X_train[i].reshape(28,28))
    index = int(y_train[i])
    plt.title(class_names[index])
    plt.xticks([])
    plt.yticks([])
plt.show()


# In[ ]:


fig, axes = plt.subplots(15, 15, figsize=(16,18))
axes = axes.ravel()
n_train = len(train_df)

for i in range(225):
    index = np.random.randint(0, n_train)
    axes[i].imshow(train_df[index,1:].reshape(28,28))
    label_index = int(train_df[index,0])
    axes[i].set_title(class_names[label_index], fontsize=9)
    axes[i].axis('off')
plt.show()
plt.tight_layout()


# ### Reshape the data to the required format for Neural Networks [batch_size, width, height, channels]

# In[ ]:


X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_valid = X_valid.reshape(X_valid.shape[0], 28, 28, 1)


# ### Define the model

# In[ ]:


def cnn_model():
    
    model = Sequential()
    
    # First Conv layer
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4), input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    
    # Second Conv layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    
     # Third, fourth, fifth convolution layer
    model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    # Fully Connected layers
    model.add(Flatten())
    
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(10, activation='softmax'))
    
    model.summary()
    
    return model


# ### Data Augmentation to introduce variations in the dataset and make the model more robust and generalize better

# In[ ]:


datagen = ImageDataGenerator(featurewise_center=False,
                            samplewise_center=False,
                            featurewise_std_normalization=False,
                            samplewise_std_normalization=False,
                            zca_whitening=False,
                            rotation_range=15,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            horizontal_flip=True,
                            vertical_flip=False)

datagen.fit(X_train)


# In[ ]:


model = cnn_model()


# ### Compile the model

# In[ ]:


model.compile(loss='sparse_categorical_crossentropy',
             optimizer=Adam(lr=0.0003, decay=1e-6),
             metrics=['accuracy'])


# ### Fit the model

# In[ ]:


history = model.fit(datagen.flow(X_train, y_train, batch_size = 64),
                    steps_per_epoch = len(X_train) // 64, 
                    epochs = 125, 
                    validation_data= (X_valid, y_valid),
                    verbose=1)


# ### Plotting the train and val accuracy & loss

# In[ ]:


pd.DataFrame(history.history).plot()


# ### Evaluate model on the test set

# In[ ]:


scores = model.evaluate(X_test, y_test)


# ### Generate predictions

# In[ ]:


pred = model.predict(X_test)


# In[ ]:


class_names = ['T_shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

y_pred = np.argmax(pred, axis=1)


# ### Classification report to understand how the model is performing on every class

# In[ ]:


print(classification_report(y_test, y_pred))


# ### Visualize the true and predicted label

# In[ ]:


fig, axes = plt.subplots(5, 5, figsize=(12,12))
axes = axes.ravel()

for i in np.arange(25):
    axes[i].imshow(X_test[i].reshape(28,28))
    axes[i].set_title('True: %s \nPredict: %s' % (class_names[int(y_test[i])], class_names[y_pred[i]]))
    axes[i].axis('off')
    plt.subplots_adjust(wspace=1)


# ### Visualize the misclassified labels 

# In[ ]:


fig, axes = plt.subplots(5, 5, figsize=(12,12))
axes = axes.ravel()

miss_pred = np.where(y_pred != y_test)[0]
for i in np.arange(25):
    axes[i].imshow(X_test[miss_pred[i]].reshape(28,28))
    axes[i].set_title('True: %s \nPredict: %s' % (class_names[int(y_test[miss_pred[i]])],
                                                 class_names[y_pred[miss_pred[i]]]))
    axes[i].axis('off')
    plt.subplots_adjust(wspace=1)


# In[ ]:





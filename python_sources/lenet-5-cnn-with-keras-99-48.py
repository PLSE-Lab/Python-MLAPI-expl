#!/usr/bin/env python
# coding: utf-8

# # LeNet-5 CNN with Keras:
# ** I am developing a series of kernels for different Deep Learning Models: **
# 
# * [L-Layered Neural Network from scratch](https://www.kaggle.com/curiousprogrammer/l-layered-neural-network-from-scratch)
# * [TensorFlow NN with Augmentation](https://www.kaggle.com/curiousprogrammer/digit-recognizer-tensorflow-nn-with-augmentation)
# * [Data Augmentation in Python, TF, Keras, Imgaug](https://www.kaggle.com/curiousprogrammer/data-augmentation-in-python-tf-keras-imgaug)
# * [Deep NN with Keras](https://www.kaggle.com/curiousprogrammer/deep-nn-with-keras-97-5) 
# * [CNN with TensorFlow](https://www.kaggle.com/curiousprogrammer/lenet-5-cnn-with-tensorflow-98-5) 
# * CNN with Keras - This one
# * AutoEncoders with TensorFlow
# * AutoEncoders with Keras
# * GANs with TensorFlow
# * GANs with Keras

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras 
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Load Data

# In[ ]:


df_train = pd.read_csv('../input/train.csv')
X_train = df_train.iloc[:, 1:]
Y_train = df_train.iloc[:, 0]


# In[ ]:


X_train.head()


# In[ ]:


Y_train.head()


# In[ ]:


X_train = np.array(X_train)
Y_train = np.array(Y_train)


# In[ ]:


# Normalize inputs
X_train = X_train / 255.0


# # Plot Digits

# In[ ]:


def plot_digits(X, Y):
    for i in range(20):
        plt.subplot(5, 4, i+1)
        plt.tight_layout()
        plt.imshow(X[i].reshape(28, 28), cmap='gray')
        plt.title('Digit:{}'.format(Y[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


# In[ ]:


plot_digits(X_train, Y_train)


# In[ ]:


fig, ax = plt.subplots(figsize=(8,8))
sns.countplot(Y_train)
ax.set_title('Distribution of Digits', fontsize=14)
ax.set_xlabel('Digits', fontsize=12)
ax.set_ylabel('Count', fontsize=14)
plt.show()


# In[ ]:


#Train-Test Split
X_dev, X_val, Y_dev, Y_val = train_test_split(X_train, Y_train, test_size=0.03, shuffle=True, random_state=2019)
T_dev = pd.get_dummies(Y_dev).values
T_val = pd.get_dummies(Y_val).values


# In[ ]:


#Reshape the input 
X_dev = X_dev.reshape(X_dev.shape[0], 28, 28, 1)
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)


# # CNN Architecture
# 
# We will LeNet-5 CNN architeture to build our model.
# 
# ** LeNet - 5 Architecture: **
# 
# ![LeNet-5 Architecture](https://engmrk.com/wp-content/uploads/2018/09/LeNet_Original_Image.jpg)
# 
# ** Convolution Operation: **
# 
# ![Convolution Operation](https://www.researchgate.net/profile/Ihab_S_Mohamed/publication/324165524/figure/fig3/AS:611103423860736@1522709818959/An-example-of-convolution-operation-in-2D-2.png)
# 
# ### Input : Flattened 784px grayscale images, which can be represented as dimension (n, 28, 28, 1)
# ### Output: 0 - 9 
# 
# ### Let's decode the operations we will be performing in each layer 
# ** First Layer:  Convolutional Layer (CONV1): **
# * Parameters: Input (N) = 28, Padding (P) = 2, Filter (F) = 5 x 5, Stride (S) = 1
# * Conv Operation: ((N + 2P - F) / S) + 1 = ((28 + 4 - 5) / 1) + 1 = 28 x 28 
# * We will apply 6 filters / kernels so we will get a 28 x 28 x 6 dimensional output
# 
# ** Second Layer:  Average Pooling Layer (POOL1): **
# * Parameters: Input (N) = 28, Filter (F) = 2 x 2, Stride (S) = 2
# * AVG Pooling Operation: ((N + 2P -F) / S) + 1 = ((28 - 2) / 2) + 1 = 14 x 14
# * We will have a 14 x 14 x 6 dimensional output at the end of this pooling
# 
# ** Third Layer:  Convolutional Layer (CONV2): **
# * Parameters: Input (N) = 14, Filter (F) = 5 x 5, Stride (S) = 1
# * Conv Operation: ((N + 2P - F) / S) + 1 = ((14 - 5) / 1) + 1 = 10 x 10
# * We will apply 16 filters / kernels so we will get a 10 x 10 x 16 dimensional output 
# 
# ** Fourth Layer: Average Pooling Layer (POOL2): **
# * Parameters: Input (N) = 10, Filter (F) = 2 x 2, Stride (S) = 2
# * AVG Pooling Operation: ((N + 2P -F) / S) + 1 = ((10 - 2) / 2) + 1 = 5 x 5
# * We will have a 5 x 5 x 16 dimensional output at the end of this pooling
# 
# ** Fifth Layer: Fully Connected layer(FC1): **
# * Parameters: W: 400 * 120, b: 120
# * We will have an output of 120 x 1 dimension
# 
# ** Sixth Layer: Fully Connected layer(FC2): **
# * Parameters: W: 120 * 84, b: 84
# * We will have an output of 84 x 1 dimension
# 
# ** Seventh Layer: Output layer(Softmax): **
# * Parameters: W: 84 * 10, b: 10
# * We will get an output of 10 x 1 dimension
# 
# We will tweak the pooling layers from average to max and activation functions. With this architecture as per book, I was not able to achieve accuracy > 98.5%. Let's imcrease the filters and check.

# In[ ]:


model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(strides=2))
model.add(Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
model.add(MaxPool2D(strides=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))


# In[ ]:


model.build()
model.summary()


# In[ ]:


adam = Adam(lr=5e-4)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)


# In[ ]:


# Set a learning rate annealer
reduce_lr = ReduceLROnPlateau(monitor='val_acc', 
                                patience=3, 
                                verbose=1, 
                                factor=0.2, 
                                min_lr=1e-6)


# In[ ]:


# Data Augmentation
datagen = ImageDataGenerator(
            rotation_range=10, 
            width_shift_range=0.1, 
            height_shift_range=0.1, 
            zoom_range=0.1)
datagen.fit(X_dev)


# In[ ]:


model.fit_generator(datagen.flow(X_dev, T_dev, batch_size=100), steps_per_epoch=len(X_dev)/100, 
                    epochs=30, validation_data=(X_val, T_val), callbacks=[reduce_lr])


# In[ ]:


score = model.evaluate(X_val, T_val, batch_size=32)


# In[ ]:


score


# # Let's predict test data

# In[ ]:


df_test = pd.read_csv('../input/test.csv')
X_test = np.array(df_test)
X_test = X_test/255.0


# In[ ]:


X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
Y_test = model.predict(X_test)


# In[ ]:


Y_test = np.argmax(Y_test, axis=1)
Y_test[:5]


# # Create submission file

# In[ ]:


df_out = pd.read_csv('../input/sample_submission.csv')
df_out['Label'] = Y_test
df_out.head()


# In[ ]:


df_out.to_csv('out.csv', index=False)


# In[ ]:





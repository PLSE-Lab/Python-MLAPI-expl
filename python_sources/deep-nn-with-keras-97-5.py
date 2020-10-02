#!/usr/bin/env python
# coding: utf-8

# # Deep NN with Keras:
# 
# I developing a series of kernels for different Deep Learning Models:
# 
# * [L-Layered Neural Network from scratch](https://www.kaggle.com/curiousprogrammer/l-layered-neural-network-from-scratch)
# * [TensorFlow NN with Augmentation](https://www.kaggle.com/curiousprogrammer/digit-recognizer-tensorflow-nn-with-augmentation)
# * [Data Augmentation in Python, TF, Keras, Imgaug](https://www.kaggle.com/curiousprogrammer/data-augmentation-in-python-tf-keras-imgaug)
# * Deep NN with Keras -This one
# * CNN with TensorFlow
# * CNN with Keras
# * AutoEncoders with TensorFlow
# * AutoEncoders with Keras
# * GANs with TensorFlow
# * GANs with Keras
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Load Data

# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_train.head()


# In[ ]:


# Split the training data to X and Y
X_train = df_train.iloc[:, 1:]
Y_train = df_train.iloc[:, 0]


# In[ ]:


#Convert input to arrays
X_train = np.array(X_train)
Y_train = np.array(Y_train)


# In[ ]:


#Normalize inputs
X_train = X_train/255.0


# # Plot images

# In[ ]:


def plot_digits(X, Y):
    for i in range(20):
        plt.subplot(5, 4, i+1)
        plt.tight_layout()
        plt.imshow(X[i].reshape((28, 28)), cmap='gray')
        plt.title('Digit:{}'.format(Y[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


# In[ ]:


plot_digits(X_train, Y_train)


# In[ ]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.countplot(Y_train)
ax.set_title('Distribution of digits', fontsize=12)
ax.set_xlabel('Digits', fontsize=10)
ax.set_ylabel('Count', fontsize=10)
plt.show()


# # Split data to dev and validation sets

# In[ ]:


X_dev, X_val, Y_dev, Y_val = train_test_split(X_train, Y_train, test_size=0.03, shuffle=True)


# # Build Model

# In[ ]:


model = Sequential()
model.add(Dense(300, input_dim=784))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))


# In[ ]:


model.build()


# In[ ]:


model.summary()


# In[ ]:


adam = Adam(lr=5e-4)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


T_dev = keras.utils.to_categorical(Y_dev, num_classes=10)
T_val = keras.utils.to_categorical(Y_val, num_classes=10)
model.fit(X_dev, T_dev, epochs=20, batch_size=100)


# # Check Validation score

# In[ ]:


score = model.evaluate(X_val, T_val, batch_size=100)


# In[ ]:


score


# # Load Test Data and Predict

# In[ ]:


df_test = pd.read_csv('../input/test.csv')
X_test = np.array(df_test)
X_test = X_test/255.0


# In[ ]:


T_test = model.predict(X_test)


# In[ ]:


Y_test = np.argmax(T_test, axis=1)
Y_test[:5]


# In[ ]:


df_sample = pd.read_csv('../input/sample_submission.csv')
df_sample['Label'] = Y_test
df_sample.head()


# In[ ]:


df_sample.to_csv('out.csv', index=False)


# In[ ]:





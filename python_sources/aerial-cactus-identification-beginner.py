#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# Any results you write to the current directory are saved as output.


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(1)
#Sklearn
from sklearn.model_selection import train_test_split
#Keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras import regularizers


# In[ ]:


train = pd.read_csv('/kaggle/input/aerial-cactus-identification/train.csv')
df = train.copy()


# In[ ]:


df.head()


# Creating labels

# In[ ]:


def get_binary(image):
    x = int(df.loc[ df['id'] == image ].has_cactus)
    return x


# In[ ]:


images = []
labels = []
for img in os.listdir('/kaggle/input/aerial-cactus-identification/train/train/'):
    binary = get_binary(img)
    image = cv2.imread('/kaggle/input/aerial-cactus-identification/train/train/' + img)
    image = cv2.resize(image,(50, 50))
    images.append(image)
    labels.append(binary)


# In[ ]:


fig = plt.figure(figsize=(20, 20))
for x in range(1, 21):
    ax = fig.add_subplot(5, 4, x)
    ax.set_title('Label = ' + str(labels[x-1]))
    ax.imshow(images[x-1])


# In[ ]:


X = np.array(images)/255
Y = np.array(labels)


# In[ ]:


print ('Train Y shape ', X.shape)
print ('Test Y shape ', Y.shape)


# In[ ]:


Y = Y.reshape(Y.shape[0], 1)


# In[ ]:


print ('Test Y shape ', Y.shape)


# In[ ]:


train_x ,test_x, train_y, test_y = train_test_split(X, Y, test_size=0.25, random_state=12)
print("Shape of Images Train X:",train_x.shape)
print("Shape of Labels Train Y:",train_y.shape)
print()
print("Shape of Images Test X:",test_x.shape)
print("Shape of Labels Test Y:",test_y.shape)


# In[ ]:


del images
del labels
del X
del Y


# In[ ]:


model = Sequential()
model.add(Conv2D(200, kernel_size=3, activation='relu', input_shape=(50, 50, 3)))
model.add(Conv2D(100, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(1,1)))
model.add(BatchNormalization(momentum=0.99, epsilon=0.01))
model.add(Dropout(0.2))

model.add(Conv2D(80, kernel_size=3, activation='relu'))
model.add(Conv2D(70, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(1,1)))
model.add(BatchNormalization(momentum=0.99, epsilon=0.01))
model.add(Dropout(0.2))

model.add(Conv2D(60, kernel_size=3, activation='relu'))
model.add(Conv2D(50, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(1,1)))
model.add(BatchNormalization(momentum=0.99, epsilon=0.01))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(45, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization(momentum=0.99, epsilon=0.01))

model.add(Dense(35, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization(momentum=0.99, epsilon=0.01))

model.add(Dense(15, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization(momentum=0.99, epsilon=0.01))

model.add(Dense(1, activation='sigmoid'))

learning_rate = 0.0001
opt = Adam(lr=learning_rate)
model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()


# In[ ]:


np.random.seed(1)

history = model.fit(train_x, train_y, batch_size=32, epochs=50)
results = model.evaluate(test_x, test_y)


# In[ ]:


plt.plot(np.squeeze(history.history["loss"]))
plt.ylabel('cost')
plt.xlabel('iterations (per tens)')
plt.title("Learning rate =" + str(learning_rate))
plt.show()
    
print("\n\nAccuracy on training set is {}".format(history.history["acc"][-1]))
print("\nAccuracy on test set is {}".format(results[1]))


# In[ ]:


images_pred = []
for img in os.listdir('/kaggle/input/aerial-cactus-identification/test/test/'):
    image = cv2.imread('/kaggle/input/aerial-cactus-identification/test/test/' + img)
    image = cv2.resize(image,(50, 50))
    images_pred.append(image)


# In[ ]:


pred = np.array(images_pred)/255
print("The Predict Dataset Shape: ", pred.shape)


# In[ ]:


predicting = model.predict_classes(pred, verbose=1)


# In[ ]:


predict_labels = []
for i in predicting:
    if i == 0:
        predict_labels.append('Not Aerial Cactus')
    elif i == 1:
        predict_labels.append('Aerial Cactus')


# In[ ]:


fig = plt.figure(figsize=(20, 20))
for x in range(1, 21):
    ax = fig.add_subplot(5, 4, x)
    ax.set_title('Model Predicted = ' + predict_labels[x-1])
    ax.imshow(images_pred[x-1])


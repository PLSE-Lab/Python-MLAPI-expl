#!/usr/bin/env python
# coding: utf-8

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


test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')


# In[ ]:


X_train = train.iloc[:,1:]
Y_train = train.iloc[:,0]
X_train /= 255
test /= 255


# In[ ]:


import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train.iloc[i,:].to_numpy().reshape((28,28)), cmap=plt.cm.binary)
    plt.xlabel(Y_train[i])
plt.show()


# In[ ]:


model = Sequential()

model.add(Conv2D(28, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10))


# In[ ]:


model.summary()


# In[ ]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
             )

X_train = X_train.to_numpy().reshape((-1, 28, 28, 1))
Y_train = Y_train.to_numpy().reshape((-1, 1))


# In[ ]:


X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=3)


# In[ ]:


epochs = 10


# In[ ]:


history = model.fit(X_train,Y_train,
                    epochs = epochs,
                    validation_data = (X_val,Y_val),
                    steps_per_epoch=X_train.shape[0]
                   )


# In[ ]:


plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')


# In[ ]:


test_data = test.to_numpy().reshape((len(test), 28, 28, 1))

output = model.predict(test_data)


# In[ ]:


output_labels = np.argmax(output, axis =1).reshape((len(test), 1))


# In[ ]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_data[i].reshape(28,28), cmap=plt.cm.binary)
    plt.xlabel(output_labels[i])
plt.show()


# In[ ]:


df = pd.DataFrame(output_labels)
submission = pd.DataFrame({'ImageId': df.index+1, 'Label': df[0]})
submission.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")


# # Credits
# 
# Adapted from [TensorFlow: Convolutional Neural Network (CNN)](https://www.tensorflow.org/tutorials/images/cnn)

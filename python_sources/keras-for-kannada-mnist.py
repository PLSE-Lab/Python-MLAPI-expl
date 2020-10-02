#!/usr/bin/env python
# coding: utf-8

# <font size='5'>Welcome to my revised submission for Kannada MNIST!</font><br>After submitting my very first submission I started looking for the most successful models out here. This time I also want to try and write "my" first CNN.<br><b>Edit:</b> <i>Actually, many commits later I can finally say I understand what is going on in each and every aspect of this kernel. It's amazing how just month ago it all still seemed quite like magic! So the model you see below, it's one of the first models I wrote from scratch (also similar to my other MNIST models).</i><br>First, let's import the tools we're going to be using this time around.
# <br><b>Note: Pandas and NumPy alreadt imported in the cell below.</b>

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Now it's time to read our data.

# In[ ]:


train_data = pd.read_csv(r'/kaggle/input/Kannada-MNIST/train.csv')


# In[ ]:


test_data = pd.read_csv(r'/kaggle/input/Kannada-MNIST/test.csv')


# <font size='5'>Let's have a look at our data!</font>

# In[ ]:


train_data.head()


# In[ ]:


test_data.describe()


# <font size=2>To properly prepare our data we choose the columns that are of our interest, divide them into training and test sets and for our X inputs, we also decrease the values of our tensors dividing the data by 255 (so that the values are between 0 and 1).<br>For labels I used the Keras utility <b>to_categorical</b> which given <b>num_classes</b> encodes the labels for us.</font>

# In[ ]:


X = train_data.drop('label', axis=1)
y = train_data['label']


X_test = test_data.drop('id', axis=1)
X_test = np.array(X_test/255.0)
X_test = X_test.reshape(-1, 28, 28,1)
X_train = np.array(X/255.0)
X_train = X_train.reshape(-1, 28, 28,1)
y_train = to_categorical(y, num_classes=10)

print (y.shape)


# In[ ]:


X_test[0]


# <font size=2>As for my model, I find that <b>Average Pooling</b> works best with digits, as it takes in the average and not the maximum value of the area, which helps when for example pressure applied by a pen changes (happened to me when I was verifying my standard MNIST code, available as my <b>DigitMNIST</b> submission).<br>
#     Increasing the amount of convolutional layers did not seem to help much, but the <b>fully-connected (dense)</b> did change the outcome quite a bit. I decided to go with a couple of low <b>Dropouts</b>, as they didn't affect the model in a way that would change the whole process, but the values are high enough to boost its accuracy.<br>
#     <b>Softmax</b> was used at the very end to render the class values and the other activations were chosen mostly through trial and error.<br>
#     I decided to train the model for 30 <b>epochs</b> because it seemed beneficial for consistency of the model.

# In[ ]:


model = keras.models.Sequential()

model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(4, 4), activation='relu'))
model.add(keras.layers.AveragePooling2D(pool_size=(3, 3),strides=(1, 1), padding='same'))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(filters=16, kernel_size=(3, 2), activation='tanh', padding='same'))
model.add(keras.layers.Conv2D(filters=16, kernel_size=(3, 2), activation='tanh', padding='valid'))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(700, activation='relu'))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(500, activation='relu'))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(300, activation='tanh'))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(200, activation='tanh'))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(100, activation='softsign'))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(50, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=1000, shuffle=True)


# In[ ]:


test_data.head()


# In[ ]:


test_pred = model.predict(X_test)

solution = []
for i in test_pred:
    solution.append(np.argmax(i))

solution[0]


# Let's save our predictions into a **Pandas DataFrame**, then export into a **CSV** file.

# In[ ]:


output = pd.DataFrame({'id': test_data.id, 'label': solution})
output.to_csv('submission.csv', index=False)


# Thank you for reading! <br>
# References: 
# Advanced Deep Learning with Keras by Rowel Atienza
# <br>https://www.kaggle.com/bustam/cnn-in-keras-for-kannada-digits
# <br>https://keras.io/examples/mnist_cnn/
# 

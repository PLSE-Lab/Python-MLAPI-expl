#!/usr/bin/env python
# coding: utf-8

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


# <font size=5>Welcome!</font><br>
# <font size=3>As MNIST is the dataset that most of us use to begin the Machine Learning journey with and its application is pretty simple, I thought it would be fun to test it with an input of my own. This is going to be both fun and wholesome, as through this exercise we can learn how to properly prepare data, so that it's consistent with what the model was trained on.

# In[ ]:


train_data = pd.read_csv(r'/kaggle/input/digit-recognizer/train.csv')
test_data = pd.read_csv(r'/kaggle/input/digit-recognizer/test.csv')


# In[ ]:


y_train = train_data['label']
X_train = train_data.drop('label', axis=1)

X_train = np.array(X_train/255.0)
y_train = to_categorical(y_train, num_classes=10)

X_test = np.array(test_data/255.0)
X_reshaped = X_train.reshape(-1, 28, 28, 1)
X_t_reshaped = X_test.reshape(-1, 28, 28, 1)


# In[ ]:


model = keras.models.Sequential()

model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(4, 4), activation='relu'))
model.add(keras.layers.AveragePooling2D(pool_size=(3, 3),strides=(1, 1), padding='same'))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(filters=16, kernel_size=(3, 2), activation='tanh', padding='same'))
model.add(keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='tanh', padding='same'))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(1024, activation='relu'))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(256, activation='tanh'))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(128, activation='tanh'))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(64, activation='softsign'))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
model.fit(X_reshaped, y_train, epochs=30, batch_size=1000, shuffle=True)


# In[ ]:


test_pred = model.predict_classes(X_t_reshaped)


# In[ ]:


submission = pd.DataFrame({'ImageID':test_data.index+1, 'Label':test_pred})
final = submission.to_csv('submission.csv', index=False)


# <font size=5>Now let's test the model on my own handwritten digits!</font><br>
# <font size=3>I used my iPad to draw the digits on a 28x28 canvas. The files are saved as JPG and are using RGB, so below we'll use a keras preprocessing tool to convert our images into greyscale.<font>

# In[ ]:


my_images=[]
#I will store all of my digits converted into arrays in the array instantiated above. 
def verify_digits(img_dir):
    for i in range(0, 10):
        #We use img_to_array to convert the loaded greyscale images into arrays, then divide the matrices by 255 as with the train_data and lastly reshape them into 4D
        my_images.append((keras.preprocessing.image.img_to_array(keras.preprocessing.image.load_img(img_dir+str(i)+'.jpg', grayscale=True))/255).reshape(-1, 28, 28, 1))


# In[ ]:


my_dir=r'/kaggle/input/verifying-mnist/'
verify_digits(my_dir)


# In[ ]:


eight = keras.preprocessing.image.load_img(my_dir+'8.jpg')
eight


# In[ ]:


my_img_preds=[]
for i in range(0, len(my_images)):
    my_img_preds.append(model.predict(my_images[i]).argmax())
my_img_preds


# <font size=3>Seems like the only often mislabeled digit is <b>8</b>, as we loaded the digits in this exact order. It was really fun nonetheless!</font>

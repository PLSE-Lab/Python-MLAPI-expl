#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import sklearn 
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
print(train.shape)
train.head()


# In[ ]:


test= pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
print(test.shape)
test.head()


# In[ ]:


Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1) 


# In[ ]:


# Normalize the data
X_train = X_train / 255.0
test = test / 255.0
print("x_train shape: ",X_train.shape)
print("test shape: ",test.shape)


# In[ ]:


# Reshape
X_train = X_train.values.reshape(-1,28,28, 1)
test = test.values.reshape(-1,28,28, 1)
print("x_train shape: ",X_train.shape)
print("test shape: ",test.shape)


# ## Show Some Data

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import sample

def show_some_images(source, titles, s):
    random_indexes = sample(range(1, 3500), 16)
    print(random_indexes)

    # Parameters for our graph; we'll output images in a 4x4 configuration
    nrows = 4
    ncols = 4

    # Index for iterating over images


    # Set up matplotlib fig, and size it to fit 4x4 pics
    fig = plt.gcf()
    fig.set_size_inches(ncols * 4, nrows * 4)



    for i, img_index in enumerate(random_indexes):
      # Set up subplot; subplot indices start at 1
      sp = plt.subplot(nrows, ncols, i + 1)
      sp.set_title("The " + s + " Value is " + str(np.argmax(titles[img_index])))
      sp.axis('Off') # Don't show axes (or gridlines)
      img = source[img_index]
      img = img.reshape((28, 28))

      plt.imshow(img)


    plt.show()


# In[ ]:


show_some_images(X_train, Y_train, "Actual")


# In[ ]:


# Label Encoding 
from tensorflow.keras.utils import to_categorical # convert to one-hot-encoding
Y_train = to_categorical(Y_train, num_classes = 10)
print(Y_train[0])


# Converted to categorical array

# ## Data Augmentation

# In[ ]:


import keras_preprocessing
training_datagen = keras_preprocessing.image.ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

training_datagen.fit(X_train)


# In[ ]:


X_train.shape[1:]


# In[ ]:


def get_base_model():
    model = tf.keras.models.Sequential([
          tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
          tf.keras.layers.MaxPooling2D(2, 2),
        
          tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
          tf.keras.layers.MaxPooling2D(2, 2),
          tf.keras.layers.Dropout(0.2),

          tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
          tf.keras.layers.MaxPooling2D(2, 2),
        
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(512, activation='relu'),
          tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    print(model.summary())
    
    return model


# In[ ]:


model = get_base_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


epochs=20
batch_size=32
history = model.fit(X_train, Y_train,
                              epochs = epochs, verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size)


# **Just a simple implementation of a neural network**

# In[ ]:


predictions = model.predict(test)
predictions_numbers = np.argmax(predictions, axis=1)
predictions_numbers


# ### LETS SEE SOME PREDICTIONS

# In[ ]:


show_some_images(test, predictions, "Predicted")


# In[ ]:


imageIds = np.arange(1,28001)
output = pd.DataFrame({'ImageId':imageIds, 'Label':predictions_numbers})
output.to_csv('output.csv', index=False)
print(output)


# In[ ]:


submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
submission.head()


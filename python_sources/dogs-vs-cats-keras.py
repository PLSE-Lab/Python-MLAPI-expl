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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Explore the data

# In[ ]:


base_dir = '../input/cat-and-dog/'

train_dir = os.path.join(base_dir, 'training_set/training_set')
test_dir = os.path.join(base_dir, 'test_set/test_set')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')


# In[ ]:


train_cats_fnames = os.listdir(train_cats_dir)
train_dogs_fnames = os.listdir(train_dogs_dir)

print(train_cats_fnames[:10])
print(train_dogs_fnames[:10])


# In[ ]:


print("total training cat images: ", len(train_cats_fnames))
print("total training dog images: ", len(train_dogs_fnames))
print("total test cat images: ", len(os.listdir(test_cats_dir)))
print("total test dogs images: ", len(os.listdir(test_dogs_dir)))


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

nrows = 4
ncols = 4

pic_index = 0


# In[ ]:


fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)

pic_index+=8

next_cat_pix = [os.path.join(train_cats_dir, fname) 
                for fname in train_cats_fnames[ pic_index-8:pic_index] 
               ]

next_dog_pix = [os.path.join(train_dogs_dir, fname) 
                for fname in train_dogs_fnames[ pic_index-8:pic_index]
               ]

for i, img_path in enumerate(next_cat_pix+next_dog_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()


# ## Build A Model

# In[ ]:


import tensorflow as tf


# In[ ]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# In[ ]:


model.summary()


# In[ ]:


from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(lr=0.001),
             loss="binary_crossentropy",
             metrics=['acc'])


# ### Data Preprocessing

# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Updated to do image augmentation
train_datagen = ImageDataGenerator(
   rescale=1./255,
   rotation_range=40,
   width_shift_range=0.2,
   height_shift_range=0.2,
   shear_range=0.2,
   zoom_range=0.2,
   horizontal_flip=True,
   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, 
                                                   batch_size=50,
                                                   class_mode='binary',
                                                   target_size=(150,150)
                                                  )

test_generator = test_datagen.flow_from_directory(test_dir,
                                                 batch_size=50,
                                                 class_mode='binary',
                                                 target_size=(150, 150))


# ### Training

# In[ ]:


model = model.fit_generator(train_generator,
                           validation_data=test_generator,
                           steps_per_epoch=160,
                           epochs=15,
                           validation_steps=40,
                           verbose=2)


# ### Evaluate Accuracy and Loss

# In[ ]:


acc = model.history['acc']
test_acc = model.history['val_acc']
loss = model.history['loss']
test_loss = model.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc)
plt.plot(epochs, test_acc)
plt.title("Training and Test Accuracy")
plt.figure()

plt.plot(epochs, loss)
plt.plot(epochs, test_loss)
plt.title("Training and Test Loss")


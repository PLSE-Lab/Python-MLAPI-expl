#!/usr/bin/env python
# coding: utf-8

# ## References
# 
# - The competition is https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition (it ended long time ago)
# - This implementation is based on https://www.tensorflow.org/tutorials/images/classification

# ## Rearrange file structure
# 
# To make it fit to Keras [flow_from_directory](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#flow_from_directory). See [this article](https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720) for more details. It is assumed that this script runs on Google Colab which mounts Google Drive and have `/Data Set/<data>.zip` below it. You can find the zip file [here](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data).

# In[ ]:


get_ipython().system("unzip '/content/drive/My Drive/Data Set/dogs-vs-cats-redux-kernels-edition.zip'")
get_ipython().system("unzip 'test.zip'")
get_ipython().system("unzip 'train.zip'")

get_ipython().system('mkdir train/{dog,cat}')
get_ipython().system("find train -maxdepth 1 -type f | grep 'dog' | xargs -I{} mv {} train/dog")
get_ipython().system("find train -maxdepth 1 -type f | grep 'cat' | xargs -I{} mv {} train/cat")

get_ipython().system('mkdir test/images')
get_ipython().system('find test -maxdepth 1 -type f | xargs -I{} mv {} test/images')


# ## Import libraries

# In[ ]:


import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import pathlib


# ## Define helper functions and global variables

# In[ ]:


# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# In[ ]:


train_dir = './train'
total_train = len(os.listdir('./train/dog')) + len(os.listdir('./train/cat'))
total_test = len(os.listdir('./test/images/'))
validation_split = 0.2

batch_size = 128
epochs = 10
IMG_HEIGHT = 150
IMG_WIDTH = 150


# ## Build model and train

# In[ ]:


train_image_generator = ImageDataGenerator(rescale=1./255,
                                           validation_split=validation_split)
train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary',
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           subset='training')
val_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary',
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           subset='validation')


# In[ ]:


model = Sequential([
    Conv2D(32, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),

    Conv2D(64, 3, padding='same', activation='relu'),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),

    Conv2D(128, 3, padding='same', activation='relu'),
    Conv2D(128, 3, padding='same', activation='relu'),
    MaxPooling2D(),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


# In[ ]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


history = model.fit(
    train_data_gen,
    epochs=epochs,
    validation_data=val_data_gen,
    steps_per_epoch=(total_train * (1 - validation_split) // batch_size),
    validation_steps=(total_train * validation_split // batch_size)
)


# ## Visualize training results

# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# ## Predict

# In[ ]:


test_image_generator = ImageDataGenerator(rescale=1./255)
test_data_gen = test_image_generator.flow_from_directory(directory='./test/',
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           shuffle=False,
                                                           class_mode=None,
                                                           batch_size=1)


# In[ ]:


test_data_gen.reset()
predictions = model.predict(test_data_gen,
                            steps=total_test,
                            verbose=1)


# Classes are `{'cat': 0, 'dog': 1}`. That matches output format so we can just round predictions value.

# In[ ]:


train_data_gen.class_indices


# ## Check accuracy by comparing test files and corresponded predictions

# In[ ]:


test_path = pathlib.Path('./test/images')
paths = [str(f) for f in test_path.glob('*')]
for path, pred in list(zip(paths, np.round(predictions)))[:10]:
  plt.imshow(Image.open(path))
  plt.show()
  print(pred)


# ## Generate submittion.csv

# In[ ]:


def extract_test_id(path):
   file = path.split('/')[1]
   return file.split('.')[0]

test_ids = [extract_test_id(f) for f in test_data_gen.filenames]


# In[ ]:


zipped = zip(test_ids, np.round(predictions.flatten()))
records = sorted(list(zipped), key=lambda kv: int(kv[0]))


# In[ ]:


output = pd.DataFrame.from_records(records, columns=['id', 'label'])
output.to_csv('submission.csv', index=False)


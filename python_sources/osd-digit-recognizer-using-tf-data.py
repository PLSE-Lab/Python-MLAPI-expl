#!/usr/bin/env python
# coding: utf-8

# ### A simple digit recognizer
# A very simple example. I chose not to do any augmentation as the orientation, size and position of the digits are constant. The model gets around 98% accuracy, but when i tested it out on a timer it only got one of 1500 frames wrong.
# Import tensorflow, numpy, os and plt

# In[ ]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


# Setting static variables

# In[ ]:


np.set_printoptions(precision = 4)
ds_dir = '../input/drone-osddvr-num-dataset/OSD_NUM_DATASET'
train_dir = ds_dir + '/train/*/*'
test_dir =  ds_dir + '/test/*/*'
CLASS_NAMES = np.array(['0','1', '2','3','4','5','6', '7','8','9','J'])
IMG_HEIGHT = 25
IMG_WIDTH = 20
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 1177


# Create a dataset of the files

# In[ ]:


traindata = tf.data.Dataset.list_files(train_dir)
testdata = tf.data.Dataset.list_files(test_dir)


# A simple labeling function that gets the label from the directory

# In[ ]:


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2] == CLASS_NAMES


# Convert the compressed string to a 3D uint8 tensor. Then change the dtype to float

# In[ ]:


def decode_img(img):
    img = tf.image.decode_jpeg(img, channels = 3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


# Use the last two functions to create a label and an image from a file path

# In[ ]:


def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


# Prepare the datasets for training by setting its attributes

# In[ ]:


def prepare_for_training(ds, cache = True, shuffle_buffer_size = 1000):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size = shuffle_buffer_size)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size = AUTOTUNE)
    return ds


# Create the datasets using the file datasets and the process_path function

# In[ ]:


train_DS = traindata.map(process_path, num_parallel_calls=AUTOTUNE)
test_DS = testdata.map(process_path, num_parallel_calls=AUTOTUNE)


# Set the attributes

# In[ ]:


train_DS = prepare_for_training(train_DS)
test_DS = prepare_for_training(test_DS)


# View some of the training data and their lables

# In[ ]:


def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(CLASS_NAMES[label_batch[n]==1])
      plt.axis('off')
  plt.show()

image_batch, label_batch = next(iter(train_DS))
show_batch(image_batch.numpy(), label_batch.numpy())


# Create a model

# In[ ]:


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(25,20,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(128, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(11, activation = tf.nn.softmax)
])


# Compile and train the model

# In[ ]:


model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['acc'])
his = model.fit(train_DS, steps_per_epoch = 10, epochs = 10, validation_data=test_DS, validation_steps = 1, shuffle=False, verbose=1)


# Plot the loss and accuracy. 

# In[ ]:


acc=his.history['acc']
val_acc=his.history['val_acc']
loss=his.history['loss']
val_loss=his.history['val_loss']
epochs=range(len(acc)) # Get number of epochs

plt.plot(epochs, acc, 'r', label = "Training Accuracy")
plt.plot(epochs, val_acc, 'b',label = "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.show()

plt.plot(epochs, loss, 'r', label = "Training Loss")
plt.plot(epochs, val_loss, 'b', label = "Validation Loss")
plt.title('Training and validation loss')
plt.show()


# The model is now trained and ready to read OSD values such as power, voltage, amps or GPS coordinates

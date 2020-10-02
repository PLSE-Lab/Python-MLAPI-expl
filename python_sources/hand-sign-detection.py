#!/usr/bin/env python
# coding: utf-8

# # Loading the libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')

import string

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# Any results you write to the current directory are saved as output.


# * **Preparing the data generator**

# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# **Normalizing the data before feeding to model**

# In[ ]:


train_datagen = ImageDataGenerator(rescale = 1/255, validation_split = 0.2)
test_datagen = ImageDataGenerator(rescale = 1/255)


# **Loading data as 28 * 28 grayscale images**

# In[ ]:


train_generator = train_datagen.flow_from_directory(
    '/kaggle/input/handsignimages/Train',
    target_size = (28, 28),
    batch_size = 128,
    class_mode = "sparse",
    color_mode='grayscale',
    subset = 'training'
    )

validation_generator = train_datagen.flow_from_directory(
    '/kaggle/input/handsignimages/Train',
    target_size = (28, 28),
    batch_size = 128,
    class_mode = "sparse",
    color_mode='grayscale',
    subset = 'validation'
    )

test_generator = test_datagen.flow_from_directory(
    '/kaggle/input/handsignimages/Test',
    target_size = (28, 28),
    batch_size = 128,
    class_mode = "sparse",
    color_mode='grayscale'
    )


# # Class Labels
# **24 classes excluding J and Z**

# In[ ]:


classes = [char for char in string.ascii_uppercase if char != "J" if char != "Z"]
print(classes, end = " ")


# In[ ]:


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(10,10))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img[:,:,0])
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# **Visualizing the dataset**

# In[ ]:


sample_training_images, _ = next(train_generator)
plotImages(sample_training_images[:5])


# # **Preparing the CNN model**

# In[ ]:


import tensorflow as tf


# **A small network of single convolution and 3 Dense layers**

# In[ ]:


model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), activation = "relu", input_shape = (28,28,1)),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation = "relu"),
        tf.keras.layers.Dense(256, activation = "relu"),
        tf.keras.layers.Dense(len(classes), activation = "softmax")
])


# In[ ]:


model.summary()


# **Callback to stop training on 99.8% accuracy**

# In[ ]:


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        if logs.get("loss") < 0.004:
            print("\nReached 99.6% accuracy so cancelling training!")
            self.model.stop_training = True

callback = myCallback()


# **Using RMSprop with learning rate = 0.01 and loss as categorical cross entropy**

# In[ ]:


from tensorflow.keras.optimizers import RMSprop

model.compile(
    optimizer = RMSprop(lr = 0.001),
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)


# # **Training**
# **Training for 10 epochs**

# In[ ]:


history = model.fit(
    train_generator,
    epochs=10,
    callbacks = [callback],
    validation_data = validation_generator
)


# # **Evaluating**
# **Testing the model on unseen dataset of 7k images**

# In[ ]:


results = model.evaluate(test_generator)


# # **Visualizing the results**

# In[ ]:


# PLOT LOSS AND ACCURACY
# %matplotlib inline

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt


# In[ ]:


#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")


plt.title('Training and validation loss')

# Desired output. Charts with training and validation metrics. No crash :)


# # **Predicting**
# **Randomly choose an alphabet from folder and display its prediction**

# In[ ]:


from random import randint
import cv2 as cv

def testModel(alphabet = "A"):
    dirname, _, filenames = list(os.walk(f'/kaggle/input/handsignimages/Test/{alphabet.upper()}'))[0]
    img_path = os.path.join(dirname, filenames[randint(0, len(filenames))])
    print(img_path)
    img = cv.imread(img_path, 0).reshape(1, 28, 28, 1)
    pred = model.predict(img)
    pred_label = classes[np.argmax(pred)]

    plt.title(pred_label)
    plt.imshow(img[0,:,:,0], cmap = "gray")


# In[ ]:


testModel("m")


# In[ ]:





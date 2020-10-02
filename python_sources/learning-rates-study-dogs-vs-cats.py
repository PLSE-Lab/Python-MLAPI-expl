#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This notebook provides working examples of some learning rate functionality that should be familiar to users of the excellent [fast.ai](https://docs.fast.ai/) library, but which have a less-than-standard implementation in Tensorflow/Keras. The goal here is to examine how these features may be implemented in Keras in a user-friendly way.
# 
# Some features, such as learning rate tuning, turn out to have a very brief and natural Keras analogue. Others, such as cyclic learning rates, require more in-depth implementation, and the corresponding functions have been packaged in the `lrutils.py` [script](https://www.kaggle.com/jjbuchanan/lrutils) attached to this notebook.

# In[ ]:


import pickle

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from lrutils import *


# In[ ]:


IMG_SHAPE = (128, 128, 3)


# ## Load the data
# 
# In previous work I have already divided the Dogs vs. Cats dataset into training (9/10ths) and validation (1/10th) sets, randomly selected from the full training data. Each contains a 50/50 mix of dog and cat images.

# In[ ]:


# Load data
train_df = pd.read_pickle('../input/cat-vs-dog-data/train_df.pkl')
train_df.filename = train_df.filename.map(lambda s: s.split('\\')[1])
validate_df = pd.read_pickle('../input/cat-vs-dog-data/validate_df.pkl')
validate_df.filename = validate_df.filename.map(lambda s: s.split('\\')[1])
train_dir = '../input/dogs-vs-cats/train/train'


# During training, apply various random transformations to the images, as a form of dataset augmentation.

# In[ ]:


train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale = 1./255.)

train_generator = train_datagen.flow_from_dataframe(train_df,
                                                    train_dir,
                                                    x_col='filename',
                                                    y_col='category',
                                                    batch_size=50,
                                                    class_mode='binary',
                                                    target_size=(128,128))     
validation_generator =  validation_datagen.flow_from_dataframe(validate_df,
                                                          train_dir,
                                                          x_col='filename',
                                                          y_col='category',
                                                         batch_size=50,
                                                         class_mode  = 'binary',
                                                         target_size = (128,128))


# ## Utility methods

# In[ ]:


def plot_history(history):
    #-----------------------------------------------------------
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    #-----------------------------------------------------------
    acc      = history.history[     'acc' ]
    val_acc  = history.history[ 'val_acc' ]
    loss     = history.history[    'loss' ]
    val_loss = history.history['val_loss' ]

    epochs   = range(len(acc)) # Get number of epochs

    #------------------------------------------------
    # Plot training and validation accuracy per epoch
    #------------------------------------------------
    plt.plot  ( epochs,     acc , label='Training')
    plt.plot  ( epochs, val_acc , label='Validation')
    plt.xlabel ('Epoch')
    plt.ylabel ('Accuracy')
    plt.legend ()
    plt.title ('Training and validation accuracy')
    plt.figure()

    #------------------------------------------------
    # Plot training and validation loss per epoch
    #------------------------------------------------
    plt.plot  ( epochs,     loss, label='Training')
    plt.plot  ( epochs, val_loss, label='Validation')
    plt.xlabel ('Epoch')
    plt.ylabel ('Loss')
    plt.legend ()
    plt.title ('Training and validation loss'   )


# In[ ]:


def report(model, history=None, validation_generator=None):
    if history is not None:
        plot_history(history)
    
    if validation_generator is not None:
        # Evaluate trained model on validation set
        validation_generator.reset()
        [val_loss, val_acc] = model.evaluate_generator(validation_generator)
        print('Model evaluation')
        print(f'val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}')
        print()


# ## Pre-trained image feature extractor
# 
# Keras comes packaged with many popular models.
# 
# https://keras.io/applications
# 
# By specifying the `include_top=False` argument, we load just the bottom convolutional layers, and not the classification layer(s) at the top.

# In[ ]:


mobilenet = keras.applications.MobileNetV2(input_shape = IMG_SHAPE,
                                                   include_top = False,
                                                   weights = 'imagenet')


# Freeze feature extractor params for now.

# In[ ]:


mobilenet.trainable = False


# This is a complex model with a large number of filters in the final layer. I will use the activations of those filters as the starting point for my classifier, which will be a dense neural network (with batch normalization and dropout).

# In[ ]:


classifier_mobilenet = keras.Sequential([
    mobilenet,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(1, activation='sigmoid')
])


# In[ ]:


classifier_mobilenet.summary()


# The classifier has this architecture for a few reasons:
# 1. Large enough number of parameters so that tuning the learning rate becomes interesting.
# 2. Few enough parameters that it trains in decent time on my laptop.
# 3. Layer widths go down by factors of two, following general convention.
# 4. The lowest layer has close to the number of units as the final number of filters in the feature extractor, which gives the model a chance to reorganize the filter outputs without losing much information.
# 5. It has about as many parameters as the feature extractor, and symmetry is nice.

# ## Test an initial learning rate

# In[ ]:


base_learning_rate = 0.0001


# In[ ]:


classifier_mobilenet.compile(optimizer=keras.optimizers.RMSprop(learning_rate=base_learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[ ]:


savebest = keras.callbacks.ModelCheckpoint('mobilenet_weights.h5', monitor='val_loss', mode='min',
                                             save_best_only=True, save_weights_only=True,
                                             verbose=1, save_freq='epoch')


# In[ ]:


history = classifier_mobilenet.fit_generator(train_generator,
                              epochs=10,
                              validation_data=validation_generator,
                                callbacks=[savebest])


# In[ ]:


classifier_mobilenet.load_weights('mobilenet_weights.h5')
report(classifier_mobilenet, history, validation_generator)


# ## Tuning the learning rate

# The following snippets implement, with a simple Keras callback (supplemented by an illustration drawn using matplotlib), the essential functionality of `lr_find()` in the [fast.ai library](https://docs.fast.ai/basic_train.html#lr_find).
# 
# For reference, the learning rate tuning method and its fast.ai implementation is introduced starting in this video:<br>https://www.youtube.com/watch?v=BWWm4AzsdLk&feature=youtu.be&t=4978.

# In[ ]:


classifier_mobilenet = keras.Sequential([
    mobilenet,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(1, activation='sigmoid')
])


# In[ ]:


classifier_mobilenet.compile(optimizer=keras.optimizers.RMSprop(),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# Start with a learning rate of 1e-6, and multiply this by sqrt(2) every batch.

# In[ ]:


lr_schedule = keras.callbacks.LearningRateScheduler(lambda epoch: 1e-6 * 2**(epoch/2))


# In[ ]:


history = classifier_mobilenet.fit_generator(train_generator,
                              epochs=30,
                              validation_data=validation_generator,
                                callbacks=[lr_schedule],
                                steps_per_epoch=1)


# In[ ]:


# Plot the tuning history
lrs = 1e-6 * 2**(np.arange(30)/2)
plt.semilogx(lrs, history.history["val_loss"])
plt.axis([1e-5, 1e-2, 0.2, 1.0])


# In[ ]:


for epoch in range(20,25):
    print(f'lr = {history.history["lr"][epoch]}, val_loss: {history.history["val_loss"][epoch]}')


# Learning starts to become unstable for lr above about 0.0014, so I will consider this the highest "good" learning rate.

# In[ ]:


tuned_learning_rate = 0.0014


# ## Train model with tuned learning rate

# In[ ]:


classifier_mobilenet = keras.Sequential([
    mobilenet,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(1, activation='sigmoid')
])


# In[ ]:


classifier_mobilenet.compile(optimizer=keras.optimizers.RMSprop(learning_rate=tuned_learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[ ]:


savebest = keras.callbacks.ModelCheckpoint('tunedlr_weights.h5', monitor='val_loss', mode='min',
                                             save_best_only=True, save_weights_only=True,
                                             verbose=1, save_freq='epoch')


# In[ ]:


history = classifier_mobilenet.fit_generator(train_generator,
                              epochs=10,
                              validation_data=validation_generator,
                                callbacks=[savebest])


# In[ ]:


classifier_mobilenet.load_weights('tunedlr_weights.h5')
report(classifier_mobilenet, history, validation_generator)


# **NOTE:** There is a notable degree of random variation if the model is independently trained multiple times - I have seen the validation set accuracy vary by roughly three quarters of a percent for the models in this notebook. A model that does better after training with one strategy may not always do better than if it were trained with another strategy. For a more rigorous comparison of different training strategies, multiple training runs are essential.

# ## Cyclical learning rates
# 
# In this strategy, the learning rate follows (the first half of) a cosine curve over the course of an epoch, and then resets and starts the cycle again for the next epoch, etc.
# 
# Original code for `LR_Updater` and `LR_Cycle` from this repo:<br>
# https://github.com/gunchagarg/learning-rate-techniques-keras
# 
# My implementation of `LR_Updater` and `LR_Cycle`, which has been clarified and expanded with some additional customization options, is stored in the `lrutils` module (imported above).
# 
# The Cyclic LR strategy being implemented is described here:<br>
# http://course18.fast.ai/lessons/lesson2.html

# In[ ]:


def compute_lrs(max_lr, epochs, batches_per_epoch, cycle_mult=1):
    lrs = []
    cycle_iterations = 0
    epoch_iterations = batches_per_epoch
    for _ in range(epochs*batches_per_epoch):
        decay_phase = np.pi*cycle_iterations/epoch_iterations
        decay = (np.cos(decay_phase) + 1.) / 2.
        lrs.append(max_lr*decay)
        cycle_iterations += 1
        if cycle_iterations == epoch_iterations:
            cycle_iterations = 0
            epoch_iterations *= cycle_mult
    return lrs


# During training, the learning rate will look like this:

# In[ ]:


lrs = compute_lrs(0.0014, 10, 450)
plt.plot(np.arange(10*450), lrs)
plt.xlabel('Batch number')
plt.ylabel('Learning rate')
plt.show()


# In[ ]:


classifier_mobilenet = keras.Sequential([
    mobilenet,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(1, activation='sigmoid')
])


# In[ ]:


classifier_mobilenet.compile(optimizer=keras.optimizers.RMSprop(learning_rate=tuned_learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# Let the cycle length be one epoch. There are 450 batches in one training epoch (22500 training images / 50 images per batch), so:

# In[ ]:


cyclic_lr = LR_Cycle(450)


# In[ ]:


savebest = keras.callbacks.ModelCheckpoint('cyclic_weights.h5', monitor='val_loss', mode='min',
                                             save_best_only=True, save_weights_only=True,
                                             verbose=1, save_freq='epoch')


# In[ ]:


history = classifier_mobilenet.fit_generator(train_generator,
                              epochs=10,
                              validation_data=validation_generator,
                                callbacks=[cyclic_lr, savebest])


# In[ ]:


classifier_mobilenet.load_weights('cyclic_weights.h5')
report(classifier_mobilenet, history, validation_generator)


# ### Cycle_mult = 2
# 
# Setting cycle_mult=2 and running for 15 epochs, the learning rate looks like this:

# In[ ]:


lrs = compute_lrs(0.0014, 15, 450, 2)
plt.plot(np.arange(15*450), lrs)
plt.xlabel('Batch number')
plt.ylabel('Learning rate')
plt.show()


# Over 15 epochs, the learning rate goes through 4 complete annealing cycles, of lengths 1, 2, 4, and finally 8.

# In[ ]:


classifier_mobilenet = keras.Sequential([
    mobilenet,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(1, activation='sigmoid')
])


# In[ ]:


classifier_mobilenet.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.0014),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[ ]:


cyclic_lr = LR_Cycle(450, cycle_mult=2)


# In[ ]:


savebest = keras.callbacks.ModelCheckpoint('cycleMult2_weights.h5', monitor='val_loss', mode='min',
                                             save_best_only=True, save_weights_only=True,
                                             verbose=1, save_freq='epoch')


# In[ ]:


history = classifier_mobilenet.fit_generator(train_generator,
                              epochs=15,
                              validation_data=validation_generator,
                                callbacks=[cyclic_lr, savebest])


# In[ ]:


classifier_mobilenet.load_weights('cycleMult2_weights.h5')
report(classifier_mobilenet, history, validation_generator)


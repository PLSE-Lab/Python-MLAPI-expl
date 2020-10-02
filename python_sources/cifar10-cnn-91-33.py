#!/usr/bin/env python
# coding: utf-8

# # The CIFAR-10 dataset
# 
# The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
# 
# The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.
# 
# The classes in the dataset are **airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck**.
# 
# The classes are completely mutually exclusive. There is no overlap between automobiles and trucks. **Automobile** includes sedans, SUVs, things of that sort. **Truck** includes only big trucks. Neither includes pickup trucks.

# In[ ]:


import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from collections import Counter
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

print(tf.__version__)


# # Dataset layout
# 
# The archive contains the files data_batch_1, data_batch_2, ..., data_batch_5, as well as test_batch. Each of these files is a Python "pickled" object produced with [cPickle](http://www.python.org/doc/2.5/lib/module-cPickle.html).
# 
# Loaded in this way, each of the batch files contains a dictionary with the following elements:
# * data -- a 10000x3072 [numpy](http://numpy.scipy.org/) array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
# * labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
# 
# The dataset contains another file, called batches.meta. It too contains a Python dictionary object. It has the following entries:
# * label_names -- a 10-element list which gives meaningful names to the numeric labels in the labels array described above. For example, label_names[0] == "airplane", label_names[1] == "automobile", etc.
# 

# We will load this dataset using Tensorflow. The classes will be loaded directly from [Alex Krizhevsky](https://www.cs.toronto.edu/~kriz/cifar.html) homepage. 

# In[ ]:


def load_dataset():
  df = pd.read_html("https://www.cs.toronto.edu/~kriz/cifar.html")
  cifar10_classes = df[0][0].values.tolist()
  num_classes = len(cifar10_classes)

  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

  y_train = tf.keras.utils.to_categorical(y_train, num_classes)
  y_test = tf.keras.utils.to_categorical(y_test, num_classes)

  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train = x_train / 255.0
  x_test = x_test / 255.0

  return x_train, y_train, x_test, y_test, np.array(cifar10_classes)


# Since we're going to classify these images, some normalization is necessary to keep data within a range.

# In[ ]:


def normalize_images(train, test):
  mean = np.mean(train, axis=(0,1,2,3))
  std = np.std(train, axis=(0,1,2,3))
  train_norm = (train - mean)/(std + 1e-7)
  test_norm = (test - mean)/(std + 1e-7)
  
  return train_norm, test_norm


# The model consists of 6 convolutional layers. I've noticed that this problem responds really well to convolutional filters. After each filter I'm applying a normalization to keep data within a range. I've tried different amount of dense layers, four responded well in an acceptible processing time.

# In[ ]:


def define_model():
    weight_decay = 1e-4
    L2 = tf.keras.regularizers.l2(weight_decay)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu", kernel_regularizer=L2, input_shape=x_train.shape[1:]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu", kernel_regularizer=L2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'), 
        tf.keras.layers.Dropout(0.2), 

        tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu", kernel_regularizer=L2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu", kernel_regularizer=L2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'),
        tf.keras.layers.Dropout(0.3), 

        tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", activation="relu", kernel_regularizer=L2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", activation="relu", kernel_regularizer=L2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'),
        tf.keras.layers.Dropout(0.4), 

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation='relu'), 
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=128, activation='relu'), 
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(units=128, activation='relu'), 
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(units=128, activation='relu'), 
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=10, activation='softmax')
    ])

    opt = tf.keras.optimizers.RMSprop(lr=0.001, decay=1e-6)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Function to plot loss x val_loss and accuracy vs val_accuracy so we can see how the learning curve progressed.

# In[ ]:


def summarize_diagnostics(history):
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')

    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')


# This function is the exact same one found on tensorflow 2.1 website. It is used to plot some random images from the dataset so we can understand what we are dealing with.

# In[ ]:


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# Now it is time to load the datasets from the disk.

# In[ ]:


x_train_df, y_train, x_test_df, y_test, classes_names = load_dataset()
x_train_df.shape, y_train.shape, x_test_df.shape, y_test.shape


# Let's check some images to verify if everything is okay so far.

# In[ ]:


sample_training_images, _ = next(tf.keras.preprocessing.image.ImageDataGenerator().flow(x_train_df, y_train, batch_size=64))
plotImages(sample_training_images[:5])


# Data normalization will restrain data to a certain range.

# In[ ]:


x_train, x_test = normalize_images(x_train_df, x_test_df)


# I've instantiated a ImageDataGenerator to augment data. Nothing fancy, just a little bit of shift, flipping and rotation. We will train the model in batches of 64 samples.

# In[ ]:


train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    rotation_range=15
)

batch_size = 64

train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)


# These callback functions are what let us achieve more than 90% of accuracy. The learning rate is reduced by a factor of 50% every 5 epochs if the model doesn't improve. If 12 epochs go by and no improvement is seen, the model stops. As such, we don't have to worry with the amount of epochs in total as we will see later.

# In[ ]:


reduce_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, verbose=1, 
    mode='auto', min_delta=1e-10, cooldown=0, min_lr=0
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=12, verbose=1, mode='auto',
    baseline=None, restore_best_weights=False
)


# These callback functions are responsible for logging accuracy and loss results for each epoch and for checkpoint the model's weights, in case we have to stop and resume training later.

# In[ ]:


csv_logger = tf.keras.callbacks.CSVLogger(
    'cifar10.epoch.results.csv', separator='|', append=False)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "cifar10.partial.hdf5", save_weights_only=True, mode='auto', 
    save_freq='epoch', verbose=0
)


# Time to define the model and print summary to check if everything is correctly set before training.

# In[ ]:


model = define_model()
model.summary()


# # Baseline results
# 
# You can find some baseline replicable results [on this dataset on the project page for cuda-convnet](http://code.google.com/p/cuda-convnet/). These results were obtained with a convolutional neural network. Briefly, they are 18% test error without data augmentation and 11% with. Additionally, [Jasper Snoek](http://www.cs.toronto.edu/~jasper/) has a [paper](http://hips.seas.harvard.edu/content/practical-bayesian-optimization-machine-learning-algorithms) in which he used Bayesian hyperparameter optimization to find nice settings of the weight decay and other hyperparameters, which allowed him to obtain a test error rate of 15% (without data augmentation) using the architecture of the net that got 18%.
# 
# [Rodrigo Benenson](http://rodrigob.github.com/) has been kind enough to collect results on CIFAR-10/100 and other datasets on his website; [click here](http://rodrigob.github.com/are_we_there_yet/build/classification_datasets_results.html) to view.

# The callback functions will take care that the model doesn't train for a long time, so we will pick a large number to use as total training epochs. If I'm not mistaken, this model will stop before reaching 90 epochs.

# In[ ]:


epochs = 1000

history = model.fit(
    train_generator, 
    steps_per_epoch=x_train.shape[0]//batch_size, 
    epochs=epochs,  
    validation_data=(x_test, y_test), 
    callbacks=[csv_logger, reduce_learning_rate, early_stopping, model_checkpoint],
    verbose=1
)


# Let's evaluate the results.

# In[ ]:


_, acc = model.evaluate(x_test, y_test, verbose=0)
print('> %.3f' % (acc * 100.0))


# As you can see, with this simple model and little processing time we are not far from the best performers.

# I like to save the model just to keep track of changes.

# In[ ]:


model.save('cifar10.h5', overwrite=True, include_optimizer=True, save_format='h5')


# Let's print the learning evolution.

# In[ ]:


summarize_diagnostics(history)


# And check if the logging of accuracy and loss is okay.

# In[ ]:


res = pd.read_csv('cifar10.epoch.results.csv', sep='|')
res.tail()


# Let's check if the saved model is okay.

# In[ ]:


model_load_tf = tf.keras.models.load_model('cifar10.h5')
model_load_tf.summary()


# In[ ]:


test_loss, test_acc = model_load_tf.evaluate(x_test, y_test)
print(f"Accuracy: {test_acc} Loss: {test_loss}")


# Let's check how the model learn about the classes.

# In[ ]:


Y_test = np.argmax(y_test, axis=1)
y_pred = model.predict_classes(x_test)
print(classification_report(Y_test, y_pred))


# Doubts, suggestions, rants, please, comment down below. Best regards to all.

# In[ ]:





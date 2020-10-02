#!/usr/bin/env python
# coding: utf-8

# # MNIST solution exploration [0.99542]

# This kernel was created with the intention on exploring different CNNs to have a benchmark on what works best on a dataset of this kind of images (i.e. numbers, characters, etc.). I started using a fully connected network followed by a CNN network I had used in the past for the MNIST dataset when I was studing a Machine Learning Nanodegree from Udacity. From that point on, I wanted to try to extend  that CNN with new ideas, which I mainly took from some notebooks (mentioned below).<br>
# I saw a couple of different notebooks but decided to try to reproduce on my own those ideas that I found to be more interesting and that were something completely new to me. I was actually expecting to see some improvements on my predictions as I was introducing these ideas to this kernel (or a combination of them). Unfortunatelly I found out (as it should be expected by some other more experienced kaggler), that a combination of such ideas do not necessarily bring an improvement or sometimes the improvement comes with a (significant) additional computational cost.<br><br>
# The kernels from which I decided to  take some ideas are:<br>
# * [deep-neural-network-keras-way](https://www.kaggle.com/poonaml/deep-neural-network-keras-way/notebook)
# * [25 Million Images! \[0.99757\] MNIST](https://www.kaggle.com/cdeotte/25-million-images-0-99757-mnist)
# * [How to choose CNN Architecture MNIST](https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist)
# 
# I found particularly interesting and full of insight those notebooks from Chris Deotte.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import utils as tf_utils
from keras.callbacks.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Lambda
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras import regularizers


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Loading the data...

# In[ ]:


mnist_train_complete = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
mnist_test_complete = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

mnist_train_complete.head(5)


# In[ ]:


# preparing the training and testing sets, separating the training pictures of the numbers (i.e. train_x)
# from their label (i.e train_y).
# We set here also the data types as int32
train_y = mnist_train_complete.iloc[:, 0].values.astype('int32')
train_x = mnist_train_complete.iloc[:, 1:].values.astype('float32')
test_x = mnist_test_complete.values.astype('float32')

# reshaping the training and testing sets to have each digit image of 28 by 28 pixels
train_x = train_x.reshape(train_x.shape[0], 28, 28)
test_x = test_x.reshape(test_x.shape[0], 28, 28)


# # Visualizing some digit images

# How does an image of this dataset looks like actually?

# In[ ]:


for i in range (10,14):
    plt.subplot(330 + i+1)
    plt.imshow(train_x[i], cmap=plt.get_cmap('gray'))
    plt.title(train_y[i])


# In[ ]:


def visualize_detail(img, ax):
    ax.imshow(img, cmap='gray')
    width, height = img.shape
    threshold = img.max()/2.5
    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y], 2)), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img[x][y] < threshold else 'black')

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111)
    
visualize_detail(train_x[10], ax)


# # Preprocessing

# In[ ]:


# Normalizing the training and testing sets
train_x = train_x.astype('float32')/np.max(train_x)
test_x = test_x.astype('float32')/np.max(test_x)

# center the normalized data around zero
mean = np.std(train_x)
train_x -= mean
mean = np.std(test_x)
test_x -= mean


# In[ ]:


# creating the training and validationg sets
splitted_train_X, splitted_test_X, splitted_train_y, splitted_test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=81)

# one-hot encoding the training and validation sets
ohe_splitted_train_y = tf_utils.to_categorical(splitted_train_y, 10)
ohe_splitted_test_y = tf_utils.to_categorical(splitted_test_y, 10)

# print first one-hot training labels
print('One-hot labels:')
print(splitted_train_y[:10])


# # Solution 1. Model using fully connected NNs

# To begin with, I will try a fully connected NN consisting of a layer of 512 neurons, followed by a dropout of 20%, followed by another layer of 512 neurons and another dropout of 20% and finally a layer of 10 neurons with softmax activation. Both the first layers of 512 neurons have 'relu' activation.

# In[ ]:


# define a fully connected NNs model
model_sol_1 = tf.keras.models.Sequential()
model_sol_1.add(tf.keras.layers.Flatten(input_shape = splitted_train_X.shape[1:]))
model_sol_1.add(tf.keras.layers.Dense(512, activation='relu'))
model_sol_1.add(tf.keras.layers.Dropout(0.2))
model_sol_1.add(tf.keras.layers.Dense(512, activation='relu'))
model_sol_1.add(tf.keras.layers.Dropout(0.2))
model_sol_1.add(tf.keras.layers.Dense(10, activation='softmax'))

# summary of model
model_sol_1.summary()


# I compiled this model using 'rmsprop' optimization, 'categorical_crossentropy' for loss measurement and accuracy as metrics measurement. I am also curious about the accuracy of this model before it has being even trained.

# In[ ]:


# compile the model
model_sol_1.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[ ]:


# evaluate test accuracy
score = model_sol_1.evaluate(splitted_test_X, ohe_splitted_test_y, verbose=0)
accuracy = 100 * score[1]

# print test accuracy
print('Test accuracy: %4f%%' % accuracy)


# Train the model:

# In[ ]:


# checkpointer to save the best weihts
checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5', verbose=1, save_best_only=True)

hist_sol_1 = model_sol_1.fit(splitted_train_X, ohe_splitted_train_y, batch_size=128, epochs=10,
                 validation_split=0.2, callbacks=[checkpointer],
                 verbose=2, shuffle=True)


# ## Complexity graph of Solution 1

# In[ ]:


# plot the losses
plt.figure(figsize=(10,5))
plt.plot(hist_sol_1.history['loss'], linestyle="--")
plt.plot(hist_sol_1.history['val_loss'], linestyle="-.")
plt.title('model losses')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(["loss", "val_loss"], loc='upper left')
axes = plt.gca()
plt.show()


# Accuracy gotten after training:

# In[ ]:


#load the weights that resulted in the minimal validation loss
model_sol_1.load_weights('mnist.model.best.hdf5')

score = model_sol_1.evaluate(splitted_test_X, ohe_splitted_test_y, verbose=0)
accuracy = 100 * score[1]

#print test accuracy
print('Test accuracy: %.4f%%' % accuracy)


# ## Making predictions using Solution 1

# In[ ]:


predictions = model_sol_1.predict(test_x)
predictions = [ np.argmax(x) for x in predictions ]


# In[ ]:


# prepare submission
submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
submission.drop('Label', axis=1, inplace=True)
submission['Label'] = predictions
submission.to_csv('submission1.csv', index=False)


# The prediction obtained by this solution yielded a score of 0.96385.

# # Solution 2. Model using Convolutional NNs

# In this solution I implement a Convolutional Network to replace my Fully Connected Neural Network from solution1.<br>
# It is now well known that convolutional neural networks achieve a superior performance than fully connected networks with way less computational cost. <br>
# The architectures designed for any type of classification contain a couple of layers of fully connected nodes at the end, just before the output. This is still needed because even though CNNs can learn more significant information, fully connected nodes are in charge -at least- of the classification part - by means of softmax activation.

# In[ ]:


extended_splitted_train_X = splitted_train_X[..., tf.newaxis]
extended_splitted_test_X = splitted_test_X[..., tf.newaxis]
extended_splitted_test_X.shape


# In[ ]:


# define a Convolutional NNs model
model_sol_2 = Sequential()
model_sol_2.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=extended_splitted_train_X.shape[1:]))
model_sol_2.add(MaxPooling2D(pool_size=2))
model_sol_2.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model_sol_2.add(MaxPooling2D(pool_size=2))
model_sol_2.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model_sol_2.add(MaxPooling2D(pool_size=2))

# Converts our 3D feature maps to 1D features vectors
model_sol_2.add(Flatten())
model_sol_2.add(Dense(64))
model_sol_2.add(Activation('relu'))
model_sol_2.add(Dropout(0.2))
model_sol_2.add(Dense(10, activation='softmax'))

# summary of model
#model_sol_2.summary()


# I compiled this model using 'rmsprop' optimization, 'categorical_crossentropy' for loss measurement and accuracy as metrics measurement. I am also curious about the accuracy of this model before it has being even trained.

# In[ ]:


# compile the model
model_sol_2.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[ ]:


# evaluate test accuracy
score = model_sol_2.evaluate(extended_splitted_test_X, ohe_splitted_test_y, verbose=0)
accuracy = 100 * score[1]

# print test accuracy
print('Test accuracy: %4f%%' % accuracy)


# In[ ]:


# checkpointer to save the best weihts
checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5', verbose=1, save_best_only=True)

hist_sol_2 = model_sol_2.fit(extended_splitted_train_X, ohe_splitted_train_y, batch_size=128,
                             epochs=10, callbacks=[checkpointer],
                             verbose=2, validation_data=(extended_splitted_test_X, ohe_splitted_test_y), shuffle=True)


# ## Complexity graph of Solution 2

# In[ ]:


# plot the losses
plt.figure(figsize=(10,5))
plt.plot(hist_sol_2.history['loss'], linestyle="--")
plt.plot(hist_sol_2.history['val_loss'], linestyle="-.")
plt.title('model losses')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(["loss", "val_loss"], loc='upper left')
axes = plt.gca()
plt.show()


# In[ ]:


#load the weights that resulted in the minimal validation loss
model_sol_2.load_weights('mnist.model.best.hdf5')

score = model_sol_2.evaluate(extended_splitted_test_X, ohe_splitted_test_y, verbose=0)
accuracy = 100 * score[1]

#print test accuracy
print('Test accuracy: %.4f%%' % accuracy)


# ## Making predictions using Solution 2

# In[ ]:


# extend the test imagae set with an additional dimension
extended_test_x = test_x[..., tf.newaxis]
predictions = model_sol_2.predict(extended_test_x)
predictions = [ np.argmax(x) for x in predictions ]

# prepare submission
submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
submission.drop('Label', axis=1, inplace=True)
submission['Label'] = predictions
submission.to_csv('submission2.csv', index=False)


# The prediction obtained by this solution yielded a score of 0.98671.

# # Solution 3. Model using Convolutional NNs with data augmentation

# In this solution I make use of the model architecture of solution number 2 but I also implement data augmentation for the training. <br>
# Data augmentation that I am implementing here consist of:
# * rotating the image with a range of -10 to 10 degrees
# * shifting 10 percent the image both widthwise and heightwise directions
# * zooming up to a 10 percent the images
# 
# These augmentation is applied randomly and differently to each image.

# ## Augmenting an image

# In[ ]:


# define a data augmentator for our images
image_augmentator = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    # rescale=1./255,
    shear_range=0.2,
    zoom_range=0.1,
    fill_mode='nearest')

# define size of batch
batch_size = 32

train_batches = image_augmentator.flow(extended_splitted_train_X, ohe_splitted_train_y, batch_size=batch_size)
val_batches = image_augmentator.flow(extended_splitted_test_X, ohe_splitted_test_y, batch_size=batch_size)


# Let's look at an example of data augmentation:

# In[ ]:


example_img = train_x[10][..., tf.newaxis]
transf_params = { 'theta':15., 'tx':0.1, 'ty':0.1, 'shear':0.2 }
augmented_image = image_augmentator.apply_transform(example_img, transf_params)

# reducing dimensinoality to two
twoDim_image = augmented_image[:, :, 0]

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111)
visualize_detail(twoDim_image, ax)


# In[ ]:


# define a Convolutional NNs model (solution number 3)
model_sol_3 = Sequential()
model_sol_3.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu', input_shape=extended_splitted_train_X.shape[1:]))
model_sol_3.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
model_sol_3.add(MaxPooling2D(pool_size=2))
model_sol_3.add(Dropout(0.1))
model_sol_3.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
model_sol_3.add(MaxPooling2D(pool_size=2))

# Converts our 3D feature maps to 1D features vectors
model_sol_3.add(Flatten())
model_sol_3.add(Dense(64))
model_sol_3.add(Activation('relu'))
model_sol_3.add(Dropout(0.2))
model_sol_3.add(Dense(10, activation='softmax'))

# summary of model
#model_sol_3.summary()


# In[ ]:


# compile the model
model_sol_3.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[ ]:


# checkpointer to save the best weihts
checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5', verbose=1, save_best_only=True)

hist_sol_3 = model_sol_3.fit_generator(generator=train_batches, steps_per_epoch =extended_splitted_train_X.shape[0] // batch_size,
                                       epochs=32, callbacks=[checkpointer],
                                       validation_data=val_batches, validation_steps=extended_splitted_test_X.shape[0] // batch_size,
                                       verbose=2)


# ## Complexity graph of Solution 3

# In[ ]:


# plot the losses
plt.figure(figsize=(10,5))
plt.plot(hist_sol_3.history['loss'], linestyle="--")
plt.plot(hist_sol_3.history['val_loss'], linestyle="-.")
plt.title('model losses')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(["loss", "val_loss"], loc='upper left')
axes = plt.gca()
plt.show()


# In[ ]:


#load the weights that resulted in the minimal validation loss
model_sol_3.load_weights('mnist.model.best.hdf5')

score = model_sol_3.evaluate(extended_splitted_test_X, ohe_splitted_test_y, verbose=0)
accuracy = 100 * score[1]

#print test accuracy
print('Test accuracy: %.4f%%' % accuracy)


# ## Making predictions using Solution 3

# In[ ]:


# extend the test imagae set with an additional dimension
extended_test_x = test_x[..., tf.newaxis]
predictions = model_sol_3.predict(extended_test_x)
predictions = [ np.argmax(x) for x in predictions ]

# prepare submission
submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
submission.drop('Label', axis=1, inplace=True)
submission['Label'] = predictions
submission.to_csv('submission3.csv', index=False)


# The prediction obtained by this solution yielded a score of 0.98885.

# # Solution 4. Model using Convolutional NNs with data augmentation and batch normalization

# For this solution I wanted to add Batch normalization to my previous solution. I wanted to see if there were any improvement on doing this. Here I want to see / compare if there are any significant difference in adding more feature maps to my CNN. So I prepared two CNNs with batch normalization (again, taking my previous solution as a base). The first CNN (i.e. solution 4_1) with 16 feature maps in all my convolutional layers an the second solution (i.e. solution 4_2) with 32 fetaure maps in its convolutional layers.

# ## Solution 4_1 - Convolutional layers with 16 feature maps

# In[ ]:


# define a Convolutional NNs model (solution number 4)
model_sol_4_1 = Sequential()
model_sol_4_1.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu', input_shape=extended_splitted_train_X.shape[1:]))
model_sol_4_1.add(BatchNormalization())
model_sol_4_1.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
model_sol_4_1.add(MaxPooling2D(pool_size=2))
model_sol_4_1.add(Dropout(0.1))
model_sol_4_1.add(BatchNormalization())
model_sol_4_1.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
model_sol_4_1.add(MaxPooling2D(pool_size=2))

# Converts our 3D feature maps to 1D features vectors
model_sol_4_1.add(Flatten())
model_sol_4_1.add(BatchNormalization())
model_sol_4_1.add(Dense(64))
model_sol_4_1.add(Activation('relu'))
model_sol_4_1.add(Dropout(0.2))
model_sol_4_1.add(BatchNormalization())
model_sol_4_1.add(Dense(10, activation='softmax'))

# summary of model
#model_sol_4_1.summary()


# In[ ]:


# compile the model
model_sol_4_1.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[ ]:


# checkpointer to save the best weihts
checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5', verbose=1, save_best_only=True)

hist_sol_4 = model_sol_4_1.fit_generator(generator=train_batches, steps_per_epoch =extended_splitted_train_X.shape[0] // batch_size,
                                       epochs=32, callbacks=[checkpointer],
                                       validation_data=val_batches, validation_steps=extended_splitted_test_X.shape[0] // batch_size,
                                       verbose=2)


# ## Complexity graph of Solution 4_1

# In[ ]:


# plot the losses
plt.figure(figsize=(10,5))
plt.plot(hist_sol_4.history['loss'], linestyle="--")
plt.plot(hist_sol_4.history['val_loss'], linestyle="-.")
plt.title('model losses')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(["loss", "val_loss"], loc='upper left')
axes = plt.gca()
plt.show()


# In[ ]:


#load the weights that resulted in the minimal validation loss
model_sol_4_1.load_weights('mnist.model.best.hdf5')

score = model_sol_4_1.evaluate(extended_splitted_test_X, ohe_splitted_test_y, verbose=0)
accuracy = 100 * score[1]

#print test accuracy
print('Test accuracy: %.4f%%' % accuracy)


# ## Solution 4_2 - Convolutional layers with 32 feature maps

# In[ ]:


model_sol_4_2 = Sequential()
model_sol_4_2.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=extended_splitted_train_X.shape[1:]))
model_sol_4_2.add(BatchNormalization())
model_sol_4_2.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model_sol_4_2.add(MaxPooling2D(pool_size=2))
model_sol_4_2.add(Dropout(0.1))
model_sol_4_2.add(BatchNormalization())
model_sol_4_2.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model_sol_4_2.add(MaxPooling2D(pool_size=2))

# Converts our 3D feature maps to 1D features vectors
model_sol_4_2.add(Flatten())
model_sol_4_2.add(BatchNormalization())
model_sol_4_2.add(Dense(64))
model_sol_4_2.add(Activation('relu'))
model_sol_4_2.add(Dropout(0.2))
model_sol_4_2.add(BatchNormalization())
model_sol_4_2.add(Dense(10, activation='softmax'))

# summary of model
#model_sol_4_2.summary()


# In[ ]:


# compile the model
model_sol_4_2.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[ ]:


# checkpointer to save the best weihts
checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5', verbose=1, save_best_only=True)

hist_sol_4 = model_sol_4_2.fit_generator(generator=train_batches, steps_per_epoch=extended_splitted_train_X.shape[0] // batch_size,
                                       epochs=32, callbacks=[checkpointer],
                                       validation_data=val_batches, validation_steps=extended_splitted_test_X.shape[0] // batch_size,
                                       verbose=2)


# ## Complexity graph of Solution 4_2

# In[ ]:


# plot the losses
plt.figure(figsize=(10,5))
plt.plot(hist_sol_4.history['loss'], linestyle="--")
plt.plot(hist_sol_4.history['val_loss'], linestyle="-.")
plt.title('model losses')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(["loss", "val_loss"], loc='upper left')
axes = plt.gca()
plt.show()


# In[ ]:


#load the weights that resulted in the minimal validation loss
model_sol_4_2.load_weights('mnist.model.best.hdf5')

score = model_sol_4_2.evaluate(extended_splitted_test_X, ohe_splitted_test_y, verbose=0)
accuracy = 100 * score[1]

#print test accuracy
print('Test accuracy: %.4f%%' % accuracy)


# ## Making predictions using Solution 4

# From these two CNNs, that with more feature maps (i.e. solution 4_2 with 32 feature maps) yielded a higher test accuracy. I used that model to make the predictions on this part of the notebook.

# In[ ]:


extended_test_x = test_x[..., tf.newaxis]
predictions = model_sol_4_2.predict(extended_test_x)
predictions = [ np.argmax(x) for x in predictions ]

# prepare submission
submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
submission.drop('Label', axis=1, inplace=True)
submission['Label'] = predictions
submission.to_csv('submission4.csv', index=False)


# **The prediction obtained by this solution yielded a score of 0.99542.**

# # Solution 5. Adding a Lambda layer to my last CNN

# I will try again, but this time I will add a Lambda layer at the input of my NN. This layer input will center the data around zero mean and unite variance (I got this from [Poonam Ligade's notebook](https://www.kaggle.com/poonaml/deep-neural-network-keras-way/notebook)). This means I have to take again the original data (not preprocessed data), and add another dimension.<br>
# The Lambda layer will perform a "Standardize" function (defined later some blocks below) which will do the preprocessing to each one of the images (i.e. as mentioned before, center the data around zero mean and unit variance).
# <br>
# I will also change the optimizer of my network in favor of [Adam](https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c) and will add [Ridge regression](https://towardsdatascience.com/ridge-regression-for-better-usage-2f19b3a202db) to the first convolutional layer of my neural network in an attempt to penalize more those feature of the images that does not help the algorithm to improve during its training.

# In[ ]:


# create new datasets out of the original files provided by kaggle (to avoid confussions with other
# variables created in other sections of this notebook and because I need this data without the first preprocessing steps
# I performed in my previous solutions)
train_y_sol5 = mnist_train_complete.iloc[:, 0].values.astype('int32')
train_x_sol5 = mnist_train_complete.iloc[:, 1:].values.astype('float32')
test_x_sol5 =  mnist_test_complete.values.astype('float32')

# reshaping the new training and testing sets to have each digit image of 28 by 28 pixels
train_x_sol5 = train_x_sol5.reshape(train_x_sol5.shape[0], 28, 28)
test_x_sol5 = test_x_sol5.reshape(test_x_sol5.shape[0], 28, 28)

# add another dimension to the training data
train_x_sol5 = train_x_sol5[..., tf.newaxis]
test_x_sol5  = test_x_sol5[..., tf.newaxis]


# Below there is the definition of the standardizer function, which will "preprocess" each one of the images as they are fed to the CNN.

# In[ ]:


# new preprocessing of data (to be applied to each individual image by the Lamda layer)
mean_px = train_x_sol5.mean().astype(np.float32)
std_px = train_x_sol5.std().astype(np.float32)

# define the function that will be performed by our Lambda layer on each of the input images
def standardize(x): 
    return (x-mean_px)/std_px


# In[ ]:


# cross validation
s5_train_x, s5_test_x, s5_train_y, s5_test_y = train_test_split(train_x_sol5, train_y_sol5,
                                                                test_size=0.2,
                                                                random_state=81)
# one-hot encoding the target labels
ohe_s5_train_y = tf_utils.to_categorical(s5_train_y, 10)
ohe_s5_test_y = tf_utils.to_categorical(s5_test_y, 10)

# create new image generators using the same image_augmentator created previously,
# but with a different number of batches (prevous batch size was 32).
train_batches_sol5 = image_augmentator.flow(s5_train_x, ohe_s5_train_y, batch_size=64)
val_batches_sol5 = image_augmentator.flow(s5_test_x, ohe_s5_test_y, batch_size=64)


# In[ ]:


model_sol_5 = Sequential()
model_sol_5.add(Lambda(standardize, input_shape=(28,28,1)))
model_sol_5.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu',
                 kernel_regularizer=regularizers.l2(0.1),
                 ))
model_sol_5.add(BatchNormalization())
model_sol_5.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'
                ))
model_sol_5.add(MaxPooling2D(pool_size=2))
#model.add(Dropout(0.1))

model_sol_5.add(BatchNormalization())
model_sol_5.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'
         ))
model_sol_5.add(MaxPooling2D(pool_size=2))

# Converts our 3D feature maps to 1D features vectors
model_sol_5.add(Flatten())
model_sol_5.add(BatchNormalization())
model_sol_5.add(Dense(64))
model_sol_5.add(Activation('relu'))
model_sol_5.add(Dropout(0.2))
model_sol_5.add(BatchNormalization())
model_sol_5.add(Dense(10, activation='softmax'))

# summary of model
#model_sol_5.summary()


# In[ ]:


# compile the model
model_sol_5.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])


# In[ ]:


# checkpointer to save the best weihts
checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5', verbose=1, save_best_only=True)

hist_sol_5 = model_sol_5.fit_generator(generator=train_batches_sol5, steps_per_epoch=s5_train_x.shape[0] // 64,
                                       epochs=32, callbacks=[checkpointer],
                                       validation_data=val_batches_sol5, validation_steps=s5_test_x.shape[0] // 64, verbose=2)


# ## Complexity graph of Solution 5

# In[ ]:


# plot the losses
plt.figure(figsize=(10,5))
plt.plot(hist_sol_5.history['loss'], linestyle="--")
plt.plot(hist_sol_5.history['val_loss'], linestyle="-.")
plt.title('model losses')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(["loss", "val_loss"], loc='upper left')
axes = plt.gca()
plt.show()


# In[ ]:


#load the weights that resulted in the minimal validation loss
model_sol_5.load_weights('mnist.model.best.hdf5')

score = model_sol_5.evaluate(s5_test_x, ohe_s5_test_y, verbose=0)
accuracy = 100 * score[1]

#print test accuracy
print('Test accuracy: %.4f%%' % accuracy)


# ## Making predictions using Solution 5

# In[ ]:


predictions = model_sol_5.predict(test_x_sol5)
predictions = [ np.argmax(x) for x in predictions ]

# prepare submission
submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
submission.drop('Label', axis=1, inplace=True)
submission['Label'] = predictions
submission.to_csv('submission5.csv', index=False)


# The prediction obtained by this solution yielded a score of 0.99257.<br>
# **Note**: After obtaining a lower score, I did try again the same network with the difference of removing the Dropout of 1% between convolutional layers (i.e. see the commented dropout), obtaining a score of **0.99342** (still lower than the score obtained with Solution 4).

# # Solution 6. Two step training approach

# The network on Solution 5 threw actually a worst score than Solution 4 even though I added Ridge regression (i.e. L2 regularization) to the first layer. To me seems to be clear that this L2 regularization hindered a little bit the training. This might be due to the fact that some "features" (i.e. pixels) do not have always the same "meaning". The same pixel will have different information on different images, depending on the position of the number that an image contains, plus the data augmentation that comes on top of some images. Therefore, it is my opinion that using L2 regularization brings not benefit at all here.
# <br><br>
# In this solution I wanted to try a 2 step training approach (which I also saw in [Poonam Ligade's notebook](https://www.kaggle.com/poonaml/deep-neural-network-keras-way/notebook) but I did not think it would be really necessary because I though that by using L2 regularization I would get way better results - I was cleary wrong about L2).<br>
# Since my las solution (i.e. Solution 5) produced a negative result - meaning no improvement in the score - I will retake solution number 4 as the base of this solution (i.e. Solution 6) with the only addition of the lambda layer at the beginning of the CNN architecture and using adam as the optimizer when compiling the model. Also, before attempting the second training step, I will set the learning rate of the optimazer to 1% instead of its default value of 0.01%.<br>
# Let's see what this two trainig approach (without L2 regularization) have to offer!

# In[ ]:


model_sol_6 = Sequential()
model_sol_6.add(Lambda(standardize, input_shape=(28,28,1)))
model_sol_6.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model_sol_6.add(BatchNormalization())
model_sol_6.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model_sol_6.add(MaxPooling2D(pool_size=2))
model_sol_6.add(Dropout(0.1))
model_sol_6.add(BatchNormalization())
model_sol_6.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model_sol_6.add(MaxPooling2D(pool_size=2))

# Converts our 3D feature maps to 1D features vectors
model_sol_6.add(Flatten())
model_sol_6.add(BatchNormalization())
model_sol_6.add(Dense(64))
model_sol_6.add(Activation('relu'))
model_sol_6.add(Dropout(0.2))
model_sol_6.add(BatchNormalization())
model_sol_6.add(Dense(10, activation='softmax'))

# summary of model
#model_sol_6.summary()


# In[ ]:


# compile the model
model_sol_6.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])


# In[ ]:


# checkpointer to save the best weihts
checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5', verbose=1, save_best_only=True)

hist_sol_6 = model_sol_6.fit_generator(generator=train_batches_sol5, steps_per_epoch=s5_train_x.shape[0] // 64,
                                       epochs=32, callbacks=[checkpointer],
                                       validation_data=val_batches_sol5, validation_steps=s5_test_x.shape[0] // 64, verbose=2)


# ## Complexity graph of Solution 6

# In[ ]:


# plot the losses
plt.figure(figsize=(10,5))
plt.plot(hist_sol_6.history['loss'], linestyle="--")
plt.plot(hist_sol_6.history['val_loss'], linestyle="-.")
plt.title('model losses')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(["loss", "val_loss"], loc='upper left')
axes = plt.gca()
plt.show()


# In[ ]:


#load the weights that resulted in the minimal validation loss
model_sol_6.load_weights('mnist.model.best.hdf5')

score = model_sol_6.evaluate(s5_test_x, ohe_s5_test_y, verbose=0)
accuracy = 100 * score[1]

#print test accuracy
print('Test accuracy: %.4f%%' % accuracy)


# Performing the second step of the training process...

# In[ ]:


model_sol_6.optimizer.lerning_rate=0.01
gen = ImageDataGenerator()
batches = gen.flow(train_x_sol5, tf_utils.to_categorical(train_y_sol5, 10), batch_size=64)
hist_sol_6 = model_sol_6.fit_generator(generator=batches, steps_per_epoch=train_x_sol5.shape[0] // 64,
                          epochs=50, verbose=2)
# I didn't use a callback on this training step becuase the 'checkpointer' callback I defined works only when
# the model produces validation loss metrics. In order to do that, I need to pass validation data to the
# fit_generator method. For this second training step I did not pass such validation data becase we do not have 
# test data to validate against - now I am using the complete set of images provided by kaggle.


# ## Making predictions using Solution 6

# In[ ]:


predictions = model_sol_6.predict(test_x_sol5)
predictions = [ np.argmax(x) for x in predictions ]

# prepare submission
submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
submission.drop('Label', axis=1, inplace=True)
submission['Label'] = predictions
submission.to_csv('submission6.csv', index=False)


# The prediction obtained by this solution yielded a score of 0.99185, but still lower than our best score of Solution 4.

# In[ ]:


os.remove('submission1.csv')
os.remove('submission2.csv')
os.remove('submission3.csv')
os.remove('submission4.csv')
os.remove('submission5.csv')
os.remove('submission6.csv')


# # Final submission

# The final submission is done using the model that yielded in the best score - in this notebook that is the model 4_2 - I just decided to compile it using 'Adam' as optimizer because it finds local minima faster (even though it has its own cons).

# In[ ]:


final_train_x = train_x[..., tf.newaxis]
final_ohe_train_y = tf_utils.to_categorical(train_y, 10)
final_train_batches = image_augmentator.flow(final_train_x, final_ohe_train_y, batch_size=64)

final_model = Sequential()
final_model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=extended_splitted_train_X.shape[1:]))
final_model.add(BatchNormalization())
final_model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
final_model.add(MaxPooling2D(pool_size=2))
final_model.add(Dropout(0.1))
final_model.add(BatchNormalization())
final_model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
final_model.add(MaxPooling2D(pool_size=2))

# Converts our 3D feature maps to 1D features vectors
final_model.add(Flatten())
final_model.add(BatchNormalization())
final_model.add(Dense(64))
final_model.add(Activation('relu'))
final_model.add(Dropout(0.2))
final_model.add(BatchNormalization())
final_model.add(Dense(10, activation='softmax'))

# compile the model
final_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

final_model.fit_generator(generator=final_train_batches, steps_per_epoch=final_train_batches.n,
                          epochs=1, verbose=1)


# In[ ]:


predictions = final_model.predict(test_x[..., tf.newaxis])
predictions = [ np.argmax(x) for x in predictions ]

# prepare submission
submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
submission.drop('Label', axis=1, inplace=True)
submission['Label'] = predictions
submission.to_csv('submission.csv', index=False)


# # Conclusion

# For me it is difficult to stop trying to expand, modify and introduce more and more ideas into my model mainly because it is not easy to "see" where the line of enough is enough is drawn. A couple of times I lost myself trying more and more things that at the end didn't bring any significan benefit. I decided to remove those from this kernel. <br><br>
# From the work on this notebook, it seems to be that both L2 regularization and the computation done with the lambda layer are not bringing benefit to the problem of image classification in the context of MNIST. I think, the mean centered might be different on each image and this might not help to the learning process of the CNN - this could be analyzed in a separate notebook.<br>
# ~~From the complexity graphs, I can see that from all the solutions implemented in this notebook, solution 2 and solution 4 presented a certain level of stabilization on the last epochs. This is an indication that the CNN reached a local minima point and it was not getting out of it. Obviously such minima point was optimal in Solution 4~~.<br>
# After running the notebook again I realized that the complexity graphs were completely different. They looked all "spiky-bumpy" all over the graph. Maybe this is due to the fact that during training the model is validating with augmented images (as it is doing with the training) and in batches. Maybe it'd be needed a longer number of epochs and to train also with the not-augmented images until I can get lines that tend to be more stable (I,d love to hear opinions/knowledge on this point from other users ).
# 
# The [kernel](https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist) from Chris Deotte inspired me to make a similar investigation on whether the lamda layer I added in my CNN number 5, the L2 regularization and the initial preprocessing of the data I did at the begining bring any real improvement to the predictions.<br><br>
# **If you have any suggestion or observation to this work, I will be more than happy to hear from you - and to learn more!**

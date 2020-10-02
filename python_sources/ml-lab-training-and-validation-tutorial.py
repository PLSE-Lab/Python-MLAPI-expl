#!/usr/bin/env python
# coding: utf-8

# ## Model creation, training and validation
# 
# Import all libraries
# - use Numpy to manipulate the pictures and input
# - use Keras to construct the model using Tensorflow under the covers
# - use sklearn.model_selection

# In[ ]:


import os
import cv2
from matplotlib import pyplot as plt

import numpy as np
import keras
from keras.models import Sequential
import keras.layers as layers
from keras import optimizers

import sklearn.model_selection as model_selection


# ## Load previously processed data
# 
# Load the arrays that we have previously processed into X and Y

# In[ ]:


#TODO load the X and Y dataset that we have saved from process_images

dataset_path = '../input/arrays/Arrays'

os.listdir(dataset_path)


# Look at np.load
# [numpy load ](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.load.html)

# In[ ]:


X = none
Y = none


# Print the shape of X and Y to make sure that they are the right shape

# In[ ]:


print('X shape : {}  Y shape: {}'.format(X.shape, Y.shape))


# As a sanity check, display the 700th picture and its label value

# In[ ]:


plt.imshow(X[700], cmap='gray')


# In[ ]:


print(Y[700]) # one-hot labels starting at zero


# ## Train and Validation
# 
# Let's split the data into training and validation sets
# 
# We should split both the input and labels into 2 sets. Typical split would be 70-30 or 80-20. 
# Let do a 80-20. 
# 
# We have imported model_selection module that has helper functions to perform the data split
# 
# Please see
# 
# http://scikit-learn.org/stable/modules/cross_validation.html#computing-cross-validated-metrics

# In[ ]:


#TODO: implement training and validation dataset split

def split_data(X, Y, validation_size):
    return None

Xtrain, Xtest, Ytrain, Ytest = split_data(X, Y, 0.2)


# In[ ]:


print('Xtrain shape {} Ytrain shape {}'.format(Xtrain.shape, Ytrain.shape))
print('Xtest shape {} Ytest shape {}'.format(Xtest.shape, Ytest.shape))


# ## Construct the model
# 
# This model will have 3 conv layers followed by 2 dense (fully connected) layers
# 
# Conv -> Max Pooling -> Conv -> Max Pooling -> Conv -> Dense -> Dense
# 
# The last Dense layer is 10 wide and has a softmax activation to determine the probablity of the 10 classes i.e.
# digits 0 to 9
# 
# Please see https://keras.io/layers/convolutional/#conv2d
# see input_shape
# 
# Please see https://keras.io/models/about-keras-models/
# see getting information about a model 

# In[ ]:


#the first Conv2D layer needs to specify what it takes as an input. Since we have resized all the images to
#a specific size, please specify the input_shape

#TODO: print out information about the model to help visualize how layers are structured

model = Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1))) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.not_implemented()


# ## Compile the model with loss function and optimizer.
# 
# Also get the model to track accuracy as a metrics.
# 
# Because we have 10 classes, we use the default loss function categorical_crossentropy an use a well-known optimizer adam.

# In[ ]:



model.compile(loss='categorical_crossentropy',
             optimizer=optimizers.adadelta(),
             metrics=['accuracy'])


# ## Shape the batch input
# 
# Keras model require the input array to have the following shape:
# (m, w, h, c) where
# m is the number of images
# w, h are the width and height of the image and in this case, we have predetermined that w and h are 64.
# c is the number of channels in the images and in our case, since we have converted the images to greyscale, we only have 1 channel. The default configurtion of a Conv2D layer uses channel last, so we need to add a single channel as the last dimension to our input data Xtrain.
# 
# So for example if Xtrain has 1649 images of shape (1649, 64, 64), after we add the channel, we should end up with an input array of shape (1649, 64, 64, 1)
# 
# Similar thing should apply to Xtest

# In[ ]:


#TODO: implement add_channel_dim so that it adds a new dimension. Look at reshape and newaxis

def add_channel_dim(X):
    return None

Xtrain_batch = add_channel_dim(Xtrain)

Xtest_batch = add_channel_dim(Xtest)


# ## Train the model with X and Y

# In[ ]:


#train our model
history = model.fit(Xtrain_batch, Ytrain, batch_size=32, epochs=9, validation_data=(Xtest_batch, Ytest))


# ## Plot to see how well we do for accuracy for training vs validation

# In[ ]:


acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# ## Evaluate
# 
# Let's evaluate how well our current model is doing against the test set Xtest batch and Ytest batch
# 
# see https://keras.io/models/sequential/#the-sequential-model-api

# In[ ]:


#TODO: evaluate the model
eval_score = None

print('Evaluation score {}'.format(eval_score))


# In[ ]:


def display_img(img_path):
    img = cv2.imread(img_path)
    color_corrected = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(color_corrected)
    plt.title(img_path)
    plt.show()

#TODO: use the same function that you've implemented in process_image
# resize to 64 x 64 and greyscale
    
def get_gsimg(image_path):
    img = cv2.imread(image_path)
    resize_img = cv2.resize(img, (64, 64))
    gs_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)
    return gs_img


# ## Now to perform some prediction using photos we've never seen

# In[ ]:


#prediction

input_path = '../input/sign_lang_dataset/Inputs'

os.listdir(input_path)


# In[ ]:


display_img(os.path.join(input_path, 'sample_1.jpg'))

sample1 = get_gsimg(os.path.join(input_path, 'sample_1.jpg'))
sample1_batch = add_channel_dim(np.array(sample1).reshape((1, 64, 64)))


# In[ ]:


model.predict(sample1_batch)


# In[ ]:


display_img(os.path.join(input_path, 'sample_3.jpg'))

sample3 = get_gsimg(os.path.join(input_path, 'sample_3.jpg'))
sample3_batch = add_channel_dim(np.array(sample3).reshape((1, 64, 64)))


# In[ ]:


model.predict(sample3_batch)


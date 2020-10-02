# Homework Assignment 3 Due: 11/27/2018 11:59PM

# Setup:
# Please review the 'Prepare' https://ben.desire2learn.com/d2l/le/content/333415/viewContent/1740859/View
# 
# Objective: 
# Let's use keras in action.  Design a CNN to categorize 10 classes of clothing types. 
#
# Goal: 
# Train the network to get at least 90% accuracy or higher with tuning.
# Prefix your versions with @V1, @V2, @V3 and place your versions of tuning.
# 
# Information:
# Use the Fashion MNIST (included in Keras) 
# 60k examples of training data
# 10k examples of test data
# 10 classes
	# various types of clothing
	# unique shapes and coloration

######
###### ORIGINAL SCORE: 91.32%
###### VERSION 1 SCORE: 91.44%
###### VERSION 2 SCORE: 91.61%
###### VERSION 3 SCORE: 91.98%
###### VERSION 4 SCORE: 92.58%
###### VERSION 5 SCORE: 92.93%
######


###### USAGE: hit cmd+f, then search for @V# in each commented block, there are two lines of code, one is commented out, and one is not. 
###### Read the description of the version, and make the change necessary, then hit cmd+a to highlight all code, and run the code.

import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# Supress warning and informational messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Number of classes
#@V4
#@Verson4
#Change number of classes to 15 from 10
num_classes = 10
#num_classes = 15
#End Version 4



# the number of passes through the data
# sizes of the batch and # of epochs of data
batch_size = 128

epochs = 24

# input image dimensions --> size of our image
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets.
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# reshape the data for the keras backend understand 
# channel_<'first'><'last'>:  Specifies which data format convention Keras will follow. 
# 1 represents the grayscale --> pass the channels first parameter
# We want to address 'channel_first' issue because this can cause dimension mismatch errors in another backend such as Theano or CNTK
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# scale the pixel intensity from 0-255 to make it fit between 0-1 because the images are grayscale
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert class vectors to binary class metrics using one-hot encoding 
# All categorical data must be converted to numbers in other words the categorical variables is represented as binary vectors.
# 3 => 0 0 0 1 0 0 0 0 0 0 and 1 => 0 1 0 0 0 0 0 0 0 0
# The first requires that the categorical values be mapped to integer values
# Then, each integer value is represented as a binary vector that is all zero values except the index of the integer which is marked with a 1. 
# This prevents the model from assuming that the numerical class designation is an ordered number
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Define a sequential model
model = Sequential()

# 32 filters
# each filter uses a 3 x 3 kernal
# activation function of relu
# pass the first layer of input_shape
model.add(Conv2D(32, (3,3), activation='relu', input_shape = input_shape))



#@V1
#@Version 1
#Set strides value in the max pooling layer to 2.
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(MaxPooling2D(pool_size=(2,2), strides=2))
#End V1




# repeat the Conv2D and the pool to learn more about the data. 
model.add(Conv2D(32,(3,3), activation='relu'))





# Now we know the image features, we can use them to categorize our images into our classes of shirts, sneakers, purses, and etc
# Categorize the classes by using a Dense fully connected layers at the end of the network

# Implement a flatten layer to convert the 2D output of the convolution and max pool into a 1D feature vector we can classify. 
model.add(Flatten())

# Use a dense layer to combine the information
# @V5
# @Version5
# Change the size of the Dense Layer output arrays from 128 to 512
model.add(Dense(128, activation='relu'))
#model.add(Dense(512, activation='relu'))
# End @V5



# Add a drop out to avoid the overfitting in the network

# @Version 2
# @V2
#Change the drop out to 0.1 from 0.5
model.add(Dropout(0.5))
# model.add(Dropout(0.1))
# End V2

# @V3
# @Version3
# Change dropout from 0.5 to 0.25
# model.add(Dropout(0.25))
# End Version3

# Last layer we want to output dense fully connected layer provides probability of the classification
#model.add(Dense(num_classes, activation='softmax'))
model.add(Dense(num_classes, activation="softmax"))

# Measure loss, which is categorical across entropy and how we are going to optimize to minimize loss. 
# We train the data and evaluate how well each epoch of the training data fits the validation data
model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy'])

# Do this using the fit function. 
# Specify the training data, size of the batch, number of epochs, amount of data to display and the data used to validated the training after each epoch. 
# Train the model and test/validate the model with the test data after each cycle (epoch) through the training data
# Return the history of loss and accuracy for each epoch.
# A history dictionary that stores how the model changed as it was trained on the data. 
hist = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))

# Evaluate the model with the test data to get the scores on the 'real' data
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Visualize it.
import numpy as np
import matplotlib.pyplot as plt
epoch_list = list(range(1, len(hist.history['acc']) + 1)) # values for x axis [1, 2, ..., # of epochs]
plt.plot(epoch_list, hist.history['acc'], epoch_list, hist.history['val_acc'])
plt.legend(('Training Accuracy', 'Validation Accuracy'))
plt.show()

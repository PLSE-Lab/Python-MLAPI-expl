import pandas as pd
import numpy as np

from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv").values
test  = pd.read_csv("../input/test.csv").values

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

# input image dimensions
img_rows, img_cols = 28, 28

batch_size = 128 # Number of images used in each optimization step
nb_classes = 10 # One class per digit
nb_epoch = 20 #70 # Number of times the whole data is used to learn

# Reshape the data to be used by a Tensorflow CNN. Shape is
# (nb_of_samples, img_width, img_heigh, nb_of_color_channels)
X_train = train[:, 1:].reshape(train.shape[0], img_rows, img_cols, 1)
X_test = test.reshape(test.shape[0], img_rows, img_cols, 1)
in_shape = (img_rows, img_cols, 1)
y_train = train[:, 0] # First data is label (already removed from X_train)

# Make the value floats in [0;1] instead of int in [0;255]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices (ie one-hot vectors)
Y_train = np_utils.to_categorical(y_train, nb_classes)

#Display the shapes to check if everything's ok
# Write to the log:
print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print('X_test shape:', X_test.shape)

model = Sequential()
# For an explanation on conv layers see http://cs231n.github.io/convolutional-networks/#conv
# By default the stride/subsample is 1 and there is no zero-padding.
# If you want zero-padding add a ZeroPadding layer or, if stride is 1 use border_mode="same"
model.add(Convolution2D(12, 5, 5, activation = 'relu', input_shape=in_shape, init='he_normal'))

# For an explanation on pooling layers see http://cs231n.github.io/convolutional-networks/#pool
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(25, 5, 5, activation = 'relu', init='he_normal'))

model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the 3D output to 1D tensor for a fully connected layer to accept the input
model.add(Flatten())
model.add(Dense(480, activation = 'relu', init='he_normal'))
model.add(Dropout(0.5))
model.add(Dense(400, activation = 'relu', init='he_normal'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation = 'softmax', init='he_normal')) #Last layer with one output per class

# The function to optimize is the cross entropy between the true label and the output (softmax) of the model
# We will use adadelta to do the gradient descent see http://cs231n.github.io/neural-networks-3/#ada
model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=["accuracy"])

# Make the model learn
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)

# Scores
loss_and_metrics = model.evaluate(X_train, Y_train, batch_size=128)
# Write to the log:
print("LOSS AND METRICS :")
print(loss_and_metrics)
print("**********")

# Predict the label for X_test
Y_pred = model.predict_classes(X_test)

# Submission
submission = pd.DataFrame({
        "ImageId": np.arange(1,len(Y_pred)+1),
        "label": Y_pred
    })
submission.to_csv('digits_CNN.csv', index=False)

__author__ = 'KevinPatel: https://kaggle.com/patelkevin'
# DogsCats VS Keras

# This is simple script with many limitation due to run on Kaggle CPU server.
# Here we use simple CNN with low number of conv layers and filters.

# Include/Import Stuff :)

import os, cv2, random
import numpy as np
import pandas as pd

import matplotlib.pylab as plt
from random import shuffle

from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop # Also try with SGD and Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

# Preparing Data for consumption
# We resize the images here to 64x64 and sample 8% of the data to run on Kaggle Kernel servers.

train_path = "../input/train/"
test_path = "../input/test/"

ROWS = 64
COLS = 64
CHANNELS = 3

nb_epoch = 3 # Change to 100
batch_size = 16

nb_filters_1 = 32 # 64
nb_filters_2 = 64 # 128
nb_filters_3 = 128 # 256
nb_conv = 3

images      = [img for img in os.listdir(train_path)]
images_dog  = [img for img in os.listdir(train_path) if "dog" in img]
images_cat  = [img for img in os.listdir(train_path) if "cat" in img]
images_test = [img for img in os.listdir(test_path)]

# We now just take 8% subset of the entire data due to the limitation of the kaggle kernels.
# We will get less accuracy here, the more data we provide here the better aou accuracy will be but the training becomes slow.

train_dog = images_dog[:1000]
train_cat = images_cat[:1000]
valid_dog = images_dog[1000:1100]
valid_cat = images_cat[1000:1100]

train_list = train_dog + train_cat
valid_list = valid_dog + valid_cat
test_list = images_test[0:]

shuffle(train_list)

train = np.ndarray(shape=(len(train_list),ROWS, COLS))
train_color = np.ndarray(shape=(len(train_list), ROWS, COLS, CHANNELS), dtype=np.uint16)

valid = np.ndarray(shape=(len(valid_list),ROWS, COLS))
valid_color = np.ndarray(shape=(len(valid_list), ROWS, COLS, CHANNELS), dtype=np.uint16)

test = np.ndarray(shape=(len(test_list),ROWS, COLS))
test_color = np.ndarray(shape=(len(test_list), ROWS, COLS, CHANNELS), dtype=np.uint16)

# Labels for train_list

labels = np.ndarray(len(train_list))

for i, img_path in enumerate(train_list):
    img_color = cv2.imread(os.path.join(train_path, img_path), 1)
    img_color = cv2.resize(img_color, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
train[i] = img
train_color[i] = img_color

if "dog" in img_path:
    labels[i] = 0
else:
    labels[i] = 1

# Labels for valid_list

valid_labels = np.ndarray(len(valid_list))

for i, img_path in enumerate(valid_list):
    img_color = cv2.imread(os.path.join(train_path, img_path), 1)
    img_color = cv2.resize(img_color, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

valid[i] = img
valid_color[i] = img_color

if "dog" in img_path:
    valid_labels[i] = 0
else:
    valid_labels[i] = 1

# Test_list

for i, img_path in enumerate(test_list):
    img_color = cv2.imread(os.path.join(test_path, img_path), 1)
    img_color = cv2.resize(img_color, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

test[i] = img
test_color[i] = img_color
	
# Comments en cours

from keras.utils import np_utils

X_train = train_color / 255
X_valid = valid_color / 255
X_test  = test_color  / 255

y_train = np_utils.to_categorical(labels)
y_valid = np_utils.to_categorical(valid_labels)
num_classes = y_train.shape[1]

# Now that we have prepared our data, time to create our model!!

def cnn_model():
    model = Sequential()
    model.add(Convolution2D(nb_filters_1, nb_conv, nb_conv, activation="relu", input_shape=(64, 64, 3), border_mode='same'))
    model.add(Convolution2D(nb_filters_1, nb_conv, nb_conv, activation="relu", border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(nb_filters_2, nb_conv, nb_conv, activation="relu", border_mode='same'))
    model.add(Convolution2D(nb_filters_2, nb_conv, nb_conv, activation="relu", border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same'))
    model.add(Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same'))
# model.add(conv.Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same'))
# model.add(conv.Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu")) # 4096
    model.add(Dense(50, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

# Comipling the model.
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

# Building the model
model = cnn_model()

# Fitting the model
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=batch_size, nb_epoch=nb_epoch, verbose=2)

# Model Evaluation
scores = model.evaluate(X_valid, y_valid, verbose=0)

# Submission
submission = model.predict_classes(X_test, verbose=2)
pd.DataFrame({"id": list(range(1,len(test_color)+1)), "label": submission}).to_csv('submission.csv', index=False,header=True)

# End
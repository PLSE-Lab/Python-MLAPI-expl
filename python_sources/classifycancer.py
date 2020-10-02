#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# import necessary packages
import os

# initialize path to input directory of images
ORIG_DATASET = "/kaggle/input/dataset/Dataset"

# initialize base path to new directory after split
BASE_PATH = "datasets/idc"

# derive split directories
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])

# define training data amount
TRAIN_SPLIT = 0.8

# percentage of training data
# to become validation data
VAL_SPLIT = 0.1


# In[ ]:


pip install imutils


# In[ ]:


# USAGE
# python build_dataset.py

# import packages
from imutils import paths
import random
import shutil
import os

# grab paths to input images and shuffle
inputPaths = list(paths.list_images(ORIG_DATASET))
random.seed(42)
random.shuffle(inputPaths)

# compute training and testing
index = int(len(inputPaths) * TRAIN_SPLIT)
trainPaths = inputPaths[:index]
testPaths = inputPaths[index:]

# for validation, part of training used
index = int(len(trainPaths) * VAL_SPLIT)
valPaths = trainPaths[:index]
trainPaths = trainPaths[index:]

# define datasets
datasets = [
	("training", trainPaths, TRAIN_PATH),
	("validation", valPaths, VAL_PATH),
	("testing", testPaths, TEST_PATH)
]

# loop over datasets
for (setType, inputPaths, baseDirect) in datasets:
	# show set build
	print("Building '{}' set".format(setType))

	# if output base directory doesn't exist, create
	if not os.path.exists(baseDirect):
		print("Building '{}' directory".format(baseDirect))
		os.makedirs(baseDirect)

	# loop over input image paths
	for inPath in inputPaths:
		# extract filename of input image,extract
		# class label (0 for negative and 1 for positive)
		name = inPath.split(os.path.sep)[-1]
		label = name[-5:-4]

		# build path to label directory
		labelPath = os.path.sep.join([baseDirect, label])

		# if label output directory doesn't exist, create
		if not os.path.exists(labelPath):
			print("Building '{}' directory".format(labelPath))
			os.makedirs(labelPath)

		# construct path to destination image
		# copy image
		path = os.path.sep.join([labelPath, name])
		shutil.copy2(inPath, path)


# In[ ]:


# import necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import SeparableConv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class CancerNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize model with input shape to be
		# "channels last" and channels dimension
		model = Sequential()
		shape = (height, width, depth)
		channelDim = -1

		# if using "channels first", update input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			shape = (depth, height, width)
			channelDim = 1

		model.add(SeparableConv2D(32, (3, 3), padding="same",
			input_shape=shape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channelDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# * 2
		model.add(SeparableConv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channelDim))
		model.add(SeparableConv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channelDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# * 3
		model.add(SeparableConv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channelDim))
		model.add(SeparableConv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channelDim))
		model.add(SeparableConv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channelDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# first set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(256))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return constructed network architecture
		return model


# In[ ]:


# USAGE
# python train_model.py

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adagrad
from keras.utils import np_utils
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# initialize our number of epochs, initial learning rate, and batch
# size
NUM_EPOCHS = 40
INIT_LR = 1e-2
BS = 32

# determine the total number of image paths in training, validation,
# and testing directories
trainPaths = list(paths.list_images(TRAIN_PATH))
numTrain = len(trainPaths)
numVal = len(list(paths.list_images(VAL_PATH)))
numTest = len(list(paths.list_images(TEST_PATH)))

# account for skew in the labeled data
trainLabels = [int(p.split(os.path.sep)[-2]) for p in trainPaths]
trainLabels = np_utils.to_categorical(trainLabels)
classTotals = trainLabels.sum(axis=0)
classWeight = classTotals.max() / classTotals

# initialize the training training data augmentation object
trainAug = ImageDataGenerator(
	rescale=1 / 255.0,
	rotation_range=20,
	zoom_range=0.05,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.05,
	horizontal_flip=True,
	vertical_flip=True,
	fill_mode="nearest")

# initialize the validation (and testing) data augmentation object
valAug = ImageDataGenerator(rescale=1 / 255.0)

# initialize the training generator
trainGen = trainAug.flow_from_directory(
	TRAIN_PATH,
	class_mode="categorical",
	target_size=(48, 48),
	color_mode="rgb",
	shuffle=True,
	batch_size=BS)

# initialize the validation generator
valGen = valAug.flow_from_directory(
	VAL_PATH,
	class_mode="categorical",
	target_size=(48, 48),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)

# initialize the testing generator
testGen = valAug.flow_from_directory(
	TEST_PATH,
	class_mode="categorical",
	target_size=(48, 48),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)

# initialize our CancerNet model and compile it
model = CancerNet.build(width=48, height=48, depth=3,
	classes=2)
opt = Adagrad(lr=INIT_LR, decay=INIT_LR / NUM_EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# fit the model
M = model.fit_generator(
	trainGen,
	steps_per_epoch=numTrain // BS,
	validation_data=valGen,
	validation_steps=numVal // BS,
	class_weight=classWeight,
	epochs=NUM_EPOCHS)

# reset the testing generator and then use our trained model to
# make predictions on the data
print("Evaluating model")
testGen.reset()
predIndex = model.predict_generator(testGen,
	steps=(numTest // BS) + 1)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIndex = np.argmax(predIndex, axis=1)

# show a nicely formatted classification report
print(classification_report(testGen.classes, predIndex,
	target_names=testGen.class_indices.keys()))

# compute the confusion matrix and and use it to derive the raw
# accuracy, sensitivity, and specificity
cm = confusion_matrix(testGen.classes, predIndex)
total = sum(sum(cm))
accuracy = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

# show the confusion matrix, accuracy, sensitivity, and specificity
print(cm)
print("Accuracy: {:.4f}".format(accuracy))
print("Sensitivity: {:.4f}".format(sensitivity))
print("Specificity: {:.4f}".format(specificity))

# plot the training loss and accuracy
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), M.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), M.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), M.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), M.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on IDC Dataset")
plt.xlabel("Epoch Number")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('plot.png')


# In[ ]:





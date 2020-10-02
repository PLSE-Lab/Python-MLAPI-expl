from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.regularizers import l2
!pip install imutils
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

from keras import applications

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

#Building a model with VGG19
class test_model:
	@staticmethod
	def build(width, height, depth, classes, reg, init="he_normal"):

		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1


		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		base_model = applications.VGG19(weights='imagenet', include_top=False, input_shape = inputShape )
		for layer in base_model.layers:
		    layer.trainable = False
		model = Sequential()
		model.add(base_model)
		# model.add(Dense(512, input_shape=(6, 6, 512), activation='relu'))
		model.add(Flatten())
		model.add(Dense(256, activation='relu'))
		model.add(Dense(256, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(classes, activation='softmax'))
		
		opt = Adam(lr=1e-4, decay=1e-4 / 50)
		model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
		model.summary()

		return model
		
		

#Loading the Data
imagePaths = list(paths.list_images("/kaggle/input/101_objectcategories/101_ObjectCategories"))
data = []
labels = []

#Resizing the data to fit in the RAM
for imagePath in imagePaths:

	label = imagePath.split(os.path.sep)[-2]

	image = cv2.imread(imagePath)
	image = cv2.resize(image, (144, 144))

	data.append(image)
	labels.append(label)

data = np.array(data, dtype="float") / 255.0

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

#Splitting the data into 75% Training and 25% Testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, stratify=labels, random_state=42)

#Augmenting
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")


model = test_model.build(width=144, height=144, depth=3,classes=len(lb.classes_), reg=l2(0.0005))


H = model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // 32,
	epochs=200,verbose=2)

#Evaluating the model
scores = model.evaluate(testX, testY, verbose=2)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
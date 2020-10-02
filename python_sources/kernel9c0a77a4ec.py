# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class MiniVGGNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# first CONV => RELU => CONV => RELU => POOL layer set
		model.add(Conv2D(32, (3, 3), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(32, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# second CONV => RELU => CONV => RELU => POOL layer set
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model
    


# USAGE
# python fashion_mnist.py

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
#from pyimagesearch.minivggnet import MiniVGGNet
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras import backend as K
from imutils import build_montages
import matplotlib.pyplot as plt
import numpy as np
import cv2
import zipfile
import random

# initialize the number of epochs to train for, base learning rate,
# and batch size
NUM_EPOCHS = 10
INIT_LR = 1e-2
BS = 32

trainXX = None
trainYY = []
testXX = None
testYY = []
test_percent = 0.3
# grab the Fashion MNIST dataset (if this is your first time running
# this the dataset will be automatically downloaded)
print("[INFO] loading Fashion MNIST...")
#import os
#print(os.listdir("../input/images/images/images"))
#with zipfile.ZipFile('../input/images/images', 'r') as zfile:
dataTSV = open('../input/traindata/train.tsv', "r").read().split("\n")
for index in range(1, len(dataTSV) - 1):
	temp = dataTSV[index].split("\t")
	file_name = '../input/images/images/images/' + str(temp[0])
	category_name = str(temp[1])
	if category_name == "CoatsnJackets":
		category_count = 0
	elif category_name == "Dresses":
		category_count = 1
	elif category_name == "Hoodies":
		category_count = 2
	elif category_name == "Jeans":
		category_count = 3
	elif category_name == "Shirts":
		category_count = 4
	print (file_name)
	#print (category_name)
	random_num = random.uniform(0, 1)
	#print (random_num)
	if (".jpg" in file_name or ".JPG" in file_name):
        #data = zfile.read(file_name)
        #img = cv2.imdecode( np.frombuffer( data, np.uint8), 1)
        #img = cv2.imread(file_name)
        #img = cv2.imread(file_name)
		img = cv2.imread(file_name)
		h, w, c = 0, 0, 0
		if ( img is not None):
			h, w, c = img.shape
		if ( h >= 100 and w >= 100):
			img = cv2.resize(img, (100, 100))
			if random_num > test_percent:
				trainYY.append(category_count)
				trainYY.append(category_count)
				trainYY.append(category_count)
				if trainXX is None:
					trainXX = img
				else:
					trainXX = np.concatenate( (trainXX, img), axis = 2 )
			else:
				testYY.append(category_count)
				testYY.append(category_count)
				testYY.append(category_count)
				if testXX is None:
					testXX = img
				else:
					testXX = np.concatenate( (testXX, img), axis = 2 )
		#print (trainXX.shape)
		#if testXX is not None:
		#	print (testXX.shape)
trainYY = np.asarray(trainYY)
testYY = np.asarray(testYY)

((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()

print (trainX.shape)
print (trainY.shape)
print (testX.shape)
print (testY.shape)

print (trainXX.shape)
print (trainYY.shape)
print (testXX.shape)
print (testYY.shape)

trainX = trainXX.swapaxes(0,2)
trainY = trainYY
testX = testXX.swapaxes(0,2)
testY = testYY

print (trainX.shape)
print (trainY.shape)
print (testX.shape)
print (testY.shape)

# if we are using "channels first" ordering, then reshape the design
# matrix such that the matrix is:
# 	num_samples x depth x rows x columns
if K.image_data_format() == "channels_first":
	print (trainX.shape)
	trainX = trainX.reshape((trainX.shape[0], 1, 100, 100))
	testX = testX.reshape((testX.shape[0], 1, 100, 100))
	print (trainX.shape)
 
# otherwise, we are using "channels last" ordering, so the design
# matrix shape should be: num_samples x rows x columns x depth
else:
	print (trainX.shape)
	#print ( trainX[0])
	trainX = trainX.reshape((trainX.shape[0], 100, 100, 1))
	testX = testX.reshape((testX.shape[0], 100, 100, 1))
	print (trainX.shape)
	#print ( trainX[0])
 
# scale data to the range of [0, 1]
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# one-hot encode the training and testing labels
trainY = np_utils.to_categorical(trainY, 5)
testY = np_utils.to_categorical(testY, 5)

# initialize the label names
#labelNames = ["top", "trouser", "pullover", "dress", "coat",
#	"sandal", "shirt", "sneaker", "bag", "ankle boot"]

# initialize the label names
labelNames = ["CoatsnJackets", "Dresses", "Hoodies", "Jeans", "Shirts"]

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / NUM_EPOCHS)
model = MiniVGGNet.build(width=100, height=100, depth=1, classes=5)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training model...")
H = model.fit(trainX, trainY,
	validation_data=(testX, testY),
	batch_size=BS, epochs=NUM_EPOCHS)

# make predictions on the test set
preds = model.predict(testX)

# show a nicely formatted classification report
print("[INFO] evaluating network...")
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1),
	target_names=labelNames))

# plot the training loss and accuracy
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")

# initialize our list of output images
images = []

# randomly select a few testing fashion items
for i in np.random.choice(np.arange(0, len(testY)), size=(16,)):
	# classify the clothing
	probs = model.predict(testX[np.newaxis, i])
	prediction = probs.argmax(axis=1)
	label = labelNames[prediction[0]]
 
	# extract the image from the testData if using "channels_first"
	# ordering
	if K.image_data_format() == "channels_first":
		image = (testX[i][0] * 255).astype("uint8")
 
	# otherwise we are using "channels_last" ordering
	else:
		image = (testX[i] * 255).astype("uint8")

	# initialize the text label color as green (correct)
	color = (0, 255, 0)

	# otherwise, the class label prediction is incorrect
	if prediction[0] != np.argmax(testY[i]):
		color = (0, 0, 255)
 
	# merge the channels into one image and resize the image from
	# 28x28 to 96x96 so we can better see it and then draw the
	# predicted label on the image
	image = cv2.merge([image] * 3)
	image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
	cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
		color, 2)

	# add the image to our list of output images
	images.append(image)

# construct the montage for the images
montage = build_montages(images, (96, 96), (4, 4))[0]

# show the output montage
print ("error happens in next line")
cv2.imshow("Fashion MNIST", montage)
cv2.waitKey(0)
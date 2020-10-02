import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

np.random.seed(123)

from numpy import genfromtxt
from numpy import argmax

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from keras.datasets import mnist

from keras.utils.vis_utils import plot_model

'''
Load the MNIST Dataset and play with the data
'''

#Load the Training Data
X_train = genfromtxt('../input/train_data.csv',delimiter=',')

#Load the Training Labels
y_train = genfromtxt('../input/train_labels.csv',delimiter=',')

#Load the Test Data
X_test = genfromtxt('../input/test.csv',delimiter=',')

#Convert the 784 Values into a 28x28 Array
X_train = X_train.reshape((42000, 28, 28))
X_test = X_test.reshape((28000, 28, 28))

#Scale the image data down to between 0 & 1.
X_train = X_train.astype('float32')
X_train /= 255

X_test = X_test.astype('float32')
X_test /= 255

#Turn the Label into a Binary Catagory
Y_train = np_utils.to_categorical(y_train, 10)

#Create the Model
model = Sequential()

model.add(Dense(140, activation='relu',input_shape=(28,28)))
model.add(Flatten())
model.add(Dense(40, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


#Train/Fit the Model
model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)

#Evaluate the Model
Score = model.evaluate(X_train, Y_train, verbose=1)

print()
print('Final Score against training data:', Score[1])

#Run through the Test Data
TestResults=model.predict(X_test)

#Convert the outback back into Numbers
results=argmax(TestResults, axis=1)

#Save the Results, Do some quick Mods & Submit to Kaggle
np.savetxt("results.csv", results, delimiter=",")
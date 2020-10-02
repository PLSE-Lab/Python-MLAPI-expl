# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras.utils import to_categorical

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
""" Loads and returns the dataset """
def load_dataset(filepath):
	dataframe = pd.read_csv(filepath)
	pixels = dataframe.values[:, 1:]
	labels = dataframe.values[:, 0]
	one_hot_encoded_labels = create_one_hot_encoded_labels(labels)

	return pixels, one_hot_encoded_labels
	
""" Normalizes the pixel values of each example in the dataset """
def normalize_pixels(pixels):
	pixels = pixels.astype('float32')
	pixels /= 255.0

	return pixels
	
""" Creates one-hot encoded labels given the list of labels for the dataset """
def create_one_hot_encoded_labels(labels):
	unique_labels_count = np.unique(labels).shape[0]
	one_hot_encoded_labels = to_categorical(labels, unique_labels_count)

	return one_hot_encoded_labels
	
""" Creates the model """
def create_model():
	model = Sequential()

	model.add(Dense(512, activation='relu', input_shape=(784,)))
	model.add(Dropout(0.25))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(10, activation='softmax'))

	return model
	
if __name__ == '__main__':
	pixels_train, labels_train = load_dataset('../input/fashion-mnist_train.csv')
	pixels_train = normalize_pixels(pixels_train)

	pixels_test, labels_test = load_dataset('../input/fashion-mnist_test.csv')
	pixels_test = normalize_pixels(pixels_test)

	model = create_model()
	model.summary()

	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
	model.fit(pixels_train, labels_train, epochs=20, batch_size=128, verbose=1)
	score = model.evaluate(pixels_test, labels_test, verbose=0)
	
	print('Test loss: {}'.format(score[0]))
	print('Test accuracy: {}'.format(score[1]))

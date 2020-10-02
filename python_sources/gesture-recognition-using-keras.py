from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
import numpy as np 
import random
import os
import cv2

# Variables:
img_size = 100 
iterations = 2

training_data = []

def img_to_array():
    X = []
    Y = []
    categories = []

    for i in range(10):
    	path = os.path.join('../input/leapgestrecog/leapGestRecog','0' + str(i))

    	for cat in os.listdir(path):
    		new_path = os.path.join(path, cat)

    		if cat not in categories:
    			categories.append(cat)

    		for img in os.listdir(new_path):
    			new_new_path = os.path.join(new_path, img)
    			label = categories.index(cat)
    			
    			try:
    				img_array = cv2.imread(new_new_path)
    				resized_image = cv2.resize(img_array, (img_size, img_size)) # Resize image to make training less expensive
    				training_data.append([resized_image, label])
    			except Exception as e:
    				print(e)

    random.shuffle(training_data) # Make sure to shuffle data in order to train on heterogenous data

    for samples, labels in training_data:
    	X.append(samples)
    	Y.append(labels)

    X = np.array(X)
    Y = np.array(Y)

    # Save arrays so that we don't have to preprocess images each time
    np.save('samples', X) 
    np.save('labels', Y)


# One hot encode our labels for each sample
# Ex: 5 encodes to -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
def to_categorical(array):
	one_hot = np.zeros((len(array), 10)) # There are 10 categories

	for idx, num in enumerate(array):
		one_hot[idx][num] = 1
	return one_hot



def train():
	samples = np.load('samples.npy')
	labels = to_categorical(np.load('labels.npy'))

	samples = samples.astype('float32') / 255 # Image data so we divide by 255 to normalize data.

	network = Sequential()

	network.add(Conv2D(64, (3,3), input_shape=(100, 100, 3)))
	network.add(Activation('relu'))
	network.add(MaxPooling2D(pool_size=(2,2)))

	network.add(Conv2D(128, (3,3)))
	network.add(Activation('relu'))
	network.add(MaxPooling2D(pool_size=(2,2)))

	network.add(Conv2D(128, (3,3)))
	network.add(Activation('relu'))
	network.add(MaxPooling2D(pool_size=(2,2)))

	network.add(Flatten())

	network.add(Dense(10))
	network.add(Activation('softmax')) # Using softmax instead of sigmoid because there are more than 2 classes.

	network.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=['accuracy'])

	network.fit(samples, labels, batch_size=32, validation_split=0.3, epochs=iterations) # Train network and take 30% of samples as validation data


	# Save Model
	network.save('Network.model')

if (os.path.isfile('samples.npy') and os.path.isfile('labels.npy')):
	train()
else:
	img_to_array()
	train()
# Deep learning on time domain samples. Acc on 20% hold out test set is 91%
from __future__ import division
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

import os
import soundfile as sf
from scipy import signal
from sklearn.utils import shuffle

def get_data(target_names):

	# Read about 20K of recs from every species
	# Note: All wav files must be the same sampling frequency and number of datapoints!

	X = []                    # holds all data
	y = []                    # holds all class labels

	filenames = []            # holds all the file names
	target_count = []         # holds the counts in a class

	for i, target in enumerate(target_names):
	    target_count.append(0)  # initialize target count
	    path='./Wingbeats/' + target + '/'    # assemble path string

	    for [root, dirs, files] in os.walk(path, topdown=False):
	        for filename in files:
	            name,ext = os.path.splitext(filename)
	            if ext=='.wav':
	                name=os.path.join(root, filename)
	                data, fs = sf.read(name)
	                X.append(data)
	                y.append(i)
	                filenames.append(name)
	                target_count[i]+=1
	                if target_count[i]>20000:
	                	break
	    print (target,'#recs = ', target_count[i])

	X = np.vstack(X)
	y = np.hstack(y)

	X = X.astype("float32")
	print ("")
	print ("Total dataset size:")
	print ('# of classes: %d' % len(np.unique(y)))
	print ('total dataset size: %d' % X.shape[0])
	print ('Sampling frequency = %d Hz' % fs)
	print ("n_samples: %d" % X.shape[1])
	print ("duration (sec): %f" % (X.shape[1]/fs))

	return X, y

# main
fs = 8000
target_names = ['Ae. aegypti', 'Ae. albopictus', 'An. gambiae', 'An. arabiensis', 'C. pipiens', 'C. quinquefasciatus']

X, y = get_data(target_names)

# Just to be sure
X, y = shuffle(X,y, random_state=2018)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2018)

# Convert label to onehot
y_train = keras.utils.to_categorical(y_train, num_classes=6)
y_test = keras.utils.to_categorical(y_test, num_classes=6)

X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# Build the Neural Network
model = Sequential()

model.add(Conv1D(16, 3, activation='relu', input_shape=(5000, 1)))
model.add(Conv1D(16, 3, activation='relu'))
model.add(BatchNormalization())

model.add(Conv1D(32, 3, activation='relu'))
model.add(Conv1D(32, 3, activation='relu'))
model.add(BatchNormalization())

model.add(MaxPooling1D(2))
model.add(Conv1D(64, 3, activation='relu'))
model.add(Conv1D(64, 3, activation='relu'))
model.add(BatchNormalization())

model.add(MaxPooling1D(2))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(BatchNormalization())

model.add(MaxPooling1D(2))
model.add(Conv1D(256, 3, activation='relu'))
model.add(Conv1D(256, 3, activation='relu'))
model.add(BatchNormalization())
model.add(GlobalAveragePooling1D())

model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Plot model
# from keras.utils import plot_model
# plot_model(model, to_file='model.png')


model_name = 'deep_1'
top_weights_path = 'model_' + str(model_name) + '.h5'

callbacks_list = [ModelCheckpoint(top_weights_path, monitor = 'val_acc', verbose = 1, save_best_only = True, save_weights_only = True), 
    EarlyStopping(monitor = 'val_acc', patience = 6, verbose = 1),
    ReduceLROnPlateau(monitor = 'val_acc', factor = 0.1, patience = 3, verbose = 1),
    CSVLogger('model_' + str(model_name) + '.log')]

model.fit(X_train, y_train, batch_size=128, epochs=100, validation_data = [X_test, y_test], callbacks = callbacks_list)


model.load_weights(top_weights_path)
loss, acc = model.evaluate(X_test, y_test, batch_size=16)

#print('loss', loss)
print('Test accuracy:', acc)
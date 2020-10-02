import numpy as np
import pandas as pd

np.random.seed(1996)

from keras.models import Sequential #model of the NN
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape #core layers
from keras.layers import Convolution2D, MaxPooling2D #convolutionnal layers


#import the dataset
f = open("../input/train.csv")
L = f.read().split('\n')[1:-1]
f.close()

L = list(map(lambda txt: txt.split(','), L))
L = [list(map(int, lst)) for lst in L]

#spliting the dataset into a training and testing set
training_set = L[:int(.8 * len(L))]
testing_set  = L[int(.8 * len(L)):]

#separing data and labels...
training_set_data   = [L[1:] for L in training_set]
training_set_labels = [L[0]  for L in training_set]

testing_set_data   = [L[1:] for L in testing_set]
testing_set_labels = [L[0]  for L in testing_set]

def one_hot(v):
    L = [0] * 10
    L[v] = 1
    return L

#testing_set_labels  = map(one_hot, testing_set_labels)    
training_set_labels = list(map(one_hot, training_set_labels))

#change formats
y_train_onehot = np.array(training_set_labels)
y_test_onehot  = np.array(testing_set_labels)
X_train = np.array(training_set)
X_test  = np.array(testing_set)
X_train = X_train[:,1:].reshape(X_train.shape[0], 28, 28, 1)
X_test  = X_test[:,1:].reshape(X_test.shape[0], 28, 28, 1)

#normalize data
X_train = X_train.astype('float32')
X_test  = X_test.astype('float32')
X_train /= 255
X_test  /= 255

model = Sequential()

model.add(Convolution2D(25, 3, 3, activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))


#prepare the model
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

#train the model
model.fit(X_train, y_train_onehot, batch_size=32, nb_epoch=5)


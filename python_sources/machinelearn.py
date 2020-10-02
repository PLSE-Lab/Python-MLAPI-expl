import numpy as np 
import pandas as pd 

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import matplotlib.pyplot as plt

train = pd.read_csv("../input/train.csv")

X=train.values[:,1:].astype('float32')
Y=train.values[:,0].astype('int32')

cut = int(0.8*len(X))
X_train = X[:cut]
X_test = X[cut:]
Y_train = Y[:cut]
Y_test = Y[cut:]

num_pixels = X_train.shape[1]

X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
num_classes = Y_test.shape[1]

def basic_model():
    model = Sequential()
    model.add(Dense(int(num_pixels*1.5), input_dim = num_pixels, kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dense(int(num_pixels*2), kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dense(int(num_pixels*2), kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dense(int(num_pixels*1.5), kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dense(num_classes, kernel_initializer = 'normal', activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model


model = basic_model()
model.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = 100, batch_size = 500, verbose = 2) 
scores = model.evaluate(X_test, Y_test)
print(scores)

p = model.predict(X_test[39].reshape(1, 784))
plt.imshow(X_test[39].reshape(28, 28))
plt.savefig('test56.png')
print(p)
print (Y_test[39])




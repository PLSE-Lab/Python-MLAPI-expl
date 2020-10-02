import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
import numpy

#Data: 42,000 input vectors each of size 784 represented with its own grayscale intensity
#Split the first 2000 examples, about 5% of our data, to be our validation set and everything after will be used for training.
#Call Sklearns StandardScaler() for zero mean and unit variance scaling to transform the data.

data = pd.read_csv('../input/train.csv') # 42000x784
target = data.pop('label').values # (42000,)
y_train = target[2000:] # (37800,)
y_valid = target[:2000] # (8000,)
X_train = data[2000:].values.astype('float32') # (34000,784)
X_valid = data[:2000].values.astype('float32') # (8000, 784)
# preprocessing
X_train = StandardScaler().fit(X_train).transform(X_train)
X_valid = StandardScaler().fit(X_valid).transform(X_valid)

#Reshape the training and validation inputs from 756 input vectors to 28x28 matrices for compatibility with Keras.
#Call to_categorical() to transform each target into one hot vector notation for classification

X_train = X_train.reshape(-1, 28, 28, 1) # (40000, 28, 28, 1)
X_valid = X_valid.reshape(-1, 28, 28, 1) # (2000, 28, 28, 1)

# one hot vector utility
y_train = np_utils.to_categorical(y_train, 10)
y_valid = np_utils.to_categorical(y_valid, 10)

# Convolutional architecture ~conv, conv, pool, drop, flatten, dense, drop, dense~
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(28,28,1), activation='relu',border_mode = 'valid'))
model.add(Convolution2D(32, 3, 3, border_mode='same'))
model.add(MaxPooling2D(pool_size=(2,2), dim_ordering='th'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

num_epochs = 6
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(X_train,y_train, batch_size=128, nb_epoch=6, 
                validation_data=(X_valid,y_valid))
scores = model.evaluate(X_valid, y_valid, verbose=0)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss Rate')
plt.ylabel('Loss')
plt.xlabel('Training interations')
plt.legend(['Training', 'Testing'], loc='upper left')
plt.show()
#plt.savefig('MNIST_loss_plot1.png')

plt.plot(hist.history['val_acc'])
plt.title('Accuracy Rate')
plt.ylabel('Accuracy %')
plt.xlabel('Training iterations')
plt.legend(['Testing'], loc='upper left')
plt.show()
#plt.savefig('MNIST_acc_plot1.png')

# load test data
test_data = pd.read_csv('../input/test.csv').values.astype('float32')
test_data = StandardScaler().fit(test_data).transform(test_data)
test_data = test_data.reshape(-1,28,28,1)

# predictive model
test_submission = model.predict_classes(test_data, verbose=2)

# save submission to csv
pd.DataFrame({"ImageId": list(range(1,len(test_data)+1)), 
              "Label": test_submission}).to_csv('MNIST-submission_1-3-2017.csv', index=False,header=True)
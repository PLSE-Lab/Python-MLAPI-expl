import numpy as np 
import pandas as pd 

import os
print(os.listdir("../input"))

import keras
from keras.datasets import cifar100
from keras.models import Sequential
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
import matplotlib.pyplot as plt
import numpy as np

weight_decay = 0.0005
x_shape = [32,32,3]
batch_size = 128
num_classes = 100

def normalize(X_train,X_test):
    mean = np.mean(X_train,axis=(0,1,2,3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train-mean)/(std+1e-7)
    X_test = (X_test-mean)/(std+1e-7)
    return X_train, X_test

def normalize_production(x):
    mean = 120.707
    std = 64.15
    return (x-mean)/(std+1e-7)
    
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(96, (3,3), strides=(2,2), activation='relu', padding='same', input_shape=(32,32,3)))
# Pooling 
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
# Batch Normalisation before passing it to the next layer
model.add(BatchNormalization())

# 2nd Convolutional Layer
model.add(Conv2D(256, (5,5), activation='relu', padding='same'))
# Pooling
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Convolutional Layer
model.add(Conv2D(384, (3,3), activation='relu', padding='same'))
# 4th Convolutional Layer
model.add(Conv2D(384, (3,3), activation='relu', padding='same'))
# 5th Convolutional Layer
model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
# Pooling
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
# Batch Normalisation
model.add(BatchNormalization())

# Passing it to a dense layer
model.add(Flatten())
model.add(Dense(4096, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='softmax'))

model.summary()

#training parameters
epochnum = 300
learning_rate = 0.1
lr_decay = 1e-6
lr_drop = 20
batch_size = 128

def lr_scheduler(epoch):
    return learning_rate * (0.5 ** (epoch // lr_drop))
reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

#data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train)

#optimization
sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

#training
hist = model.fit_generator(datagen.flow(x_train, y_train,
                                 batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    epochs=epochnum,
                    validation_data=(x_test, y_test),callbacks=[reduce_lr],verbose=2)
model.save_weights('cifar100-vgg.h5')
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (hist.history['acc'][epochnum - 1]*100))



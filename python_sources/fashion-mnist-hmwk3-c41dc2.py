# Homework Assignment 3 Due: 11/27/2018 11:59PM

# Setup:
# Please review the 'Prepare' https://ben.desire2learn.com/d2l/le/content/333415/viewContent/1740859/View
# 
# Objective: 
# Let's use keras in action.  Design a CNN to categorize 10 classes of clothing types. 
#
# Goal: 
# Train the network to get at least 90% accuracy or higher with tuning.
# Prefix your versions with @V1, @V2, @V3 and place your versions of tuning.
# 
# Information:
# Use the Fashion MNIST (included in Keras) 
# 60k examples of training data
# 10k examples of test data
# 10 classes
	# various types of clothing
	# unique shapes and coloration
	
#ACCURACY BEFORE CHANGES
    #LOSS = .2756265490412712
    #ACC = .9093
#ACCURACY VERSION 1:
    #change the number of epochs to 40
    #LOSS = 0.28120831694602966
    #ACC = 0.9177
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# num_classes = 10
# batch_size = 128
# epochs = 24
# img_rows, img_cols = 28, 28
# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
# model = Sequential()
# model.add(Conv2D(32, (3,3), activation='relu', input_shape = input_shape))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(32,(3,3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
# model.compile(loss=keras.losses.categorical_crossentropy,
#             optimizer=keras.optimizers.Adadelta(),
#             metrics=['accuracy'])
# hist = model.fit(x_train, y_train,
#             batch_size=batch_size,
#             epochs=epochs,
#             verbose=1,
#             validation_data=(x_test, y_test))
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
# import numpy as np
# import matplotlib.pyplot as plt
# epoch_list = list(range(1, len(hist.history['acc']) + 1)) # values for x axis [1, 2, ..., # of epochs]
# plt.plot(epoch_list, hist.history['acc'], epoch_list, hist.history['val_acc'])
# plt.legend(('Training Accuracy', 'Validation Accuracy'))
# plt.show()

#@V1
# num_classes = 10
# batch_size = 128
# epochs = 40
# img_rows, img_cols = 28, 28
# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
# model = Sequential()
# model.add(Conv2D(32, (3,3), activation='relu', input_shape = input_shape))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(32,(3,3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
# model.compile(loss=keras.losses.categorical_crossentropy,
#             optimizer=keras.optimizers.Adadelta(),
#             metrics=['accuracy'])
# hist = model.fit(x_train, y_train,
#             batch_size=batch_size,
#             epochs=epochs,
#             verbose=1,
#             validation_data=(x_test, y_test))
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
# import numpy as np
# import matplotlib.pyplot as plt
# epoch_list = list(range(1, len(hist.history['acc']) + 1)) # values for x axis [1, 2, ..., # of epochs]
# plt.plot(epoch_list, hist.history['acc'], epoch_list, hist.history['val_acc'])
# plt.legend(('Training Accuracy', 'Validation Accuracy'))
# plt.show()
#END @V1

#@V2
# num_classes = 10
# batch_size = 128
# epochs = 24
# img_rows, img_cols = 28, 28
# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
# model = Sequential()
# model.add(Conv2D(32, (5,5), activation='relu', input_shape = input_shape))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(64,(5,5), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
# model.compile(loss=keras.losses.categorical_crossentropy,
#             optimizer=keras.optimizers.Adadelta(),
#             metrics=['accuracy'])
# hist = model.fit(x_train, y_train,
#             batch_size=batch_size,
#             epochs=epochs,
#             verbose=1,
#             validation_data=(x_test, y_test))
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
# import numpy as np
# import matplotlib.pyplot as plt
# epoch_list = list(range(1, len(hist.history['acc']) + 1)) # values for x axis [1, 2, ..., # of epochs]
# plt.plot(epoch_list, hist.history['acc'], epoch_list, hist.history['val_acc'])
# plt.legend(('Training Accuracy', 'Validation Accuracy'))
# plt.show()
#END @V2

#@V3
# num_classes = 10
# batch_size = 128
# epochs = 24
# img_rows, img_cols = 28, 28
# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
# model = Sequential()
# model.add(Conv2D(32, (3,3), activation='relu', input_shape = input_shape))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(64,(3,3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
# model.compile(loss=keras.losses.categorical_crossentropy,
#             optimizer=keras.optimizers.Adadelta(),
#             metrics=['accuracy'])
# hist = model.fit(x_train, y_train,
#             batch_size=batch_size,
#             epochs=epochs,
#             verbose=1,
#             validation_data=(x_test, y_test))
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
# import numpy as np
# import matplotlib.pyplot as plt
# epoch_list = list(range(1, len(hist.history['acc']) + 1)) # values for x axis [1, 2, ..., # of epochs]
# plt.plot(epoch_list, hist.history['acc'], epoch_list, hist.history['val_acc'])
# plt.legend(('Training Accuracy', 'Validation Accuracy'))
# plt.show()
#END @V3

#@V4
# num_classes = 10
# batch_size = 256
# epochs = 30
# img_rows, img_cols = 28, 28
# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
# model = Sequential()
# model.add(Conv2D(64, (5,5), activation='relu', input_shape = input_shape))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(512,(5,5), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.35))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.35))
# model.add(Dense(num_classes, activation='softmax'))
# model.compile(loss=keras.losses.categorical_crossentropy,
#             optimizer=keras.optimizers.Adadelta(),
#             metrics=['accuracy'])
# hist = model.fit(x_train, y_train,
#             batch_size=batch_size,
#             epochs=epochs,
#             verbose=1,
#             validation_data=(x_test, y_test))
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
# import numpy as np
# import matplotlib.pyplot as plt
# epoch_list = list(range(1, len(hist.history['acc']) + 1)) # values for x axis [1, 2, ..., # of epochs]
# plt.plot(epoch_list, hist.history['acc'], epoch_list, hist.history['val_acc'])
# plt.legend(('Training Accuracy', 'Validation Accuracy'))
# plt.show()
#END @V4

#@V5
# num_classes = 10
# batch_size = 512
# epochs = 55
# img_rows, img_cols = 28, 28
# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
# model = Sequential()
# model.add(Conv2D(64, (5,5), activation='relu', input_shape = input_shape))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(64,(5,5), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
# model.compile(loss=keras.losses.categorical_crossentropy,
#             optimizer=keras.optimizers.Adadelta(),
#             metrics=['accuracy'])
# hist = model.fit(x_train, y_train,
#             batch_size=batch_size,
#             epochs=epochs,
#             verbose=1,
#             validation_data=(x_test, y_test))
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
# import numpy as np
# import matplotlib.pyplot as plt
# epoch_list = list(range(1, len(hist.history['acc']) + 1)) # values for x axis [1, 2, ..., # of epochs]
# plt.plot(epoch_list, hist.history['acc'], epoch_list, hist.history['val_acc'])
# plt.legend(('Training Accuracy', 'Validation Accuracy'))
# plt.show()
#END @V5


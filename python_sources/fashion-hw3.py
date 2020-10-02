"""
With 24 epochs and 128 size batches,
original achieves roughly 90% accuracy

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split

import keras
from keras.datasets import fashion_mnist
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Dropout, Flatten, Add, Concatenate
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K

# Supress warning and informational messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# hyperparameters
num_classes = 10
batch_size = 128
epochs = 24

# img dimensions
img_rows, img_cols = 28, 28

data_train = pd.read_csv('../input/fashion-mnist_train.csv')
#data_test = pd.read_csv('../input/fashion-mnist_test.csv')

x = np.array(data_train.iloc[:, 1:])
y = np.array(data_train.iloc[:, 0])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=13)

# reshape the data for the keras backend understand 
# channel_<'first'><'last'>:  Specifies which data format convention Keras will follow. 
# 1 represents the grayscale --> pass the channels first parameter
# We want to address 'channel_first' issue because this can cause dimension mismatch errors in another backend such as Theano or CNTK
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# scale the pixel intensity from 0-255 to make it fit between 0-1 because the images are grayscale
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# one hot encode
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
I used different method of initializing the models for more flexibility
"""

"""
# Version 1 - Create deeper network
# Achieves rouhly 91% validation accuracy with 24 epochs
"""
def model_1():
    inputs = Input(shape = input_shape)
    
    t = Conv2D(50, (3,3), activation='relu', padding='same')(inputs)
    t = BatchNormalization()(t)
    t = Dropout(0.2)(t)
    t = MaxPooling2D(pool_size=(2,2))(t)
    
    t = Conv2D(46, (3,3), activation='relu', padding='same')(t)
    t = BatchNormalization()(t)
    t = Dropout(0.2)(t)
    t = MaxPooling2D(pool_size=(2,2))(t)
    
    t = Conv2D(38, (3,3), activation='relu', padding='same')(t)
    t = BatchNormalization()(t)
    t = Dropout(0.2)(t)
    t = MaxPooling2D(pool_size=(2,2))(t)
    
    # Dense layers
    t = Flatten()(t)
    
    t = Dense(units = 300, activation = 'relu')(t)
    t = Dropout(0.3)(t)
    
    t = Dense(units = 100, activation = 'relu')(t)
    
    outputs = Dense(units = num_classes, activation = 'softmax')(t)
    
    model = Model(inputs = inputs, outputs = outputs)
    
    return model
    
"""
# Version 2 - Create deeper network with only 2 pooling layers

# Achieves 92.9% validation accuracy with 48 epochs
"""
def model_1():
    inputs = Input(shape = input_shape)
    
    t = Conv2D(50, (3,3), activation='relu', padding='same')(inputs)
    t = BatchNormalization()(t)
    t = Dropout(0.2)(t)
    t = MaxPooling2D(pool_size=(2,2))(t)
    
    t = Conv2D(46, (3,3), activation='relu', padding='same')(t)
    t = BatchNormalization()(t)
    t = Dropout(0.2)(t)
    t = MaxPooling2D(pool_size=(2,2))(t)
    
    t = Conv2D(38, (3,3), activation='relu', padding='same')(t)
    t = BatchNormalization()(t)
    t = Dropout(0.2)(t)
    
    # Dense layers
    t = Flatten()(t)
    
    t = Dense(units = 300, activation = 'relu')(t)
    t = Dropout(0.3)(t)
    
    t = Dense(units = 100, activation = 'relu')(t)
    
    outputs = Dense(units = num_classes, activation = 'softmax')(t)
    
    model = Model(inputs = inputs, outputs = outputs)
    
    return model

"""
# Version 3 - Create deeper network with 2 inception modules each with 2 paths (towers)

# Achieves 93.3% validation accuracy with 34 epochs
"""
def model_3():
    inputs = Input(shape = input_shape)
    
    # tower 1
    t1 = Conv2D(50, (3,3), activation='relu', padding='same')(inputs)
    t1 = BatchNormalization()(t1)
    t1 = Dropout(0.2)(t1)
    t1 = MaxPooling2D(pool_size=(2,2))(t1)
    
    # tower 2
    t2 = Conv2D(46, (4,4), activation='relu', padding='same')(inputs)
    t2 = BatchNormalization()(t2)
    t2 = Dropout(0.2)(t2)
    t2 = MaxPooling2D(pool_size=(2,2))(t2)
    
    # Combine
    combined = Concatenate()([t1, t2])
    
    # tower 1
    t1 = Conv2D(42, (3,3), activation='relu', padding='same')(combined)
    t1 = BatchNormalization()(t1)
    t1 = Dropout(0.2)(t1)
    t1 = MaxPooling2D(pool_size=(2,2))(t1)
    
    # tower 2
    t2 = Conv2D(38, (4,4), activation='relu', padding='same')(combined)
    t2 = BatchNormalization()(t2)
    t2 = Dropout(0.2)(t2)
    t2 = MaxPooling2D(pool_size=(2,2))(t2)
    
    # Combine
    combined = Concatenate()([t1, t2])
    
    # Final layer
    t = Conv2D(38, (3,3), activation='relu', padding='same')(combined)
    t = BatchNormalization()(t)
    t = Dropout(0.2)(t)
    
    # Dense layers
    t = Flatten()(t)
    
    t = Dense(units = 300, activation = 'relu')(t)
    t = Dropout(0.3)(t)
    
    t = Dense(units = 100, activation = 'relu')(t)
    
    outputs = Dense(units = num_classes, activation = 'softmax')(t)
    
    model = Model(inputs = inputs, outputs = outputs)
    
    return model

"""
# Version 4 - Create deeper network on top of model 3 with residual blocks

# Achieves 92.9% validation accuracy with 48 epochs 
"""
def model_4():
    inputs = Input(shape = input_shape)
    
    t = Conv2D(50, (3,3), activation='relu', padding='same')(inputs)
    t = BatchNormalization()(t)
    t = Dropout(0.2)(t)
    t = MaxPooling2D(pool_size=(2,2))(t)
    
    t = Conv2D(46, (3,3), activation='relu', padding='same')(t)
    t = BatchNormalization()(t)
    t = Dropout(0.2)(t)
    t = MaxPooling2D(pool_size=(2,2))(t)
    
    # Res block like structure
    t = Conv2D(30, (3,3), activation='relu', padding='same')(t)
    v = BatchNormalization()(t)
    
    t = Conv2D(30, (3,3), activation='relu', padding='same')(t)
    t = BatchNormalization()(t)
    
    t = Add()([t, v])
    
    t = Conv2D(30, (3,3), activation='relu', padding='same')(t)
    t = BatchNormalization()(t)
    
    v = Add()([t, v])
    
    # Dense layers
    t = Flatten()(t)
    
    t = Dense(units = 300, activation = 'relu')(t)
    t = Dropout(0.3)(t)
    
    t = Dense(units = 100, activation = 'relu')(t)
    
    outputs = Dense(units = num_classes, activation = 'softmax')(t)
    
    model = Model(inputs = inputs, outputs = outputs)
    
    return model

"""
Version 5 wider 

Achieves 92.7% validation accuracy with 24 epochs (Could probably get higher with more epochs since more complex than others)
"""
def model_5():
    # Create deeper network without dropout
    inputs = Input(shape = input_shape)
    
    # tower 1
    tower1 = Conv2D(36, (5,5), activation='relu', padding='same')(inputs)
    tower1 = Dropout(0.1)(tower1)
    tower1 = MaxPooling2D(pool_size=(2,2))(tower1)
    
    # tower 2
    tower2 = Conv2D(42, (3,3), activation='relu', padding='same')(inputs)
    tower2 = Dropout(0.1)(tower2)
    tower2 = MaxPooling2D(pool_size=(2,2))(tower2)
    
    tower2 = Conv2D(36, (3,3), activation='relu', padding='same')(tower2)
    tower2 = Dropout(0.1)(tower2)
    
    combine = Concatenate()([tower1, tower2])
    
    # tower 1
    tower1 = Conv2D(30, (5,5), activation='relu', padding='same')(combine)
    tower1 = Dropout(0.1)(tower1)
    tower1 = MaxPooling2D(pool_size=(2,2))(tower1)
    
    # tower 2
    tower2 = Conv2D(38, (3,3), activation='relu', padding='same')(combine)
    tower2 = Dropout(0.1)(tower2)
    tower2 = MaxPooling2D(pool_size=(2,2))(tower2)
    
    tower2 = Conv2D(30, (3,3), activation='relu', padding='same')(tower2)
    tower2 = Dropout(0.1)(tower2)
    
    t = Concatenate()([tower1, tower2])
    
    t = Conv2D(30, (3,3), activation='relu', padding='same')(t)
    
    t = Flatten()(t)
    
    t = Dense(units = 256, activation = 'relu')(t)
    t = Dropout(0.3)(t)
    t = Dense(units = 100, activation = 'relu')(t)
    t = Dropout(0.1)(t)
    
    outputs = Dense(units = num_classes, activation = 'softmax')(t)
    
    return Model(inputs = inputs, outputs = outputs)
# Compile
def compile_model(model):
    model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy'])

    model.summary()

# Train
def train_model(model, epochs=24):
    hist = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
    return hist

"""
# Evaluate the model with the test data to get the scores on the 'real' data
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Visualize it.
epoch_list = list(range(1, len(hist.history['acc']) + 1)) # values for x axis [1, 2, ..., # of epochs]
plt.plot(epoch_list, hist.history['acc'], epoch_list, hist.history['val_acc'])
plt.legend(('Training Accuracy', 'Validation Accuracy'))
plt.show()
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.advanced_activations import PReLU
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau
K.set_image_dim_ordering('tf')
import time
import argparse
from scipy.io import loadmat
import pickle
import random
import struct
np.random.seed(1337)
def load_data(mat_file_path, width=28, height=28, max_=None, verbose=True):
    ''' Load data in from .mat file as specified by the paper.

        Arguments:
            mat_file_path: path to the .mat, should be in sample/

        Optional Arguments:
            width: specified width
            height: specified height
            max_: the max number of samples to load
            verbose: enable verbose printing

        Returns:
            A tuple of training and test data, and the mapping for class code to ascii value,
            in the following format:
                - ((training_images, training_labels), (testing_images, testing_labels), mapping)

    '''
    # Local functions
    def rotate(img):
        # Used to rotate images (for some reason they are transposed on read-in)
        flipped = np.fliplr(img)
        return np.rot90(flipped)

    def display(img, threshold=0.5):
        # Debugging only
        render = ''
        for row in img:
            for col in row:
                if col > threshold:
                    render += '@'
                else:
                    render += '.'
            render += '\n'
        return render

    # Load convoluted list structure form loadmat
    mat = loadmat(mat_file_path)

    # Load char mapping
    mapping = {kv[0]:kv[1:][0] for kv in mat['dataset'][0][0][2]}
    pickle.dump(mapping, open('mapping.p', 'wb' ))

    # Load training data
    if max_ == None:
        max_ = len(mat['dataset'][0][0][0][0][0][0])
    training_images = mat['dataset'][0][0][0][0][0][0][:max_].reshape(max_, height, width, 1)
    training_labels = mat['dataset'][0][0][0][0][0][1][:max_]

    # Load testing data
    if max_ == None:
        max_ = len(mat['dataset'][0][0][1][0][0][0])
    else:
        max_ = int(max_ / 6)
    testing_images = mat['dataset'][0][0][1][0][0][0][:max_].reshape(max_, height, width, 1)
    testing_labels = mat['dataset'][0][0][1][0][0][1][:max_]

    # Reshape training data to be valid
    if verbose == True: _len = len(training_images)
    for i in range(len(training_images)):
        if verbose == True: print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1)/_len) * 100), end='\r')
        training_images[i] = rotate(training_images[i])
    if verbose == True: print('')

    # Reshape testing data to be valid
    if verbose == True: _len = len(testing_images)
    for i in range(len(testing_images)):
        if verbose == True: print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1)/_len) * 100), end='\r')
        testing_images[i] = rotate(testing_images[i])
    if verbose == True: print('')

    # Convert type to float32
    training_images = training_images.astype('float32')
    testing_images = testing_images.astype('float32')

    # Normalize to prevent issues with model
    training_images /= 255
    testing_images /= 255

    nb_classes = len(mapping)

    return ((training_images, training_labels), (testing_images, testing_labels), mapping, nb_classes)
(x_train,y_train),(x_test,y_test),mapping,number_of_classes=load_data('../input/emnistbyclassmat/emnist-byclass.mat')
epochs=70
batch_size = 64
num_fil=128
mod_fil='check1.h5'
csv_1='check1.csv'
csv_2='check-pred.csv'
Y_train = np_utils.to_categorical(y_train, number_of_classes)
Y_test = np_utils.to_categorical(y_test, number_of_classes)

y_train[0], Y_train[0]

#data pre
model = Sequential()
model.add(Conv2D(num_fil,(5, 5),padding='valid', input_shape=(28,28,1),kernel_initializer='he_normal'))
model.add(PReLU(weights=None, alpha_initializer="zero"))
BatchNormalization(axis=-1)
#model.add(Dropout(0.1))
model.add(Conv2D(num_fil, (5, 5),kernel_initializer='he_normal'))
model.add(PReLU(weights=None, alpha_initializer="zero"))
model.add(AveragePooling2D(pool_size=(2,2)))
BatchNormalization(axis=-1)
#model.add(Dropout(0.2))
#num_fil*=2
model.add(Conv2D(num_fil,(5, 5),padding='valid',kernel_initializer='he_normal'))
model.add(PReLU(weights=None, alpha_initializer="zero"))
BatchNormalization(axis=-1)
#model.add(Dropout(0.3))
model.add(Conv2D(num_fil, (5, 5),kernel_initializer='he_normal'))
model.add(PReLU(weights=None, alpha_initializer="zero"))
model.add(MaxPooling2D(pool_size=(2,2)))
BatchNormalization(axis=-1)
#model.add(Dropout(0.4))
model.add(Flatten())
# Fully connected layer

model.add(Dense(1024,kernel_initializer='he_normal'))
model.add(PReLU(weights=None, alpha_initializer="zero"))
#BatchNormalization(axis=-1)
model.add(Dropout(0.5))
model.add(Dense(512,kernel_initializer='he_normal'))
model.add(PReLU(weights=None, alpha_initializer="zero"))
#BatchNormalization(axis=-1)
model.add(Dropout(0.5))
model.add(Dense(256, kernel_initializer='he_normal'))
model.add(PReLU(weights=None, alpha_initializer="zero"))
#BatchNormalization(axis=-1)
model.add(Dropout(0.5))

model.add(Dense(number_of_classes))

# model.add(Convolution2D(10,3,3, border_mode='same'))
# model.add(GlobalAveragePooling2D())
model.add(Activation('softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(momentum=0.9, nesterov=True), metrics=['accuracy'])

gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)

test_gen = ImageDataGenerator()
train_generator = gen.flow(x_train, Y_train, batch_size=batch_size)
test_generator = test_gen.flow(x_test, Y_test, batch_size=batch_size)
t0=time.time()
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
model.fit_generator(train_generator, steps_per_epoch=len(x_train)//batch_size, epochs=epochs, 
                    validation_data=test_generator, validation_steps=len(x_test)//batch_size,callbacks=[reduce_lr])
model.save(mod_fil)
t1=time.time()
print("Training completed in " + str(t1-t0) + " seconds")
model_yaml = model.to_yaml()
with open("bin/model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)

score = model.evaluate(x_test, Y_test)
print('Test loss:', score[0])
print('Test accuracy: ', score[1])
predictions = model.predict_classes(x_test)

predictions = list(predictions)
actuals = list(y_test)
p=model.predict(x_test)

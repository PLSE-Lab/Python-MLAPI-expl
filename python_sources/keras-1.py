'''
To do:
Create a validation set, and run cross validation tests
Add additional layers to the neural network
Add code to identify the fish in each of the images
Train model on additional pictures of fish
'''

import numpy as np
np.random.seed(5)

import os
import glob
import cv2
import datetime
import pandas as pd
import time

from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import log_loss
from keras import __version__ as keras_version
from keras import backend

# dimensions of our images.
img_width, img_height = 32 , 32  #images will be resized to this

train_data_dir = '../input/train'
#validation_data_dir =
test_data_dir='../input/test_stg1'
nb_train_samples = 3777
#nb_validation_samples = None
nb_epoch = 10
batch_size=32

#Function to load training data
def generate_train():
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical')

    return train_generator

#Functions to load test data

def get_im_cv2(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (img_width, img_height), cv2.INTER_AREA)
    return resized

def load_test():
    path = os.path.join('..', 'input', 'test_stg1', '*.jpg')
    files = sorted(glob.glob(path))

    X_test = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl)
        X_test.append(img)
        X_test_id.append(flbase)

    return X_test, X_test_id

def read_and_normalize_test_data():
    start_time = time.time()
    test_data, test_id = load_test()

    test_data = np.array(test_data, dtype=np.uint8)

    test_data = test_data.astype('float32')
    test_data = test_data / 255

    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return test_data, test_id

#Function to create submission
def create_submission(predictions, test_id):
    result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_'  + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)

def normalisation(x):
    return (x - backend.mean(x)) / backend.std(x)
    
#Function to make model
def create_model():
    model = Sequential()

    model.add(Activation(activation=normalisation,input_shape=(img_width, img_height, 3)))
    
    model.add(Convolution2D(2, 3, 3, activation='relu',init = 'he_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Convolution2D(4, 3, 3, activation='relu', init='he_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Convolution2D(4, 3, 3, activation='relu', init='he_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Flatten())
    model.add(Dense(32,activation='relu', init='he_uniform'))
    model.add(Dropout(0.5))
    
    model.add(Dense(8,activation='relu', init='he_uniform'))
    model.add(Dropout(0.5))

    model.compile(loss='mean_squared_logarithmic_error',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model

def fit_model(train_generator, model):
    model.fit_generator(
            train_generator,
            samples_per_epoch=nb_train_samples,
            nb_epoch=nb_epoch,
            validation_data=None,
            nb_val_samples=None)

    return model

def run_model():
    train_generator=generate_train()
    test_data, test_id=read_and_normalize_test_data()
    model=create_model()
    model=fit_model(train_generator,model)
    predictions=model.predict_proba(test_data,batch_size=batch_size,verbose=1)
    create_submission(predictions,test_id)


run_model()

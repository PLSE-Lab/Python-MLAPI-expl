import os
import cv2
import numpy as np
# from IPython.display import clear_output

from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras.layers import Input, Dense, concatenate, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, GlobalMaxPool2D, Flatten
from keras.layers.normalization import BatchNormalization

from scipy.stats import zscore

rootdir = '../input/autocolor-dataset/repository/NikhilCodes-AutoColor-TrainIMG-91e1f92/'
training_img = os.listdir(rootdir)
# Img to Matirx

training_img_data = []
for img in training_img:
    training_img_data.append(cv2.resize(cv2.imread(rootdir + img), (300, 250)))
    training_img_data.append(cv2.flip(training_img_data[-1],1))
    training_img_data.append(cv2.flip(training_img_data[-1],0))
    training_img_data.append(cv2.flip(training_img_data[-1],-1))
    
Y = np.array(training_img_data)
X = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in Y])
X = np.expand_dims(X, axis=-1)

model = Sequential([
    Conv2D(32, (2,2), activation='relu', input_shape=(250,300,1),data_format='channels_last', padding='same', kernel_initializer='he_normal'),
    Conv2D(32, (2,2), activation='relu', padding='same', kernel_initializer='he_normal'),
    BatchNormalization(),
    
    Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_normal'),
    Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_normal'),
    BatchNormalization(),
    
    Conv2D(128, (5,5), activation='relu', padding='same', kernel_initializer='he_normal'),
    Conv2D(128, (5,5), activation='relu', padding='same', kernel_initializer='he_normal'),
    BatchNormalization(),
    
    Conv2D(3, (1,1), activation='tanh', padding='same', kernel_initializer='he_normal'),
    Conv2D(3, (1,1), activation='sigmoid', padding='same', kernel_initializer='he_normal'),
])

model.compile(
    optimizer='rmsprop', # Adadelta is Robust Optimizer, but faster learning
    loss='mse', # mse Faster Convergence(Regression)
    metrics=['accuracy']
)

model = load_model("../input/autocolor-model/my_model.h5")

model.fit(X/255, Y/255, epochs=1000, verbose=2, validation_data=(X/255,Y/255))
model.save('my_model.h5')

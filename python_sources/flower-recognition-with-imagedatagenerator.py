import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
print("No warnings!")
################################################################################
##-------------------------------Lot of imports-------------------------------##
################################################################################
print("---Import modules---")
import numpy as np 
import pandas as pd
import h5py

import matplotlib.pylab as plt
from matplotlib import cm
%matplotlib inline

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical
from keras.preprocessing import image as keras_image
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.metrics import top_k_categorical_accuracy, categorical_accuracy

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, GlobalAveragePooling1D, GlobalAveragePooling2D
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers import Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D

from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D
import os
#print(os.listdir("../input/"))
from os.path import join
print("---Succeded---")

###############################################################################
#--------------------------------Plot function--------------------------------#
###############################################################################
def history_plot(fit_history, n):
    plt.figure(figsize=(18, 12))
    
    plt.subplot(211)
    plt.plot(fit_history.history['loss'][n:], color='slategray', label = 'train')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title('Loss Function');  
    
    plt.subplot(212)
    plt.plot(fit_history.history['categorical_accuracy'][n:], color='slategray', label = 'train')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")    
    plt.legend()
    plt.title('Accuracy');

print("Sorts of flowers are:")
train_path = "../input/flowers/flowers"
types_of_flowers = os.listdir(train_path)
print(types_of_flowers)

#model.add(Conv2D(32,kernel_size=(3), padding='same' ,input_shape=(64,64,3,)))
#    model.add(MaxPooling2D((2,2)))
 #   model.add(Conv2D(32,kernel_size=(3), padding='same',activation = 'relu'))
  #  model.add(MaxPooling2D((2,2)))
   # model.add(Flatten())  
    #model.add(Dense(128))
    #model.add(Dense(5,activation='softmax'))

def model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(32, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Flatten())
    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dense(units = 5, activation = 'softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy',categorical_accuracy])
    return model

print("Creating model")
model = model()
print("Model created")
model.summary()

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   horizontal_flip=True,
                                   vertical_flip=False)
training_set = train_datagen.flow_from_directory('../input/flowers/flowers',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
                                                 
                                                 

print("start fitting")
#history = model.fit(train_X, train_data_temp_y, epochs = 10, steps_per_epoch=100)
#history = model.fit(train_X,train_y,epochs=25,batch_size=100,verbose=2,validation_data=(val_X,val_y))
#history = model.fit_generator(train_generator,steps_per_epoch=100,epochs=5,verbose=0,shuffle=True)
model.fit_generator(training_set,steps_per_epoch = 100,verbose=0,epochs = 100)
print("Not crashed")

# Plot the training history
history_plot(history, 0)


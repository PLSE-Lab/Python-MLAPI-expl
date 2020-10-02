import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sys
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras import backend as K
import tensorflow as tf


K.clear_session()



data_train = '/kaggle/input/diabeticretinopathytrainvalidation/Diabetic Retinopathy/cropped/train'
data_validation = '/kaggle/input/diabeticretinopathytrainvalidation/Diabetic Retinopathy/cropped/validation'

#PARAMETROS

epocas = 10
altura, longitud = 200,200
batch_size = 32
pasos = 150
pasos_validation = 200
filtrosConv1 = 32
filtrosConv2 = 64
size_filtro1 = (3,3)
size_filtro2 = (2,2)
size_pool = (2,2)
clases = 2
learning_rate = 0.0025

#PRE-PROCESAMIENTO DE IMÁGENES

entrenamiento_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True
)
validation_datagen = ImageDataGenerator(
    rescale=1./255
)

imagen_train = entrenamiento_datagen.flow_from_directory(
    data_train,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
)

imagen_validation = validation_datagen.flow_from_directory(
    data_validation,
    target_size=(altura,longitud),
    batch_size=batch_size,
    class_mode='categorical'
)

#Crear la red CNN

cnn = Sequential()

cnn.add(Convolution2D(filtrosConv1, size_filtro1, padding='same', input_shape=(altura,longitud,3), activation='relu'))

cnn.add(MaxPooling2D(pool_size=size_pool))

cnn.add(Convolution2D(filtrosConv2, size_filtro2, padding='same', activation='relu'))

cnn.add(MaxPooling2D(pool_size=size_pool))

cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.5)) #Evitar overfitting, evitar memorizar
cnn.add(Dense(clases, activation='sigmoid'))

cnn.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=learning_rate), metrics=['accuracy'])

cnn.fit(imagen_train,
        steps_per_epoch=pasos,
        epochs=epocas,
        validation_data=imagen_validation,
        validation_steps=pasos_validation)

dir ='/kaggle/working/modelo/'

if not os.path.exists(dir):
    os.mkdir(dir)
cnn.save('/kaggle/working/modelo/modeloDRPrediction1.h5')
cnn.save_weights('/kaggle/working/modelo/pesosDRPrediction1.h5')
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import cv2
import os
from os import listdir
import time
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import clear_output
# try:
#   %tensorflow_version only exists in Colab. 
#  !pip install tf-nightly-gpu-2.0-preview
# except Exception:
# pass
import numpy as np
import keras
from keras.models import *
from keras import backend as K
from keras.callbacks import *
from keras.optimizers import *
from keras.preprocessing.image import *
from keras.layers import *
from sklearn.utils import class_weight

print(tf.__version__)
print(keras.__version__)


# In[ ]:


TRAIN = '../input/chest-xray-pneumonia/chest_xray/chest_xray/train/'
TEST = '../input/chest-xray-pneumonia/chest_xray/chest_xray/test/'
VAL = '../input/chest-xray-pneumonia/chest_xray/chest_xray/val/'


# In[ ]:


ALTO, ANCHO, CANALES = 300, 300, 1
BATCH_SIZE = 32


# In[ ]:


datos_e = ImageDataGenerator(rescale=1./255, rotation_range = 90)
train = datos_e.flow_from_directory(directory = TRAIN, batch_size = BATCH_SIZE, shuffle = True,
                                               target_size = (ALTO, ANCHO), class_mode = 'binary', color_mode='grayscale')
datos_t = ImageDataGenerator(rescale=1./255)
test = datos_t.flow_from_directory(directory = TEST, batch_size = BATCH_SIZE, shuffle = True,
                                               target_size = (ALTO, ANCHO), class_mode = 'binary', color_mode='grayscale')
datos_v = ImageDataGenerator(rescale=1./255)
validacion = datos_v.flow_from_directory(directory = VAL, batch_size = BATCH_SIZE, shuffle = True,
                                               target_size = (ALTO, ANCHO), class_mode = 'binary', color_mode='grayscale')


# In[ ]:


x,y = next(train)
print(x.shape)


# In[ ]:


result = x[2][:, :, 0]
result = result*255
print(result.shape)
print(y[2])
plt.imshow(result,cmap = 'gray')
plt.show()


# Definimos red neuronal, usaremos la estructura de Xception. Ajustare la cantidad de parametros segun mi intuicion, los cuales son menores a los de la estructura original

# In[ ]:


inputs = tf.keras.layers.Input(shape = [ALTO, ANCHO, CANALES])

# Bloque 1


x = tf.keras.layers.Conv2D(32,(3,3),strides=(2,2),use_bias=False)(inputs) #32
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Conv2D(51,(3,3),strides=(2,2),use_bias=False)(x) #64
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU()(x)

residual = tf.keras.layers.Conv2D(82,(1,1),strides=(2,2),padding='same', use_bias=False)(x) #128
residual = tf.keras.layers.BatchNormalization()(residual)

# Bloque 2

x = tf.keras.layers.SeparableConv2D(82,(3,3), padding='same',use_bias=False)(x) #128
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.SeparableConv2D(82,(3,3), padding='same',use_bias=False)(x) #128
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.MaxPool2D((3,3),strides=(2,2),padding='same')(x)
x = tf.keras.layers.add([x,residual])

residual = tf.keras.layers.Conv2D(132,(1,1),strides=(2,2),padding='same',use_bias=False)(x) #256
residual = tf.keras.layers.BatchNormalization()(residual)

# Bloque 3

x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.SeparableConv2D(132,(3,3),padding='same',use_bias=False)(x) #256
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.SeparableConv2D(132,(3,3),padding='same',use_bias=False)(x) #256
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.MaxPool2D((3,3),strides=(2,2),padding='same')(x)
x = tf.keras.layers.add([x,residual])

residual = tf.keras.layers.Conv2D(213,(1,1),strides=(2,2),padding='same',use_bias=False)(x) #728
residual = tf.keras.layers.BatchNormalization()(residual)

# Bloque 4

x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.SeparableConv2D(213,(3,3),padding='same',use_bias=False)(x) #728
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.SeparableConv2D(213,(3,3),padding='same',use_bias=False)(x) #728
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.MaxPool2D((3,3),strides=(2,2),padding='same')(x)
x = tf.keras.layers.add([x,residual])

# Bloque 5 - 12

for i in range(0):
    residual = x

    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.SeparableConv2D(213,(3,3),padding='same',use_bias=False)(x) #728
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.SeparableConv2D(213,(3,3),padding='same',use_bias=False)(x) #728
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.SeparableConv2D(213,(3,3),padding='same',use_bias=False)(x) #728
    x = tf.keras.layers.BatchNormalization()(x)  
    x = tf.keras.layers.add([x,residual])

residual = tf.keras.layers.Conv2D(344,(1,1),strides=(2,2),padding='same',use_bias=False)(x) #1024
residual = tf.keras.layers.BatchNormalization()(residual)

# Bloque 13

x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.SeparableConv2D(213,(3,3),padding='same',use_bias=False)(x) #728
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.SeparableConv2D(344,(3,3),padding='same',use_bias=False)(x) #1024
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.MaxPool2D((3,3),strides=(2,2),padding='same')(x)
x = tf.keras.layers.add([x,residual])

# Bloque 14

x = tf.keras.layers.SeparableConv2D(556,(3,3),padding='same',use_bias=False)(x) #1536
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.SeparableConv2D(556,(3,3),padding='same',use_bias=False)(x) #2048
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU()(x)

# Fin 
x = tf.keras.layers.AveragePooling2D()(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(556)(x)
x = tf.keras.layers.BatchNormalization()(x)#Adicional agregado x mi
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Dropout(0.5)(x) #Adicional agregado x mi

x = tf.keras.layers.Dense(344)(x)
x = tf.keras.layers.BatchNormalization()(x)#Adicional agregado x mi
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Dropout(0.5)(x) #Adicional agregado x mi

x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

modelo = tf.keras.Model(inputs = inputs, outputs = x)

modelo.summary()


# In[ ]:


# 0.001 valor tipico

def scheduler(epoch):
    if epoch < 5:
        return float(0.005)
    if epoch < 20:
        return float(0.001)
    else:
        return float(0.001 * tf.math.exp(0.1 * (10 - epoch)))

tasa = tf.keras.callbacks.LearningRateScheduler(scheduler)
# guardar = tf.keras.callbacks.ModelCheckpoint(filepath=CK, monitor='val_loss', mode='auto', verbose=0, save_best_only=True)
ajuste = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,patience=5, min_lr=0.0,verbose=1)
ajuste2 = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.01,patience=10, min_lr=0.0,verbose=1)
parar = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.01, patience=20, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
callbacks = [ajuste]


# In[ ]:


adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
rmsprop = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
adadelta = tf.keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)
nadam = tf.keras.optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)


# In[ ]:


modelo.compile(optimizer = nadam,loss = 'binary_crossentropy',metrics = ['accuracy'])


# Debido a que tenemos desbalance de clases, debemos nivelar

# In[ ]:


class_weights = class_weight.compute_class_weight('balanced',np.unique(train.classes),train.classes)
print('====')
print(class_weights)
print(len(class_weights))


# In[ ]:


epochs = 500
history = modelo.fit_generator(train, epochs=epochs, validation_data = test, callbacks=callbacks, class_weight = class_weights)


# In[ ]:


evaluacion = modelo.evaluate(validacion)
print(evaluacion)


# Entrenando en mi pc, he logrado los siguientes resultados (no lo hago aca por que seria muy lento):
# 
# 16/1 [================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================] - 0s 14ms/sample - loss: 0.1149 - accuracy: 0.9375
# [0.11485753953456879, 0.9375]

# In[ ]:





# In[ ]:





# In[ ]:





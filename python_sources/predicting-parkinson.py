#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd

from keras.layers import *
from keras import backend as K
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import Model,load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau,EarlyStopping

from scipy.misc import toimage,imresize
from skimage import exposure
from PIL import Image
import cv2


# ### Image generators

# In[ ]:


spirals_train_folder = '../input/drawings/spiral/training'
spirals_val_folder = '../input/drawings/spiral/testing'
waves_train_folder = '../input/drawings/wave/training'
waves_val_folder = '../input/drawings/wave/testing'

batch_size = 24

# histogram equalizer
def eqz_plz(img):
    return exposure.equalize_hist(img)


spiral_datagen = ImageDataGenerator(rotation_range=360, # they're spirals.
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    brightness_range=(0.5,1.5),
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    preprocessing_function=eqz_plz,
                                    vertical_flip=True)

wave_datagen = ImageDataGenerator(rotation_range=5,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  brightness_range=(0.5,1.5),
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  preprocessing_function=eqz_plz,
                                  vertical_flip=True)


spiral_train_generator = spiral_datagen.flow_from_directory(directory=os.path.abspath(spirals_train_folder),
                                                            target_size=(256, 256),
                                                            color_mode="grayscale",
                                                            batch_size=batch_size,
                                                            class_mode="binary",
                                                            shuffle=True,
                                                            seed=666)

spiral_val_generator = spiral_datagen.flow_from_directory(directory=os.path.abspath(spirals_val_folder),
                                                            target_size=(256, 256),
                                                            color_mode="grayscale",
                                                            batch_size=batch_size,
                                                            class_mode="binary",
                                                            shuffle=True,
                                                            seed=710)

wave_train_generator = wave_datagen.flow_from_directory(directory=os.path.abspath(waves_train_folder),
                                                        target_size=(256, 512), # HxW in machine learning, WxH in computer vision
                                                        color_mode="grayscale",
                                                        batch_size=batch_size,
                                                        class_mode="binary",
                                                        shuffle=True,
                                                        seed=420)

wave_val_generator = wave_datagen.flow_from_directory(directory=os.path.abspath(waves_val_folder),
                                                        target_size=(256, 512),
                                                        color_mode="grayscale",
                                                        batch_size=batch_size,
                                                        class_mode="binary",
                                                        shuffle=True,
                                                        seed=420)


# In[ ]:


reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=12,min_lr=1e-9,verbose=1)
early_stop = EarlyStopping(monitor='val_loss',patience=16,verbose=1)


# ## DNN

# In[ ]:


K.clear_session()

def nopamine_model(mode):
    if (mode == 'spirals') or (mode == 'spiral'):
        input_layer = Input(shape=(256,256,1),name=f'{mode}_input_layer')
    elif (mode == 'waves') or (mode == 'wave'):
        input_layer = Input(shape=(256,512,1),name=f'{mode}_input_layer')

    m1 = Conv2D(256,(5,5),dilation_rate=4,kernel_initializer='glorot_normal',kernel_regularizer=l2(0.001),activation='relu',padding='same')(input_layer)
    p1 = MaxPool2D((9,9),strides=3)(m1)
    m2 = Conv2D(128,(5,5),dilation_rate=2,kernel_initializer='glorot_normal',kernel_regularizer=l2(0.001),activation='relu',padding='same')(p1)
    p2 = MaxPool2D((7,7),strides=3)(m2)
    m3 = Conv2D(64,(3,3),kernel_initializer='glorot_normal',kernel_regularizer=l2(0.001),activation='relu',padding='same')(p2)
    p3 = MaxPool2D((5,5),strides=2)(m3)
    g1=Dropout(0.2)(p3)
    f1 = Flatten()(g1)
    d1 = Dense(666,activation='relu')(f1)
    g2=Dropout(0.5)(d1)
    d2 = Dense(1,activation='sigmoid')(g2)
    

    
    this_model = Model(input_layer,d2)
    this_model.summary()
    return this_model


# ### Network for spirals

# In[ ]:


spiral_model = nopamine_model(mode='spirals') # early stopping epoch 89: val_loss 0.4796, val_acc 0.8274
spiral_model.compile(optimizer=Adam(lr=3.15e-5), loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


result=spiral_model.fit_generator(spiral_train_generator,
                           validation_data=spiral_val_generator,
                           epochs=666,
                           steps_per_epoch=(2000//batch_size),
                           validation_steps=(800//batch_size),
                           callbacks=[reduce_lr,early_stop],
                           verbose=1)


# In[ ]:


# here's how to load/save
spiral_model.save('../nopamine_model_spirals.h5')


# ### Network for waves

# In[ ]:


waves_model = nopamine_model(mode='waves')
waves_model.compile(optimizer=Adam(lr=3.15e-5),loss='binary_crossentropy',metrics=['accuracy'])


# In[ ]:


waves_model.fit_generator(wave_train_generator,
                          validation_data=wave_val_generator,
                          epochs=666,
                          steps_per_epoch=(2000//batch_size),
                          validation_steps=(800//batch_size),
                          callbacks=[reduce_lr,early_stop],
                          verbose=1)


# In[ ]:


waves_model.save('../nopamine_model_waves.h5')


#!/usr/bin/env python
# coding: utf-8

# # Import packages

# In[ ]:


get_ipython().system('pip install -q efficientnet')
import efficientnet.keras as efn
from keras import backend as K
from keras import regularizers
from keras.applications import densenet
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint, Callback
from keras.layers import Activation, Conv1D, Conv2D, Dense, Dropout, Flatten, Input, average, concatenate, multiply, MaxPool1D, MaxPooling2D, MaxPool2D, BatchNormalization
from keras.models import Sequential, Model, load_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.utils.np_utils import to_categorical
from sklearn import *
from sklearn.metrics import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


# In[ ]:


trainTab = pd.read_csv("../input/jpeg-melanoma-256x256/train.csv")
testTab = pd.read_csv("../input/jpeg-melanoma-256x256/test.csv")

dim=256
bs=64

def append_ext(fn):
    return fn+".jpg"

trainTab["image_name"]=trainTab["image_name"].apply(append_ext)
testTab["image_name"]=testTab["image_name"].apply(append_ext)

Rows = np.arange(0, trainTab.shape[0], 1, dtype='int')

trainRows = np.random.choice(Rows, size=int(trainTab.shape[0]*0.7), replace=False)
valRows = [a for a in Rows if not a in trainRows]

print(len(trainRows))
print(len(valRows))


# In[ ]:


train_args = dict(brightness_range = [0.8,1.2], rescale=1./255, width_shift_range=0.3,
                             height_shift_range=0.3, shear_range=0.3, zoom_range=0.3, validation_split=0.3,
                             horizontal_flip = True, rotation_range = 40)

val_args = dict(rescale=1./255)

datagen = ImageDataGenerator(**train_args)

Vdatagen = ImageDataGenerator(**val_args)

Traingenerator = datagen.flow_from_dataframe(
        dataframe=trainTab.iloc[trainRows],
        directory='../input/jpeg-melanoma-256x256/train',
        x_col= 'image_name', # features
        y_col= ['target'], # labels
        class_mode="raw", # 'target' column should be in train_df
        batch_size= bs, # images per batch
        shuffle=True, # shuffle the rows or not
        target_size=(dim,dim) # width and height of output image
)

Valgenerator = Vdatagen.flow_from_dataframe(
        dataframe=trainTab.iloc[valRows],
        directory='../input/jpeg-melanoma-256x256/train',
        x_col= 'image_name', # features
        y_col= ['target'], # labels
        class_mode="raw", # 'target' column should be in train_df
        batch_size= bs, # images per batch
        shuffle=False, # shuffle the rows or not
        target_size=(dim,dim) # width and height of output image
)

Testgenerator = Vdatagen.flow_from_dataframe(
        dataframe=testTab,
        directory='../input/jpeg-melanoma-256x256/test',
        x_col= 'image_name', # features
        class_mode=None, # 'target' column should be in train_df
        batch_size= 1, # images per batch
        shuffle=False, # shuffle the rows or not
        target_size=(dim,dim) # width and height of output image
)


# In[ ]:


xu=Traingenerator.next()
xu[0].shape


# In[ ]:


base_model = efn.EfficientNetB0(input_shape=(dim, dim, 3), weights='imagenet', include_top=False, pooling='avg')
# for layer in base_model.layers:
#     layer.trainable = False
y = base_model.output
y = Dropout(0.3)(y)
y = Dense(256, activation="relu")(y)
y = Dense(1, activation='sigmoid')(y)
model = Model(inputs=base_model.input, outputs=y) 


# In[ ]:


def binary_focal_loss(gamma=2., alpha=.75):
    def binary_focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))                -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed

# COMPILE WITH ADAM OPTIMIZER AND CROSS ENTROPY COST
model.compile(optimizer= 'adam', loss=binary_focal_loss(), metrics=[tf.keras.metrics.AUC()])
model.load_weights("../input/mixedmodel/modelGenerator.h5")


# In[ ]:


filepath='modelGenerator.h5'

# CUSTOM LEARNING SCHEUDLE
LR_START = 1e-5
LR_MAX = 0.000177978515625
LR_RAMPUP_EPOCHS = 5
LR_SUSTAIN_EPOCHS = 0
LR_STEP_DECAY = 0.75

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = LR_MAX * LR_STEP_DECAY**((epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS)//10)
    return lr
  
lr2 = LearningRateScheduler(lrfn, verbose = True)
lr_reducer = LearningRateScheduler(lambda x: 0.001 * 0.99 ** x, verbose=True)

EarLY=EarlyStopping(monitor='val_loss', mode='min', min_delta=0, patience=30, verbose=0,
                                    restore_best_weights=True)

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', mode='min', verbose=1, save_best_only=True)

epochs = 40
tsteps = int(trainTab.shape[0]*0.7/bs)
vsteps = int(trainTab.shape[0]*0.3/bs)

history = model.fit_generator(Traingenerator, epochs = epochs,  
                              validation_data = Valgenerator,
                              steps_per_epoch = tsteps, validation_steps=vsteps,
                              callbacks=[lr2, EarLY, checkpoint])


# In[ ]:


predi=model.predict_generator(Testgenerator, steps=testTab.shape[0], verbose=1)
submi = pd.read_csv("../input/jpeg-melanoma-256x256/sample_submission.csv")
submi["target"]=predi
submi.to_csv("submiJPG.csv", index=False)


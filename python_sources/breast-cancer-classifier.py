#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer Classification
# 
# I don't know much about breast cancer, but I do know how to do... this.

# In[31]:


import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import set_random_seed

np.random.seed(666)
set_random_seed(666)

from keras import backend as K
from keras.models import Sequential,load_model
from keras.optimizers import Adam,SGD
from keras.layers import InputLayer,Dense,Dropout
from keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint

from sklearn.metrics import roc_auc_score


# In[32]:


X = pd.read_csv('../input/data.csv')
del X['id']
y = X.pop('diagnosis')
y = [0 if (val == 'M') else 1 for val in y] # 0 = M, 1 = B


# In[33]:


def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def binary_focal_loss(gamma=2., alpha=1e-3):
    def binary_focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        epsilon = K.epsilon()
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))                -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return binary_focal_loss_fixed

# callbacks
model_ckpt = ModelCheckpoint('BreastNet_weights.hdf5',save_weights_only=True)
reduce_lr = ReduceLROnPlateau(patience=4,factor=0.8,min_lr=1e-9)
early_stop = EarlyStopping(patience=7)


# I'll add KFolds tomorrow

# In[34]:


def build_model():
    K.clear_session()
    model = Sequential()
    model.add(InputLayer(input_shape=(X.shape[1],)))
    model.add(Dropout(0.4))
    model.add(Dense(666, activation='relu'))
    model.add(Dense(69, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=binary_focal_loss(),optimizer=Adam(lr=8e-6),metrics=['accuracy',sensitivity,specificity]) # 5e-6 @ 0.977 with some tweaks
    return model

model = build_model()


# In[35]:


model.fit(X, y, validation_split=0.5, batch_size=24, epochs=666, callbacks=[model_ckpt,reduce_lr,early_stop],verbose=0)
preds = model.predict(X)[:,0]
roc_auc_score(y,preds)


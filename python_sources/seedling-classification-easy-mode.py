#!/usr/bin/env python
# coding: utf-8

# # Plant Seedling Classification
# I don't know much about plant seedlings, but I do know how to do... this.

# In[ ]:


import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import set_random_seed
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export
from keras.optimizers import Optimizer

from skimage import exposure

from keras.applications.nasnet import NASNetMobile
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint
from keras.models import Model
from keras.layers import Dense
from keras import backend as K
from keras.optimizers import Adam


# In[ ]:


def categorical_focal_loss(gamma=2., alpha=.25):
    def categorical_focal_loss_fixed(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.sum(loss, axis=1)
    return categorical_focal_loss_fixed

class Yogi(Optimizer):
    def __init__(self,
               lr=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=None,
               decay=0.00000001,
               amsgrad=False,
               **kwargs):
        super(Yogi, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [state_ops.assign_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (  # pylint: disable=g-no-augmented-assignment
                1. / (1. + self.decay * math_ops.cast(self.iterations,
                                                    K.dtype(self.decay))))

        t = math_ops.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (
            K.sqrt(1. - math_ops.pow(self.beta_2, t)) /
            (1. - math_ops.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            #v_t = (self.beta_2 * v) + (1. - self.beta_2) * math_ops.square(g) # from amsgrad
            v_t = v - (1-self.beta_2)*K.sign(v-math_ops.square(g))*math_ops.square(g)
            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(state_ops.assign(m, m_t))
            self.updates.append(state_ops.assign(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(state_ops.assign(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'lr': float(K.get_value(self.lr)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'decay': float(K.get_value(self.decay)),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad
            }
        base_config = super(Yogi, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# In[ ]:


np.random.seed(666)
set_random_seed(666)

base_path = '../input/nonsegmentedv2'

datagen = ImageDataGenerator(rotation_range=360,
                            width_shift_range=0.15,
                            height_shift_range=0.15,
                            brightness_range=(0.5,1.5),
                            shear_range=0.15,
                            zoom_range=0.25,
                            horizontal_flip=True,
                            vertical_flip=True,
                            preprocessing_function=exposure.equalize_hist,
                            validation_split=0.5)

train = datagen.flow_from_directory(directory=os.path.abspath(base_path),
                                    target_size=(224,224),
                                    batch_size=16,subset='training')
val = datagen.flow_from_directory(directory=os.path.abspath(base_path),
                                  target_size=(224,224),
                                  batch_size=16,subset='validation')

model_ckpt = ModelCheckpoint('SeedNet_weights.hdf5',save_weights_only=True)
reduce_lr = ReduceLROnPlateau(patience=3, verbose=1)
early_stop = EarlyStopping(patience=7)

base_model = NASNetMobile(include_top=False,pooling='avg')
x = base_model.layers[-1].output
x = Dense(12,activation='softmax')(x)
model = Model(base_model.input,x)

model.compile(loss=categorical_focal_loss(),optimizer=Yogi(),metrics=['accuracy'])
model.fit_generator(train, steps_per_epoch=500, epochs=666,
                    validation_data=val, validation_steps=100,
                    callbacks=[model_ckpt,reduce_lr,early_stop],
                    verbose=2)


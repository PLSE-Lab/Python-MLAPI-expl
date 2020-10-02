#!/usr/bin/env python
# coding: utf-8

# # Probability of heart disease
# Based on 76 attributes.

# ## imports

# In[35]:


import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint
from keras.models import Model,Sequential
from keras.layers import Dense,InputLayer,LeakyReLU
from keras import backend as K
from keras.optimizers import Optimizer

import tensorflow as tf
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export
from tensorflow import set_random_seed

np.random.seed(666)
set_random_seed(666)


# In[36]:


def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def binary_focal_loss(gamma=2., alpha=.6):
    def binary_focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        epsilon = K.epsilon()
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))                -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return binary_focal_loss_fixed

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

# callbacks
model_ckpt = ModelCheckpoint('Pulstar_weights.hdf5',save_weights_only=True)
reduce_lr = ReduceLROnPlateau(patience=6,factor=0.6,min_lr=1e-12,verbose=0)
early_stop = EarlyStopping(patience=16,verbose=0)


# ## data

# In[37]:


X = pd.read_csv('../input/heart.csv')
y = X.pop('target')

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.05)


# ## neural network

# In[38]:


def build_model():
    K.clear_session()
    model = Sequential()
    model.add(InputLayer(input_shape=(X.shape[1],)))
    model.add(Dense(710, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=binary_focal_loss(),optimizer=Yogi(lr=1e-4,epsilon=0.01),metrics=['accuracy',sensitivity,specificity])
    return model

model = build_model()
model.fit(X_train, y_train, validation_split=0.4, batch_size=32, epochs=666,
          callbacks=[model_ckpt,reduce_lr,early_stop],shuffle=False,verbose=0)


# ## result

# In[39]:


# isolated test data
preds = model.predict(X_test)[:,0]
roc_auc_score(y_test,preds)


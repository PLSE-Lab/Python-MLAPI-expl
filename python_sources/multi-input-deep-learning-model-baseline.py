#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

from sklearn.metrics import *
from sklearn.model_selection import *

from tqdm import tqdm_notebook

import keras
from keras.layers import *
from keras.callbacks import *
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from numpy.random import seed
seed(42)


# ### Load Data

# In[ ]:


X_train = pd.read_csv("../input/X_train.csv")
X_test = pd.read_csv("../input/X_test.csv")
y_train = pd.read_csv("../input/y_train.csv")
sub = pd.read_csv("../input/sample_submission.csv")


# In[ ]:


X_train.head()


# In[ ]:


plt.figure(figsize=(15, 5))
sns.countplot(y_train['surface'])
plt.title('Target distribution', size=15)
plt.show()


# ## Make Data for the Network

# ### Input

# In[ ]:


X_train.drop(['row_id', "series_id", "measurement_number"], axis=1, inplace=True)
X_train = X_train.values.reshape((3810, 128, 10))


# In[ ]:


X_test.drop(['row_id', "series_id", "measurement_number"], axis=1, inplace=True)
X_test = X_test.values.reshape((3816, 128, 10))


# In[ ]:


for j in range(2):
    plt.figure(figsize=(15, 5))
    plt.title("Target : " + y_train['surface'][j], size=15)
    for i in range(10):
        plt.plot(X_train[j, :, i], label=i)
    plt.legend()
    plt.show()


# ### Ouput
# 
# We encode our targets

# In[ ]:


encode_dic = {'fine_concrete': 0, 
              'concrete': 1, 
              'soft_tiles': 2, 
              'tiled': 3, 
              'soft_pvc': 4,
              'hard_tiles_large_space': 5, 
              'carpet': 6, 
              'hard_tiles': 7, 
              'wood': 8}


# In[ ]:


decode_dic = {0: 'fine_concrete',
              1: 'concrete',
              2: 'soft_tiles',
              3: 'tiled',
              4: 'soft_pvc',
              5: 'hard_tiles_large_space',
              6: 'carpet',
              7: 'hard_tiles',
              8: 'wood'}


# In[ ]:


y_train = y_train['surface'].map(encode_dic).astype(int)


# ## Modeling

# ###  Attention Layer
# Because that's fancy

# In[ ]:


class Attention(Layer):
    def __init__(self, step_dim, W_regularizer=None, b_regularizer=None, W_constraint=None, b_constraint=None, bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight((input_shape[-1],), initializer=self.init, name='{}_W'.format(self.name), regularizer=self.W_regularizer, constraint=self.W_constraint)
        self.features_dim = input_shape[-1]
        if self.bias:
            self.b = self.add_weight((input_shape[1],), initializer='zero', name='{}_b'.format(self.name), regularizer=self.b_regularizer, constraint=self.b_constraint)
        else:
            self.b = None
        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias: eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)
        if mask is not None: a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


# ### Model

# In[ ]:


def make_model():
    x_inp = Input(shape=(128, 4))
    x = BatchNormalization()(x_inp)
    x = Bidirectional(CuDNNGRU(128, return_sequences=True, recurrent_regularizer=regularizers.l2(0.01)))(x)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True, recurrent_regularizer=regularizers.l2(0.01)))(x)
    x = Attention(128)(x)
    
    y_inp = Input(shape=(128, 3))
    y = BatchNormalization()(y_inp)
    y = Bidirectional(CuDNNGRU(128, return_sequences=True, recurrent_regularizer=regularizers.l2(0.01)))(y)
    y = Bidirectional(CuDNNGRU(64, return_sequences=True, recurrent_regularizer=regularizers.l2(0.01)))(y)
    y = Attention(128)(y)
    
    z_inp = Input(shape=(128, 3))
    z = BatchNormalization()(z_inp)
    z = Bidirectional(CuDNNGRU(128, return_sequences=True, recurrent_regularizer=regularizers.l2(0.01)))(z)
    z = Bidirectional(CuDNNGRU(64, return_sequences=True, recurrent_regularizer=regularizers.l2(0.01)))(z)
    z = Attention(128)(z)
    
    q = concatenate([x, y, z])
    q = BatchNormalization()(q)
    f = Dense(128, activation='relu')(q)
    f = Dropout(0.5)(f)
    f = Dense(384, activation='relu')(f)
    f = Add()([q, f])
    f = Dense(32, activation=None)(f)
    f = Dense(9, activation="softmax")(f)
    
    model = Model(inputs=[x_inp, y_inp, z_inp], outputs=f)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[ ]:


model = make_model()
model.summary()


# ### Plot Accuracy and Loss

# In[ ]:


def plot_acc(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.ylim([0, 3])
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# ### $k$-Folds

# In[ ]:


def k_folds(X, y, X_test, k=5):
    folds = list(StratifiedKFold(n_splits=k).split(X, y))
    y_test = np.zeros((X_test.shape[0], 9))
    y_oof = np.zeros((X.shape[0]))
    
    for i, (train_idx, val_idx) in  enumerate(folds):
        print(f"Fold {i+1}")
        model = make_model()
        chk = ModelCheckpoint("best_weight.wt", monitor='val_acc', mode = 'max', save_best_only = True, verbose = 1)
        history = model.fit([X[train_idx, :, :4], X[train_idx, :, 4:7], X[train_idx, :, 7:]], 
                             y[train_idx], 
                             batch_size=256, 
                             epochs=100, 
                             validation_data=[[X[val_idx, :, :4], X[val_idx, :, 4:7], X[val_idx, :, 7:]], y[val_idx]],
                             callbacks=[chk])
        model.load_weights("best_weight.wt")
        
        pred_val = np.argmax(model.predict([X[val_idx, :, :4], X[val_idx, :, 4:7], X[val_idx, :, 7:]]), axis=1)
        score = accuracy_score(pred_val, y[val_idx])
        y_oof[val_idx] = pred_val
        
        print(f'Scored {score:.3f} on validation data')
        plot_loss(history)
        
        y_test += model.predict([X_test[:, :, :4], X_test[:, :, 4:7], X_test[:, :, 7:]]) / k
        
    return y_oof, y_test                                                                          


# In[ ]:


y_oof, y_test = k_folds(X_train, y_train, X_test, k=5)


# In[ ]:


print(f'Local CV is {accuracy_score(y_oof, y_train): .4f}')


# ### Confusion Matrix

# In[ ]:


def plot_confusion_matrix(truth, pred, classes, normalize=False, title=''):
    cm = confusion_matrix(truth, pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(15, 15))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix', size=15)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid(False)
    plt.tight_layout()


# In[ ]:


plot_confusion_matrix(y_train, y_oof, encode_dic.keys())


# ### Submission

# In[ ]:


y_test = np.argmax(y_test, axis=1)


# In[ ]:


sub['surface'] = y_test
sub['surface'] = sub['surface'].map(decode_dic)
sub.head()


# In[ ]:


sub.to_csv('submission.csv', index=False)


# ### Thanks for reading ! 
# ##### Please leave an upvote, it is always appreciated!

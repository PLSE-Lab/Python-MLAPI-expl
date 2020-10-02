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

import keras
from keras.layers import *
from keras.callbacks import *
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.preprocessing import scale, MinMaxScaler


# In[ ]:


Thanks to all kagglers
i've started from the multi input RNN kernel of https://www.kaggle.com/jackg0/multi-input-deep-learning-model-baseline
then learning from that kernel about wavelet : https://www.kaggle.com/asauve/smart-robots-with-wavelets
learn a lot with this blog and github about wavelet :http://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/
and https://github.com/taspinar/siml/blob/master/notebooks/WV1%20-%20Using%20PyWavelets%20for%20Wavelet%20Analysis.ipynb

just gave a 0.64 score on public lb and 0.6503 on private lb (not selected ;-) )

but i learnt a lot it's was very fun.


# In[ ]:


import tensorflow as tf
import keras as Ker
print(tf.__version__)
print(Ker.__version__)
print(tf.keras.__version__)


# In[ ]:


import pywt
from collections import defaultdict, Counter


# In[ ]:


import random as rn

# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/fchollet/keras/issues/2280#issuecomment-306959926

import os

def init_seeds(seed):
    os.environ['PYTHONHASHSEED'] = '0'

    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.

    np.random.seed(seed)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.

    rn.seed(seed)

    tf.set_random_seed(seed)


# In[ ]:


SEED = 21

init_seeds(SEED)


# In[ ]:


X_train = pd.read_csv("../input/X_train.csv")
X_test = pd.read_csv("../input/X_test.csv")
y_train = pd.read_csv("../input/y_train.csv")
sub = pd.read_csv("../input/sample_submission.csv")


# In[ ]:


X_train.head()


# In[ ]:


X_train.shape


# In[ ]:


X_test.head()


# In[ ]:


X_test.shape


# In[ ]:


plt.figure(figsize=(15, 5))
sns.countplot(y_train['surface'])
plt.title('Target distribution', size=15)
plt.show()


# In[ ]:


X_train.drop(['row_id', "series_id", "measurement_number"], axis=1, inplace=True)
X_test.drop(['row_id', "series_id", "measurement_number"], axis=1, inplace=True)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


X_train.fillna(0, inplace = True)
X_test.fillna(0, inplace = True)


# In[ ]:


X_train.replace(-np.inf, 0, inplace = True)
X_train.replace(np.inf, 0, inplace = True)
X_test.replace(-np.inf, 0, inplace = True)
X_test.replace(np.inf, 0, inplace = True)


# In[ ]:


X_train.describe()


# In[ ]:


X_tr = X_train.values
X_te = X_test.values


# In[ ]:


X_tr = X_tr.reshape((3810, 128, 10))
X_te = X_te.reshape((3816, 128, 10))


# In[ ]:


X_tr.shape


# In[ ]:


signal_test = X_tr[0,:,2]
print(np.mean(signal_test))
print(np.min(signal_test))
print(np.max(signal_test))


# In[ ]:


print(signal_test)


# In[ ]:


#compute wavelet
scales = range(1,128)
waveletname = 'morl'
train_size = X_tr.shape[0]
train_data_cwt = np.ndarray(shape=(train_size, 127, 127, 10))


# In[ ]:


for ii in range(0,train_size):
    if ii % 1000 == 0:
        print(ii)
    for jj in range(0,10):
        signal = X_tr[ii, :, jj]
        signal_mean = np.mean(signal)
        signal = signal -signal_mean
        coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
        coeff_ = coeff[:,:127]
        train_data_cwt[ii, :, :, jj] = coeff_


# In[ ]:


test_size = X_te.shape[0]
test_data_cwt = np.ndarray(shape=(test_size, 127, 127, 10))


# In[ ]:


for ii in range(0,test_size):
    if ii % 1000 == 0:
        print(ii)
    for jj in range(0,10):
        signal = X_te[ii, :, jj]
        signal_mean = np.mean(signal)
        signal = signal -signal_mean        
        coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
        coeff_ = coeff[:,:127]
        test_data_cwt[ii, :, :, jj] = coeff_


# In[ ]:





# In[ ]:


encode_dic = {'carpet': 0, 
              'concrete': 1, 
              'fine_concrete': 2, 
              'hard_tiles': 3, 
              'hard_tiles_large_space': 4,
              'soft_pvc': 5, 
              'soft_tiles': 6, 
              'tiled': 7, 
              'wood': 8}


# In[ ]:


decode_dic = {0: 'carpet',
              1: 'concrete',
              2: 'fine_concrete',
              3: 'hard_tiles',
              4: 'hard_tiles_large_space',
              5: 'soft_pvc',
              6: 'soft_tiles',
              7: 'tiled',
              8: 'wood'}


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


# In[ ]:


y_train = y_train['surface'].map(encode_dic).astype(int)


# In[ ]:


from keras.initializers import he_normal, he_uniform,  glorot_normal,  glorot_uniform


# In[ ]:


def get_Conv2D_model(input_shape):
    
    conv1_inp = Input(shape=(input_shape[1], input_shape[2], input_shape[3]))
    
    conv1 = Conv2D(filters=16, kernel_size=(5,5), kernel_initializer=he_uniform(seed=SEED), activation='relu')(conv1_inp)
    conv1 = MaxPooling2D(pool_size=(2,2))(conv1)
    conv1 = Conv2D(filters=32, kernel_size=(5,5), kernel_initializer=he_uniform(seed=SEED), activation='relu')(conv1)
    conv1 = MaxPooling2D(pool_size=(2,2))(conv1)
    conv1 = Conv2D(filters=64, kernel_size=(5,5), kernel_initializer=he_uniform(seed=SEED), activation='relu')(conv1)
    conv1 = MaxPooling2D(pool_size=(2,2))(conv1)    
    avg_pool1 = GlobalAveragePooling2D()(conv1)
    max_pool1 = GlobalMaxPooling2D()(conv1)

    
    conv2 = Conv2D(filters=16, kernel_size=(3,3), kernel_initializer=he_uniform(seed=SEED), activation='relu')(conv1_inp)
    conv2 = MaxPooling2D(pool_size=(2,2))(conv2)
    conv2 = Conv2D(filters=32, kernel_size=(3,3), kernel_initializer=he_uniform(seed=SEED), activation='relu')(conv2)
    conv2 = MaxPooling2D(pool_size=(2,2))(conv2)
    conv2 = Conv2D(filters=64, kernel_size=(3,3), kernel_initializer=he_uniform(seed=SEED), activation='relu')(conv2)
    conv2 = MaxPooling2D(pool_size=(2,2))(conv2)    

    avg_pool2 = GlobalAveragePooling2D()(conv2)
    max_pool2 = GlobalMaxPooling2D()(conv2)    

    
    concat = concatenate([avg_pool1, max_pool1, avg_pool2, max_pool2])
    last = Dropout(0.2)(concat)

    last = Dense(128, activation='relu')(last)
    last = Dense(9, activation="softmax")(last)
    
    model = Model(inputs=[conv1_inp], outputs=last)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[ ]:


def get_multi_RNN_Conv2D_model(input_shape):

    x_inp1 = Input(shape=(128, 4))
    x1 = BatchNormalization()(x_inp1)
    x1 = Bidirectional(CuDNNGRU(128, kernel_initializer=glorot_uniform(seed=SEED), return_sequences=True, recurrent_regularizer=regularizers.l2(0.01)))(x1)
    x1 = Bidirectional(CuDNNGRU(64, kernel_initializer=glorot_uniform(seed=SEED), return_sequences=True, recurrent_regularizer=regularizers.l2(0.01)))(x1)
    x1 = Attention(128)(x1)
    
    y_inp1 = Input(shape=(128, 3))
    y1 = BatchNormalization()(y_inp1)
    y1 = Bidirectional(CuDNNGRU(128, kernel_initializer=glorot_uniform(seed=SEED), return_sequences=True, recurrent_regularizer=regularizers.l2(0.01)))(y1)
    y1 = Bidirectional(CuDNNGRU(64, kernel_initializer=glorot_uniform(seed=SEED), return_sequences=True, recurrent_regularizer=regularizers.l2(0.01)))(y1)
    y1 = Attention(128)(y1)
    
    z_inp1 = Input(shape=(128, 3))
    z1 = BatchNormalization()(z_inp1)
    z1 = Bidirectional(CuDNNGRU(128, kernel_initializer=glorot_uniform(seed=SEED), return_sequences=True, recurrent_regularizer=regularizers.l2(0.01)))(z1)
    z1 = Bidirectional(CuDNNGRU(64, kernel_initializer=glorot_uniform(seed=SEED), return_sequences=True, recurrent_regularizer=regularizers.l2(0.01)))(z1)
    z1 = Attention(128)(z1)
           
    q = concatenate([x1, y1, z1])
    q = BatchNormalization()(q)
    f = Dense(128, kernel_initializer=he_uniform(seed=SEED), activation='relu')(q)
    f = Dropout(0.5)(f)
    f = Dense(384, kernel_initializer=he_uniform(seed=SEED), activation='relu')(f)
    f = Add()([q, f])    
    
    conv1_inp = Input(shape=(input_shape[1], input_shape[2], input_shape[3]))
    
    conv1 = Conv2D(filters=16, kernel_size=(5,5), kernel_initializer=he_uniform(seed=SEED), activation='relu')(conv1_inp)
    conv1 = MaxPooling2D(pool_size=(2,2))(conv1)
    conv1 = Conv2D(filters=32, kernel_size=(5,5), kernel_initializer=he_uniform(seed=SEED), activation='relu')(conv1)
    conv1 = MaxPooling2D(pool_size=(2,2))(conv1)
    conv1 = Conv2D(filters=64, kernel_size=(5,5), kernel_initializer=he_uniform(seed=SEED), activation='relu')(conv1)
    conv1 = MaxPooling2D(pool_size=(2,2))(conv1)    
    avg_pool1 = GlobalAveragePooling2D()(conv1)
    max_pool1 = GlobalMaxPooling2D()(conv1)

    
    conv2 = Conv2D(filters=16, kernel_size=(3,3), kernel_initializer=he_uniform(seed=SEED), activation='relu')(conv1_inp)
    conv2 = MaxPooling2D(pool_size=(2,2))(conv2)
    conv2 = Conv2D(filters=32, kernel_size=(3,3), kernel_initializer=he_uniform(seed=SEED), activation='relu')(conv2)
    conv2 = MaxPooling2D(pool_size=(2,2))(conv2)
    conv2 = Conv2D(filters=64, kernel_size=(3,3), kernel_initializer=he_uniform(seed=SEED), activation='relu')(conv2)
    conv2 = MaxPooling2D(pool_size=(2,2))(conv2)    

    avg_pool2 = GlobalAveragePooling2D()(conv2)
    max_pool2 = GlobalMaxPooling2D()(conv2)    

    
    concat = concatenate([f, avg_pool1, max_pool1, avg_pool2, max_pool2])
    last = Dropout(0.5)(concat)

    last = Dense(128, activation='relu')(last)
    last = Dense(9, activation="softmax")(last)
    
    model = Model(inputs=[x_inp1, y_inp1, z_inp1, conv1_inp], outputs=last)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[ ]:


model = get_multi_RNN_Conv2D_model(train_data_cwt.shape)


# In[ ]:





# In[ ]:


model.summary()


# In[ ]:


from keras.utils import plot_model
plot_model(model, to_file='model.png')

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))


# In[ ]:



def k_folds(X, y, X_test, k=5):
    folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED).split(X, y))
    y_test = np.zeros((X_test.shape[0], 9))
    y_oof = np.zeros((X.shape[0]))
    
    for i, (train_idx, val_idx) in  enumerate(folds):
        K.clear_session()
        print(f"Fold {i+1}")
        model = get_Conv2D_model(X.shape)

        ckpt = ModelCheckpoint('weights_{}.h5'.format(i), save_best_only=True, save_weights_only=True, verbose=1, monitor='val_acc', mode='max')
                # use the indexes to extract the folds in the train and validation data
        train_X, train_y, val_X, val_y = X[train_idx], y[train_idx], X[val_idx], y[val_idx]
        history = model.fit([train_X], train_y, batch_size=64, epochs=48, 
                  validation_data=([val_X], val_y), verbose=1, callbacks=[ckpt])
        
        model.load_weights('weights_{}.h5'.format(i))
        pred_val = np.argmax(model.predict([val_X]), axis=1)
        score = accuracy_score(pred_val, val_y)
        y_oof[val_idx] = pred_val
        
        print(f'Scored {score:.3f} on validation data')
        
        y_test += model.predict([X_test])/k
        history_list.append(history)
        
    return y_oof, y_test


# In[ ]:



def k_folds_multi(X, data_train, y, X_test, data_test, k=5):
    folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED).split(X, y))
    y_test = np.zeros((X_test.shape[0], 9))
    y_oof = np.zeros((X.shape[0]))
    
    for i, (train_idx, val_idx) in  enumerate(folds):
        K.clear_session()
        print(f"Fold {i+1}")
        model = get_multi_RNN_Conv2D_model(data_train.shape)
        ckpt = ModelCheckpoint('weights_{}.h5'.format(i), save_best_only=True, save_weights_only=True, verbose=1, monitor='val_acc', mode='max')
        # use the indexes to extract the folds in the train and validation data
        
        history = model.fit([X[train_idx, :, :4], X[train_idx, :, 4:7], X[train_idx, :, 7:10], data_train[train_idx]], 
                             y[train_idx], 
                             batch_size=256, 
                             epochs=100, 
                             validation_data=[[X[val_idx, :, :4], X[val_idx, :, 4:7], X[val_idx, :, 7:10], data_train[val_idx]], y[val_idx]],
                             verbose=1, callbacks=[ckpt])
        
        model.load_weights('weights_{}.h5'.format(i))
        
        pred_val = np.argmax(model.predict([X[val_idx, :, :4], X[val_idx, :, 4:7], X[val_idx, :, 7:10], data_train[val_idx]]), axis=1)          

        score = accuracy_score(pred_val, y[val_idx])
        y_oof[val_idx] = pred_val
        
        print(f'Scored {score:.3f} on validation data')
        
        y_test += model.predict([X_test[:, :, :4], X_test[:, :, 4:7], X_test[:, :, 7:10], data_test]) / k        
        
        history_list.append(history)
        
    return y_oof, y_test


# In[ ]:


history_list = []
y_oof, y_test = k_folds_multi(X_tr, train_data_cwt, y_train, X_te, test_data_cwt, k=5)


# In[ ]:


print(f'Local CV is {accuracy_score(y_oof, y_train): .4f}')


# In[ ]:


import matplotlib.pyplot as plt

def graphical_analysis(hist):
      
    #summarize history of loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc='upper left')
    plt.show() 
    
    
    #summarize history mcc
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])

    plt.title('accuracy')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc='upper left')
    plt.show()    


# In[ ]:


print(history_list[0].history.keys())


# In[ ]:


print(len(history_list))
for i in range(len(history_list)):
    graphical_analysis(history_list[i])


# In[ ]:


# I use code from this kernel: https://www.kaggle.com/theoviel/deep-learning-starter
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix

def plot_confusion_matrix(truth, pred, classes, normalize=False, title=''):
    cm = confusion_matrix(truth, pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 10))
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


# In[ ]:


y_test = np.argmax(y_test, axis=1)


# In[ ]:


sub['surface'] = y_test
sub['surface'] = sub['surface'].map(decode_dic)
sub.head()


# In[ ]:


sub.to_csv('submission-MultiRNN-Mean-Wavelet163264-5et3filters-128relu.csv', index=False)


# In[ ]:


print("done")


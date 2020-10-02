#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Thanks to these kernel 
# Multi input RNN : https://www.kaggle.com/jackg0/multi-input-deep-learning-model-baseline
# then trying to learn about wavelet some discussion threads like that : https://www.kaggle.com/c/career-con-2019/discussion/87239#latest-510635
# giving willing to experiment (and mostly learn) with wavelet : https://www.kaggle.com/asauve/a-gentle-introduction-to-wavelet-for-data-analysis
# found a blog tutorial about wavelet : http://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/
# with git repo https://github.com/taspinar/siml/blob/master/notebooks/WV1%20-%20Using%20PyWavelets%20for%20Wavelet%20Analysis.ipynb
# 
# So i like kaggle competition because its a very mighty way to learn and experiment new things!!
# 
# Thanks for all people above and all kagglers sharing their knowledge and experience
# 
# I decide to try the multi input RNN and CNN with wavelets; not a high score (about 0.64 on LB) but quite a fun journey in ML lands :-)

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


#in an attempt to have some reproductability using keras
SEED = 21

init_seeds(SEED)


# In[ ]:


X_train = pd.read_csv("../input/X_train.csv")
X_test = pd.read_csv("../input/X_test.csv")
y_train = pd.read_csv("../input/y_train.csv")
sub = pd.read_csv("../input/sample_submission.csv")


# In[ ]:


plt.figure(figsize=(15, 5))
sns.countplot(y_train['surface'])
plt.title('Target distribution', size=15)
plt.show()


# In[ ]:


X_train.drop(['row_id', "series_id", "measurement_number"], axis=1, inplace=True)
X_test.drop(['row_id', "series_id", "measurement_number"], axis=1, inplace=True)


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


y_train = y_train['surface'].map(encode_dic).astype(int)


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


from keras.initializers import he_normal, he_uniform,  glorot_normal,  glorot_uniform


# In[ ]:


def get_multi_RNN_multi_Conv2D_model(input_shape):

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
    
    conv1_inp1 = Input(shape=(input_shape[1], input_shape[2], 4))
    
    conv11 = Conv2D(filters=1, kernel_size=(1,1), kernel_initializer=he_uniform(seed=SEED), activation='relu')(conv1_inp1)    
    conv11 = Conv2D(filters=16, kernel_size=(5,5), kernel_initializer=he_uniform(seed=SEED), activation='relu')(conv11)
    conv11 = MaxPooling2D(pool_size=(2,2))(conv11)
    conv11 = Conv2D(filters=32, kernel_size=(5,5), kernel_initializer=he_uniform(seed=SEED), activation='relu')(conv11)
    conv11 = MaxPooling2D(pool_size=(2,2))(conv11)
    conv11 = Conv2D(filters=64, kernel_size=(5,5), kernel_initializer=he_uniform(seed=SEED), activation='relu')(conv11)
    conv11 = MaxPooling2D(pool_size=(2,2))(conv11)    
    avg_pool11 = GlobalAveragePooling2D()(conv11)
    max_pool11 = GlobalMaxPooling2D()(conv11)

    
    conv21 = Conv2D(filters=16, kernel_size=(3,3), kernel_initializer=he_uniform(seed=SEED), activation='relu')(conv1_inp1)
    conv21 = MaxPooling2D(pool_size=(2,2))(conv21)
    conv21 = Conv2D(filters=32, kernel_size=(3,3), kernel_initializer=he_uniform(seed=SEED), activation='relu')(conv21)
    conv21 = MaxPooling2D(pool_size=(2,2))(conv21)
    conv21 = Conv2D(filters=64, kernel_size=(3,3), kernel_initializer=he_uniform(seed=SEED), activation='relu')(conv21)
    conv21 = MaxPooling2D(pool_size=(2,2))(conv21)    

    avg_pool21 = GlobalAveragePooling2D()(conv21)
    max_pool21 = GlobalMaxPooling2D()(conv21)    

    conv1_inp2 = Input(shape=(input_shape[1], input_shape[2], 3))

    conv12 = Conv2D(filters=1, kernel_size=(1,1), kernel_initializer=he_uniform(seed=SEED), activation='relu')(conv1_inp2)    
    conv12 = Conv2D(filters=16, kernel_size=(5,5), kernel_initializer=he_uniform(seed=SEED), activation='relu')(conv12)
    conv12 = MaxPooling2D(pool_size=(2,2))(conv12)
    conv12 = Conv2D(filters=32, kernel_size=(5,5), kernel_initializer=he_uniform(seed=SEED), activation='relu')(conv12)
    conv12 = MaxPooling2D(pool_size=(2,2))(conv12)
    conv12 = Conv2D(filters=64, kernel_size=(5,5), kernel_initializer=he_uniform(seed=SEED), activation='relu')(conv12)
    conv12 = MaxPooling2D(pool_size=(2,2))(conv12)    
    avg_pool12 = GlobalAveragePooling2D()(conv12)
    max_pool12 = GlobalMaxPooling2D()(conv12)

    
    conv22 = Conv2D(filters=16, kernel_size=(3,3), kernel_initializer=he_uniform(seed=SEED), activation='relu')(conv1_inp2)
    conv22 = MaxPooling2D(pool_size=(2,2))(conv22)
    conv22 = Conv2D(filters=32, kernel_size=(3,3), kernel_initializer=he_uniform(seed=SEED), activation='relu')(conv22)
    conv22 = MaxPooling2D(pool_size=(2,2))(conv22)
    conv22 = Conv2D(filters=64, kernel_size=(3,3), kernel_initializer=he_uniform(seed=SEED), activation='relu')(conv22)
    conv22 = MaxPooling2D(pool_size=(2,2))(conv22)    

    avg_pool22 = GlobalAveragePooling2D()(conv22)
    max_pool22 = GlobalMaxPooling2D()(conv22) 
    
    conv1_inp3 = Input(shape=(input_shape[1], input_shape[2], 3))

    conv13 = Conv2D(filters=1, kernel_size=(1,1), kernel_initializer=he_uniform(seed=SEED), activation='relu')(conv1_inp3)     
    conv13 = Conv2D(filters=16, kernel_size=(5,5), kernel_initializer=he_uniform(seed=SEED), activation='relu')(conv13)
    conv13 = MaxPooling2D(pool_size=(2,2))(conv13)
    conv13 = Conv2D(filters=32, kernel_size=(5,5), kernel_initializer=he_uniform(seed=SEED), activation='relu')(conv13)
    conv13 = MaxPooling2D(pool_size=(2,2))(conv13)
    conv13 = Conv2D(filters=64, kernel_size=(5,5), kernel_initializer=he_uniform(seed=SEED), activation='relu')(conv13)
    conv13 = MaxPooling2D(pool_size=(2,2))(conv13)    
    avg_pool13 = GlobalAveragePooling2D()(conv13)
    max_pool13 = GlobalMaxPooling2D()(conv13)

    
    conv23 = Conv2D(filters=16, kernel_size=(3,3), kernel_initializer=he_uniform(seed=SEED), activation='relu')(conv1_inp3)
    conv23 = MaxPooling2D(pool_size=(2,2))(conv23)
    conv23 = Conv2D(filters=32, kernel_size=(3,3), kernel_initializer=he_uniform(seed=SEED), activation='relu')(conv23)
    conv23 = MaxPooling2D(pool_size=(2,2))(conv23)
    conv23 = Conv2D(filters=64, kernel_size=(3,3), kernel_initializer=he_uniform(seed=SEED), activation='relu')(conv23)
    conv23 = MaxPooling2D(pool_size=(2,2))(conv23)    

    avg_pool23 = GlobalAveragePooling2D()(conv23)
    max_pool23 = GlobalMaxPooling2D()(conv23)    
    
    concat = concatenate([f, avg_pool11, max_pool11, avg_pool21, max_pool21, avg_pool12, max_pool12, avg_pool22, max_pool22, avg_pool13, max_pool13, avg_pool23, max_pool23])
    last = Dropout(0.5)(concat)

    last = Dense(128, activation='relu')(last)
    last = Dense(9, activation="softmax")(last)
    
    model = Model(inputs=[x_inp1, y_inp1, z_inp1, conv1_inp1, conv1_inp2, conv1_inp3], outputs=last)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[ ]:


model = get_multi_RNN_multi_Conv2D_model(train_data_cwt.shape)


# In[ ]:


model.summary()


# In[ ]:



def k_folds_multi(X, data_train, y, X_test, data_test, k=5):
    folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED).split(X, y))
    y_test = np.zeros((X_test.shape[0], 9))
    y_oof = np.zeros((X.shape[0]))
    
    for i, (train_idx, val_idx) in  enumerate(folds):
        K.clear_session()
        print(f"Fold {i+1}")
        model = get_multi_RNN_multi_Conv2D_model(data_train.shape)
        ckpt = ModelCheckpoint('weights_{}.h5'.format(i), save_best_only=True, save_weights_only=True, verbose=1, monitor='val_acc', mode='max')
        # use the indexes to extract the folds in the train and validation data
        
        history = model.fit([X[train_idx, :, :4], X[train_idx, :, 4:7], X[train_idx, :, 7:10], data_train[train_idx, :,:, :4], data_train[train_idx, :,:, 4:7], data_train[train_idx, :,:, 7:10]], 
                             y[train_idx], 
                             batch_size=256, 
                             epochs=100, 
                             validation_data=[[X[val_idx, :, :4], X[val_idx, :, 4:7], X[val_idx, :, 7:10], data_train[val_idx, :,:, :4], data_train[val_idx, :,:, 4:7], data_train[val_idx, :,:, 7:10]], y[val_idx]],
                             verbose=1, callbacks=[ckpt])
        
        model.load_weights('weights_{}.h5'.format(i))
        
        pred_val = np.argmax(model.predict([X[val_idx, :, :4], X[val_idx, :, 4:7], X[val_idx, :, 7:10], data_train[val_idx, :,:, :4], data_train[val_idx, :,:, 4:7], data_train[val_idx, :,:, 7:10]]), axis=1)          

        score = accuracy_score(pred_val, y[val_idx])
        y_oof[val_idx] = pred_val
        
        print(f'Scored {score:.3f} on validation data')
        
        y_test += model.predict([X_test[:, :, :4], X_test[:, :, 4:7], X_test[:, :, 7:10], data_test[:, :,:, :4], data_test[:, :,:, 4:7], data_test[:, :,:, 7:10]]) / k        
        
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


sub.to_csv('submission-Try-MultiRNN-Mean-Wavelet163264-5et3filters-multiCNN-conv11-128relu.csv', index=False)


# In[ ]:


print("done")


#!/usr/bin/env python
# coding: utf-8

# # Character level only model - LB 0.630
# 
# An attempt at seeing how good a character only model could be: TL; DR not great but it does actually work
# 
# Many of Quora's rules about asking questions deal with capitalisation and if the question is in fact, a question at all. Many of the models made public do things like strip `?` ([the default of Keras' tokenizer, which you must specifcally tell it not to do](https://www.kaggle.com/hamishdickson/using-keras-oov-tokens)) and lower the case of all text - that removes this information. My intuition here is a model just focusing on those aspects might do well.
# 
# I wanted to use this as part of an ensemble model, but it's very slow - with the architecture below you need about 20 epochs to get anything useful - so that might be a no-goer
# 
# The model and clr used below is all based on [one of shujain's kernels](https://www.kaggle.com/shujian/single-rnn-with-4-folds-clr). I've make a quick tokenizer, but if you're careful with the filters there's no reason you couldn't use keras' one instead. The other change made is the embeddings are trainable, given the embedding space is 15 dimensions and it overly covers about 200 features you can get away with that
# 
# **update:** it turns out using batch norm helps you get to a bad answer quicker, in this case about 10 epochs rather than the 20 before

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


# In[ ]:


import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
import random

from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

train_df, val_df = train_test_split(train_df, test_size=0.03)

train_df['l'] = train_df['question_text'].apply(lambda t: len(str(t)))


# In[ ]:


train_df.l.describe()


# In[ ]:


maxlen = int(train_df.quantile(0.99)[1])
maxlen


# In[ ]:


chars = set([])

for a in train_df.question_text.values:
    for c in str(a):
        chars.add(c)


# In[ ]:


count = 2
chars_to_token = {}
token_to_char = {1: "OOV"}

for c in chars:
    chars_to_token[c] = count
    token_to_char[count] = c
    count += 1


def texts_to_sequences(texts):
    out = []
    for text in texts:
        out1 = []
        for t in text:
            if t in chars_to_token:
                out1.append(chars_to_token[t])
            else:
                out1.append(1)
        out.append(out1)
        
    return np.array(out)


# In[ ]:


val_df = val_df.sample(frac=1)

train_X = texts_to_sequences(train_df.question_text.values)
val_X = texts_to_sequences(val_df.question_text.values)
test_X = texts_to_sequences(test_df.question_text.values)

train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

## Get the target values
train_y = train_df['target'].values
val_y = val_df['target'].values


# In[ ]:


# https://www.kaggle.com/suicaokhoailang/lstm-attention-baseline-0-652-lb

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
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

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


# In[ ]:


# https://www.kaggle.com/hireme/fun-api-keras-f1-metric-cyclical-learning-rate/code
from keras.callbacks import Callback
class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())
    

def f1(y_true, y_pred):
    '''
    metric from here 
    https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
    '''
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# In[ ]:


max_features = len(chars) + 1

def build_model(embed_dim=15, trainable=True, lstm_dim=20, gru_dim=20, dense_dim=4, learning_rate=0.001):

    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_dim, input_length=train_X.shape[1], trainable=trainable)(inp)
    x = SpatialDropout1D(0.1)(x)
    
    x = Bidirectional(CuDNNLSTM(lstm_dim, return_sequences=True))(x)
    y = Bidirectional(CuDNNGRU(gru_dim, return_sequences=True))(x)
    
    atten_1 = Attention(maxlen)(x)
    atten_2 = Attention(maxlen)(y)
    avg_pool = GlobalAveragePooling1D()(y)
    max_pool = GlobalMaxPooling1D()(y)
    
    conc = concatenate([atten_1, atten_2, avg_pool, max_pool])
    conc = Dense(dense_dim, activation="relu")(conc)
    conc = BatchNormalization()(conc)
    outp = Dense(1, activation="sigmoid")(conc) 
    
    model = Model(inputs=inp, outputs=outp)
    
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00, amsgrad=False)
    
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[f1])
    
    return model


# In[ ]:


def print_it(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['f1'])
    plt.plot(history.history['val_f1'])
    plt.title('model f1')
    plt.ylabel('f1')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# In[ ]:


from sklearn import metrics

def find_threshold(y_hat, y):
    max_threshold = 0.
    max_value = 0.
    
    for thresh in np.arange(0.01, 0.99, 0.01):
        thresh = np.round(thresh, 2)
        v = metrics.f1_score(y, (y_hat > thresh).astype(int))
        if v > max_value:
            max_value = v
            max_threshold = thresh

    print("best f1 score: " + str(max_value) + " at: " + str(max_threshold))
    
    return max_threshold, max_value


# In[ ]:


model = build_model()
model.summary()


# In[ ]:


batch_size = 1024

step_size = float(4 * len(train_X)) / float(batch_size)

clr = CyclicLR(base_lr=0.001, max_lr=0.01,
               step_size=step_size, mode='exp_range',
               gamma=0.99994)


# In[ ]:


history = model.fit(train_X, train_y, batch_size=batch_size, epochs=10, validation_data=(val_X, val_y), callbacks=[clr])


# In[ ]:


print_it(history)


# In[ ]:


y_val_pred = model.predict([val_X])
f1_threshold, f1_value = find_threshold(y_val_pred, val_y)
print(f1_threshold, f1_value)


# In[ ]:


print("precision", metrics.precision_score(val_y, (y_val_pred > f1_threshold).astype(int)))
print("recall", metrics.recall_score(val_y, (y_val_pred > f1_threshold).astype(int)))


# In[ ]:


pd.set_option('display.max_colwidth', -1)

val_df['y_hat'] = (y_val_pred > f1_threshold).astype(int)
val_df[val_df['y_hat'] != val_df['target']].sample(20)


# In[ ]:


val_df[val_df['y_hat'] == 1].sample(20)


# In[ ]:


pred_test_y = model.predict([test_X])


# In[ ]:


pred_test_y = (pred_test_y > f1_threshold).astype(int)
test_df = pd.read_csv("../input/test.csv", usecols=["qid"])
out_df = pd.DataFrame({"qid":test_df["qid"].values})
out_df['prediction'] = pred_test_y
out_df.to_csv("submission.csv", index=False)


# In[ ]:





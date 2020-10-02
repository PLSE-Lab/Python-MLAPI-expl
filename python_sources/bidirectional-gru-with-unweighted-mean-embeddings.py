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
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from sklearn.model_selection import GridSearchCV,StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score,train_test_split
from scipy import stats
from sklearn import metrics

from keras.models import Sequential # initialize neural network library
from keras.layers import Dense # build our layers library
from keras.preprocessing.text import Tokenizer

import string,re
from collections import Counter
import nltk
from nltk.corpus import stopwords

#spaCy based imports
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English


# In[ ]:


import time
from tqdm import tqdm
import math
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D,CuDNNLSTM
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from sklearn import model_selection as ms
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from keras.layers import Input, Dense, Embedding, concatenate
from keras.layers import CuDNNGRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers import *
from keras.callbacks import *


# In[ ]:


df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")

print("train data shape --",df_train.shape)
print("test data shape --",df_test.shape)


# In[ ]:


df_train["question_text"] = df_train["question_text"].apply(lambda x: x.replace('.',' fullstop '))
df_train["question_text"] = df_train["question_text"].apply(lambda x: x.replace('?',' endofquestion '))
df_train["question_text"] = df_train["question_text"].apply(lambda x: x.replace(',',' comma '))
df_train["question_text"] = df_train["question_text"].apply(lambda x: x.replace('!',' exclamationmark '))
df_train["question_text"] = df_train["question_text"].apply(lambda x: x.replace('-',' hyphen '))
df_train["question_text"] = df_train["question_text"].apply(lambda x: x.replace('/',' backslash '))


# In[ ]:


df_train["question_text"] = df_train["question_text"].apply(lambda x: x.replace('fullstop','.'))
df_train["question_text"] = df_train["question_text"].apply(lambda x: x.replace('endofquestion','?'))
df_train["question_text"] = df_train["question_text"].apply(lambda x: x.replace('comma',','))
df_train["question_text"] = df_train["question_text"].apply(lambda x: x.replace('exclamationmark','!'))
df_train["question_text"] = df_train["question_text"].apply(lambda x: x.replace('hyphen','-'))
df_train["question_text"] = df_train["question_text"].apply(lambda x: x.replace('backslash','/'))


# In[ ]:


df_test["question_text"] = df_test["question_text"].apply(lambda x: x.replace('.',' fullstop '))
df_test["question_text"] = df_test["question_text"].apply(lambda x: x.replace('?',' endofquestion '))
df_test["question_text"] = df_test["question_text"].apply(lambda x: x.replace(',',' comma '))
df_test["question_text"] = df_test["question_text"].apply(lambda x: x.replace('!',' exclamationmark '))
df_test["question_text"] = df_test["question_text"].apply(lambda x: x.replace('-',' hyphen '))
df_test["question_text"] = df_test["question_text"].apply(lambda x: x.replace('/',' backslash '))


# In[ ]:


df_test["question_text"] = df_test["question_text"].apply(lambda x: x.replace('fullstop','.'))
df_test["question_text"] = df_test["question_text"].apply(lambda x: x.replace('endofquestion','?'))
df_test["question_text"] = df_test["question_text"].apply(lambda x: x.replace('comma',','))
df_test["question_text"] = df_test["question_text"].apply(lambda x: x.replace('exclamationmark','!'))
df_test["question_text"] = df_test["question_text"].apply(lambda x: x.replace('hyphen','-'))
df_test["question_text"] = df_test["question_text"].apply(lambda x: x.replace('backslash','/'))


# In[ ]:


df_combined = pd.concat([df_train,df_test],axis=0)

print("combined shape ",df_combined.shape)


# In[ ]:


questions = list(df_combined['question_text'].values)
length_of_sentences = []
for s in questions:
    length_of_sentences.append(len(s.split()))
print(questions[:5])
print("\nlength of sentences",length_of_sentences[:5])
print("\nstats",stats.describe(length_of_sentences))
print("\npercentile","-->",np.percentile(length_of_sentences,99.99))


# In[ ]:


## some config values 
embed_size = 300 # how big is each word vector
max_features = 60000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 60 # max number of words in a question to use

total_X = df_combined["question_text"].values


# In[ ]:


tokenizer = Tokenizer(num_words=max_features,filters='"#$%()+-:;<=>@[\\]^_`{|}~\t\n',)

tokenizer.fit_on_texts(list(total_X))


# In[ ]:


WORDS = tokenizer.word_counts
print(len(WORDS))


# In[ ]:


train_X = df_train["question_text"].values
test_X = df_test["question_text"].values


# In[ ]:


train_X = tokenizer.texts_to_sequences(train_X)
test_X = tokenizer.texts_to_sequences(test_X)


# In[ ]:


## Pad the sentences 
train_X = pad_sequences(train_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

## Get the target values
train_y = df_train['target'].values


# Trying different Embeddings and compare the results on basis of val_loss.

# In[ ]:


def get_embeddings(embedtype):
    if embedtype is "glove":
        EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    elif embedtype is "fastext":
            EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
    elif embedtype is "paragram":
                EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
    
    def get_coefs(word,*arr): 
        return word, np.asarray(arr, dtype='float32')
    
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE,encoding="latin") if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    
    vocab_size = all_embs.shape[0]
    embed_size = all_embs.shape[1]
    word_index = tokenizer.word_index
    
    print("embed size-->",embed_size)
    print("total words in embeddings-->",vocab_size)
    print("total words in data-->",len(word_index))
    
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    
    count_common_words =0
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            count_common_words = count_common_words+1
    print(count_common_words," common words fount in ",embedtype)
    print("{0}% common words found in {1}".format((count_common_words*100/len(word_index)),embedtype))
    
    return embedding_matrix
                


# In[ ]:


embedding_glove = get_embeddings(embedtype="glove")


# In[ ]:


embedding_paragram = get_embeddings(embedtype="paragram")


# In[ ]:


mean_gl_par_embedding = np.mean([embedding_glove,embedding_paragram],axis=0)
print("mean glove paragram embedding shape--> ",mean_gl_par_embedding.shape)


# In[ ]:



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


# In[ ]:


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
    

clr = CyclicLR(base_lr=0.001, max_lr=0.01,
                        step_size=300., mode='exp_range',
                        gamma=0.99994)


# In[ ]:


def build_model():
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[mean_gl_par_embedding],trainable=False)(inp)
    x = SpatialDropout1D(rate=0.1)(x)
    x1 = Bidirectional(CuDNNGRU(200, return_sequences=True))(x)
    x2 = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)

    atten_1 = Attention(maxlen)(x1) # skip connect
    #avg_pool = GlobalAveragePooling1D()(x1)
    atten_2 = Attention(maxlen)(x2)
    #max_pool = GlobalMaxPooling1D()(x2)
    
    x = concatenate([atten_1,atten_2])
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(),)
    
    return model


# In[ ]:


def f1_smart(y_true, y_pred):
    args = np.argsort(y_pred)
    tp = y_true.sum()
    fs = (tp - np.cumsum(y_true[args[:-1]])) / np.arange(y_true.shape[0] + tp - 1, tp, -1)
    res_idx = np.argmax(fs)
    return 2 * fs[res_idx], (y_pred[args[res_idx]] + y_pred[args[res_idx + 1]]) / 2


# In[ ]:


kfold = StratifiedKFold(n_splits=5, random_state=1990, shuffle=True)
bestscore = []
y_test = np.zeros((test_X.shape[0], ))
filepath="weights_best_mean.h5"
for i, (train_index, valid_index) in enumerate(kfold.split(train_X, train_y)):
    X_train, X_val, Y_train, Y_val = train_X[train_index], train_X[valid_index], train_y[train_index], train_y[valid_index]
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=1, min_lr=0.0001, verbose=2)
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=2, mode='auto')
    callbacks = [clr,]
    model = build_model()
    if i == 0:print(model.summary()) 
    model.fit(X_train, Y_train, batch_size=512, epochs=2, validation_data=(X_val, Y_val), verbose=1, callbacks=[checkpoint,clr,])
    model.load_weights(filepath)
    y_pred = model.predict([X_val], batch_size=1024, verbose=1)
    y_test += np.squeeze(model.predict([test_X], batch_size=1024, verbose=2))/5
    f1, threshold = f1_smart(np.squeeze(Y_val), np.squeeze(y_pred))
    print('Optimal F1: {:.4f} at threshold: {:.4f}'.format(f1, threshold))
    bestscore.append(threshold)


# In[ ]:


print("mean threshold--> ",np.mean(bestscore))


# In[ ]:


print(y_test.shape)
pred_test_y = (y_test>np.mean(bestscore)).astype(int)
out_df = pd.DataFrame({"qid":df_test["qid"].values})
out_df['prediction'] = pred_test_y
out_df.to_csv("submission.csv", index=False)


# In[ ]:





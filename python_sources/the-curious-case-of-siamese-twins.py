#!/usr/bin/env python
# coding: utf-8

# # An Attempt at using Siamese network:
# 
# ## Intuition :
# 
# My Intuition was to overcome the huge class imbalance in the dataset with the use of Siamese network. Instead of trying to understand sincere and insincere question, Siamese networks approaches the problem as a similarity problem. It tries to understand how similar a given pair is. Since it is easier to genrate as many combination of similar and non similar pairs as we would need from the dataset ( theoreticaly we can augment pairs in the order of Billions ). I thought the class imbalance would not be a problem and can lead to a better result.
# 
# ## Learning :
# 
# 1. The 2 hour constrain of kaggle kernels restricted me from using only 3-5 million pairs of generated data. But these were unfortunately not sufficient for the network to understand the nuances of many ambigous and difficult questions. I learnt that siamese are a good choice when we have much lesser samples per class. As the sample size increases , the triplets / pair sizes increases quadratically making training impossible.
# 
# 2. As the training size increases , the number of easy pairs ( pairs which are easier to differentiate / identify ) aslo increases in a given batch. This affects the gradient update as the average loss per batch is much less when majority of samples in a batch are easy pairs. This stalls the model from learning further after a certain point.
# 
# ## Things to try:
# 
# 1.  To use only hard samples during loss calculation. As stated in https://arxiv.org/pdf/1703.07737.pdf . For the dataset of this magnitude , I presume it will be still be impossible to get a decent result with a 2 hour time constraint. ( Implementation of Hard pair selector - https://github.com/adambielski/siamese-triplet/blob/master/utils.py )
# 
# 
# ### References:
# 
# https://www.kaggle.com/shujian/single-rnn-with-4-folds-clr
# 
# 

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


## some config values 
embed_size = 300 # how big is each word vector
max_features = 95000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 70 # max number of words in a question to use


# In[ ]:


import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D,merge
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate,dot
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.optimizers import Adam
from keras.models import Model,Sequential
from keras.layers.core import Lambda, Flatten, Dense
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers import concatenate
from keras.callbacks import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time
from datetime import timedelta
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


def split_df(df):
    qid = df["qid"].values
    qid_train,qid_val = train_test_split(qid,test_size=0.20,random_state=2019)
    train_df = df[df["qid"].isin(qid_train)]
    val_df = df[df["qid"].isin(qid_val)]
    return train_df.reset_index(drop=True),val_df.reset_index(drop=True)


# In[ ]:


def generate_pairs(df,augment=False):
    df1 = df.copy()
    df2 = df.copy()
    df2 = sklearn.utils.shuffle(df2,random_state=2018)
    df1 = sklearn.utils.shuffle(df1,random_state=2019)
    df2.columns = ["qid2","question_text2","target2"]
    df2 = df2.reset_index(drop=True)
    df2 = pd.concat([df1, df2], axis=1)
    df2["similarityTarget"] = df2.apply(lambda row: 1-np.logical_xor(row.target,row.target2),axis=1 )
    single_incincere = None
    single_sincere = None
    if augment:
        df_augmented_non_similar = generate_non_similar_samples(df,portion_of_incincere=15)
        df_augmented_similar = generate_similar_incincere_samples(df,portion_of_incincere=10)
        df_augmented_single_incincere,single_incincere = generate_samples_against_single_incincere(df)
        df_augmented_single_sincere,single_sincere = generate_samples_against_single_sincere(df)
        df2 = pd.concat([df2,df_augmented_non_similar,df_augmented_similar,df_augmented_single_sincere,df_augmented_single_incincere])
    df2 = df2.drop_duplicates()
    df2 = sklearn.utils.shuffle(df2,random_state=2020)
    print("Data not similar: {}".format(len(df2[df2["similarityTarget"]==0])))
    print("Data similar: {}".format(len(df2[df2["similarityTarget"]==1])))
    print("Not similar to Similar ratio: {}".format(len(df2[df2["similarityTarget"]==0])/len(df2[df2["similarityTarget"]==1])))
#     return df2
    return df2["question_text"].fillna("_##_").values,df2["question_text2"].fillna("_##_").values,df2["target"].values,df2["target2"].values,df2["similarityTarget"].values,single_incincere,single_sincere


# In[ ]:


## Helper functions to generate pairs

def generate_non_similar_samples(data_df,portion_of_incincere = 3):
    data_df_1 = data_df[data_df["target"]==1].copy()
    data_df_1 = pd.concat([data_df_1]*portion_of_incincere)
    data_df_0 = data_df[data_df["target"]==0].sample(n=len(data_df_1),random_state=2018).copy()
    data_df_1.columns = ["qid2","question_text2","target2"]
    data_df_1 = sklearn.utils.shuffle(data_df_1,random_state=2018)
    data_df_0 = sklearn.utils.shuffle(data_df_0,random_state=2019)
    data_df_1 = data_df_1.reset_index(drop=True)
    data_df_0 = data_df_0.reset_index(drop=True)
    data_df_1 = pd.concat([data_df_0, data_df_1], axis=1)
    data_df_1["similarityTarget"] = data_df_1.apply(lambda row: 1-np.logical_xor(row.target,row.target2),axis=1 )
    return data_df_1

def generate_similar_incincere_samples(data_df,portion_of_incincere = 3):
    data_df_1 = data_df[data_df["target"]==1].copy()
    data_df_1 = pd.concat([data_df_1]*portion_of_incincere)
    data_df_0 = data_df_1.copy()
    data_df_1.columns = ["qid2","question_text2","target2"]
    data_df_1 = sklearn.utils.shuffle(data_df_1,random_state=2018)
    data_df_0 = sklearn.utils.shuffle(data_df_0,random_state=2019)
    data_df_1 = data_df_1.reset_index(drop=True)
    data_df_0 = data_df_0.reset_index(drop=True)
    data_df_1 = pd.concat([data_df_0, data_df_1], axis=1)
    data_df_1 = data_df_1.drop_duplicates()
    data_df_1 = data_df_1[data_df_1["question_text"]!=data_df_1["question_text2"]]
    data_df_1["similarityTarget"] = data_df_1.apply(lambda row: 1-np.logical_xor(row.target,row.target2),axis=1 )
    return data_df_1

def _fuse_dataframes(single,data_df_1):
    data_df_0 = pd.concat([single]*len(data_df_1))
    data_df_1.columns = ["qid2","question_text2","target2"]
    data_df_1 = sklearn.utils.shuffle(data_df_1,random_state=2018)
    data_df_1 = data_df_1.reset_index(drop=True)
    data_df_0 = data_df_0.reset_index(drop=True)
    data_df_1 = pd.concat([data_df_0, data_df_1], axis=1)
    data_df_1 = data_df_1[data_df_1["question_text"]!=data_df_1["question_text2"]]
    data_df_1["similarityTarget"] = data_df_1.apply(lambda row: 1-np.logical_xor(row.target,row.target2),axis=1 )
    return data_df_1

def generate_samples_against_single_incincere(data_df,portion_of_incincere = 1):
    data_df_1 = data_df[data_df["target"]==1].copy()
    data_df_2 = data_df[data_df["target"]==0].copy()
    data_df_1 = pd.concat([data_df_1]*portion_of_incincere)
    single_incincere = data_df_1.sample(n=1,random_state=2018).copy()
    data_df_1 = _fuse_dataframes(single_incincere,data_df_1)
#     data_df_2 = _fuse_dataframes(single_incincere,data_df_2)
#     data_df_1 = pd.concat([data_df_1, data_df_2])
    return data_df_1,single_incincere

def generate_samples_against_single_sincere(data_df,portion_of_incincere = 1):
    data_df_1 = data_df[data_df["target"]==1].copy()
    data_df_2 = data_df[data_df["target"]==0].copy()
    data_df_1 = pd.concat([data_df_1]*portion_of_incincere)
    single_sincincere = data_df[data_df["target"]==0].sample(n=1,random_state=2018).copy()
    data_df_1 = _fuse_dataframes(single_sincincere,data_df_1)
#     data_df_2 = _fuse_dataframes(single_sincincere,data_df_2)
#     data_df_1 = pd.concat([data_df_1, data_df_2])
    return data_df_1,single_sincincere


# In[ ]:


def load_and_prec():
    data_df = pd.read_csv("../input/train.csv")
    test_df = pd.read_csv("../input/test.csv")
    print("Train shape : ",data_df.shape)
    print("Test shape : ",test_df.shape)
        
    X = data_df["question_text"].fillna("_##_").values
    train_df,val_df = split_df(data_df)
    train_left_X,train_right_X,train_left_Y,train_right_Y,train_Y,single_incincere,single_sincere = generate_pairs(train_df,augment=True)
    val_left_X,val_right_X,val_left_Y,val_right_Y,val_Y,_,_ = generate_pairs(val_df)
    test_X = test_df["question_text"].fillna("_##_").values
    print("Single sincere -",single_sincere["question_text"])
    print("Single incincere -",single_incincere["question_text"])

    ## Tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(X))
    train_left_X = tokenizer.texts_to_sequences(train_left_X)
    train_right_X = tokenizer.texts_to_sequences(train_right_X)
    val_left_X = tokenizer.texts_to_sequences(val_left_X)
    val_right_X = tokenizer.texts_to_sequences(val_right_X)
    test_X = tokenizer.texts_to_sequences(test_X)
    single_incincere_X = tokenizer.texts_to_sequences(single_incincere["question_text"].fillna("_##_").values)
    single_sincere_X = tokenizer.texts_to_sequences(single_sincere["question_text"].fillna("_##_").values)

    ## Pad the sentences 
    train_left_X = pad_sequences(train_left_X, maxlen=maxlen)
    train_right_X = pad_sequences(train_right_X, maxlen=maxlen)
    val_left_X = pad_sequences(val_left_X, maxlen=maxlen)
    val_right_X = pad_sequences(val_right_X, maxlen=maxlen)
    test_X = pad_sequences(test_X, maxlen=maxlen)
    single_incincere_X = pad_sequences(single_incincere_X, maxlen=maxlen)
    single_sincere_X = pad_sequences(single_sincere_X, maxlen=maxlen)
    
    return test_X,train_left_X,val_left_X,train_right_X,val_right_X,train_left_Y,val_left_Y,train_right_Y,val_right_Y,train_Y,val_Y, tokenizer.word_index,single_incincere_X,single_sincere_X


# In[ ]:


def load_glove(word_index):
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
            
    return embedding_matrix 
    
def load_fasttext(word_index):    
    EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix

def load_para(word_index):
    EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE,encoding="utf8", errors='ignore') if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
    return embedding_matrix


# In[ ]:


start_time = time.time()
test_X,train_left_X,val_left_X,train_right_X,val_right_X,train_left_Y,val_left_Y,train_right_Y,val_right_Y,train_Y,val_Y,word_index,single_incincere,single_sincere = load_and_prec()
embedding_matrix_1 = load_glove(word_index)
# embedding_matrix_2 = load_fasttext(word_index)
embedding_matrix_3 = load_para(word_index)
print(timedelta(seconds=time.time() - start_time))


# In[ ]:


import gc

embedding_matrix = np.mean([embedding_matrix_1, embedding_matrix_3], axis = 0)
np.shape(embedding_matrix)
del(embedding_matrix_1)
del(embedding_matrix_3)
gc.collect()


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
                                 name='{}_W'.format(self.name))
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((self.step_dim,),
                                     initializer=initializers.get('zero'),
                                     name='{}_W'.format(self.name))
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


def create_skip_model():
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.1)(x)
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    y = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
    
    atten_1 = Attention(maxlen)(x) # skip connect
    atten_2 = Attention(maxlen)(y)
    avg_pool = GlobalAveragePooling1D()(y)
    max_pool = GlobalMaxPooling1D()(y) 
    conc = concatenate([atten_1, atten_2, avg_pool, max_pool])
    conc = Dropout(0.1)(conc)
    outp = Dense(128, activation="sigmoid")(conc)
    return Model(inputs=inp, outputs=outp,name="skip_model")


# In[ ]:


from keras import initializers, regularizers, constraints, optimizers

def model_lstm_atten(embedding_matrix):
    
    left_inp = Input(shape=(maxlen,),name="left_input")
    right_inp = Input(shape=(maxlen,),name="right_input")
    inp = Input(shape=(maxlen,))
    rnn = create_skip_model()
    
    left_rnn = rnn(left_inp)
    right_rnn = rnn(right_inp)

    both = dot([left_rnn,right_rnn],axes=-1,normalize=True) # https://stackoverflow.com/a/52021481
    prediction_similarity = Dense(1,activation='sigmoid',name="similarity_classification")(both)
    siamese_net = Model(input=[left_inp,right_inp],output=[prediction_similarity])#,left_prediction,right_prediction])
    siamese_net.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1,'accuracy'])
    return siamese_net


# In[ ]:


model = model_lstm_atten(embedding_matrix)
model.summary()


# In[ ]:


reduce_err = EarlyStopping(monitor="val_loss",
                              min_delta=0,
                              patience=1,
                              verbose=0, mode='auto')

model.fit([train_left_X,train_right_X],train_Y,batch_size=512, epochs=10, validation_data=([val_left_X,val_right_X],val_Y),callbacks=[reduce_err])


# In[ ]:


## Extracting siamese embeddings

siamese_embedding = Model(inputs=model.get_layer("skip_model").get_input_at(0),output=model.get_layer("skip_model").get_output_at(0))
siamese_embedding.summary()
start_time = time.time()
# Using Kmeans for clustering the embeddings and prediction of test set.
train_left_X = np.unique(train_left_X,axis=0)
encoded_embeddings_train_left= siamese_embedding.predict(train_left_X,batch_size=512)
print(timedelta(seconds=time.time() - start_time))
start_time = time.time()
encoded_embeddings_val_left = siamese_embedding.predict(val_left_X,batch_size=512)
print(timedelta(seconds=time.time() - start_time))
encoded_embeddings_test = siamese_embedding.predict(test_X,batch_size=512)


# In[ ]:



start_time = time.time()
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0,n_jobs=-1,max_iter=5000).fit(encoded_embeddings_train_left)
print(timedelta(seconds=time.time() - start_time))


# In[ ]:


# metrics.f1_score(kmeans.predict(encoded_embeddings_val_left),val_left_Y)


# In[ ]:


kmeans_predictions = kmeans.predict(encoded_embeddings_test)
print(kmeans_predictions[:300])


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub.prediction = kmeans_predictions
sub.to_csv("submission.csv", index=False)


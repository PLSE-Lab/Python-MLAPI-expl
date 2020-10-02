#!/usr/bin/env python
# coding: utf-8

# This is a bare bones kernel that shows that even a very simple NN pipeline is non deterministic on kernels. 
# 
# This is without any Cudnn usage so that is not the reason
# 
#  Once everything is seeded properly I am unable to get non deterministic behaviour locally. So it must be something in the kernel environment.
#  
#  Even Cudnn behaves deterministically in my local environment.  Would love if others can verify running this kernel locally on a GPU.
#  
#  Things to keep in mind when you try to make your code deterministic. Cudnn has 2 initializers that need to be seeded. Also .fit has shuffle by default so I turn that off and shuffle manually. Also dropouts take a seed value. With all these things I can get deterministic behaviour (locally) even when running a loop to try hyper parameter changes.
# 
#  For a bit I thought it was the versioning since I thought I had non deterministic behaviour locally on 2.2.4 but I have not been able to duplicate that so seems to be a deadend. At this point I am throwing in the towel. Maybe someone else has some more ideas on the cause and remedy. At least we should stop blamming Cudnn.
# 
#  The biggest issue with this is that they presumablly will only be running these one time for final submissions. With this much variance I think it will put alot of luck in play.**
# 

# In[ ]:


#import sys
#!{sys.executable} -m conda install tensorflow-gpu -y
#!{sys.executable} -m pip install keras==2.1.4


# In[ ]:


import keras
import tensorflow as tf
print(keras.__version__)
print(tf.__version__)


# In[ ]:


## some config values 
embed_size = 300 # how big is each word vector
max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 50 # max number of words in a question to use

RANDOM_SEED = 42


# **Load packages and data**

# In[ ]:


import os
import time
import numpy as np # linear algebra
import random as rn

np.random.seed(RANDOM_SEED)
rn.seed(RANDOM_SEED)

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.optimizers import *
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import *
from keras.initializers import *

from tensorflow import set_random_seed
 
set_random_seed(RANDOM_SEED)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from tensorflow.python.client import device_lib
device_lib.list_local_devices()


# In[ ]:


def load_and_prec():
    train_df = pd.read_csv("../input/train.csv")
    test_df = pd.read_csv("../input/test.csv")
    print("Train shape : ",train_df.shape)
    print("Test shape : ",test_df.shape)
    
    ## fill up the missing values
    train_X = train_df["question_text"].fillna("_##_").values
    test_X = test_df["question_text"].fillna("_##_").values

    ## Tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    test_X = tokenizer.texts_to_sequences(test_X)

    ## Pad the sentences 
    train_X = pad_sequences(train_X, maxlen=maxlen)
    test_X = pad_sequences(test_X, maxlen=maxlen)

    ## Get the target values
    train_y = train_df['target'].values
    
    #shuffling the data
    np.random.seed(RANDOM_SEED)
    trn_idx = np.random.permutation(len(train_X))

    train_X = train_X[trn_idx]
    train_y = train_y[trn_idx]
    
    return train_X, test_X, train_y, tokenizer.word_index


# **Load embeddings**

# In[ ]:


def load_glove(word_index):
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding='utf-8'))

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    np.random.seed(RANDOM_SEED)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
            
    return embedding_matrix 
    


# **LSTM models**

# In[ ]:


def get_model(embedding_matrix):
    
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    #x = SpatialDropout1D(0.1, seed = RANDOM_SEED)(x)
    
    #x = Bidirectional(CuDNNLSTM(40, return_sequences=True, kernel_initializer=glorot_uniform(seed = RANDOM_SEED), \
    #                                recurrent_initializer=Orthogonal(seed = RANDOM_SEED)))(x)
        
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    
    conc = concatenate([avg_pool, max_pool])
    
    x = Dense(16, activation="relu", kernel_initializer=he_uniform(seed=RANDOM_SEED))(conc)
    #x = Dropout(0.1, seed = RANDOM_SEED)(x)
    outp = Dense(1, activation="sigmoid", kernel_initializer=he_uniform(seed=RANDOM_SEED))(x)    

    model = Model(inputs=inp, outputs=outp)
    optimizer = Adam()
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model


# **Train and predict**

# In[ ]:


# https://www.kaggle.com/strideradu/word2vec-and-gensim-go-go-go
def train_pred(model, train_X, train_y, val_X, val_y, epochs=2, callback=None):
    for e in range(epochs):
        
        np.random.seed(RANDOM_SEED + e)
        trn_idx = np.random.permutation(len(train_X))
        #print('trn_idx[:1000].sum()', trn_idx[:10000].sum())
        X = train_X[trn_idx]
        y = train_y[trn_idx]
        
        model.fit(X, y, batch_size=512, epochs=1, validation_data=(val_X, val_y), callbacks = callback, verbose=0, shuffle = False)
        pred_val_y = model.predict([val_X], batch_size=1024, verbose=0)
        
        best_score = 0.0
        best_score = metrics.f1_score(val_y, (pred_val_y > 0.33).astype(int))
        print("Epoch: ", e, "-    Val F1 Score: {:.4f}".format(best_score))

    pred_test_y = model.predict([test_X], batch_size=1024, verbose=0)
    print('='*100)
    return pred_val_y, pred_test_y, best_score


# **Main part: load, train, pred and blend**

# In[ ]:


train_X, test_X, train_y, word_index = load_and_prec()
embedding_matrix = load_glove(word_index)


# In[ ]:


epochs = 5


# In[ ]:


get_ipython().run_cell_magic('time', '', 'set_random_seed(RANDOM_SEED)\n\nsplits = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_SEED).split(train_X, train_y))\nfor idx, (train_idx, valid_idx) in list(enumerate(splits))[:1]:\n        X_train = train_X[train_idx]\n        y_train = train_y[train_idx]\n        X_val = train_X[valid_idx]\n        y_val = train_y[valid_idx]\n        model = get_model(embedding_matrix)\n        \n        pred_val_y, pred_test_y, best_score = train_pred(model, X_train, y_train, X_val, y_val, epochs = epochs)\n        \n        for l in model.layers:\n            for w in l.get_weights():\n                print(w.shape, w.sum())')


# # This should output the same weights if everything is deterministic. 
# # This works fine for me on local kernels even after adding more depth/epochs/Cudnn etc
# 
# This actually just ran deterministically for the first time I have seen in many many tries. Right before this and I was going to publish it. Good thing I checked.

# In[ ]:


set_random_seed(RANDOM_SEED)

splits = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_SEED).split(train_X, train_y))
for idx, (train_idx, valid_idx) in list(enumerate(splits))[:1]:
        X_train = train_X[train_idx]
        y_train = train_y[train_idx]
        X_val = train_X[valid_idx]
        y_val = train_y[valid_idx]
        model = get_model(embedding_matrix)
        
        pred_val_y, pred_test_y, best_score = train_pred(model, X_train, y_train, X_val, y_val, epochs = epochs)
        
        for l in model.layers:
            for w in l.get_weights():
                print(w.shape, w.sum())


# In[ ]:





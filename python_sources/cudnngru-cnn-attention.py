#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import time
import numpy as np 
import pandas as pd 
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, concatenate
from keras.layers import ConvRNN2D, SpatialDropout1D, Reshape, MaxPool2D, Concatenate, Flatten, Conv2D
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
get_ipython().run_line_magic('matplotlib', 'inline')

from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.engine.topology import Layer

#GPU configs
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

gpu_options = K.tf.GPUOptions(per_process_gpu_memory_fraction = 1)
config = K.tf.ConfigProto(gpu_options = gpu_options, allow_soft_placement = True)
K.set_session(K.tf.Session(config = config))

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#Source: https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings
print([dev.name for dev in device_lib.list_local_devices()])


# ### Load the Data

# In[ ]:


get_ipython().run_cell_magic('time', '', 'base_path = "../input/"\ntrain_df = pd.read_csv(base_path+"train.csv")\ntest_df = pd.read_csv(base_path+"test.csv")\nprint("Train shape : ",train_df.shape)\nprint("Test shape : ",test_df.shape)\n\n\n## split to train and val\ntrain_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2018)\n\n## some config values \nembed_size = 300 # how big is each word vector\nmax_features = 50000 # how many unique words to use (i.e num rows in embedding vector)\nmaxlen = 100 # max number of words in a question to use\n\n## fill up the missing values\ntrain_X = train_df["question_text"].fillna("_na_").values\nval_X = val_df["question_text"].fillna("_na_").values\ntest_X = test_df["question_text"].fillna("_na_").values\n\n## Tokenize the sentences\ntokenizer = Tokenizer(num_words=max_features)\ntokenizer.fit_on_texts(list(train_X))\ntrain_X = tokenizer.texts_to_sequences(train_X)\nval_X = tokenizer.texts_to_sequences(val_X)\ntest_X = tokenizer.texts_to_sequences(test_X)\n\n## Pad the sentences \ntrain_X = pad_sequences(train_X, maxlen=maxlen)\nval_X = pad_sequences(val_X, maxlen=maxlen)\ntest_X = pad_sequences(test_X, maxlen=maxlen)\n\n## Get the target values\ntrain_y = train_df[\'target\'].values\nval_y = val_df[\'target\'].values\n\nprint("Train: ", train_X.shape, train_y.shape)\nprint("Validation: ", val_X.shape, val_y.shape)\nprint("Test :", test_X.shape)')


# ### Load the Embeddings

# In[ ]:


get_ipython().run_cell_magic('time', '', 'def get_coefs(word,*arr): \n    return word, np.asarray(arr, dtype=\'float32\')\n\nbase_path = "../input/embeddings/"\nfiles = {"glove": "glove.840B.300d/glove.840B.300d.txt",\n         "wiki_news": "wiki-news-300d-1M/wiki-news-300d-1M.vec",\n         "paragram": "paragram_300_sl999/paragram_300_sl999.txt"}\n\nembedding_matrices = {}\n\nfor emb in files:\n    if emb=="glove":\n        emb_index = dict(get_coefs(*o.split(" ")) for o in open(base_path+files[emb]))\n    elif emb=="wiki_news":\n        emb_index = dict(get_coefs(*o.split(" ")) for o in open(base_path+files[emb]) \\\n                         if len(o)>100)\n    elif emb=="paragram":\n        emb_index = dict(get_coefs(*o.split(" ")) for o in open(base_path+files[emb],\n                                                             encoding="utf8", \n                                                             errors=\'ignore\') \\\n                         if len(o)>100)\n    all_embs = np.stack(emb_index.values())\n    emb_mean, emb_std = all_embs.mean(), all_embs.std()\n    embed_size = all_embs.shape[1]\n    \n    word_index = tokenizer.word_index\n    nb_words = min(max_features, len(word_index))\n    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))\n    for word, i in word_index.items():\n        if i >= max_features: continue\n        embedding_vector = emb_index.get(word)\n        if embedding_vector is not None: embedding_matrix[i] = embedding_vector\n            \n    embedding_matrices[emb] = embedding_matrix')


# ### CuDNNGRU network

# In[ ]:


get_ipython().run_cell_magic('time', '', 'inp = Input(shape=(maxlen,))\n\ninp_glove = Embedding(max_features, embed_size, weights=[embedding_matrices["glove"]])(inp)\ninp_glove = Bidirectional(CuDNNGRU(128, return_sequences=True))(inp_glove)\ninp_glove = GlobalMaxPool1D()(inp_glove)\n\ninp_wiki = Embedding(max_features, embed_size, weights=[embedding_matrices["wiki_news"]])(inp)\ninp_wiki = Bidirectional(CuDNNGRU(128, return_sequences=True))(inp_wiki)\ninp_wiki = GlobalMaxPool1D()(inp_wiki)\n\ninp_paragram = Embedding(max_features, embed_size, weights=[embedding_matrices["paragram"]])(inp)\ninp_paragram = Bidirectional(CuDNNGRU(128, return_sequences=True))(inp_paragram)\ninp_paragram = GlobalMaxPool1D()(inp_paragram)\n\nmerged  = concatenate([inp_glove, inp_wiki, inp_paragram])\nx = Dense(32, activation="relu")(merged)\nx = Dropout(0.1)(x)\nx = Dense(1, activation="sigmoid")(x)\nmodel = Model(inputs=inp, outputs=x)\nmodel.compile(loss=\'binary_crossentropy\', optimizer=\'adam\', metrics=[\'accuracy\'])\nprint(model.summary())\n\n#train\nmodel.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))\n\n#model performance \npred_val_y = model.predict([val_X], batch_size=1024, verbose=1)\nf1_scores = []\nthreshs = np.arange(0.1, 0.9, 0.01)\nfor thresh in threshs:\n    thresh = np.round(thresh, 2)\n    f1_score = metrics.f1_score(val_y, (pred_val_y>thresh).astype(int))\n    f1_scores.append(f1_score)\n    \nplt.plot(threshs, f1_scores)\nmax_fscore = np.round(f1_scores[np.argmax(f1_scores)], 3)\nmax_thresh = np.round(threshs[np.argmax(f1_scores)], 3)\nplt.title("F scores at different values of thresholds | Max: {} | Thresh {}".format(max_fscore, max_thresh))')


# In[ ]:


thresh = max_thresh

#predictions on the validation set
pred_val_y_1 = model.predict([val_X], batch_size=1024, verbose=1)
#pred_val_y_1 = (pred_val_y_1>=thresh).astype(int)

#predictions on the test set
pred_test_y_1 = model.predict([test_X], batch_size=1024, verbose=1)
#pred_test_y_1 = (pred_test_y_1>=thresh).astype(int)


# ### CNN architecture

# In[ ]:


get_ipython().run_cell_magic('time', '', 'filter_sizes = [1, 2, 3, 5]\nnum_filt = 36\n\ninp = Input(shape=(maxlen,))\n\nembeddings = [Embedding(max_features, embed_size, weights=[embedding_matrices[emb]])(inp) \\\n              for emb in embedding_matrices]\n\nembed_maxpools = []\nfor embed in embeddings:\n    embed = SpatialDropout1D(0.1)(embed)\n    embed = Reshape((maxlen, embed_size, 1))(embed)\n    \n    maxpools = []\n    for i in range(len(filter_sizes)):\n        conv = Conv2D(num_filt, kernel_size=(filter_sizes[i], embed_size),\n                  kernel_initializer="he_normal", activation="elu")(embed)\n        maxpools.append(MaxPool2D(pool_size=(maxlen - filter_sizes[i] + 1, 1))(conv))\n    merged = Concatenate(axis=1)(maxpools)\n    merged = Flatten()(merged)\n    merged = Dropout(0.2)(merged)\n    \n    embed_maxpools.append(merged)\n\nembed_maxpools = Concatenate(axis=1)(embed_maxpools)\n   \nx = Dense(32, activation="relu")(embed_maxpools)\nx = Dropout(0.1)(x)\nx = Dense(1, activation="sigmoid")(x)\nmodel = Model(inputs=inp, outputs=x)\nmodel.compile(loss=\'binary_crossentropy\', optimizer=\'adam\', metrics=[\'accuracy\'])\nprint(model.summary())\n\n#train\nmodel.fit(train_X, train_y, batch_size=256, epochs=2, validation_data=(val_X, val_y))\n\n#model performance \npred_val_y = model.predict([val_X], batch_size=1024, verbose=1)\nf1_scores = []\nthreshs = np.arange(0.1, 0.9, 0.01)\nfor thresh in threshs:\n    thresh = np.round(thresh, 2)\n    f1_score = metrics.f1_score(val_y, (pred_val_y>thresh).astype(int))\n    f1_scores.append(f1_score)\n    \nplt.plot(threshs, f1_scores)\nmax_fscore = np.round(f1_scores[np.argmax(f1_scores)], 3)\nmax_thresh = np.round(threshs[np.argmax(f1_scores)], 3)\nplt.title("F scores at different values of thresholds | Max: {} | Thresh {}".format(max_fscore, max_thresh))')


# In[ ]:


thresh = max_thresh

#predictions on the validation set
pred_val_y_2 = model.predict([val_X], batch_size=1024, verbose=1)
#pred_val_y_2 = (pred_val_y_2>=thresh).astype(int)

#predictions on the test set
pred_test_y_2 = model.predict([test_X], batch_size=1024, verbose=1)
#pred_test_y_2 = (pred_test_y_2>=thresh).astype(int)


# ### CuDNNGRU architecture with Attention Layer

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


get_ipython().run_cell_magic('time', '', 'inp = Input(shape=(maxlen,))\n\ninp_glove = Embedding(max_features, embed_size, weights=[embedding_matrices["glove"]])(inp)\ninp_glove = Bidirectional(CuDNNGRU(128, return_sequences=True))(inp_glove)\ninp_glove = Attention(maxlen)(inp_glove)\n\ninp_wiki = Embedding(max_features, embed_size, weights=[embedding_matrices["wiki_news"]])(inp)\ninp_wiki = Bidirectional(CuDNNGRU(128, return_sequences=True))(inp_wiki)\ninp_wiki = Attention(maxlen)(inp_wiki)\n\ninp_paragram = Embedding(max_features, embed_size, weights=[embedding_matrices["paragram"]])(inp)\ninp_paragram = Bidirectional(CuDNNGRU(128, return_sequences=True))(inp_paragram)\ninp_paragram = Attention(maxlen)(inp_paragram)\n\nmerged  = concatenate([inp_glove, inp_wiki, inp_paragram])\nx = Dense(32, activation="relu")(merged)\nx = Dropout(0.1)(x)\nx = Dense(16, activation="relu")(x)\nx = Dropout(0.1)(x)\nx = Dense(1, activation="sigmoid")(x)\nmodel = Model(inputs=inp, outputs=x)\nmodel.compile(loss=\'binary_crossentropy\', optimizer=\'adam\', metrics=[\'accuracy\'])\nprint(model.summary())\n\n#train\nmodel.fit(train_X, train_y, batch_size=256, epochs=2, validation_data=(val_X, val_y))\n\n#model performance \npred_val_y = model.predict([val_X], batch_size=1024, verbose=1)\nf1_scores = []\nthreshs = np.arange(0.1, 0.9, 0.01)\nfor thresh in threshs:\n    thresh = np.round(thresh, 2)\n    f1_score = metrics.f1_score(val_y, (pred_val_y>thresh).astype(int))\n    f1_scores.append(f1_score)\n    \nplt.plot(threshs, f1_scores)\nmax_fscore = np.round(f1_scores[np.argmax(f1_scores)], 3)\nmax_thresh = np.round(threshs[np.argmax(f1_scores)], 3)\nplt.title("F scores at different values of thresholds | Max: {} | Thresh {}".format(max_fscore, max_thresh))')


# In[ ]:


thresh = max_thresh

#predictions on the validation set
pred_val_y_3 = model.predict([val_X], batch_size=1024, verbose=1)
#pred_val_y_3 = (pred_val_y_3>=thresh).astype(int)

#predictions on the test set
pred_test_y_3 = model.predict([test_X], batch_size=1024, verbose=1)
#pred_test_y_3 = (pred_test_y_3>=thresh).astype(int)


# #### Ensemble Method on the Predictions

# In[ ]:


pred_val_y = 0.5*pred_val_y_1 + 0.3*pred_val_y_2 + 0.2*pred_val_y_3
f1_scores = []
threshs = np.arange(0.1, 0.9, 0.01)
for thresh in threshs:
    thresh = np.round(thresh, 2)
    f1_score = metrics.f1_score(val_y, (pred_val_y>thresh).astype(int))
    f1_scores.append(f1_score)
    
plt.plot(threshs, f1_scores)
max_fscore = np.round(f1_scores[np.argmax(f1_scores)], 3)
max_thresh = np.round(threshs[np.argmax(f1_scores)], 3)
plt.title("F scores at different values of thresholds | Max: {} | Thresh {}".format(max_fscore, max_thresh))


# In[ ]:


pred_test_y = 0.5*pred_test_y_1 + 0.3*pred_test_y_2 + 0.2*pred_test_y_3
pred_test_y = (pred_test_y>=max_thresh).astype(int)
out_df = pd.DataFrame({"qid":test_df["qid"].values})
out_df['prediction'] = pred_test_y
out_df.to_csv("submission.csv", index=False)


# In[ ]:





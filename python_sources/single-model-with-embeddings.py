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


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


X_train = train_df["question_text"].fillna("na").values
X_test = test_df["question_text"].fillna("na").values
y = train_df["target"]


# In[ ]:


from keras.models import Model
from keras.layers import Input, Dense, Embedding, concatenate
from keras.layers import CuDNNGRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, Conv1D
from keras.layers import Add, BatchNormalization, Activation, CuDNNLSTM, Dropout
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import gc
from sklearn import metrics
from keras import regularizers


# In[ ]:


maxlen = 50
max_features = 50000
embed_size = 300

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)


# In[ ]:


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    TIME_STEPS = inputs.shape[1].value
    SINGLE_ATTENTION_VECTOR = False
    
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


# In[ ]:


from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


# # MODELS
# 1. CNN
# 2. LSTM/GRU
# 
# For me LSTM work better than GRU

# In[ ]:


def model_cnn(embedding_matrix):
    filter_sizes = [1, 2, 3, 5]
    num_filters = 64
    inp = Input(shape=(maxlen, ))
    embed = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.1)(embed)
    
    mpool = []
    x = Reshape((maxlen, embed_size, 1))(x)
    for fil in filter_sizes:
        conv = Conv2D(num_filters, (fil, embed_size), kernel_initializer='he_normal', activation='relu')(x)
        pool = MaxPool2D(pool_size=(maxlen - fil + 1, 1))(conv)
        mpool.append(pool)
        
    x = Concatenate(axis=1)(mpool)
    x = Flatten()(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation='sigmoid')(x)
    
    model = Model(inp, x)
    return model
    

def model_bgru(embedding_matrix):
    inp = Input(shape=(maxlen, ))
    embed = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = embed
    x = SpatialDropout1D(0.2)(x)
    
    x1 = Bidirectional(CuDNNLSTM(64, return_sequences=True ))(x)
    x2 = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x1)
    x = Add()([x1, x2])
    x = AttLayer(maxlen)(x)
    x = Dropout(0.4)(x)
    x = Dense(1, activation='sigmoid')(x)
    
    model = Model(inp, x)
    return model


# In[ ]:


from sklearn.model_selection import train_test_split
X_tra, X_val, y_tra, y_val = train_test_split(x_train, y, test_size = 0.1, random_state=42)


# In[ ]:


VAL_Y = []
TEST_Y = []


# # GLOVE EMBEDDING

# In[ ]:


EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# # GLOVE  - CNN

# In[ ]:


model = model_cnn(embedding_matrix)
#model.summary()
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])
batch_size = 2048
epochs = 5
model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), verbose=True)
model.save('./model_cnn_glove.h5')


# In[ ]:


val_pred_cnn_glove = model.predict([X_val], batch_size=1024, verbose=1)
thresholds = []
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    res = metrics.f1_score(y_val, (val_pred_cnn_glove > thresh).astype(int))
    thresholds.append([thresh, res])
    print("F1 score at threshold {0} is {1}".format(thresh, res))
    
thresholds.sort(key=lambda x: x[1], reverse=True)
best_thresh = thresholds[0][0]
print("Best threshold: ", best_thresh)

y_pred_cnn_glove = model.predict(x_test, batch_size=1024, verbose=True)

VAL_Y.append(val_pred_cnn_glove)
TEST_Y.append(y_pred_cnn_glove)


# # GLOVE  - BGRU

# In[ ]:


model = model_bgru(embedding_matrix)
#model.summary()
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])
batch_size = 2048
epochs = 5
model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), verbose=True)
model.save('./model_bgru_glove.h5')


# In[ ]:


val_pred_bgru_glove = model.predict([X_val], batch_size=1024, verbose=1)
thresholds = []
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    res = metrics.f1_score(y_val, (val_pred_bgru_glove > thresh).astype(int))
    thresholds.append([thresh, res])
    print("F1 score at threshold {0} is {1}".format(thresh, res))
    
thresholds.sort(key=lambda x: x[1], reverse=True)
best_thresh = thresholds[0][0]
print("Best threshold: ", best_thresh)

y_pred_bgru_glove = model.predict(x_test, batch_size=1024, verbose=True)

VAL_Y.append(val_pred_bgru_glove)
TEST_Y.append(y_pred_bgru_glove)


# In[ ]:


del embeddings_index;
del embedding_matrix
gc.collect()  


# # WIKI_NEWS - BGRU

# In[ ]:


EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# In[ ]:


model = model_bgru(embedding_matrix)
#model.summary()
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])
batch_size = 2048
epochs = 5
model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), verbose=True)
model.save('./model_bgru_wiki.h5')


# In[ ]:


val_pred_bgru_wiki = model.predict([X_val], batch_size=1024, verbose=1)
thresholds = []
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    res = metrics.f1_score(y_val, (val_pred_bgru_wiki > thresh).astype(int))
    thresholds.append([thresh, res])
    print("F1 score at threshold {0} is {1}".format(thresh, res))
    
thresholds.sort(key=lambda x: x[1], reverse=True)
best_thresh = thresholds[0][0]
print("Best threshold: ", best_thresh)

y_pred_bgru_wiki = model.predict(x_test, batch_size=1024, verbose=True)

VAL_Y.append(val_pred_bgru_wiki)
TEST_Y.append(y_pred_bgru_wiki)


# In[ ]:


del embeddings_index;
del embedding_matrix
gc.collect()


# # PARAGRAM - BGRU 

# In[ ]:


EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# In[ ]:


model = model_bgru(embedding_matrix)
#model.summary()
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])
batch_size = 2048
epochs = 5
model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), verbose=True)
model.save('./model_bgru_paragram.h5')


# In[ ]:


val_pred_bgru_paragram = model.predict([X_val], batch_size=1024, verbose=1)
thresholds = []
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    res = metrics.f1_score(y_val, (val_pred_bgru_paragram > thresh).astype(int))
    thresholds.append([thresh, res])
    print("F1 score at threshold {0} is {1}".format(thresh, res))
    
thresholds.sort(key=lambda x: x[1], reverse=True)
best_thresh = thresholds[0][0]
print("Best threshold: ", best_thresh)

y_pred_bgru_paragram = model.predict(x_test, batch_size=1024, verbose=True)

VAL_Y.append(val_pred_bgru_paragram)
TEST_Y.append(y_pred_bgru_paragram)


# # Concat Result & Best Threshold

# In[ ]:



pred_val_y = (2.5*val_pred_cnn_glove + 2.5*val_pred_bgru_glove + 2.5*val_pred_bgru_wiki + 2.5*val_pred_bgru_paragram)/10

thresholds = []
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    res = metrics.f1_score(y_val, (pred_val_y > thresh).astype(int))
    thresholds.append([thresh, res])
    print("F1 score at threshold {0} is {1}".format(thresh, res))
    
thresholds.sort(key=lambda x: x[1], reverse=True)
best_thresh = thresholds[0][0]
print("Best threshold: ", best_thresh)


# # Submission

# In[ ]:


# y_pred = 0
# for i in range(len(y_th)):
#     y_pred += (y_th[i] * TEST_Y[i])
# y_pred = y_pred / 10

y_pred = (2.5*y_pred_cnn_glove + 2.5*y_pred_bgru_glove + 2.5*y_pred_bgru_wiki + 2.5*y_pred_bgru_paragram)/10

y_te = (y_pred[:,0] > best_thresh).astype(np.int)

submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": y_te})
submit_df.to_csv("submission.csv", index=False)


# In[ ]:


from IPython.display import HTML
import base64  
import pandas as pd  

def create_download_link( df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv(index =False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

create_download_link(submit_df)


# In[ ]:





# In[ ]:





# In[ ]:




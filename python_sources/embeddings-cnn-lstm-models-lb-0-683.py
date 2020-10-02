#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow import set_random_seed

np.random.seed(1)
set_random_seed(1)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import nltk, re, string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


# In[ ]:


def clean_text(text):
    print(text)
    
    ## Remove puncuation
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower()
    
    ## Remove stop words
    #text = text.split()
    #stops = set(stopwords.words("english"))
    #text = [w for w in text if not w in stops and len(w) >= 3]
    
    #text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    #text = text.split()
    #stemmer = SnowballStemmer('english')
    #stemmed_words = [stemmer.stem(word) for word in text]
    #text = " ".join(stemmed_words)

    print(text)
    print("")
    return text


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
#train_df["question_text"] = train_df["question_text"].map(lambda x: clean_text(x))

test_df = pd.read_csv('../input/test.csv')
#test_df["question_text"] = test_df["question_text"].map(lambda x: clean_text(x))


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
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import gc
from sklearn import metrics


# In[ ]:


maxlen = 70
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


# In[ ]:


EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix_1 = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix_1[i] = embedding_vector

del embeddings_index; gc.collect() 


# In[ ]:


EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix_2 = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix_2[i] = embedding_vector
del embeddings_index; gc.collect()


# In[ ]:


EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix_3 = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix_3[i] = embedding_vector
        
del embeddings_index; gc.collect()   


# In[ ]:


# # https://www.kaggle.com/strideradu/word2vec-and-gensim-go-go-go
# from gensim.models import KeyedVectors

# EMBEDDING_FILE = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
# embeddings_index = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

# word_index = tokenizer.word_index
# nb_words = min(max_features, len(word_index))
# embedding_matrix_4 = (np.random.rand(nb_words, embed_size) - 0.5) / 5.0
# for word, i in word_index.items():
#     if i >= max_features: continue
#     if word in embeddings_index:
#         embedding_vector = embeddings_index.get_vector(word)
#         embedding_matrix_4[i] = embedding_vector
        
# del embeddings_index; gc.collect()        


# # Concatenating the embeddings

# In[ ]:


embedding_matrix = np.concatenate((embedding_matrix_1, embedding_matrix_2, embedding_matrix_3), axis=1)  
del embedding_matrix_1, embedding_matrix_2, embedding_matrix_3
gc.collect()
np.shape(embedding_matrix)


# In[ ]:


from sklearn.model_selection import train_test_split
X_tra, X_val, y_tra, y_val = train_test_split(x_train, y, test_size = 0.1, random_state=42)


# # Capsule Net Block

# In[ ]:


# def squash(x, axis=-1):
#     # s_squared_norm is really small
#     # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
#     # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
#     # return scale * x
#     s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
#     scale = K.sqrt(s_squared_norm + K.epsilon())
#     return x / scale


# # A Capsule Implement with Pure Keras
# class Capsule(Layer):
#     def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
#                  activation='default', **kwargs):
#         super(Capsule, self).__init__(**kwargs)
#         self.num_capsule = num_capsule
#         self.dim_capsule = dim_capsule
#         self.routings = routings
#         self.kernel_size = kernel_size
#         self.share_weights = share_weights
#         if activation == 'default':
#             self.activation = squash
#         else:
#             self.activation = Activation(activation)

#     def build(self, input_shape):
#         super(Capsule, self).build(input_shape)
#         input_dim_capsule = input_shape[-1]
#         if self.share_weights:
#             self.W = self.add_weight(name='capsule_kernel',
#                                      shape=(1, input_dim_capsule,
#                                             self.num_capsule * self.dim_capsule),
#                                      # shape=self.kernel_size,
#                                      initializer='glorot_uniform',
#                                      trainable=True)
#         else:
#             input_num_capsule = input_shape[-2]
#             self.W = self.add_weight(name='capsule_kernel',
#                                      shape=(input_num_capsule,
#                                             input_dim_capsule,
#                                             self.num_capsule * self.dim_capsule),
#                                      initializer='glorot_uniform',
#                                      trainable=True)

#     def call(self, u_vecs):
#         if self.share_weights:
#             u_hat_vecs = K.conv1d(u_vecs, self.W)
#         else:
#             u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

#         batch_size = K.shape(u_vecs)[0]
#         input_num_capsule = K.shape(u_vecs)[1]
#         u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
#                                             self.num_capsule, self.dim_capsule))
#         u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
#         # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

#         b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
#         for i in range(self.routings):
#             b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
#             c = K.softmax(b)
#             c = K.permute_dimensions(c, (0, 2, 1))
#             b = K.permute_dimensions(b, (0, 2, 1))
#             outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
#             if i < self.routings - 1:
#                 b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

#         return outputs

#     def compute_output_shape(self, input_shape):
#         return (None, self.num_capsule, self.dim_capsule)


# # MODEL 1: Conv2D

# In[ ]:


# def model1():
#     inp = Input(shape=(maxlen, ))
#     embed = Embedding(max_features, embed_size * 3, weights=[embedding_matrix], trainable=False)(inp)
    
#     #x = Reshape((maxlen, embed_size * 4, 1))(embed)
    
#     filter_sizes = [1,2,3,5]
#     num_filters = 64
    
#     conv_0 = Conv1D(num_filters, filter_sizes[0], padding='valid', kernel_initializer='normal', activation='relu')(embed)
#     conv_1 = Conv1D(num_filters, filter_sizes[1], padding='valid', kernel_initializer='normal', activation='relu')(embed)
#     conv_2 = Conv1D(num_filters, filter_sizes[2], padding='valid', kernel_initializer='normal', activation='relu')(embed)
#     conv_3 = Conv1D(num_filters, filter_sizes[3], padding='valid', kernel_initializer='normal', activation='relu')(embed)

#     maxpool_0 = MaxPool1D(pool_size=(maxlen - filter_sizes[0] + 1), strides=(1), padding='valid')(conv_0)
#     maxpool_1 = MaxPool1D(pool_size=(maxlen - filter_sizes[1] + 1), strides=(1), padding='valid')(conv_1)
#     maxpool_2 = MaxPool1D(pool_size=(maxlen - filter_sizes[2] + 1), strides=(1), padding='valid')(conv_2)
#     maxpool_3 = MaxPool1D(pool_size=(maxlen - filter_sizes[3] + 1), strides=(1), padding='valid')(conv_3)

#     concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])
    
#     #gmp = GlobalMaxPooling1D()(concatenated_tensor)
#     #gap = GlobalAveragePooling1D()(concatenated_tensor)
    
#     #conc = Concatenate(axis=1)([gmp, gap])
    
#     flatten = Flatten()(concatenated_tensor)
    
#     x = flatten
#     x = Dropout(0.3)(x)
#     x = Dense(128, activation='relu')(x)
#     outp = Dense(1, activation="sigmoid")(x)
#     model = Model(inputs=inp, outputs=outp)
#     model.compile(loss='binary_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'])    

#     return model


# In[ ]:


# MODEL1 = model1()
# MODEL1.summary()

# batch_size = 1536
# epochs = 3

# early_stopping = EarlyStopping(patience=3, verbose=1, monitor='val_loss', mode='min')
# model_checkpoint = ModelCheckpoint('./model1.model', save_best_only=True, verbose=1, monitor='val_loss', mode='min')
# reduce_lr = ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.0001, verbose=1)

# hist = MODEL1.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), verbose=True)
# MODEL1.save('./model1.h5')


# In[ ]:


# pred_val_y_1 = MODEL1.predict([X_val], batch_size=1024, verbose=1)
# thresholds = []
# for thresh in np.arange(0.1, 0.501, 0.01):
#     thresh = np.round(thresh, 2)
#     res = metrics.f1_score(y_val, (pred_val_y_1 > thresh).astype(int))
#     thresholds.append([thresh, res])
#     print("F1 score at threshold {0} is {1}".format(thresh, res))
    
# thresholds.sort(key=lambda x: x[1], reverse=True)
# best_thresh_1 = thresholds[0][0]
# print("Best threshold: ", best_thresh_1)

# y_pred_1 = MODEL1.predict(x_test, batch_size=1024, verbose=True)


# # MODEL 2: LSTM

# In[ ]:


def model2():
    inp = Input(shape=(maxlen, ))
    embed = Embedding(max_features, embed_size * 3, weights=[embedding_matrix], trainable=False)(inp)
    x = embed
    
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    x = attention_3d_block(x)
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    x = AttLayer(64)(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    outp = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])    

    return model


# In[ ]:


MODEL2 = model2()
MODEL2.summary()

batch_size = 2048
epochs = 3

early_stopping = EarlyStopping(patience=3, verbose=1, monitor='val_loss', mode='min')
model_checkpoint = ModelCheckpoint('./model2.model', save_best_only=True, verbose=1, monitor='val_loss', mode='min')
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.0001, verbose=1)

hist = MODEL2.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), verbose=True)
MODEL2.save('./model2.h5')


# In[ ]:


pred_val_y_2 = MODEL2.predict([X_val], batch_size=1024, verbose=1)
thresholds = []
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    res = metrics.f1_score(y_val, (pred_val_y_2 > thresh).astype(int))
    thresholds.append([thresh, res])
    print("F1 score at threshold {0} is {1}".format(thresh, res))
    
thresholds.sort(key=lambda x: x[1], reverse=True)
best_thresh_2 = thresholds[0][0]
print("Best threshold: ", best_thresh_2)

y_pred_2 = MODEL2.predict(x_test, batch_size=1024, verbose=True)


# # MODEL 3: Conv1D

# In[ ]:


# def model3():
#     filters = 128
    
#     inp = Input(shape=(maxlen, ))
#     embed = Embedding(max_features, embed_size * 3, weights=[embedding_matrix], trainable=False)(inp)
#     x = embed
    
#     x = Conv1D(filters, 1, activation='relu')(x)
#     x = Dropout(0.1)(x)
    
#     x = Conv1D(filters, 2, activation='relu')(x)
#     x = Dropout(0.1)(x)
    
#     x = Conv1D(filters, 3, activation='relu')(x)
#     x = Dropout(0.1)(x)
    
#     x = Conv1D(filters, 5, activation='relu')(x)
#     x = Dropout(0.1)(x)
    
#     #x = Flatten()(x)
#     x = GlobalAveragePooling1D()(x)
    
#     x = Dropout(0.3)(x)
#     x = Dense(128, activation='relu')(x)
#     outp = Dense(1, activation="sigmoid")(x)
#     model = Model(inputs=inp, outputs=outp)
#     model.compile(loss='binary_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'])    
#     return model


# In[ ]:


# MODEL3 = model3()
# MODEL3.summary()

# batch_size = 2048
# epochs = 1

# early_stopping = EarlyStopping(patience=3, verbose=1, monitor='val_loss', mode='min')
# model_checkpoint = ModelCheckpoint('./model3.model', save_best_only=True, verbose=1, monitor='val_loss', mode='min')
# reduce_lr = ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.0001, verbose=1)

# hist = MODEL3.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), verbose=True)
# MODEL3.save('./model3.h5')


# In[ ]:


# pred_val_y_3 = MODEL3.predict([X_val], batch_size=1024, verbose=1)
# thresholds = []
# for thresh in np.arange(0.1, 0.501, 0.01):
#     thresh = np.round(thresh, 2)
#     res = metrics.f1_score(y_val, (pred_val_y_3 > thresh).astype(int))
#     thresholds.append([thresh, res])
#     print("F1 score at threshold {0} is {1}".format(thresh, res))
    
# thresholds.sort(key=lambda x: x[1], reverse=True)
# best_thresh_3 = thresholds[0][0]
# print("Best threshold: ", best_thresh_3)

# y_pred_3 = MODEL3.predict(x_test, batch_size=1024, verbose=True)


# # MODEL 4: GRU

# In[ ]:


def model4():
    inp = Input(shape=(maxlen, ))
    embed = Embedding(max_features, embed_size * 3, weights=[embedding_matrix], trainable=False)(inp)
    x = embed
    
    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
    x = attention_3d_block(x)
    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
    x = AttLayer(64)(x)
    
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    outp = Dense(1, activation="sigmoid")(x)
    outp = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])    

    return model


# In[ ]:


MODEL4 = model4()
MODEL4.summary()

batch_size = 1536
epochs = 3

early_stopping = EarlyStopping(patience=3, verbose=1, monitor='val_loss', mode='min')
model_checkpoint = ModelCheckpoint('./model4.model', save_best_only=True, verbose=1, monitor='val_loss', mode='min')
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.0001, verbose=1)


hist = MODEL4.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), verbose=True)
MODEL4.save('./model4.h5')


# In[ ]:


pred_val_y_4 = MODEL4.predict([X_val], batch_size=1024, verbose=1)
thresholds = []
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    res = metrics.f1_score(y_val, (pred_val_y_4 > thresh).astype(int))
    thresholds.append([thresh, res])
    print("F1 score at threshold {0} is {1}".format(thresh, res))
    
thresholds.sort(key=lambda x: x[1], reverse=True)
best_thresh_4 = thresholds[0][0]
print("Best threshold: ", best_thresh_4)

y_pred_4 = MODEL4.predict(x_test, batch_size=1024, verbose=True)


# # Model 5: GRU Add

# In[ ]:


def model5():
    inp = Input(shape=(maxlen, ))
    embed = Embedding(max_features, embed_size * 3, weights=[embedding_matrix], trainable=False)(inp)
    x = embed
    
    x0 = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
    x1 = attention_3d_block(x0)
    x2 = Bidirectional(CuDNNGRU(128, return_sequences=True))(x1)
    x3 = Add()([x0, x2])
    x4 = Bidirectional(CuDNNGRU(64, return_sequences=True))(x3)
    x5 = AttLayer(64)(x4)
    #x5 = Capsule(num_capsule=5, dim_capsule=32, routings=5, share_weights=True)(x4)
    
    x = Dropout(0.3)(x5)
    x = Dense(128, activation='relu')(x)
    outp = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])    

    return model


# In[ ]:


MODEL5 = model5()
MODEL5.summary()

batch_size = 1536
epochs = 3

early_stopping = EarlyStopping(patience=3, verbose=1, monitor='val_loss', mode='min')
model_checkpoint = ModelCheckpoint('./model5.model', save_best_only=True, verbose=1, monitor='val_loss', mode='min')
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.0001, verbose=1)

hist = MODEL5.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), verbose=True)
MODEL5.save('./model5.h5')


# In[ ]:


pred_val_y_5 = MODEL5.predict([X_val], batch_size=1024, verbose=1)
thresholds = []
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    res = metrics.f1_score(y_val, (pred_val_y_5 > thresh).astype(int))
    thresholds.append([thresh, res])
    print("F1 score at threshold {0} is {1}".format(thresh, res))
    
thresholds.sort(key=lambda x: x[1], reverse=True)
best_thresh_5 = thresholds[0][0]
print("Best threshold: ", best_thresh_5)

y_pred_5 = MODEL5.predict(x_test, batch_size=1024, verbose=True)


# # Concat Result & Best Threshold

# In[ ]:


pred_val_y = (3*pred_val_y_2 + 4*pred_val_y_4 + 3*pred_val_y_5)/10

thresholds = []
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    res = metrics.f1_score(y_val, (pred_val_y > thresh).astype(int))
    thresholds.append([thresh, res])
    print("F1 score at threshold {0} is {1}".format(thresh, res))
    
thresholds.sort(key=lambda x: x[1], reverse=True)
best_thresh = thresholds[0][0]
print("Best threshold: ", best_thresh)


# # Submission File

# In[ ]:


y_pred = (3*y_pred_2 + 4*y_pred_4 + 3*y_pred_5)/10
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





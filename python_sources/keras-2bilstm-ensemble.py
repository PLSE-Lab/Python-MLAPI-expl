#!/usr/bin/env python
# coding: utf-8

# Hello fellow kagglers. Here is my take on Quora Insincere Questions Classification. Unfortunately I've failed to submit this kernel to the second round :/  It scored 6,88 in the first stage(version 1). 
# 
# There is nothing extravagant about my architecture, although I've spend some time tunning it. Enjoy.

# All the standard preprocesing and preparing the embeding matrix are in the first block. 
# 
# These parameters are important to note
# 
#     embed_size = 300              number of dimensions of a embeding vector
# 
#     max_features = 100000     number of unique words to use
# 
#     maxlen = 60                       maximum number of words in a sample
#  

# In[ ]:


import os
import time
import numpy as np 
import pandas as pd 
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, CuDNNLSTM, Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Train shape : ",train_df.shape)
print("Test shape : ",test_df.shape)




embed_size = 300
max_features = 100000 
maxlen = 60 

train_X = train_df["question_text"].fillna("_na_").values
test_X = test_df["question_text"].fillna("_na_").values


tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
test_X = tokenizer.texts_to_sequences(test_X)


train_X = pad_sequences(train_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)


train_y = train_df['target'].values


EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
f = open(EMBEDDING_FILE, encoding="utf8", errors='ignore')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in f)
f.close()

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


del train_df, test_df, tokenizer, embeddings_index, all_embs,word_index
import gc; gc.collect()
time.sleep(10)


# Inital learning rate is set to 0,002.  ReduceLROnPlateau calback patience parametar is set to 1.

# In[ ]:


from keras import regularizers 
from keras.layers import BatchNormalization,Activation
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import keras

adam = keras.optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=1, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0001)
callbacks = [EarlyStopping(monitor='val_loss', patience=5),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True), learning_rate_reduction]


# Chain of two Bidirectional LSTM layers are used with outputs having a 128 number of dimensions(there is no reason for being a power of  2). I've tried a number of combinations with different number of dimensions and layers, this one seems most performant. Also, it took a more than a few tries to tune the dropout rate, model has a tendency to overfit easily.

# In[ ]:


def get_model():
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Dropout(0.3)(x)
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation="elu")(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation="elu")(x)
    x = Dropout(0.3)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    return model


# Here, number of folds is set to 10, although two folds are only used. I've wanted to have more training samples per fold. 

# In[ ]:


from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=10, shuffle=True)
skf.get_n_splits(train_X, train_y)


# In[ ]:


from sklearn.metrics import f1_score
def thresh_search(y_true, y_proba):          #jpmiller
    best_thresh = 0
    best_score = 0
    for thresh in np.arange(0, 1, 0.01):
        score = f1_score(y_true, y_proba > thresh)
        if score > best_score:
            best_thresh = thresh
            best_score = score
    return best_thresh, best_score


# Two models are trained for 7 epochs.

# In[ ]:


from sklearn.metrics import f1_score

models = []
y_pred = []
pred_val_y = []
tresh_f1 = []
for i,(train_index, test_index) in enumerate(skf.split(train_X, train_y)):
    if 1<i:
        kws=1
    else:
        X_train, X_val = train_X[train_index], train_X[test_index]
        y_train, y_val = train_y[train_index], train_y[test_index]
        models.append(get_model())
        models[i].compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        history = models[i].fit(X_train, y_train, batch_size=512, callbacks = callbacks, epochs=7, validation_data=(X_val, y_val))
        y_pred.append(models[i].predict(test_X, batch_size=1024, verbose = True))
        pred_val_y.append(models[i].predict([X_val], batch_size=1024, verbose=1))
        tresh_f1.append(thresh_search(y_val, pred_val_y[i]))


# In[ ]:


for i,j in tresh_f1:
    print('{}\n'.format((i,j)))


# In[ ]:


tresh_final = 0.5*(tresh_f1[0][0]+tresh_f1[1][0])
tresh_final


# Finaly, the ensemble of the two trained models is created.

# In[ ]:


pred_test = 0.5*( y_pred[0]+y_pred[1])


# In[ ]:


#best_threshold = thresh_search(val_y, pred_val_y)[0]

submission = pd.read_csv('../input/sample_submission.csv')
pred_test = (pred_test > tresh_final).astype(int)
submission['prediction'] = pred_test
submission.to_csv('submission.csv', index=False)


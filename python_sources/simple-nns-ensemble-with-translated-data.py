#!/usr/bin/env python
# coding: utf-8

# # About
# This notebook uses simple NN models:
# * CNN
# * BiLSTM
# * MLP
# 
# The final prediction is build on weighted average of the model's predictions.
# The notebook doesn't use any pretrained models.

# # References
# 
# * [Jigsaw Multilingual Toxicity : EDA + Models](https://www.kaggle.com/tarunpaparaju/jigsaw-multilingual-toxicity-eda-models)
# * [WordCNNwithoutPreTrainEmbed](https://www.kaggle.com/davelo12/wordcnnwithoutpretrainembed)
# * [Translated test data](https://www.kaggle.com/bamps53/test-en-df)
# * [Translated val data](https://www.kaggle.com/bamps53/val-en-df)

# In[ ]:


import warnings

def warn(*args, **kwargs):
    pass

warnings.warn = warn

import os
import numpy as np
import pandas as pd
import gc

import keras
from keras import *
from keras.preprocessing import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Embedding, Conv1D, GlobalMaxPooling1D,Dropout, LSTM, MaxPooling1D, SpatialDropout1D, Bidirectional, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
import transformers
from tokenizers import BertWordPieceTokenizer


# # Utils

# In[ ]:


# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


# In[ ]:


class RocCallback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
    
    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\r\n ROC_AUC_VAL: {0}\n'.format(roc_val))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
    

def get_auc(valid_features, valid_labels):
    y_pred = pd.DataFrame(model.predict(valid_features))   
    auc = roc_auc_score(valid_labels, y_pred)
    print('AUC: %.3f' % auc)


# In[ ]:


train_1 = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
train_2 = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")
train_2.toxic = train_2.toxic.round().astype(int)

train = pd.concat([
    train_1[['comment_text', 'toxic']],
    train_2[['comment_text', 'toxic']]
])


# In[ ]:


X_train = train.comment_text
y_train = train.toxic


# In[ ]:


valid = pd.read_csv('../input/val-en-df/validation_en.csv')
X_val = valid.comment_text_en
y_val = valid.toxic


# In[ ]:


test = pd.read_csv('../input/test-en-df/test_en.csv')
X_test = test.content_en


# # Features

# In[ ]:


VOCABULARY_SIZE = 8000

tokenizer = Tokenizer(num_words=VOCABULARY_SIZE)
tokenizer.fit_on_texts(X_train.append(X_test).append(X_val))

length_train = [len(item.split(" ")) for item in X_train]
length_test = [len(item.split(" ")) for item in X_test]
length_val = [len(item.split(" ")) for item in X_val]

print(np.percentile(length_train+length_test+length_val, 95))

sns.set()
plt.hist(length_train, label='Train', range=(0,100), density=True)
plt.hist(length_test, label='Test', range=(0,100), density=True)
plt.hist(length_val, label='Val', range=(0,100), density=True)
plt.legend()
plt.show()

MAX_SENTENCE_LENGTH = int(np.percentile(length_train+length_test+length_val, 95))

train_features = tokenizer.texts_to_sequences(X_train)
train_features = pad_sequences(train_features, MAX_SENTENCE_LENGTH, padding='post')

valid_features = tokenizer.texts_to_sequences(X_val)
valid_features = pad_sequences(valid_features, MAX_SENTENCE_LENGTH, padding='post')

test_features = tokenizer.texts_to_sequences(X_test)
test_features = pad_sequences(test_features, MAX_SENTENCE_LENGTH, padding='post')


# In[ ]:


del X_train, X_val, X_test, train_1, train_2, test, valid
gc.collect()


# In[ ]:


roc = RocCallback(training_data=(train_features, y_train),
                  validation_data=(valid_features, y_val))


# In[ ]:


def build_cnn(transformer=None, max_len=512):
    EMBEDDINGS_DIM = 50
    FILTERS_SIZE = 64
    KERNEL_SIZE = 3
    HIDDEN_DIMS = 64
    LEARNING_RATE = 0.0001
    
    input_word_ids = Input(shape=(MAX_SENTENCE_LENGTH,), dtype=tf.int32, name="input_word_ids")
    embedding = Embedding(input_dim=VOCABULARY_SIZE + 1,
                            output_dim=EMBEDDINGS_DIM,
                            input_length=MAX_SENTENCE_LENGTH)(input_word_ids)

    conv_1 = Conv1D(FILTERS_SIZE, KERNEL_SIZE, activation='relu')(embedding)
    conv_1 = Dropout(0.5)(conv_1)
    max_pool = GlobalMaxPooling1D()(conv_1)
    max_pool = Dropout(0.5)(max_pool)
    dense_1 = Dense(HIDDEN_DIMS, activation='relu')(max_pool)
    output = Dense(1, activation='sigmoid')(dense_1)
    
    optimizer = optimizers.Adam(lr=LEARNING_RATE)
    model = Model(inputs=input_word_ids, outputs=output)
    model.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


# # CNN

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nmodel = build_cnn()\nmodel.summary()')


# In[ ]:


EPOCHS = 3

train_history = model.fit(
    train_features, y_train,
    validation_data = (valid_features, y_val),
    epochs=EPOCHS,
    batch_size=256,
    callbacks=[roc]
 )

train_history_2 = model.fit(
    valid_features, y_val,
    epochs=EPOCHS,
    batch_size=256,
    callbacks=[roc]
 )


# In[ ]:


get_auc(valid_features, y_val)


# In[ ]:


gc.collect()


# In[ ]:


cnn_pred = model.predict(test_features, verbose=1)


# # BiLSTM

# In[ ]:


def build_lstm(transformer=None, max_len=512):
    EMBEDDINGS_DIM = 50
    RNN_SIZE = 256
    HIDDEN_DIMS = 64
    LEARNING_RATE = 0.0001
    
    input_word_ids = Input(shape=(MAX_SENTENCE_LENGTH,), dtype=tf.int32, name="input_word_ids")
    embedding = Embedding(input_dim=VOCABULARY_SIZE + 1,
                            output_dim=EMBEDDINGS_DIM,
                            input_length=MAX_SENTENCE_LENGTH)(input_word_ids)

    lstm = Bidirectional(LSTM(RNN_SIZE))(embedding)
    dense_1 = Dense(HIDDEN_DIMS, activation='relu')(lstm)
    dense_1 = Dropout(0.5)(dense_1)
    output = Dense(1, activation='sigmoid')(dense_1)
    
    optimizer = optimizers.Adam(lr=LEARNING_RATE)
    model = Model(inputs=input_word_ids, outputs=output)
    model.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])
    
    return model


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nmodel = build_lstm()\nmodel.summary()')


# In[ ]:


EPOCHS = 2

train_history = model.fit(
    train_features, y_train,
    validation_data = (valid_features, y_val),
    epochs=EPOCHS,
    batch_size=256,
    callbacks=[roc]
 )

train_history_2 = model.fit(
    valid_features, y_val,
    epochs=EPOCHS,
    batch_size=256,
    callbacks=[roc]
 )


# In[ ]:


get_auc(valid_features, y_val)


# In[ ]:


gc.collect()


# In[ ]:


lstm_pred = model.predict(test_features, verbose=1)


# # MLP

# In[ ]:


def build_mlp(transformer=None, max_len=512):
    EMBEDDINGS_DIM = 50
    HIDDEN_DIMS = 128
    LEARNING_RATE = 0.0001
    
    input_word_ids = Input(shape=(MAX_SENTENCE_LENGTH,), dtype=tf.int32, name="input_word_ids")
    embedding = Embedding(input_dim=VOCABULARY_SIZE + 1,
                            output_dim=EMBEDDINGS_DIM,
                            input_length=MAX_SENTENCE_LENGTH)(input_word_ids)
    
    embedding = K.sum(embedding, axis=2)
    dense_1 = Dense(HIDDEN_DIMS, activation='relu')(embedding)
    dense_1 = Dropout(0.5)(dense_1)
    output = Dense(1, activation='sigmoid')(dense_1)
    
    optimizer = optimizers.Adam(lr=LEARNING_RATE)
    model = Model(inputs=input_word_ids, outputs=output)
    model.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nmodel = build_mlp()\nmodel.summary()')


# In[ ]:


EPOCHS = 2

train_history = model.fit(
    train_features, y_train,
    validation_data = (valid_features, y_val),
    epochs=EPOCHS,
    batch_size=256,
    callbacks=[roc]
 )
train_history_2 = model.fit(
    valid_features, y_val,
    epochs=EPOCHS,
    batch_size=256,
    callbacks=[roc]
 )


# In[ ]:


get_auc(valid_features, y_val)


# In[ ]:


gc.collect()


# In[ ]:


mlp_pred = model.predict(test_features, verbose=1)


# In[ ]:


sample = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv")
sample.toxic = mlp_pred*0.3 + cnn_pred*0.4 + lstm_pred*0.3


# In[ ]:


sample.to_csv("submission.csv", index=False)


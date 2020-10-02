#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import re
import gc
import pickle  
import random

import lightgbm as lgbm


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


get_ipython().system('pip install ../input/sacremoses/sacremoses-master/ > /dev/null')
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import tensorflow_hub as hub
import tensorflow as tf
import bert_tokenization as tokenization
import tensorflow.keras.backend as K
import gc
import os
from scipy.stats import spearmanr
from math import floor, ceil

import sys
sys.path.insert(0, "../input/transformers/transformers-master/")
import transformers
import os

import glob
import torch


# In[ ]:


import numpy as np
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import gensim
from nltk.corpus import brown
import random
from sklearn.model_selection import KFold
import lightgbm as lgb
import gc
from keras.callbacks.callbacks import EarlyStopping
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks.callbacks import EarlyStopping
from scipy.stats import spearmanr
from nltk.corpus import wordnet as wn
import tqdm
from sklearn.model_selection import StratifiedKFold


# In[ ]:


np.set_printoptions(suppress=True)
tokenizer = transformers.DistilBertTokenizer.from_pretrained("../input/distilbertbaseuncased/")
model = transformers.DistilBertModel.from_pretrained("../input/distilbertbaseuncased/")


# In[ ]:


model.cuda()


# In[ ]:


PATH = '../input/google-quest-challenge/'
BERT_PATH = '../input/bert-base-from-tfhub/bert_en_uncased_L-12_H-768_A-12'
tokenizer = transformers.DistilBertTokenizer.from_pretrained(BERT_PATH+'/assets/vocab.txt')
MAX_SEQUENCE_LENGTH = 512

train = df_train = pd.read_csv(PATH+'train.csv')
test = df_test = pd.read_csv(PATH+'test.csv')
df_sub = pd.read_csv(PATH+'sample_submission.csv')
print('train shape =', df_train.shape)
print('test shape =', df_test.shape)

output_categories = list(df_train.columns[11:])
input_categories = list(df_train.columns[[1,2,5]])
print('\noutput categories:\n\t', output_categories)
print('\ninput categories:\n\t', input_categories)


# In[ ]:


import tensorflow_hub as hub
import keras.backend as K

from keras.models import Model
from keras.layers import Dense, Input, Dropout, Lambda
from keras.optimizers import Adam
from keras.callbacks import Callback
from scipy.stats import spearmanr, rankdata
from os.path import join as path_join
from numpy.random import seed
from urllib.parse import urlparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import MultiTaskElasticNet
import lightgbm as lgb
from sklearn import metrics

seed(42)
tf.random.set_seed(42)
random.seed(42)


# In[ ]:


data_dir = '../input/google-quest-challenge/'


# In[ ]:


targets = [
        'question_asker_intent_understanding',
        'question_body_critical',
        'question_conversational',
        'question_expect_short_answer',
        'question_fact_seeking',
        'question_has_commonly_accepted_answer',
        'question_interestingness_others',
        'question_interestingness_self',
        'question_multi_intent',
        'question_not_really_a_question',
        'question_opinion_seeking',
        'question_type_choice',
        'question_type_compare',
        'question_type_consequence',
        'question_type_definition',
        'question_type_entity',
        'question_type_instructions',
        'question_type_procedure',
        'question_type_reason_explanation',
        'question_type_spelling',
        'question_well_written',
        'answer_helpful',
        'answer_level_of_information',
        'answer_plausible',
        'answer_relevance',
        'answer_satisfaction',
        'answer_type_instructions',
        'answer_type_procedure',
        'answer_type_reason_explanation',
        'answer_well_written'    
    ]

input_columns = ['question_title', 'question_body', 'answer']


# In[ ]:


def simple_prepro(s):
    return [w for w in s.replace("\n"," ").replace(","," , ").replace("("," ( ").replace(")"," ) ").
            replace("."," . ").replace("?"," ? ").replace(":"," : ").replace("n't"," not").
            replace("'ve"," have").replace("'re"," are").replace("'s"," is").split(" ") if w != ""]
def simple_prepro_tfidf(s):
    return " ".join([w for w in s.lower().replace("\n"," ").replace(","," , ").replace("("," ( ").replace(")"," ) ").
            replace("."," . ").replace("?"," ? ").replace(":"," : ").replace("n't"," not").
            replace("'ve"," have").replace("'re"," are").replace("'s"," is").split(" ") if w != ""])
#This is basic preprocessing. This time, symbols and words are attached, so they are separated here.

qt_max = max([len(simple_prepro(l)) for l in list(train["question_title"].values)])
qb_max = max([len(simple_prepro(l))  for l in list(train["question_body"].values)])
an_max = max([len(simple_prepro(l))  for l in list(train["answer"].values)])
print("max lenght of question_title is",qt_max)
print("max lenght of question_body is",qb_max)
print("max lenght of question_answer is",an_max)


# In[ ]:


gc.collect()
tfidf = TfidfVectorizer(ngram_range=(1, 3))
tsvd = TruncatedSVD(n_components = 60)
tfidf_question_title = tfidf.fit_transform([simple_prepro_tfidf(l) for l in tqdm.tqdm(train["question_title"].values)])
tfidf_question_title_test = tfidf.transform([simple_prepro_tfidf(l) for l in tqdm.tqdm(test["question_title"].values)])
tfidf_question_title = tsvd.fit_transform(tfidf_question_title)
tfidf_question_title_test = tsvd.transform(tfidf_question_title_test)

tfidf_question_body = tfidf.fit_transform([simple_prepro_tfidf(l) for l in tqdm.tqdm(train["question_body"].values)])
tfidf_question_body_test = tfidf.transform([simple_prepro_tfidf(l) for l in tqdm.tqdm(test["question_body"].values)])
tfidf_question_body = tsvd.fit_transform(tfidf_question_body)
tfidf_question_body_test = tsvd.transform(tfidf_question_body_test)

tfidf_answer = tfidf.fit_transform([simple_prepro_tfidf(l) for l in tqdm.tqdm(train["answer"].values)])
tfidf_answer_test = tfidf.transform([simple_prepro_tfidf(l) for l in tqdm.tqdm(test["answer"].values)])
tfidf_answer = tsvd.fit_transform(tfidf_answer)
tfidf_answer_test = tsvd.transform(tfidf_answer_test)
type2int = {type:i for i,type in enumerate(list(set(train["category"])))}
cate = np.identity(5)[np.array(train["category"].apply(lambda x:type2int[x]))].astype(np.float64)
cate_test = np.identity(5)[np.array(test["category"].apply(lambda x:type2int[x]))].astype(np.float64)


# In[ ]:


w2v_model = gensim.models.Word2Vec(brown.sents())


# In[ ]:


def get_word_embeddings(text):
    np.random.seed(abs(hash(text)) % (10 ** 8))
    words = simple_prepro(text)
    vectors = np.zeros((len(words),100))
    if len(words)==0:
        vectors = np.zeros((1,100))
    for i,word in enumerate(simple_prepro(text)):
        try:
            vectors[i]=w2v_model[word]
        except:
            vectors[i]=np.random.uniform(-0.01, 0.01,100)
    return np.concatenate([np.max(np.array(vectors), axis=0),
                          np.array([min(len(text),5000)/5000,
                                    min(text.count(" "),5000)/5000,
                                    min(len(words),1000)/1000,
                                    min(text.count("\n"),100)/100,
                                   min(text.count("!"),20)/20,
                                   min(text.count("?"),20)/20])])
                           
question_title = [get_word_embeddings(l) for l in tqdm.tqdm(train["question_title"].values)]
question_title_test = [get_word_embeddings(l) for l in tqdm.tqdm(test["question_title"].values)]

question_body = [get_word_embeddings(l) for l in tqdm.tqdm(train["question_body"].values)]
question_body_test = [get_word_embeddings(l) for l in tqdm.tqdm(test["question_body"].values)]

answer = [get_word_embeddings(l) for l in tqdm.tqdm(train["answer"].values)]
answer_test = [get_word_embeddings(l) for l in tqdm.tqdm(test["answer"].values)]
#From here on, I'm quite referring to https://www.kaggle.com/ryches/tfidf-benchmark.

gc.collect()
tfidf = TfidfVectorizer(ngram_range=(1, 3))
tsvd = TruncatedSVD(n_components = 60)
tfidf_question_title = tfidf.fit_transform([simple_prepro_tfidf(l) for l in tqdm.tqdm(train["question_title"].values)])
tfidf_question_title_test = tfidf.transform([simple_prepro_tfidf(l) for l in tqdm.tqdm(test["question_title"].values)])
tfidf_question_title = tsvd.fit_transform(tfidf_question_title)
tfidf_question_title_test = tsvd.transform(tfidf_question_title_test)

tfidf_question_body = tfidf.fit_transform([simple_prepro_tfidf(l) for l in tqdm.tqdm(train["question_body"].values)])
tfidf_question_body_test = tfidf.transform([simple_prepro_tfidf(l) for l in tqdm.tqdm(test["question_body"].values)])
tfidf_question_body = tsvd.fit_transform(tfidf_question_body)
tfidf_question_body_test = tsvd.transform(tfidf_question_body_test)

tfidf_answer = tfidf.fit_transform([simple_prepro_tfidf(l) for l in tqdm.tqdm(train["answer"].values)])
tfidf_answer_test = tfidf.transform([simple_prepro_tfidf(l) for l in tqdm.tqdm(test["answer"].values)])
tfidf_answer = tsvd.fit_transform(tfidf_answer)
tfidf_answer_test = tsvd.transform(tfidf_answer_test)
type2int = {type:i for i,type in enumerate(list(set(train["category"])))}
cate = np.identity(5)[np.array(train["category"].apply(lambda x:type2int[x]))].astype(np.float64)
cate_test = np.identity(5)[np.array(test["category"].apply(lambda x:type2int[x]))].astype(np.float64)
train_features = np.concatenate([question_title, question_body, answer,
                                 tfidf_question_title, tfidf_question_body, tfidf_answer, 
                                 cate
                                ], axis=1)
test_features = np.concatenate([question_title_test, question_body_test, answer_test, 
                               tfidf_question_title_test, tfidf_question_body_test, tfidf_answer_test,
                                cate_test
                                ], axis=1)


# In[ ]:


def _get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))

def _get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    first_sep = True
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            if first_sep:
                first_sep = False 
            else:
                current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))

def _get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids

def _trim_input(title, question, answer, max_sequence_length, 
                t_max_len=30, q_max_len=239, a_max_len=239):

    t = tokenizer.tokenize(title)
    q = tokenizer.tokenize(question)
    a = tokenizer.tokenize(answer)
    
    t_len = len(t)
    q_len = len(q)
    a_len = len(a)

    if (t_len+q_len+a_len+4) > max_sequence_length:
        
        if t_max_len > t_len:
            t_new_len = t_len
            a_max_len = a_max_len + floor((t_max_len - t_len)/2)
            q_max_len = q_max_len + ceil((t_max_len - t_len)/2)
        else:
            t_new_len = t_max_len
      
        if a_max_len > a_len:
            a_new_len = a_len 
            q_new_len = q_max_len + (a_max_len - a_len)
        elif q_max_len > q_len:
            a_new_len = a_max_len + (q_max_len - q_len)
            q_new_len = q_len
        else:
            a_new_len = a_max_len
            q_new_len = q_max_len
            
            
        if t_new_len+a_new_len+q_new_len+4 != max_sequence_length:
            raise ValueError("New sequence length should be %d, but is %d" 
                             % (max_sequence_length, (t_new_len+a_new_len+q_new_len+4)))
        
        t = t[:t_new_len]
        q = q[:q_new_len]
        a = a[:a_new_len]
    
    return t, q, a

def _convert_to_bert_inputs(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for BERT"""
    
    stoken = ["[CLS]"] + title + ["[SEP]"] + question + ["[SEP]"] + answer + ["[SEP]"]

    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
    input_masks = _get_masks(stoken, max_sequence_length)
    input_segments = _get_segments(stoken, max_sequence_length)

    return [input_ids, input_masks, input_segments]

def compute_input_arays(df, columns, tokenizer, max_sequence_length):
    input_ids, input_masks, input_segments = [], [], []
    for _, instance in tqdm.tqdm(df[columns].iterrows()):
        t, q, a = instance.question_title, instance.question_body, instance.answer

        t, q, a = _trim_input(t, q, a, max_sequence_length)

        ids, masks, segments = _convert_to_bert_inputs(t, q, a, tokenizer, max_sequence_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
        
    return [np.asarray(input_ids, dtype=np.int32), 
            np.asarray(input_masks, dtype=np.int32), 
            np.asarray(input_segments, dtype=np.int32)]


def compute_output_arrays(df, columns):
    return np.asarray(df[columns])


# In[ ]:


def compute_spearmanr(trues, preds):
    rhos = []
    for col_trues, col_pred in zip(trues.T, preds.T):
        rhos.append(
            spearmanr(col_trues, col_pred + np.random.normal(0, 1e-7, col_pred.shape[0])).correlation)
    return np.mean(rhos)


class CustomCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, valid_data, test_data, batch_size=16, fold=None):

        self.valid_inputs = valid_data[0]
        self.valid_outputs = valid_data[1]
        self.test_inputs = test_data
        
        self.batch_size = batch_size
        self.fold = fold
        
    def on_train_begin(self, logs={}):
        self.valid_predictions = []
        self.test_predictions = []
        
    def on_epoch_end(self, epoch, logs={}):
        self.valid_predictions.append(
            self.model.predict(self.valid_inputs, batch_size=self.batch_size))
        
        rho_val = compute_spearmanr(
            self.valid_outputs, np.average(self.valid_predictions, axis=0))
        
        print("\nvalidation rho: %.4f" % rho_val)
        
        if self.fold is not None:
            self.model.save_weights(f'bert-base-{fold}-{epoch}.h5py')
        
        self.test_predictions.append(
            self.model.predict(self.test_inputs, batch_size=self.batch_size)
        )

def bert_model():
    
    input_word_ids = tf.keras.layers.Input(
        (MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_word_ids')
    input_masks = tf.keras.layers.Input(
        (MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_masks')
    input_segments = tf.keras.layers.Input(
        (MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_segments')
    
    bert_layer = hub.KerasLayer(BERT_PATH, trainable=True)
    
    _, sequence_output = bert_layer([input_word_ids, input_masks, input_segments])
    
    x = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(30, activation="sigmoid", name="dense_output")(x)

    model = tf.keras.models.Model(
        inputs=[input_word_ids, input_masks, input_segments], outputs=out)
    
    return model    
        
def train_and_predict(model, train_data, valid_data, test_data, 
                      learning_rate, epochs, batch_size, loss_function, fold):
        
    custom_callback = CustomCallback(
        valid_data=(valid_data[0], valid_data[1]), 
        test_data=test_data,
        batch_size=batch_size,
        fold=None)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss_function, optimizer=optimizer)
    model.fit(train_data[0], train_data[1], epochs=epochs, 
              batch_size=batch_size, callbacks=[custom_callback])
    
    return custom_callback
gkf = GroupKFold(n_splits=10).split(X=df_train.question_body, groups=df_train.question_body) ############## originaln_splits=5

outputs = compute_output_arrays(df_train, output_categories)
inputs = compute_input_arays(df_train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
test_inputs = compute_input_arays(df_test, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)


# In[ ]:


length =(np.linspace(1, 229, num=1))


# In[ ]:


import math
import fastai
from fastai.train import Learner
from fastai.train import DataBunch
from fastai.callbacks import *
from fastai.basic_data import DatasetType
import fastprogress
from fastprogress import force_console_behavior
import numpy as np
from pprint import pprint
import pandas as pd
import os
import time
import gc
import random
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
from keras.preprocessing import text, sequence
import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import seaborn as sns

class TextDataset(data.Dataset):
    def __init__(self, text, lens, y=None):
        self.text = text
        self.lens = lens
        self.y = y
    def __len__(self):
        return len(self.lens)
    def __getitem__(self, idx):
        if self.y is None:
            return self.text[idx], self.lens[idx]
        return self.text[idx], self.lens[idx], self.y[idx]
    
class Collator(object):
    def __init__(self, test=False, percentile=100):
        self.test = test
        self.percentile = percentile
        
    def __call__(self, batch):
        global MAX_LEN
        
        if self.test:
            texts, lens = zip(*batch)
        else:
            texts, lens, target = zip(*batch)
        lens = np.array(lens)
        max_len = min(int(np.percentile(lens, self.percentile)), MAX_LEN)
        texts = torch.tensor(sequence.pad_sequences(texts, maxlen=max_len), dtype=torch.long)
        
        if self.test:
            return texts
        
        return texts, torch.tensor(target, dtype=torch.float32)
train_collate = Collator(percentile=100)
train_dataset = TextDataset(train, length, test)
train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_collate)
n_repeats = 10
start_time = time.time()
for _ in range(n_repeats):
    for i in tqdm(lengths):
        pass
method1_time = (time.time() - start_time) / n_repeats


# In[ ]:


train_collator = SequenceBucketCollator(lambda lengths: lengths.max(), 
                                        sequence_index=0, 
                                        length_index=1, 
                                        label_index=2)
test_collator = SequenceBucketCollator(lambda lengths: lengths.max(), sequence_index=0, length_index=1)

valid_dataset = data.Subset(train, indices=[0, 1])
train_loader = data.DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=train_collator)
valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=train_collator)
test_loader = data.DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=test_collator)


# In[ ]:


def compute_spearmanr(trues, preds):
    rhos = []
    for col_trues, col_pred in zip(trues.T, preds.T):
        rhos.append(
            spearmanr(col_trues, col_pred + np.random.normal(0, 1e-7, col_pred.shape[0])).correlation)
    return np.mean(rhos)


class CustomCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, valid_data, test_data, batch_size=16, fold=None):

        self.valid_inputs = valid_data[0]
        self.valid_outputs = valid_data[1]
        self.test_inputs = test_data
        
        self.batch_size = batch_size
        self.fold = fold
        
    def on_train_begin(self, logs={}):
        self.valid_predictions = []
        self.test_predictions = []
        
    def on_epoch_end(self, epoch, logs={}):
        self.valid_predictions.append(
            self.model.predict(self.valid_inputs, batch_size=self.batch_size))
        
        rho_val = compute_spearmanr(
            self.valid_outputs, np.average(self.valid_predictions, axis=0))
        
        print("\nvalidation rho: %.4f" % rho_val)
        
        if self.fold is not None:
            self.model.save_weights(f'bert-base-{fold}-{epoch}.h5py')
        
        self.test_predictions.append(
            self.model.predict(self.test_inputs, batch_size=self.batch_size)
        )

def bert_model():
    
    input_word_ids = tf.keras.layers.Input(
        (MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_word_ids')
    input_masks = tf.keras.layers.Input(
        (MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_masks')
    input_segments = tf.keras.layers.Input(
        (MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_segments')
    
    bert_layer = hub.KerasLayer(BERT_PATH, trainable=True)
    
    _, sequence_output = bert_layer([input_word_ids, input_masks, input_segments])
    
    x = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
    x = tf.keras.layers.Dropout(0.2)(x)
    y = tf.keras.layers.BatchNormalization(epsilon=1e-06, momentum=0.9, weights=None)
    out = tf.keras.layers.Dense(30, activation="sigmoid", name="dense_output")(x)

    model = tf.keras.models.Model(
        inputs=[input_word_ids, input_masks, input_segments], outputs=out)
    
    return model    
        
def train_and_predict(model, train_data, valid_data, test_data, 
                      learning_rate, epochs, batch_size, loss_function, fold):
        
    custom_callback = CustomCallback(
        valid_data=(valid_data[0], valid_data[1]), 
        test_data=test_data,
        batch_size=batch_size,
        fold=None)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(loss=loss_function, optimizer=optimizer)
    model.fit(train_data[0], train_data[1], epochs=epochs, 
              batch_size=batch_size, callbacks=[custom_callback])
    
    return custom_callback

histories = []
for fold, (train_idx, valid_idx) in enumerate(gkf):

    
    # will actually only do 3 folds (out of 5) to manage < 2h
    if fold < 3:
        K.clear_session()
        model = bert_model()

        train_inputs = [inputs[i][train_idx] for i in range(3)]
        train_outputs = outputs[train_idx]

        valid_inputs = [inputs[i][valid_idx] for i in range(3)]
        valid_outputs = outputs[valid_idx]

        # history contains two lists of valid and test preds respectively:
        #  [valid_predictions_{fold}, test_predictions_{fold}]
        history = train_and_predict(model, 
                          train_data=(train_inputs, train_outputs), 
                          valid_data=(valid_inputs, valid_outputs),
                          test_data=test_inputs, 
                          learning_rate=3e-5, epochs=5, batch_size=8,
                          loss_function='binary_crossentropy', fold=fold)

        histories.append(history)


# In[ ]:


test_predictions = [histories[i].test_predictions for i in range(len(histories))]
test_predictions = [np.average(test_predictions[i], axis=0) for i in range(len(test_predictions))]
test_predictions = np.mean(test_predictions, axis=0)

df_sub.iloc[:, 1:] = test_predictions

df_sub.to_csv('submission.csv', index=False)


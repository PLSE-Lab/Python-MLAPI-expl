#!/usr/bin/env python
# coding: utf-8

# ### Bert-base TensorFlow 2.0
# 
# This kernel does not explore the data. For that you could check out some of the great EDA kernels: [introduction](https://www.kaggle.com/corochann/google-quest-first-data-introduction), [getting started](https://www.kaggle.com/phoenix9032/get-started-with-your-questions-eda-model-nn) & [another getting started](https://www.kaggle.com/hamditarek/get-started-with-nlp-lda-lsa). This kernel is an example of a TensorFlow 2.0 Bert-base implementation, using ~~TensorFow Hub~~ Huggingface transformer. <br><br>
# 
# ---
# **Update 1 (Commit 7):**
# * removing penultimate dense layer; now there's only one dense layer (output layer) for fine-tuning
# * using BERT's sequence_output instead of pooled_output as input for the dense layer
# ---
# 
# **Update 2 (Commit 8):**
# * adjusting `_trim_input()` --- now have a q_max_len and a_max_len, instead of 'keeping the ratio the same' while trimming.
# * **importantly:** now also includes question_title for the input sequence
# ---
# 
# **Update 3 (Commit 9)**
# <br><br>*A lot of experiments can be made with the title + body + answer sequence. Feel free to look into e.g. (1) inventing new tokens (add it to '../input/path-to-bert-folder/assets/vocab.txt'), (2) keeping \[SEP\] between title and body but modify `_get_segments()`, (3) using the \[PAD\] token, or (4) merging title and body without any kind of separation. In this commit I'm doing (2). I also tried (3) offline, and they both perform better than in commit 8, in terms of validation rho.*<br>
# 
# * ignoring first \[SEP\] token in `_get_segments()`.
# 
# ---
# 
# **Update 4 (Commit 11)**
# * **Now using Huggingface transformer instead of TFHub** (note major changes in the code). This creates the possibility to easily try out different architectures like XLNet, Roberta etc. As well as easily outputting the hidden states of the transformer.
# * two separate inputs (title+body and answer) for BERT
# * removed snapshot average (now only using last (third) epoch). This will likely decrease performance, but it's not feasible to use ~ 5 x 4 models for a single bert prediction in practice. 
# * only training for 2 epochs instead of 3 (to manage 2h limit)
# ---
# 
# **Update 5 (Commit 12)**
# * `transformers` now available in 'Latest Available' Docker
# * reducing input size
# ---

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import tensorflow as tf
import tensorflow.keras.backend as K
import os
from scipy.stats import spearmanr
from math import floor, ceil
from transformers import *

np.set_printoptions(suppress=True)
print(tf.__version__)


# #### 1. Read data and tokenizer
# 
# Read tokenizer and data, as well as defining the maximum sequence length that will be used for the input to Bert (maximum is usually 512 tokens)

# In[ ]:


PATH = '../input/google-quest-challenge/'

BERT_PATH = '../input/bert-base-uncased-huggingface-transformer/'
tokenizer = BertTokenizer.from_pretrained(BERT_PATH+'bert-base-uncased-vocab.txt')

MAX_SEQUENCE_LENGTH = 384

df_train = pd.read_csv(PATH+'train.csv')
df_test = pd.read_csv(PATH+'test.csv')
df_sub = pd.read_csv(PATH+'sample_submission.csv')
print('train shape =', df_train.shape)
print('test shape =', df_test.shape)

output_categories = list(df_train.columns[11:])
input_categories = list(df_train.columns[[1,2,5]])
print('\noutput categories:\n\t', output_categories)
print('\ninput categories:\n\t', input_categories)


# #### 2. Preprocessing functions
# 
# These are some functions that will be used to preprocess the raw text data into useable Bert inputs.<br>
# 
# *update 4:* credits to [Minh](https://www.kaggle.com/dathudeptrai) for this implementation. If I'm not mistaken, it could be used directly with other Huggingface transformers too! Note that due to the 2 x 512 input *(update 5: 2 x 384)*, it will require significantly more memory when finetuning BERT.

# In[ ]:


def _convert_to_transformer_inputs(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""
    
    def return_id(str1, str2, truncation_strategy, length):

        inputs = tokenizer.encode_plus(str1, str2,
            add_special_tokens=True,
            max_length=length,
            truncation_strategy=truncation_strategy)
        
        input_ids =  inputs["input_ids"]
        input_masks = [1] * len(input_ids)
        input_segments = inputs["token_type_ids"]
        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)
        
        return [input_ids, input_masks, input_segments]
    
    input_ids_q, input_masks_q, input_segments_q = return_id(
        title + ' ' + question, None, 'longest_first', max_sequence_length)
    
    input_ids_a, input_masks_a, input_segments_a = return_id(
        answer, None, 'longest_first', max_sequence_length)
    
    return [input_ids_q, input_masks_q, input_segments_q,
            input_ids_a, input_masks_a, input_segments_a]

def compute_input_arrays(df, columns, tokenizer, max_sequence_length):
    input_ids_q, input_masks_q, input_segments_q = [], [], []
    input_ids_a, input_masks_a, input_segments_a = [], [], []
    for _, instance in tqdm(df[columns].iterrows()):
        t, q, a = instance.question_title, instance.question_body, instance.answer

        ids_q, masks_q, segments_q, ids_a, masks_a, segments_a =         _convert_to_transformer_inputs(t, q, a, tokenizer, max_sequence_length)
        
        input_ids_q.append(ids_q)
        input_masks_q.append(masks_q)
        input_segments_q.append(segments_q)

        input_ids_a.append(ids_a)
        input_masks_a.append(masks_a)
        input_segments_a.append(segments_a)
        
    return [np.asarray(input_ids_q, dtype=np.int32), 
            np.asarray(input_masks_q, dtype=np.int32), 
            np.asarray(input_segments_q, dtype=np.int32),
            np.asarray(input_ids_a, dtype=np.int32), 
            np.asarray(input_masks_a, dtype=np.int32), 
            np.asarray(input_segments_a, dtype=np.int32)]

def compute_output_arrays(df, columns):
    return np.asarray(df[columns])


# #### 3. Create model
# 
# `compute_spearmanr()` is used to compute the competition metric for the validation set
# <br><br>
# `create_model()` contains the actual architecture that will be used to finetune BERT to our dataset.
# 

# In[ ]:


def compute_spearmanr_ignore_nan(trues, preds):
    rhos = []
    for tcol, pcol in zip(np.transpose(trues), np.transpose(preds)):
        rhos.append(spearmanr(tcol, pcol).correlation)
    return np.nanmean(rhos)

def create_model():
    q_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    a_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    
    q_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    a_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    
    q_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    a_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    
    config = BertConfig() # print(config) to see settings
    config.output_hidden_states = False # Set to True to obtain hidden states
    # caution: when using e.g. XLNet, XLNetConfig() will automatically use xlnet-large config
    
    # normally ".from_pretrained('bert-base-uncased')", but because of no internet, the 
    # pretrained model has been downloaded manually and uploaded to kaggle. 
    bert_model = TFBertModel.from_pretrained(
        BERT_PATH+'bert-base-uncased-tf_model.h5', config=config)
    
    # if config.output_hidden_states = True, obtain hidden states via bert_model(...)[-1]
    q_embedding = bert_model(q_id, attention_mask=q_mask, token_type_ids=q_atn)[0]
    a_embedding = bert_model(a_id, attention_mask=a_mask, token_type_ids=a_atn)[0]
    
    q = tf.keras.layers.GlobalAveragePooling1D()(q_embedding)
    a = tf.keras.layers.GlobalAveragePooling1D()(a_embedding)
    
    x = tf.keras.layers.Concatenate()([q, a])
    
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Dense(30, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs=[q_id, q_mask, q_atn, a_id, a_mask, a_atn,], outputs=x)
    
    return model


# #### 4. Obtain inputs and targets, as well as the indices of the train/validation splits

# In[ ]:


outputs = compute_output_arrays(df_train, output_categories)
inputs = compute_input_arrays(df_train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
test_inputs = compute_input_arrays(df_test, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)


# #### 5. Training, validation and testing
# 
# Loops over the folds in gkf and trains each fold for 3 epochs --- with a learning rate of 3e-5 and batch_size of 6. A simple binary crossentropy is used as the objective-/loss-function. 

# In[ ]:


gkf = GroupKFold(n_splits=5).split(X=df_train.question_body, groups=df_train.question_body)

valid_preds = []
test_preds = []
for fold, (train_idx, valid_idx) in enumerate(gkf):
    
    # will actually only do 2 folds (out of 5) to manage < 2h
    if fold in [0, 2]:

        train_inputs = [inputs[i][train_idx] for i in range(len(inputs))]
        train_outputs = outputs[train_idx]

        valid_inputs = [inputs[i][valid_idx] for i in range(len(inputs))]
        valid_outputs = outputs[valid_idx]
        
        K.clear_session()
        model = create_model()
        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
        model.compile(loss='binary_crossentropy', optimizer=optimizer)
        model.fit(train_inputs, train_outputs, epochs=3, batch_size=6)
        # model.save_weights(f'bert-{fold}.h5')
        valid_preds.append(model.predict(valid_inputs))
        test_preds.append(model.predict(test_inputs))
        
        rho_val = compute_spearmanr_ignore_nan(valid_outputs, valid_preds[-1])
        print('validation score = ', rho_val)


# #### 6. Process and submit test predictions
# 
# Average fold predictions, then save as `submission.csv`

# In[ ]:


df_sub.iloc[:, 1:] = np.average(test_preds, axis=0) # for weighted average set weights=[...]

df_sub.to_csv('submission.csv', index=False)


# In[ ]:





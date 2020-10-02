#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install ../input/huggingface-transformers/sacremoses-master/sacremoses-master')
get_ipython().system('pip install ../input/huggingface-transformers/transformers-master/transformers-master')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import gc
from sklearn.model_selection import train_test_split, GroupKFold
import bert_tokenization as tokenization
from scipy.stats import spearmanr
from math import floor, ceil
from transformers import TFPreTrainedModel, TFBertMainLayer, BertConfig, TFBertModel, BertTokenizer
from tqdm.notebook import tqdm
# import pytorch_transformers

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# tf.executing_eagerly()
import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


PATH = '../input/google-quest-challenge/'
BERT_PATH = '../input/bert-base-from-tfhub/bert_en_uncased_L-12_H-768_A-12'
tokenizer = tokenization.FullTokenizer(BERT_PATH+'/assets/vocab.txt', True)
# tokenizer2 = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
MAX_SEQUENCE_LENGTH = 512

df_train = pd.read_csv(PATH+'train.csv')
df_test = pd.read_csv(PATH+'test.csv')
df_sub = pd.read_csv(PATH+'sample_submission.csv')
print('train shape =', df_train.shape)
print('test shape =', df_test.shape)


output_categories = ['question_asker_intent_understanding', 'question_body_critical', 'question_conversational', 'question_expect_short_answer', 'question_fact_seeking', 'question_has_commonly_accepted_answer', 'question_interestingness_others', 'question_interestingness_self', 'question_multi_intent', 'question_not_really_a_question', 'question_opinion_seeking', 'question_type_choice', 'question_type_compare', 'question_type_consequence', 'question_type_definition', 'question_type_entity', 'question_type_instructions', 'question_type_procedure', 'question_type_reason_explanation', 'question_type_spelling', 'question_well_written', 'answer_helpful', 'answer_level_of_information', 'answer_plausible', 'answer_relevance', 'answer_satisfaction', 'answer_type_instructions', 'answer_type_procedure', 'answer_type_reason_explanation', 'answer_well_written']
input_categories = ['question_title', 'question_body', 'answer']
print('\noutput categories:\n\t', output_categories)
print('\ninput categories:\n\t', input_categories)


# In[ ]:


def _get_masks(tokens, max_seq_length):
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens)+[0]*(max_seq_length-len(tokens))

def _get_segments(tokens, max_seq_length):
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    
    segments=[]
    first_sep=True
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == '[SEP]':
            if first_sep:
                first_sep=False
            else:
                current_segment_id=1
    return segments+[0]*(max_seq_length-len(tokens))

def _trim_input(title, question, answer, max_sequence_length, t_max_len=30, q_max_len=239, a_max_len=239):

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

def _get_ids(tokens, tokenizer, max_seq_length):
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0]*(max_seq_length - len(token_ids))
    return input_ids

def _convert_to_bert_inputs(title, question, answer, tokenizer, max_seq_length):
    stoken = ['[CLS]']+title+['[SEP]']+question+['[SEP]']+answer+['[SEP]']
    input_ids = _get_ids(tokens=stoken, tokenizer=tokenizer,max_seq_length=max_seq_length)
    input_masks = _get_masks(tokens=stoken, max_seq_length=max_seq_length)
    input_segments = _get_segments(tokens=stoken, max_seq_length=max_seq_length)
    
    return [input_ids, input_masks, input_segments]

def compute_input_array(df, columns, tokenizer, max_sequence_length):
    input_ids, input_masks, input_segments = [], [], []
    for _, col in tqdm(df[columns].iterrows()):
        t, q, a = col['question_title'], col['question_body'], col['answer']
        t,q,a = _trim_input(t,q,a, max_sequence_length)
        ids, masks, segments = _convert_to_bert_inputs(t,q,a, tokenizer, max_sequence_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
        
    return [np.array(input_ids, dtype=np.int32), np.array(input_masks, dtype=np.int32), 
            np.array(input_segments, dtype=np.int32)]

def compute_output_arrays(df, columns):
    return np.asarray(df[columns])


# In[ ]:


def compute_spearmanr(trues, preds):
    rhos=[]
    for t, p in zip(trues.T, preds.T):
        rhos.append(
            spearmanr(t,p).correlation
        )
    return np.nanmean(rhos)

test_predictions=[]

class CustomCallback(keras.callbacks.Callback):
    def __init__(self, valid_data, test_data, test_predictions=test_predictions, batch_size=16, fold=None):
        self.valid_inputs = valid_data[0]
        self.valid_outputs = valid_data[1]
        self.test_inputs = test_data
        self.batch_size = batch_size
        self.test_predictions = test_predictions
        self.fold = fold
        
    def on_train_begin(self, logs={}):
        self.valid_predictions=[]
        #self.test_predictions=[]
        
    def on_epoch_end(self, epoch, logs={}):
        self.valid_predictions.append(self.model.predict(self.valid_inputs, batch_size=self.batch_size))
        
        rho_val = compute_spearmanr(self.valid_outputs, np.average(self.valid_predictions, axis=0))
        print(f"\nvalidation rho: {round(rho_val,4)}")
        
        if self.fold is not None and int(epoch)==5:
            self.model.save_weights(f'bert-base-{fold}-{epoch}.hdf5')
            
        self.test_predictions.append(self.model.predict(self.test_inputs, batch_size=self.batch_size))


# In[ ]:


bert_config=BertConfig.from_pretrained('../input/bert-tensorflow/bert-base-uncased-config.json',output_hidden_states=True)
def bertModel():
    input_ids = keras.layers.Input((MAX_SEQUENCE_LENGTH), dtype = tf.int32, name = 'input_word_ids')
    input_mask = keras.layers.Input((MAX_SEQUENCE_LENGTH), dtype = tf.int32, name = 'input_masks')
    input_segments = keras.layers.Input((MAX_SEQUENCE_LENGTH), dtype = tf.int32, name = 'input_segments')
    
    bert_model = TFBertModel.from_pretrained(pretrained_model_name_or_path='../input/bert-tensorflow/bert-base-uncased-tf_model.h5',config = bert_config)

    sequence_output, pooler_output, hidden_states = bert_model([input_ids,input_mask, input_segments])
    
    h12 = tf.reshape(hidden_states[-1][:,0],(-1,1,768))
    h11 = tf.reshape(hidden_states[-2][:,0],(-1,1,768))
    h10 = tf.reshape(hidden_states[-3][:,0],(-1,1,768))
    h09 = tf.reshape(hidden_states[-4][:,0],(-1,1,768))
    concat_hidden = keras.layers.Concatenate(axis=1)([h12, h11, h10, h09])
    
    x = keras.layers.GlobalAveragePooling1D()(concat_hidden)
    x = keras.layers.Dropout(0.2)(x)
    out = keras.layers.Dense(len(output_categories), activation='sigmoid', name='final_dense_output')(x)
    
    model = keras.models.Model(inputs=[input_ids, input_mask, input_segments], outputs=out)
    model.compile(loss='binary_crossentropy', optimizer = keras.optimizers.Adam(lr=3e-5))
    return model
    


# In[ ]:


# model = bertModel()
# model.summary()


# In[ ]:


gkf = GroupKFold(n_splits=10).split(X=df_train.question_body, groups=df_train.question_body)

outputs = compute_output_arrays(df_train, output_categories)
inputs = compute_input_array(df_train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
test_inputs = compute_input_array(df_test, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)


# In[ ]:


histories = []

for fold, (train_idx, valid_idx) in enumerate(gkf):
    
    if fold<3:
        keras.backend.clear_session()
        model = bertModel()
        train_inputs = [inputs[i][train_idx] for i in range(3)]
        train_outputs = outputs[train_idx]
    
        valid_inputs = [inputs[i][valid_idx] for i in range(3)]
        valid_outputs = outputs[valid_idx]
    
        custom_callback = CustomCallback(valid_data=(valid_inputs, valid_outputs), test_data=test_inputs,
                                         batch_size=8, fold=fold)
        H = model.fit(train_inputs, train_outputs, batch_size=8, epochs=4, callbacks=[custom_callback])
        histories.append(H)
    


# In[ ]:


print(len(test_predictions))


# In[ ]:


test_preds = [test_predictions[i] for i in range(len(test_predictions))]
test_preds = [np.average(test_preds[i], axis=0) for i in range(len(test_preds))]
test_preds = np.mean(test_predictions, axis=0)

df_sub.iloc[:, 1:] = test_preds

df_sub.to_csv('submission.csv', index=False)


# In[ ]:


# del model
# gc.collect()


# In[ ]:


# zz = BertConfig.from_pretrained('../input/bert-tensorflow/bert-base-uncased-config.json',output_hidden_states=True)
# class TFBertPreTrainedModel(TFPreTrainedModel):
#     """ An abstract class to handle weights initialization and
#         a simple interface for dowloading and loading pretrained models.
#     """

#     config_class = zz
#     pretrained_model_archive_map = {"bert-base-uncased": "../input/bert-tensorflow/bert-base-uncased-tf_model.h5"}
#     base_model_prefix = "bert"


# class TFBert_v2(TFBertPreTrainedModel):
#     def __init__(self, config='../input/bert-tensorflow/bert-base-uncased-config.json', *inputs, **kwargs):
#         self.config = config
#         self.TF_BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {"bert-base-uncased": "../input/bert-tensorflow/bert-base-uncased-tf_model.h5"}
#         super(TFBert_v2, self).__init__(self.config, *inputs, **kwargs)
#         self.bert = TFBertMainLayer(config, name="bert")

#     def call(self, inputs, **kwargs):
#         outputs = self.bert(inputs, **kwargs)
#         return outputs


# In[ ]:


# bert_config=BertConfig.from_pretrained('../input/bert-tensorflow/bert-base-uncased-config.json',output_hidden_states=True)
# TFBertModel.from_pretrained(pretrained_model_name_or_path='../input/bert-tensorflow/bert-base-uncased-tf_model.h5', config = bert_config)


# In[ ]:


# tok = BertTokenizer(vocab_file=BERT_PATH+'/assets/vocab.txt')#.from_pretrained('bert-base-uncased', do_lower_case=True)
# input_ids=tf.constant(tok.encode(text="Hello, my dog is cute", add_special_tokens=True))[None, :]  # Batch size 1
# outputs = m(input_ids)


# In[ ]:





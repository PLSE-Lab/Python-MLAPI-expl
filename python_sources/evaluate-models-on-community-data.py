#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Update to transformers 2.8.0
get_ipython().system('pip install -q transformers --upgrade')
get_ipython().system('pip show transformers')
get_ipython().system('pip install -q pandas --upgrade')


# In[ ]:


import os
import pickle
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import transformers as trfm
from transformers import AutoTokenizer, TFAutoModel, TFElectraModel, ElectraTokenizer
from tqdm.notebook import tqdm
from tokenizers import BertWordPieceTokenizer


# ## Helper functions

# In[ ]:


def build_reranker(tokenizer, model):
    tokenizer.enable_padding()
    
    def rerank(question, answers):
        pairs = list(zip([question] * len(answers), answers))

        encs = tokenizer.encode_batch(pairs)
        input_ids = np.array([enc.ids for enc in encs])
        scores = model.predict(input_ids).squeeze()

        return scores
    
    return rerank


# In[ ]:


def touch_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print(f"Created directory {dirname}.")
    else:
        print(f"Directory {dirname} already exists.")


# In[ ]:


def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512, enable_padding=False):
    """
    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras
    
    ---
    Inputs:
        tokenizer: the `fast_tokenizer` that we imported from the tokenizers library
    """
    tokenizer.enable_truncation(max_length=maxlen)
    if enable_padding:
        tokenizer.enable_padding(max_length=maxlen)
    
    all_ids = []
    
    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
    
    return np.array(all_ids)


# In[ ]:


def combine_qa_ids(q_ids, a_ids, tokenizer, maxlen=512):
    """
    Given two arrays of IDs (questions and answers) created by
    `fast_encode`, we combine and pad them.
    Inputs:
        tokenizer: The original tokenizer (not the fast_tokenizer)
    """
    combined_ids = []

    for i in tqdm(range(q_ids.shape[0])):
        ids = []
        ids.append(tokenizer.cls_token_id)
        ids.extend(q_ids[i])
        ids.append(tokenizer.sep_token_id)
        ids.extend(a_ids[i])
        ids.append(tokenizer.sep_token_id)
        ids.extend([tokenizer.pad_token_id] * (maxlen - len(ids)))

        combined_ids.append(ids)
    
    return np.array(combined_ids)


# In[ ]:


def encode_qa(questions, answers, tokenizer, maxlen=512):
    """
    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras
    """
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(max_length=maxlen)
    all_ids = []
    
    for i in tqdm(range(0, len(questions))):
        q = questions[i]
        a = answers[i]
        
        encs = tokenizer.encode(q, a)
        all_ids.append(encs.ids)
        if len(encs.ids) > 512:
            return q, a
    
    return np.array(all_ids)


# In[ ]:


def build_model(transformer, max_len=None):
    """
    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras
    """
    input_ids = L.Input(shape=(max_len, ), dtype=tf.int32)
    
    x = transformer(input_ids)[0]
    x = x[:, 0, :]
    x = L.Dense(1, activation='sigmoid', name='sigmoid')(x)
    
    # BUILD AND COMPILE MODEL
    model = Model(inputs=input_ids, outputs=x)
    model.compile(
        loss='binary_crossentropy', 
        metrics=['accuracy'], 
        optimizer=Adam(lr=1e-5)
    )
    
    return model


# In[ ]:


def load_model(sigmoid_dir, transformer_dir='transformer', architecture="electra", max_len=None):
    """
    Special function to load a keras model that uses a transformer layer
    """
    sigmoid_path = os.path.join(sigmoid_dir,'sigmoid.pickle')
    
    if architecture == 'electra':
        transformer = TFElectraModel.from_pretrained(transformer_dir)
    else:
        transformer = TFAutoModel.from_pretrained(transformer_dir)
    model = build_model(transformer, max_len=max_len)
    
    sigmoid = pickle.load(open(sigmoid_path, 'rb'))
    model.get_layer('sigmoid').set_weights(sigmoid)
    
    return model


# In[ ]:


tokenizer = trfm.ElectraTokenizer.from_pretrained("google/electra-small-discriminator")
fast_tokenizer = BertWordPieceTokenizer('/kaggle/input/healthtap-joint-electra-small/vocab.txt', lowercase=True, add_special_tokens=False)


# ## Load Models

# In[ ]:


models = {}


# In[ ]:


models['electra_ht_small'] = load_model(
    sigmoid_dir='/kaggle/input/healthtap-joint-electra-small',
    transformer_dir='/kaggle/input/healthtap-joint-electra-small/transformer',
    architecture='electra',
    max_len=None
)

models['electra_ht_small'].summary()


# In[ ]:


models['electra_ht_base'] = load_model(
    sigmoid_dir='/kaggle/input/healthtap-joint-electra-base',
    transformer_dir='/kaggle/input/healthtap-joint-electra-base/transformer',
    architecture='electra',
    max_len=None
)

models['electra_ht_base'].summary()


# In[ ]:


models['electra_se_small'] = load_model(
    sigmoid_dir='/kaggle/input/stackexchange-finetune-electra-small/transformer',
    transformer_dir='/kaggle/input/stackexchange-finetune-electra-small/transformer',
    architecture='electra',
    max_len=None
)

models['electra_se_small'].summary()


# In[ ]:


models['electra_se_base'] = load_model(
    sigmoid_dir='/kaggle/input/stackexchange-finetune-electra-base/transformer',
    transformer_dir='/kaggle/input/stackexchange-finetune-electra-base/transformer',
    architecture='electra',
    max_len=None
)

models['electra_se_base'].summary()


# ## Load Data

# In[ ]:


df = pd.read_csv("/kaggle/input/covidqa/community.csv")

MAX_LEN = 512

questions = df.title + ' [SEP] ' + df.question


# In[ ]:


q_ids = fast_encode(questions.values, fast_tokenizer, maxlen=MAX_LEN//2 - 2)
a_ids = fast_encode(df.answer.values, fast_tokenizer, maxlen=MAX_LEN//2 - 2)
wa_ids = fast_encode(df.wrong_answer.values, fast_tokenizer, maxlen=MAX_LEN//2 - 2)

correct_ids = combine_qa_ids(q_ids, a_ids, tokenizer, maxlen=MAX_LEN)
wrong_ids = combine_qa_ids(q_ids, wa_ids, tokenizer, maxlen=MAX_LEN)


# In[ ]:


input_ids = np.concatenate([correct_ids, wrong_ids])

labels = np.concatenate([
    np.ones(correct_ids.shape[0]),
    np.zeros(correct_ids.shape[0])
]).astype(np.int32)


# ## Compute Scores

# In[ ]:


score_df = pd.concat([df[['url', 'source']]]*2)

for model_name, model in models.items():
    get_ipython().run_line_magic('time', 'score_df[model_name] = model.predict(input_ids, batch_size=64)')


# In[ ]:


score_df['labels'] = labels


# In[ ]:


score_df.to_csv("community.csv", index=False)


# ## Compute Prediction Results

# ### Overall

# In[ ]:


overall = {}

for model_name in models.keys():
    result = {}
    labels = score_df['labels']
    score = score_df[model_name]
    pred = score.round().astype(int)
    result['ap'] = average_precision_score(labels, score)
    result['roc_auc'] = roc_auc_score(labels, score)
    result['f1_score'] = f1_score(labels, pred)
    result['accuracy'] = accuracy_score(labels, pred)
    overall[model_name] = result

overall_df = pd.DataFrame(overall).round(4)
overall_df.to_csv("overall_results.csv")
overall_df


# In[ ]:


print(overall_df.to_latex())


# In[ ]:


print(overall_df.to_markdown())


# In[ ]:


all_sources = {}

for source in df.source.unique():
    source_results = {}
    score_source_df = score_df[score_df.source == source]

    for model_name in models.keys():
        result = {}
        labels = score_source_df['labels']
        score = score_source_df[model_name]
        pred = score.round().astype(int)
        result['ap'] = average_precision_score(labels, score)
        result['roc_auc'] = roc_auc_score(labels, score)
        result['f1_score'] = f1_score(labels, pred)
        result['accuracy'] = accuracy_score(labels, pred)
        source_results[model_name] = result
    
    all_sources[source] = pd.DataFrame(source_results).round(4)


# In[ ]:


all_sources['biomedical']


# In[ ]:


all_sources['general']


# In[ ]:


all_sources['expert']


# In[ ]:


print('biomedical')
print("-"*40)
print(all_sources['biomedical'].to_latex())
print("="*40)

print('general')
print("-"*40)
print(all_sources['general'].to_latex())
print("="*40)

print('expert')
print("-"*40)
print(all_sources['expert'].to_latex())
print("="*40)


# In[ ]:


print('biomedical')
print("-"*40)
print(all_sources['biomedical'].to_markdown())
print("="*40)

print('general')
print("-"*40)
print(all_sources['general'].to_markdown())
print("="*40)

print('expert')
print("-"*40)
print(all_sources['expert'].to_markdown())
print("="*40)


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Update to transformers 2.8.0
get_ipython().system('pip install -q transformers --upgrade')
get_ipython().system('pip show transformers')


# In[ ]:


import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score
import matplotlib.pyplot as plt
import transformers
from transformers import AutoTokenizer, TFAutoModel, TFElectraModel, ElectraTokenizer
from tqdm.notebook import tqdm
from tokenizers import BertWordPieceTokenizer


# In[ ]:


tqdm.pandas()


# ## Helper functions

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


def encode_qa(questions, answers, tokenizer, chunk_size=256, maxlen=512):
    """
    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras
    """
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(max_length=maxlen)
    all_ids = []
    
    for i in tqdm(range(0, len(questions), chunk_size)):
        q_chunk = questions[i:i+chunk_size].tolist()
        a_chunk = answers[i:i+chunk_size].tolist()
        text_chunk = list(zip(q_chunk, a_chunk))
        
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
    
    return np.array(all_ids)


# In[ ]:


def truncate_text(text, tokenizer, chunk_size=256, maxlen=256):
    """
    Ensure that the text does not have more than maxlen tokens
    """
    tokenizer.enable_truncation(max_length=maxlen)
    all_norm_str = []
    
    for i in tqdm(range(0, len(text), chunk_size)):
        chunk = text[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(chunk)
        all_norm_str.extend([str(enc.normalized_str) for enc in encs])
    
    return all_norm_str


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


def save_model(model, sigmoid_dir='transformer', transformer_dir='transformer'):
    """
    Special function to load a keras model that uses a transformer layer
    """
    os.makedirs(transformer_dir, exist_ok=True)
    os.makedirs(sigmoid_dir, exist_ok=True)
    
    transformer = model.layers[1]
    transformer.save_pretrained(transformer_dir)
    
    sigmoid_path = os.path.join(sigmoid_dir,'sigmoid.pickle')
    sigmoid = model.get_layer('sigmoid').get_weights()
    pickle.dump(sigmoid, open(sigmoid_path, 'wb'))


# In[ ]:



def load_model(sigmoid_dir='transformer', transformer_dir='transformer', 
           architecture="electra", max_len=None):
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


# ## TPU Configs

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


AUTO = tf.data.experimental.AUTOTUNE

# Configuration
EPOCHS = 8
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
MAX_LEN = 512
MODEL = 'google/electra-small-discriminator'


# ## Load data

# In[ ]:


df = pd.concat([
    pd.read_csv(f'/kaggle/input/stackexchange-qa-pairs/pre_covid/{group}.csv')
    for group in ['expert', 'biomedical', 'general']
]).reset_index(drop=True)

df.head()


# In[ ]:


questions = df.title + "[SEP]" + df.question


# ## Bert tokenizer

# In[ ]:


# First load the real tokenizer
tokenizer = ElectraTokenizer.from_pretrained(MODEL)
# Save the loaded tokenizer locally
tokenizer.save_pretrained('.')
# Reload it with the huggingface tokenizers library
fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=True, add_special_tokens=False)
fast_tokenizer


# ## Convert text to matrices
# 
# Caveat: Since a lot of the questions on stackexchange goes over 256, characters, we end up truncating a large part (if not all) of the answers. Thus, we need to "pre" truncate them by separately encode the questions and answers, and use a functions to combine them again.
# 
# Note: Here we are not actually encoding it, instead we load the encoded q&a pairs from another notebook, in order to limit memory consumption.

# In[ ]:


# q_ids = fast_encode(questions.values, fast_tokenizer, maxlen=MAX_LEN//2 - 2)
# a_ids = fast_encode(df.answer.values, fast_tokenizer, maxlen=MAX_LEN//2 - 2)
# wa_ids = fast_encode(df.wrong_answer.values, fast_tokenizer, maxlen=MAX_LEN//2 - 2)

# correct_ids = combine_qa_ids(q_ids, a_ids, tokenizer, maxlen=MAX_LEN)
# wrong_ids = combine_qa_ids(q_ids, wa_ids, tokenizer, maxlen=MAX_LEN)


# In[ ]:


correct_ids = np.load('/kaggle/input/stackexchange-encode-for-electra/correct_ids.npy')
wrong_ids = np.load('/kaggle/input/stackexchange-encode-for-electra/wrong_ids.npy')


# In[ ]:


input_ids = np.concatenate([correct_ids, wrong_ids])

labels = np.concatenate([
    np.ones(correct_ids.shape[0]),
    np.zeros(wrong_ids.shape[0])
]).astype(np.int32)


# ## Train test split

# In[ ]:


train_idx, test_idx = train_test_split(
    np.arange(input_ids.shape[0]), 
    test_size=0.3, 
    random_state=0
)

valid_idx, test_idx = train_test_split(
    test_idx, 
    test_size=0.5, 
    random_state=1
)


# In[ ]:


train_ids = input_ids[train_idx]
valid_ids = input_ids[valid_idx]
test_ids = input_ids[test_idx]

train_labels = labels[train_idx]
valid_labels = labels[valid_idx]
test_labels = labels[test_idx]


# ## Build datasets objects

# In[ ]:


train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((train_ids, train_labels))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((valid_ids, valid_labels))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(test_ids)
    .batch(BATCH_SIZE)
)


# ## Modeling

# In[ ]:


get_ipython().run_cell_magic('time', '', 'with strategy.scope():\n    transformer_layer = TFElectraModel.from_pretrained(MODEL)\n    model = build_model(transformer_layer, max_len=MAX_LEN)\nmodel.summary()')


# ### Train model

# In[ ]:


n_steps = train_labels.shape[0] // BATCH_SIZE

train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=EPOCHS
)


# In[ ]:


save_model(model)


# In[ ]:


hist_df = pd.DataFrame(train_history.history)
hist_df.to_csv('train_history.csv')
hist_df


# ## Eval

# In[ ]:


with strategy.scope():
    model = load_model(max_len=MAX_LEN)


# In[ ]:


y_score = model.predict(test_dataset, verbose=1).squeeze()
y_pred = y_score.round().astype(int)
print("AP:", average_precision_score(test_labels, y_score))
print("ROC AUC:", roc_auc_score(test_labels, y_score))
print(classification_report(test_labels, y_pred))


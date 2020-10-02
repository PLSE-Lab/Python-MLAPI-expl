#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os, gc

# Any results you write to the current directory are saved as output.


# In[ ]:


from transformers import *
from kaggle_datasets import KaggleDatasets
import tensorflow as tf
from tqdm.autonotebook import  tqdm
from ast import literal_eval
import tensorflow.keras.layers as L


# In[ ]:




AUTO = tf.data.experimental.AUTOTUNE

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.experimental.TPUStrategy(tpu)

GCS_DS_PATH = KaggleDatasets().get_gcs_path('jigsaw-multilingual-toxic-comment-classification')

print("PATH: ", GCS_DS_PATH)
print("REPLICAS: ", strategy.num_replicas_in_sync)


# In[ ]:


EPOCHS = 2
BATCH_SIZE = 64 * strategy.num_replicas_in_sync


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_nobias = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")\n\ntrain_base = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")\nvalid_base = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv")\ntest_base = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv")\n\ntrain_pre = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train-processed-seqlen128.csv")\nvalid_pre = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation-processed-seqlen128.csv")\ntest_pre = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test-processed-seqlen128.csv")\n\nsub = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv")')


# In[ ]:


MODEL_TYPE =  "bert-base-multilingual-cased"


# In[ ]:


tokenizer = BertTokenizer.from_pretrained(MODEL_TYPE)


# In[ ]:


train_base.shape


# In[ ]:


def tokenize_all(texts, tokenizer, chunk_size=512 ,max_len=512):
    ids = []
    
    for i in tqdm(range(0, len(texts), chunk_size)):
        chunk_ids = tokenizer.batch_encode_plus(texts[i: i+chunk_size])
        ids.append(chunk_ids["input_ids"])
    
    return ids
    
#train_ids = tokenize_all(train_base.comment_text.values, tokenizer)
#valid_ids = tokenize_all(valid_base.comment_text.values, tokenizer)
#test_ids = tokenize_all(test_base.comment_text.values, tokenizer)


# In[ ]:


train_pre.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_pre["input_word_ids"] = train_pre["input_word_ids"].apply(literal_eval)\nvalid_pre["input_word_ids"] = valid_pre["input_word_ids"].apply(literal_eval)\ntest_pre["input_word_ids"] = test_pre["input_word_ids"].apply(literal_eval)\n\ntrain_pre["input_mask"] = train_pre["input_mask"].apply(literal_eval)\nvalid_pre["input_mask"] = valid_pre["input_mask"].apply(literal_eval)\ntest_pre["input_mask"] = test_pre["input_mask"].apply(literal_eval)')


# In[ ]:


train_ids = train_pre.loc[:, ["input_word_ids", "input_mask"]]
valid_ids = valid_pre.loc[:, ["input_word_ids", "input_mask"]]
test_ids  = test_pre.loc[:, ["input_word_ids", "input_mask"]]


# In[ ]:


gc.collect()


# In[ ]:


train_targets = train_pre.loc[:, ["toxic"]]#, "severe_toxic", "obscene", "threat",	"insult", "identity_hate"]]
valid_targets = valid_pre.loc[:, ["toxic"]]#, "severe_toxic", "obscene", "threat",	"insult", "identity_hate"]]


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_ids["input_word_ids"] = train_ids["input_word_ids"].apply(np.array, dtype=np.int32)\ntrain_ids["input_mask"] = train_ids["input_mask"].apply(np.array, dtype=np.int32)\n\nvalid_ids["input_word_ids"] = valid_ids["input_word_ids"].apply(np.array, dtype=np.int32)\nvalid_ids["input_mask"] = valid_ids["input_mask"].apply(np.array, dtype=np.int32)\n\ntest_ids["input_word_ids"] = test_ids["input_word_ids"].apply(np.array, dtype=np.int32)\ntest_ids["input_mask"] = test_ids["input_mask"].apply(np.array, dtype=np.int32)')


# In[ ]:


train_data = (tf.convert_to_tensor(train_ids.iloc[:, 0]), tf.convert_to_tensor(train_ids.iloc[:, 1]))
valid_data = (tf.convert_to_tensor(valid_ids.iloc[:, 0]), tf.convert_to_tensor(valid_ids.iloc[:, 1]))
test_data = (tf.convert_to_tensor(test_ids.iloc[:, 0]), tf.convert_to_tensor(test_ids.iloc[:, 1]))


# In[ ]:


gc.collect()


# In[ ]:


train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((train_data[0], train_targets.values))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)
valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((valid_data[0], valid_targets.values))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)
test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(test_data[0])
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)


# Modelling 

# In[ ]:





# In[ ]:


def get_multi_classifier(base_model, n_classes, max_len=128):
    input_i = L.Input(shape=(max_len, ), dtype=tf.int32)
    #input_m = L.Input(shape=(max_len, ), dtype=tf.int32)
    
    print(input_i.shape)
    op1, op2 = base_model(input_i)
    print(op1.shape)
    print(op2.shape)
    cls_token = op1[:, 0, :]
    """
    cls_token = L.Dense(256, activation="relu")(cls_token)
    cls_token = L.Dropout(0.1)(cls_token)
    """
    out = L.Dense(n_classes, activation='sigmoid')(cls_token)
    
    model = tf.keras.models.Model(inputs = input_i, outputs = out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1.5e-5), loss="binary_crossentropy", metrics=["accuracy"])
    
    return model
    
    


# In[ ]:


import gc
gc.collect()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'with strategy.scope():\n    base_model = TFBertModel.from_pretrained(MODEL_TYPE)\n    model_ = get_multi_classifier(base_model, 1, 128)\n    \n    ')


# In[ ]:


model_.summary()


# In[ ]:


from tensorflow.keras.callbacks import Callback


# In[ ]:


model_.fit(train_dataset,
           epochs=15,
           steps_per_epoch=train_data[0].shape[0]//BATCH_SIZE,
           validation_data=valid_dataset,
           validation_steps=valid_data[0].shape[0]//BATCH_SIZE,
           callbacks = [])


# In[ ]:


ps = model_.predict(test_dataset, verbose=1)


# In[ ]:


test_pre.head()


# In[ ]:


ps_s=np.array(ps).squeeze()


# In[ ]:


final = pd.DataFrame({"id":test_pre["id"].values, "toxic":ps_s})


# In[ ]:


ps_s.mean()


# In[ ]:


(ps_s>0.5).mean()


# In[ ]:


final.head()


# In[ ]:


final.to_csv("submission.csv", index=False)


# In[ ]:





# In[ ]:





# In[ ]:





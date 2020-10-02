#!/usr/bin/env python
# coding: utf-8

# # XLM-Roberta Large tokenize dataset
# 
# This kernel tokenizes the whole (train+test) dataset ahead of time and saves it in npy file format for later loading in order to save time during training and inference.
# 
# Based on [abhishek's](https://www.kaggle.com/abhishek/bert-multi-lingual-tpu-training-8-cores-w-valid) and [xhlulu's](https://www.kaggle.com/xhlulu/jigsaw-tpu-xlm-roberta) kernels.

# In[ ]:


import os
import torch
import pandas as pd
from scipy import stats
import numpy as np

from tqdm import tqdm
from collections import OrderedDict, namedtuple
import torch.nn as nn
from torch.optim import lr_scheduler
import joblib

import logging
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
import sys
from sklearn import metrics, model_selection

import warnings

warnings.filterwarnings("ignore")


# In[ ]:


tokenizer = transformers.XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')


# In[ ]:


def regular_encode(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
        texts, 
        return_attention_masks=False, 
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=maxlen
    )
    
    return np.array(enc_di['input_ids'])


# In[ ]:


df_train1 = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv", usecols=["comment_text", "toxic"]).fillna("none")
df_train2 = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv", usecols=["comment_text", "toxic"]).fillna("none")
df_train_full = pd.concat([df_train1, df_train2], axis=0).reset_index(drop=True)
df_train = df_train_full.sample(frac=1).reset_index(drop=True).head(200000)

df_valid = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/validation.csv', 
                       usecols=["comment_text", "toxic"])

df_train = pd.concat([df_train, df_valid], axis=0).reset_index(drop=True)
df_train = df_train.sample(frac=1).reset_index(drop=True)

df_test = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/test.csv")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'x_train = regular_encode(df_train.comment_text.values, tokenizer, maxlen=192)\nx_valid = regular_encode(df_valid.comment_text.values, tokenizer, maxlen=192)\nx_test  = regular_encode(df_test.content.values,       tokenizer, maxlen=192)')


# In[ ]:


np.save('x_train',x_train)
np.save('x_valid',x_valid)
np.save('x_test',x_test)


# In[ ]:


np.save('df_train_toxic',df_train.toxic.values)
np.save('df_valid_toxic',df_valid.toxic.values)


# In[ ]:


np.save('test_df_ids',df_test.id.values)


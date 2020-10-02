#!/usr/bin/env python
# coding: utf-8

# # imports

# In[ ]:


import os
import torch
import pandas as pd
import numpy as np
import joblib
import logging
import transformers
import sys
import torch.nn as nn
import gc;
import h5py
from scipy import stats
from collections import OrderedDict, namedtuple
from torch.optim import lr_scheduler
from transformers import (
    AdamW, get_linear_schedule_with_warmup, get_constant_schedule, 
    XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaConfig,
)
from sklearn import metrics, model_selection
from tqdm.autonotebook import tqdm


# In[ ]:


get_ipython().run_cell_magic('time', '', '# load the data\n\ntrain1 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv", usecols=["comment_text", "toxic"])\ntrain2 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv", usecols=["comment_text", "toxic"])\ntrain2.toxic = train2.toxic.round().astype(int)\n\ndf_valid = pd.read_csv(\'/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv\')\ntest = pd.read_csv(\'/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv\')\nsub = pd.read_csv(\'/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv\')\n\ndf_train = pd.concat([\n    train1[[\'comment_text\', \'toxic\']],\n    train2[[\'comment_text\', \'toxic\']].query(\'toxic==1\'),\n    train2[[\'comment_text\', \'toxic\']].query(\'toxic==0\').sample(n=99937, random_state=0), # hacked to make train_data size divisible by bs;\n])\n\ndel train1, train2\ngc.collect(); gc.collect();\nprint(df_train.shape, df_valid.shape)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'tqdm.pandas()\ndf_train["comment_text"] = df_train["comment_text"].progress_apply(lambda x: " " + " ".join(str(x).split()))\ndf_valid["comment_text"] = df_valid["comment_text"].progress_apply(lambda x: " " + " ".join(str(x).split()))')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nfrom joblib import Parallel, delayed\ntokenizer = transformers.XLMRobertaTokenizer.from_pretrained(\'xlm-roberta-large\')\n\ndef regular_encode(texts, tokenizer=tokenizer, maxlen=128):\n    enc_di = tokenizer.encode_plus(\n        str(texts[0]),\n        return_attention_masks=False, \n        return_token_type_ids=False,\n        pad_to_max_length=True,\n        max_length=maxlen\n    )\n    \n    return np.array(enc_di[\'input_ids\']), np.array(enc_di["attention_mask"]), texts[1]\n\nrows = zip(df_train[\'comment_text\'].values.tolist(), df_train.toxic.values.tolist())\nx_train = Parallel(n_jobs=4, backend=\'multiprocessing\')(delayed(regular_encode)(row) for row in tqdm(rows))\n\nrows = zip(df_valid[\'comment_text\'].values.tolist(), df_valid.toxic.values.tolist())\nx_valid = Parallel(n_jobs=4, backend=\'multiprocessing\')(delayed(regular_encode)(row) for row in tqdm(rows))')


# In[ ]:


np.save("x_train_tokenized", x_train)
np.save("x_valid_tokenized", x_valid);


# In[ ]:


np.array(x_train).shape, np.array(x_valid).shape


# In[ ]:


import numpy

a = numpy.memmap('train.mymemmap', dtype='int32', mode='w+', shape=(2, np.array(x_train).shape[0], 128))
for idx in tqdm(range(np.array(x_train).shape[0])):
    a[0][idx] = np.array(x_train[idx][0], dtype=np.int32)
    a[1][idx] = np.array(x_train[idx][1], dtype=np.int32)
del a;

a = numpy.memmap('train_targets.mymemmap', dtype='int32', mode='w+', shape=(np.array(x_train).shape[0],))
for idx in tqdm(range(np.array(x_train).shape[0])):
    a[idx] = np.array(x_train[idx][2], dtype=np.int32)
del a;

a = numpy.memmap('valid.mymemmap', dtype='int32', mode='w+', shape=(2, np.array(x_valid).shape[0], 128))
for idx in tqdm(range(np.array(x_valid).shape[0])):
    a[0][idx] = np.array(x_valid[idx][0], dtype=np.int32)
    a[1][idx] = np.array(x_valid[idx][1], dtype=np.int32)
del a

a = numpy.memmap('valid_targets.mymemmap', dtype='int32', mode='w+', shape=(np.array(x_valid).shape[0],))
for idx in tqdm(range(np.array(x_valid).shape[0])):
    a[idx] = np.array(x_valid[idx][2], dtype=np.int32)
del a;


# In[ ]:


class MyIterableDataset_v1(torch.utils.data.IterableDataset):
    
    def __init__(self):
        
        self.data = np.memmap("valid.mymemmap", shape=(2, 8000, 128), mode="r", dtype="int32")
        self.target = np.memmap("valid_targets.mymemmap", shape=(8000,), mode="r", dtype="int32")
    
    def __iter__(self):
        # memmap contains input_ids, masks, targets
        return iter(zip(np.array(self.data[0]), np.array(self.data[1]), np.array(self.target)))


# In[ ]:


iterable_dataset = MyIterableDataset_v1()
loader = torch.utils.data.DataLoader(iterable_dataset, batch_size=32)


# In[ ]:


for batch in tqdm(loader):
    print(batch)
    break


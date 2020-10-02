#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import sys

package_dir_a = "../input/ppbert/pytorch-pretrained-bert/pytorch-pretrained-BERT"
sys.path.insert(0, package_dir_a)


# In[ ]:


# library
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd

from typing import *
from pathlib import Path

import torch
import torch.optim as optim

from fastai import *
from fastai.text import *
from fastai.vision import *
from fastai.callbacks import *
from sklearn.model_selection import train_test_split


# In[ ]:


from tqdm import tqdm
tqdm.pandas(desc="my bar!")
import torch.utils.data
from sklearn import metrics
from scipy.stats import rankdata
from tqdm import tqdm_notebook as tqdm
from nltk.tokenize.treebank import TreebankWordTokenizer
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam, BertConfig, convert_tf_checkpoint_to_pytorch

import re
import psutil
import multiprocessing as mp
from multiprocessing import Pool

from gensim.models import KeyedVectors

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')


# In[ ]:


def transform_bert_to_dir(BERT_MODEL_PATH, WORK_DIR, bert_model_name):
    if not os.path.exists(WORK_DIR):
        os.mkdir(WORK_DIR)
    convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(BERT_MODEL_PATH + '/bert_model.ckpt',
                                                                  BERT_MODEL_PATH + '/bert_config.json',
                                                                  WORK_DIR + f'/pytorch_model_{bert_model_name}.bin')
    shutil.copyfile(BERT_MODEL_PATH + '/bert_config.json', WORK_DIR + f'/bert_config_{bert_model_name}.json')


# In[ ]:


bert_model_path_parent = os.listdir('../input/bert-pretrained-models/')
print(bert_model_path_parent)


# In[ ]:


for i in tqdm(range(len(bert_model_path_parent))):
    path_parent = os.path.join('../input/bert-pretrained-models/', bert_model_path_parent[i])
    bert_model_path = os.path.join(path_parent, os.listdir(path_parent)[0])
    working_path = os.path.join('../working/', os.listdir(path_parent)[0])
    transform_bert_to_dir(bert_model_path, working_path, bert_model_path_parent[i])


# In[ ]:


get_ipython().system('ls ./multi_cased_L-12_H-768_A-12')


# In[ ]:





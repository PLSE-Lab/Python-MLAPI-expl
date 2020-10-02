#!/usr/bin/env python
# coding: utf-8

# One of the major breakthroughs in deep learning in 2018 was the development of effective transfer learning methods in NLP. One method that took the NLP community by storm was BERT.
# 
# BERT, or Bidirectional Encoder Representations from Transformers, is a new method of pre-training language representations which obtains state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks.
# 

# This notebook uses Python >= 3.6 and fastai >=1.0.52

# we will use a Pretrained BERT base uncased model which has 24-layers, 1024-hidden, 16-heads, 340M parameters

# 
# we will have to use the BERT tokenizer and BERT vocabulary while using the BERT model,we cant use the default tokenzer and vocabulary of Fastai. 
# 
# Hence we will have to write this custom code in Pytorch 
# 

# Install Pretrained BERT for Pytorch in the below step

# In[ ]:



get_ipython().run_cell_magic('bash', '', 'pip install pytorch-pretrained-bert')


# In[ ]:


import csv
import pandas as pd
from pathlib import Path
import matplotlib.cm as cm
from fastai import *
from fastai.text import *
from fastai.callbacks import *
from fastai.metrics import *
#import utils, bert_fastai, bert_helper
import numpy as np
import pandas as pd

from pathlib import Path
from typing import *

import torch
import torch.optim as optim


# 
# we use the same hyperparameters as mentioned in the BERT paper(https://github.com/google-research/bert)

# In[ ]:


class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)


config = Config(
    testing=False,
    bert_model_name="bert-base-uncased",
    max_lr=3e-5,
    epochs=4,
    use_fp16=True,
    bs=32,
    discriminative=False,
    max_seq_len=256,
)


# In[ ]:


from pytorch_pretrained_bert import BertTokenizer
bert_tok = BertTokenizer.from_pretrained(
    config.bert_model_name,
)


# In[ ]:


class FastAiBertTokenizer(BaseTokenizer):
    """Wrapper around BertTokenizer to be compatible with fast.ai"""
    def __init__(self, tokenizer: BertTokenizer, max_seq_len: int=128, **kwargs):
        self._pretrained_tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, *args, **kwargs):
        return self

    def tokenizer(self, t:str) -> List[str]:
        """Limits the maximum sequence length"""
        return ["[CLS]"] + self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2] + ["[SEP]"]


# In[ ]:


fastai_tokenizer = Tokenizer(tok_func=FastAiBertTokenizer(bert_tok, max_seq_len=config.max_seq_len), pre_rules=[], post_rules=[])


# In[ ]:


fastai_bert_vocab = Vocab(list(bert_tok.vocab.keys()))


# In[ ]:


#read the Training Data

train_df=pd.read_excel('../input/Train_Data.xlsx')


# In[ ]:


train_df.head()


# In[ ]:


train_df.info()


# In[ ]:


#Split data into Train and Validation 
from sklearn.model_selection import train_test_split
train, val = train_test_split(train_df)


# In[ ]:


databunch = TextDataBunch.from_df(".", train, val, 
                  tokenizer=fastai_tokenizer,
                  vocab=fastai_bert_vocab,
                  include_bos=False,
                  include_eos=False,
                  text_cols="question_text",
                  label_cols="target",
                  bs=config.bs,
                  collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),
             )


# In[ ]:


databunch.show_batch()


# In[ ]:


databunch.classes


# In[ ]:



from pytorch_pretrained_bert.modeling import BertConfig, BertForSequenceClassification
bert_model = BertForSequenceClassification.from_pretrained(config.bert_model_name, num_labels=2)


# In[ ]:


learner = Learner(
    databunch, bert_model,
    metrics=[accuracy]
)
learner.callbacks.append(ShowGraph(learner))


# In[ ]:


learner.lr_find()
learner.recorder.plot(suggestion=True)


# In[ ]:


learner.fit_one_cycle(3, max_lr=3e-5)


# **we see that just with 3 iterations, we are able to get 96% accuracy on this data set without any feature engineering and training. Hence Transfer Learning is a good approach for Text classification
# **

# Please refer https://www.kaggle.com/keitakurita/bert-with-fastai-example for the original code of BERT in Fastai

# In[ ]:





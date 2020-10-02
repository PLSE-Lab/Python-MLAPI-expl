#!/usr/bin/env python
# coding: utf-8

# # Project Description

# ## Overall objective

# In this notebook, I want to use two state of the art Natural Language Processing (NLP) techniques which have sort of revolutionalized the area of NLP in Deep Learning.
# 
# These techniques are as follows:
# 
# 1. BERT (Deep Bidirectional Transformers for Language Understanding)
# 2. Fastai ULMFiT (Universal Language Model Fine-tuning for Text Classification)
# 
# Both these techniques are very advanced and very recent NLP techniques (BERT was introduced by Google in 2018). Both of them incorporate the methods of Transfer Learning which is quite cool and are pre-trained on large corpuses of Wikipedia articles. I wanted to compare the overall performance of these two techniques.
# 
# I really like using Fastai for my deep learning projects and can't thank enough for this amazing community and our mentors - Jeremy & Rachael for creating few wonderful courses on the matters pertaining to Deep Learning. Therefore one of my aims to work on this project was to **integrate BERT with Fastai**. This means power of BERT combined with the simplicity of Fastai. It was not an easy task especially implementing Discriminative Learning Rate technique of Fastai in BERT modelling. 
# 
# In my project, below article helped me in understanding few of these integration techniques and I would like to extend my gratidue to the writer of this article:
# 
# [https://mlexplained.com/2019/05/13/a-tutorial-to-fine-tuning-bert-with-fast-ai/](http://)
# 
# 

# ## Data

# In this project, we will use BasicNeedsBasicRights Comments dataset which has categorized each text item into 4 classes: 
# 
# 1. Depression
# 2. Alcohol
# 3. Suicide
# 4. Drugs
# 
# This is a **multi-label text classification challenge**.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


test = pd.read_csv('/kaggle/input/testdata/Test_BNBR.csv')
train = pd.read_csv('/kaggle/input/encoded-train/encoded_train.csv')


# # Importing Libraries & Data Preparation

# In[ ]:


import numpy as np
import pandas as pd

from pathlib import Path
from typing import *

import torch
import torch.optim as optim

import gc
gc.collect()


# In this section, we will import Fastai libraries and few other important libraries for our task

# In[ ]:


get_ipython().system('pip install pretrainedmodels')

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().system('pip install fastai==1.0.52')
import fastai

from fastai import *
from fastai.vision import *
from fastai.text import *

from torchvision.models import *
import pretrainedmodels

from utils import *
import sys

from fastai.callbacks.tracker import EarlyStoppingCallback
from fastai.callbacks.tracker import SaveModelCallback


# Let's import Huggingface's "pytorch-pretrained-bert" model (this is now renamed as pytorch-transformers)
# 
# [https://github.com/huggingface/pytorch-transformers](http://)
# 
# This is a brilliant repository of few of amazing NLP techniques and already pre-trained.

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'pip install pytorch-pretrained-bert')


# BERT has several flavours when it comes to Tokenization. For our modelling purposes, we will use the most common and standard method named as "bert-case-uncased".
# 
# We will name this as bert_tok

# In[ ]:


from pytorch_pretrained_bert import BertTokenizer
bert_tok = BertTokenizer.from_pretrained(
    "bert-base-uncased",
)


# As mentioned in the article in first section, we will change the tokenizer of Fastai to incorporate BertTokenizer. One important thing to note here is to change the start and end of each token with [CLS] and [SEP] which is a requirement of BERT.

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


# Before we move further, lets have a look at the Data on which we have to work.
# 
# We will split the train data into two parts: Train, Validation. However, for the purpose of this project, we will not be using Test Data

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


DATA_ROOT = Path("..") / "input"

train, test = [pd.read_csv(DATA_ROOT / fname) for fname in ["train", "test"]]
train, val = train_test_split(train, shuffle=True, test_size=0.2, random_state=42)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


val.head()


# In following code snippets, we need to wrap BERT vocab and BERT tokenizer with Fastai modules

# In[ ]:


fastai_bert_vocab = Vocab(list(bert_tok.vocab.keys()))


# In[ ]:


fastai_tokenizer = Tokenizer(tok_func=FastAiBertTokenizer(bert_tok, max_seq_len=256), pre_rules=[], post_rules=[])


# Now, we can create our Databunch. Important thing to note here is to use BERT Tokenizer, BERT Vocab. And to and put include_bos and include_eos as False as Fastai puts some default values for these

# In[ ]:


label_cols = ["Depression", "Alcohol", "Suicide", "Drugs"]

databunch_1 = TextDataBunch.from_df(".", train, val, 
                  tokenizer=fastai_tokenizer,
                  vocab=fastai_bert_vocab,
                  include_bos=False,
                  include_eos=False,
                  text_cols="text",
                  label_cols=label_cols,
                  bs=32,
                  collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),
             )


# Alternatively, we can pass our own list of Preprocessors to the databunch (this is effectively what is happening behind the scenes)

# In[ ]:


class BertTokenizeProcessor(TokenizeProcessor):
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer, include_bos=False, include_eos=False)

class BertNumericalizeProcessor(NumericalizeProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, vocab=Vocab(list(bert_tok.vocab.keys())), **kwargs)

def get_bert_processor(tokenizer:Tokenizer=None, vocab:Vocab=None):
    """
    Constructing preprocessors for BERT
    We remove sos/eos tokens since we add that ourselves in the tokenizer.
    We also use a custom vocabulary to match the numericalization with the original BERT model.
    """
    return [BertTokenizeProcessor(tokenizer=tokenizer),
            NumericalizeProcessor(vocab=vocab)]


# In[ ]:


class BertDataBunch(TextDataBunch):
    @classmethod
    def from_df(cls, path:PathOrStr, train_df:DataFrame, valid_df:DataFrame, test_df:Optional[DataFrame]=None,
                tokenizer:Tokenizer=None, vocab:Vocab=None, classes:Collection[str]=None, text_cols:IntsOrStrs=1,
                label_cols:IntsOrStrs=0, label_delim:str=None, **kwargs) -> DataBunch:
        "Create a `TextDataBunch` from DataFrames."
        p_kwargs, kwargs = split_kwargs_by_func(kwargs, get_bert_processor)
        # use our custom processors while taking tokenizer and vocab as kwargs
        processor = get_bert_processor(tokenizer=tokenizer, vocab=vocab, **p_kwargs)
        if classes is None and is_listy(label_cols) and len(label_cols) > 1: classes = label_cols
        src = ItemLists(path, TextList.from_df(train_df, path, cols=text_cols, processor=processor),
                        TextList.from_df(valid_df, path, cols=text_cols, processor=processor))
        src = src.label_for_lm() if cls==TextLMDataBunch else src.label_from_df(cols=label_cols, classes=classes)
        if test_df is not None: src.add_test(TextList.from_df(test_df, path, cols=text_cols))
        return src.databunch(**kwargs)


# In[ ]:


# this will produce a virtually identical databunch to the code above
databunch_2 = BertDataBunch.from_df(".", train_df=train, valid_df=val,
                  tokenizer=fastai_tokenizer,
                  vocab=fastai_bert_vocab,
                  text_cols="text",
                  label_cols=label_cols,
                  bs=32,
                  collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),
             )


# In[ ]:


path=Path('../input/')


# In[ ]:


databunch_2.show_batch()


# In[ ]:


databunch_1.show_batch()


# Both Databunch_1 and Databunch_2 can be used for modelling purposes. In this project, we will be using Databunch_1 which is easier to create and use.

# # BERT Model

# In[ ]:


from pytorch_pretrained_bert.modeling import BertConfig, BertForSequenceClassification, BertForNextSentencePrediction, BertForMaskedLM
bert_model_class = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)


# Loss function to be used is Binary Cross Entropy with Logistic Losses

# In[ ]:


loss_func = nn.BCEWithLogitsLoss()


# Considering this is a multi-label classification problem, we cant use simple accuracy as metrics here. Instead, we will use accuracy_thresh with threshold of 25% as our metric here.

# In[ ]:


acc_02 = partial(accuracy_thresh, thresh=0.5)


# In[ ]:


model = bert_model_class


# Now, lets create learner function

# In[ ]:


from fastai.callbacks import *

learner = Learner(
    databunch_1, model,
    loss_func=loss_func, model_dir='/temp/model', metrics=acc_02,
)


# Below code will help us in splitting the model into desirable parts which will be helpful for us in Discriminative Learning i.e. setting up different learning rates and weight decays for different parts of the model.

# In[ ]:


def bert_clas_split(self) -> List[nn.Module]:
    
    bert = model.bert
    embedder = bert.embeddings
    pooler = bert.pooler
    encoder = bert.encoder
    classifier = [model.dropout, model.classifier]
    n = len(encoder.layer)//3
    print(n)
    groups = [[embedder], list(encoder.layer[:n]), list(encoder.layer[n+1:2*n]), list(encoder.layer[(2*n)+1:]), [pooler], classifier]
    return groups


# In[ ]:


x = bert_clas_split(model)


# Let's split the model now in 6 parts

# In[ ]:


learner.split([x[0], x[1], x[2], x[3]])  #  , x[5]


# In[ ]:


learner.lr_find()


# In[ ]:


learner.recorder.plot()


# In[ ]:


learner.fit_one_cycle(2, max_lr=slice(1e-5, 5e-4), moms=(0.8,0.7), pct_start=0.2, wd =(1e-5, 1e-4, 1e-3, 1e-2))


# In[ ]:


learner.save('head')
learner.load('head')


# Now, we will unfreeze last two last layers and train the model again

# In[ ]:


learner.freeze_to(-2)
learner.fit_one_cycle(2, max_lr=slice(1e-5, 5e-4), moms=(0.8,0.7), pct_start=0.2, wd =(1e-5, 1e-4, 1e-3, 1e-2))


# In[ ]:


learner.save('head-2')
learner.load('head-2')


# We will now unfreeze the entire model and train it

# In[ ]:


learner.unfreeze()
learner.lr_find()
learner.recorder.plot(suggestion=True)


# In[ ]:


learner.fit_one_cycle(2, slice(5e-6, 5e-5), moms=(0.8,0.7), pct_start=0.2, wd =(1e-5, 1e-4, 1e-3, 1e-2))


# We will now see our model's prediction power

# In[ ]:


text = 'I feel alone and unwanted by people around me'
learner.predict(text)


# In[ ]:


text = 'Lonely and happiness'
learner.predict(text)


# In[ ]:


text = 'I feel sad and lost'
learner.predict(text)


# In[ ]:


text = 'Effects of alcohol on my body health'
learner.predict(text)


# This is awesome!
# 
# With few number of epochs, we are able to get the accuracy of around 98% on this multi-label classification task.
# 
# Now, lets see how does Fastai ULMFiT fare on this task

# # Fastai - ULMFiT

# This will have two parts:
# 
# 1. Training the Language Model
# 2. Training the Classifier Model

# ## Language Model
# 

# Important thing to remember in the Language Model is that we train it without label. The basic objective by training language model is to predict the next sentence / words in a sequence of text.

# In[ ]:


src_lm = ItemLists(path, TextList.from_df(train, path=".", cols = "text"), 
                   TextList.from_df(val, path=".", cols = 'text'))


# In[ ]:


data_lm = src_lm.label_for_lm().databunch(bs=32)


# In[ ]:


data_lm.show_batch()


# In[ ]:


learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3, model_dir="/temp/model")


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(1, max_lr=slice(5e-4, 5e-3), moms=(0.8, 0.7), pct_start=0.2, wd =(1e-5, 1e-4, 1e-3))


# In[ ]:


learn.save('fit_head')
learn.load('fit_head')


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(10, max_lr = slice(1e-4, 1e-3), moms=(0.8, 0.7), pct_start=0.2, wd =(1e-7, 1e-5, 1e-4,  1e-2))


# In[ ]:


learn.save('fine-tuned')
learn.load('fine-tuned')
learn.save_encoder('fine-tuned')


# In[ ]:


TEXT = "He is a piece of"
N_WORDS = 10
N_SENTENCES = 2


# In[ ]:


print("\n".join(learn.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))


# # Classification Model

# In[ ]:


src_clas = ItemLists(path, TextList.from_df( train, path=".", cols="comment_text", vocab = data_lm.vocab),
                    TextList.from_df( val, path=".", cols="comment_text", vocab = data_lm.vocab))


# In[ ]:


data_clas = src_clas.label_from_df(cols=label_cols).databunch(bs=32)


# In[ ]:


data_clas.show_batch()


# In[ ]:


learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5, model_dir='/temp/model', metrics=acc_02, loss_func=loss_func)
learn.load_encoder('fine-tuned')


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(2, max_lr=slice(1e-3, 1e-2), moms=(0.8, 0.7), pct_start=0.2, wd =(1e-7, 1e-5, 1e-4, 1e-3, 1e-2))


# In[ ]:


learn.save('first-head')
learn.load('first-head')


# In[ ]:


learn.freeze_to(-2)
learn.fit_one_cycle(2, slice(5e-2/(2.6**4),5e-2), moms=(0.8,0.7), pct_start=0.2, wd =(1e-7, 1e-5, 1e-4, 1e-3, 1e-2))


# In[ ]:


learn.save('second')
learn.load('second')


# In[ ]:


learn.freeze_to(-3)
learn.fit_one_cycle(2, slice(5e-2/(2.6**4),5e-2), moms=(0.8,0.7), pct_start=0.2, wd =(1e-7, 1e-5, 1e-4, 1e-3, 1e-2))


# In[ ]:


learn.save('third')
learn.load('third')


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(2, slice(1e-4/(2.6**4),1e-4), moms=(0.8,0.7), pct_start=0.2, wd =(1e-7, 1e-5, 1e-4, 1e-3, 1e-2))


# In[ ]:


learn.predict('she is so sweet')


# In[ ]:


learn.predict('you are pathetic piece of shit')


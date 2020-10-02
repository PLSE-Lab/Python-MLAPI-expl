#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai.text import *
from fastai import *

import pandas as pd
import numpy as np


# In[ ]:


path = untar_data(URLs.IMDB)
path.ls()


# In[ ]:


get_ipython().system('ls {path}/train/pos | head')


# In[ ]:


from pytorch_pretrained_bert import BertTokenizer

bert_tok = BertTokenizer.from_pretrained(
    "bert-base-uncased",
)


# In[ ]:


class FastaiBertTokenizer(BaseTokenizer):
    '''wrapper for fastai tokenizer'''
    def __init__(self, tokenizer, max_seq=128, **kwargs):
        self._pretrained_tokenizer = tokenizer
        self.max_seq_length = max_seq
        
    def __call__(self,*args,**kwargs):
        return self
    
    def tokenizer(self,t):
        return ["[CLS]"] + self._pretrained_tokenizer.tokenize(t)[:self.max_seq_length - 2] + ['[SEP]']


# In[ ]:


fastai_bert_vocab = Vocab(list(bert_tok.vocab.keys()))


# In[ ]:


fastai_bert_vocab.itos[2000:2010]


# In[ ]:


fastai_tokenizer = Tokenizer(tok_func=FastaiBertTokenizer(bert_tok,max_seq=128),pre_rules=[fix_html],post_rules=[])


# In[ ]:


processor = [OpenFileProcessor(),
             TokenizeProcessor(tokenizer=fastai_tokenizer,include_bos=False,include_eos=False),
             NumericalizeProcessor(vocab=fastai_bert_vocab)
            ]


# In[ ]:


data = (TextList
        .from_folder(path/'train',vocab=fastai_bert_vocab,processor=processor)
        .split_by_rand_pct(seed=42)
        .label_from_folder()
        .databunch(bs=16,num_workers=2)
       )


# In[ ]:


fastai_bert_vocab.stoi['the']


# In[ ]:


data.vocab.stoi['the']


# In[ ]:


from pytorch_pretrained_bert.modeling import BertConfig, BertForSequenceClassification, BertForNextSentencePrediction, BertForMaskedLM
bert_model_class = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model = bert_model_class


# In[ ]:


bert_1 = model.bert
embedder = bert_1.embeddings
pooler = bert_1.pooler
encoder = bert_1.encoder
classifier = [model.dropout,model.classifier]


# In[ ]:


n = len(encoder.layer) // 3
print(n)


# In[ ]:


model


# In[ ]:


from fastai.callbacks import *
learner = Learner(
 data, model,
 model_dir='/kaggle/working', metrics=accuracy
).to_fp16()


# In[ ]:


learner.lr_find()


# In[ ]:


learner.recorder.plot(suggestion=True)


# In[ ]:


learner.fit_one_cycle(1,1e-4)


# In[ ]:


learner.fit(1,1e-4)


# In[ ]:


learner.recorder.plot_lr()


# In[ ]:





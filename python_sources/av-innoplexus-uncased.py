#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

from pathlib import Path
from typing import *

import torch
import torch.optim as optim

seed = 100

# python RNG
import random
random.seed(seed)

# pytorch RNGs
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# numpy RNG
np.random.seed(seed)


# In[ ]:


from fastai import *
from fastai.vision import *
from fastai.text import *
from fastai.callbacks import *


# In[ ]:


import pytorch_pretrained_bert
import fastai
import scipy
import random
import pathlib
import typing
print (torch.__version__)
print (fastai.__version__)
print (np.__version__)
print (pd.__version__)
print (pytorch_pretrained_bert.__version__)
print (scipy.__version__)


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


def _join_texts(texts:Collection[str], mark_fields:bool=False, sos_token:Optional[str]=BOS):
    """Borrowed from fast.ai source"""
    if not isinstance(texts, np.ndarray): texts = np.array(texts)
    if is1d(texts): texts = texts[:,None]
    df = pd.DataFrame({i:texts[:,i] for i in range(texts.shape[1])})
    text_col = f'{FLD} {1} ' + df[0].astype(str) if mark_fields else df[0].astype(str)
    if sos_token is not None: text_col = f"{sos_token} " + text_col
    for i in range(1,len(df.columns)):
        #text_col += (f' {FLD} {i+1} ' if mark_fields else ' ') + df[i]
        text_col += (f' {FLD} {i+1} ' if mark_fields else ' ') + df[i].astype(str)
    return text_col.values


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





# In[ ]:


from sklearn.model_selection import train_test_split

df_train=pd.read_csv('../input/train.csv')
df_test=pd.read_csv('../input/test.csv')

path='/kaggle/working/'
df_train.text=df_train.text+' This comment is about '+df_train.drug
df_test.text=df_test.text+' This comment is about '+df_test.drug
train_df,valid_df= train_test_split(df_train,test_size=0.2, random_state=seed)


# In[ ]:


fastai_bert_vocab = Vocab(list(bert_tok.vocab.keys()))


# In[ ]:


fastai_tokenizer = Tokenizer(tok_func=FastAiBertTokenizer(bert_tok, max_seq_len=config.max_seq_len), pre_rules=[], post_rules=[])


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


# # Model

# In[ ]:


from sklearn.model_selection import StratifiedKFold
folds=StratifiedKFold(n_splits=5,random_state=seed,shuffle=True)
folds=[(tr,val) for tr,val in folds.split(df_train,df_train.sentiment)]


# In[ ]:


def get_preds_as_nparray(learner_,dbunch_,ds_type) -> np.ndarray:
    """
    the get_preds method does not yield the elements in order by default
    we borrow the code from the RNNLearner to resort the elements into their correct order
    """
    preds = learner_.get_preds(ds_type)[0].detach().cpu().numpy()
    sampler = [i for i in dbunch_.dl(ds_type).sampler]
    reverse_sampler = np.argsort(sampler)
    return preds[reverse_sampler, :]


# In[ ]:


from pytorch_pretrained_bert.modeling import BertConfig, BertForSequenceClassification


# In[ ]:


wts=    wts=(0.25/df_train.sentiment.value_counts(1).sort_values()).tolist()


# In[ ]:


import gc
gc.collect()


# In[ ]:


from fastai.callbacks import *


# In[ ]:


test_preds_list2=[]
scores=[]
for i in range(5):
    train_df=df_train.iloc[folds[i][0],:]
    valid_df=df_train.iloc[folds[i][1],:]
    # this will produce a virtually identical databunch to the code above
    bert_model = BertForSequenceClassification.from_pretrained(config.bert_model_name, num_labels=3)
    databunch = BertDataBunch.from_df(path, train_df, valid_df,test_df= df_test,
                      tokenizer=fastai_tokenizer,
                      vocab=fastai_bert_vocab,
                      text_cols="text",
                      label_cols='sentiment',
                      bs=config.bs,
                      collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),
                 )

    learner = Learner(
        databunch, bert_model,
    metrics=[FBeta(average='macro',beta=1),accuracy]
    )
    if config.use_fp16: learner = learner.to_fp16()

    learner.loss_func=CrossEntropyFlat(weight=torch.Tensor(wts).float().cuda())    

    cb = [callbacks.tracker.SaveModelCallback(learner, every='improvement', monitor='f_beta', name='best-model_{}'.format(i))]
    learner.fit_one_cycle(10,3e-5,callbacks=cb)
    learner.load('best-model_{}'.format(i))
    scores.append(np.array(learner.recorder.metrics).max(axis=0).tolist())
    test_preds = get_preds_as_nparray(learner,databunch,DatasetType.Test)
    temp_val2=test_preds.argmax(axis=1)
    test_preds_list2.append(temp_val2)
    if i==0:
        get_ipython().system('rm -r /kaggle/working/models/best-model_0.pth')
    if i==1:
        get_ipython().system('rm -r /kaggle/working/models/best-model_1.pth')
    if i==2:
        get_ipython().system('rm -r /kaggle/working/models/best-model_2.pth')
    if i==3:
        get_ipython().system('rm -r /kaggle/working/models/best-model_3.pth')
        


# In[ ]:


print (scores)


# In[ ]:


from scipy.stats import mode
bagg_preds2=pd.DataFrame(np.column_stack(test_preds_list2))
preds2=bagg_preds2.mode(axis=1)[0]
bagg_preds2['unique_hash']=df_test.unique_hash
bagg_preds2.to_csv('raw_preds_bert_uncased.csv',index=False)
pd.DataFrame({'unique_hash':df_test.unique_hash,'sentiment':preds2}).to_csv('sub__bert_uncased.csv',index=False)


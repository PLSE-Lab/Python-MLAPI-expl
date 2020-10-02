#!/usr/bin/env python
# coding: utf-8

# Here we mostly follow the training scheme described by Jeremy Howard in fast.ai,
# taking a pretrained language model, fine-tuning it with unlabeled data, then fine-tuning classification head for our particular task.
# 
# http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html
# 
# https://docs.fast.ai/text.html#Quick-Start:-Training-an-IMDb-sentiment-model-with-ULMFiT

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# In[ ]:


train = pd.read_excel('../input/Train_Data.xlsx')


# In[ ]:


import fastai
from fastai.text import *
from fastai.callbacks import *


# In[ ]:


# check the contents of the dat set
train.head()


# Quickly check the content of some of the Questions 

# In[ ]:


train['question_text'][2]


# In[ ]:


train['question_text'][21]


# In[ ]:


from sklearn.model_selection import train_test_split
train, val = train_test_split(train)


# In[ ]:


# Language model data bunch
data_lm = TextLMDataBunch.from_df('.', train,val,text_cols='question_text',label_cols='target')


# In[ ]:


#save the preprocessed data
data_lm.save()


# In[ ]:


# Classifier model data
data_clas  = TextClasDataBunch.from_df('.', train_df=train,text_cols='question_text',label_cols='target',valid_df=val,vocab=data_lm.train_ds.vocab)


# In[ ]:


data_clas.save()


# In[ ]:


data_clas.show_batch()


# https://docs.fast.ai/text.transform.html#BaseTokenizer
# 
# **The rules are all listed below, here is the meaning of the special tokens:**
# 
# UNK (xxunk) is for an unknown word (one that isn't present in the current vocabulary)
# 
# PAD (xxpad) is the token used for padding, if we need to regroup several texts of different lengths in a batch
# 
# BOS (xxbos) represents the beginning of a text in your dataset
# 
# FLD (xxfld) is used if you set mark_fields=True in your TokenizeProcessor to separate the different fields of texts 
# (if your texts are loaded from several columns in a dataframe)
# 
# TK_MAJ (xxmaj) is used to indicate the next word begins with a capital in the original text
# 
# TK_UP (xxup) is used to indicate the next word is written in all caps in the original text
# 
# TK_REP (xxrep) is used to indicate the next character is repeated n times in the original text (usage xxrep n {char})
# 
# TK_WREP(xxwrep) is used to indicate the next word is repeated n times in the original text (usage xxwrep n {word})
# 

# In[ ]:


data_clas.vocab.itos[:10]


# In[ ]:


learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3,pretrained=True)


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(2, 5.75E-02,callbacks=[SaveModelCallback(learn, name="best_lm")], moms=(0.8,0.7))


# In[ ]:


learn.save('fit_head')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(3,3.98E-04,callbacks=[SaveModelCallback(learn, name="best_lm")], moms=(0.8,0.7))


# In[ ]:


learn.load('best_lm')


# In[ ]:


learn.save_encoder('AIBoot_enc')


# In[ ]:


learn1 = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.3)


# In[ ]:


learn1.load_encoder('AIBoot_enc')


# In[ ]:


learn1.lr_find()


# In[ ]:


learn1.recorder.plot(suggestion=True)


# In[ ]:


best_clf_lr = learn1.recorder.min_grad_lr
best_clf_lr


# In[ ]:


learn1.fit_one_cycle(1, best_clf_lr)


# In[ ]:


learn1.freeze_to(-2)


# In[ ]:


learn1.fit_one_cycle(1, best_clf_lr)


# In[ ]:


learn1.unfreeze()


# In[ ]:


learn1.lr_find()
learn1.recorder.plot(suggestion=True)


# In[ ]:


learn1.fit_one_cycle(3, 2e-3)


# In[ ]:


learn1.show_results()


# Language Model Prediction

# In[ ]:


learn.predict('Is it just me or have you ever been')


# In[ ]:


'Is it just me or have you ever been in this phase wherein you became ignorant to the people you once loved, completely disregarding their feelings/lives so you get to have something go your way and feel temporarily at ease. How did things change?'


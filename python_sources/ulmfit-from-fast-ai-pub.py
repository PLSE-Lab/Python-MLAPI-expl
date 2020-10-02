#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai.text import *
from fastai.datasets import URLs
import torch
import pandas as pd
import numpy as np
import logging
import os
import random
logging.basicConfig(filename='log.txt',level=logging.DEBUG, format='%(asctime)s %(message)s')


# In[ ]:


get_ipython().system('( head -535000 ../input/train.csv ) > train.csv # reducing size to train in less than 2 hours')
# !( head -200 ../input/test.csv ) > test.csv
# !cp ../input/train.csv train.csv
get_ipython().system('cp ../input/test.csv test.csv')
train_file = 'train.csv'
test_file = 'test.csv'
folder = '.'


# In[ ]:


# Language model data (10 min with whole training set)
data_lm = TextLMDataBunch.from_csv(folder, 
                                   train_file, 
                                   text_cols='question_text', 
                                   label_cols='target')
assert data_lm.device == torch.device('cuda')
logging.info('Language model data finish')


# In[ ]:


# Language model trainer (66 min with whole training set)
learn = language_model_learner(data_lm, 
                           drop_mult=0.5,
                           bptt=45)
learn.fit_one_cycle(1 , 1e-2)
learn.save_encoder('encoder')
logging.info('Language model trained')


# In[ ]:


# Classifier model data (10 min with whole training set)
data_clas = TextClasDataBunch.from_csv(folder, 
                                       train_file,
                                       valid_pct=0.1, #added to see if we prevent errors 
                                       vocab=data_lm.train_ds.vocab, 
                                       bs=32,
                                       text_cols='question_text', 
                                       label_cols='target')
logging.info('data_clas created')


# In[ ]:


# Classifier training (43 min with whole training set)
learn_classifier = text_classifier_learner(data_clas, drop_mult=0.5)
learn_classifier.load_encoder('encoder')
learn_classifier.fit_one_cycle(1, 1e-2)   
logging.info('learn_classifier fit')


# In[ ]:


# Do predictions and submission (67 min with whole test set)
# This is awful in terms of time and can be reduced to a few seconds (see next TODO)
# TODO: take the model out using learn_classifier.model and do steps manually. Work in progress.
test_set =  pd.read_csv(test_file)
true_threshold = 0.33
predictions = test_set['question_text'].apply(lambda x: int(learn_classifier.predict(x)[2][1]>true_threshold))
logging.info('predictions built')
submission = pd.DataFrame(test_set['qid'])
submission['prediction'] = predictions 
submission.to_csv('submission.csv',index=False)
logging.info('submission built')


#!/usr/bin/env python
# coding: utf-8

# SANTANDER SOLUTION

# We want to help our customers understand their financial health and identify which products and services might help them achieve their monetary goals.
# We want to be able to work with  binary classification problems such as: is a customer satisfied? Will a customer buy this product? Can a customer pay this loan?
# 
# In this challenge we want to be able to identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted.
# The data provided for this competition has the same structure as the real data we have available to solve this problem.
# 
# 

# In[ ]:


import sys
sys.version


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import seaborn as sns
from pathlib import Path
import PIL
import json
from fastai import *
from fastai.tabular import *
from fastai.vision import *
from fastai.metrics import error_rate


#                                                **LOADING KAGGLE DATASET **
#         
# ---
# 
# 

# In[ ]:


PATH = Path('content/kaggle/')
PATH


# In[ ]:


PATH = Path('../input')


#                                                                **LOAD DATA INTO DATAFRAME**

# In[ ]:


train_df = pd.read_csv(PATH/'train.csv')
train_df.head()


# In[ ]:


test_df = pd.read_csv(PATH/'test.csv')
test_df.head()


# In[ ]:


ss_df = pd.read_csv(PATH/'sample_submission.csv')
ss_df.head()


# In[ ]:


train_df.describe()


# In[ ]:


test_df.describe()


#                                                  ** DATA PRE-PROCESSING**

# In[ ]:


dep_var = 'target'


# In[ ]:


cat_names = []


# In[ ]:


df = train_df


# In[ ]:


cont_names = []
var_counter = 0 #creating a counter
num_of_cont_vars = len(df.columns) - 2
for _ in range(num_of_cont_vars):
    name = 'var_' + str(var_counter)
    cont_names.append(name)
    var_counter+=1


# In[ ]:


procs = [FillMissing, Normalize]


# In[ ]:


valid_idx = range(len(df)-20000, len(df))


# In[ ]:


test = TabularList.from_df(test_df, path=PATH, cont_names=cont_names, procs=procs)


# In[ ]:


path = PATH

data = (TabularList.from_df(df, path=path, cont_names=cont_names, procs=procs)
        .split_by_rand_pct(valid_pct=0.1)
        .label_from_df(cols=dep_var)
        .add_test(test)
        .databunch())

print(data.train_ds.cont_names)


# In[ ]:


data.show_batch(rows= 6)


# In[ ]:


(cat_x,cont_x),y = next(iter(data.train_dl))
for o in (cat_x, cont_x, y): print(to_np(o[:5]))


#                                             **DEFINING AND TRAIN  MODEL**

# In[ ]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


# In[ ]:


learn = tabular_learner(data, layers=[ 200 , 100], ps=[0.001,0.01], emb_drop=0.04, emb_szs={'ID_code': 20}, metrics=accuracy , path='.')
#to change to get rsqme just change accuracy to rmspe


# In[ ]:


learn.model


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr = 1e-02
learn.fit_one_cycle(7, lr , wd = 0.3)
#wd=0.2


# In[ ]:


learn.show_results()


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.data.batch_size


# In[ ]:


test_preds = learn.get_preds(ds_type=DatasetType.Test)
test_preds


# In[ ]:


target_preds = test_preds[0][:,1]
test_df['target'] = target_preds


# In[ ]:


target_preds


# In[ ]:


test_df.to_csv('submission.csv', columns=['ID_code', 'target'], index=False)


# In[ ]:


sub = pd.read_csv('submission.csv')
sub.head()


# In[ ]:


preds = learn.get_preds()
pred_tensors = preds[0]
actual_labels = preds[1].numpy()


# In[ ]:


pred_tensors, actual_labels

total_to_test = 20000
correct = 0
for i in range(total_to_test):
    if(pred_tensors[i][0] > 0.5 and actual_labels[i] == 0):
        correct = correct + 1

print(f"{correct}/{total_to_test} correct")


# In[ ]:


learn.save("trained_model", return_path=True)


# In[ ]:


learn = learn.load("trained_model" )


# In[ ]:


get_ipython().system('kaggle competitions submit -c santander-customer-transaction-prediction -f {PATH/\'submission.csv\'} -m "initial submission"')


# In[ ]:


test_df.head()


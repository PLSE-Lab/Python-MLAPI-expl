#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy as sp
import pandas as pd
import os

import random

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.display.precision = 15

import gc
import warnings
warnings.filterwarnings("ignore")

from fastai.tabular import * 
from tqdm import tqdm_notebook
from fastai.callbacks import *


# In[2]:


get_ipython().run_cell_magic('time', '', "train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32},nrows=6e8)")


# In[3]:


min = -100
max = 100
spread = 110
def get_counts(sequence):     
    counts = [0]*spread
    unique_count = np.unique(sequence, return_counts=True)
    for i in range(0,len(unique_count[0])):
        val = unique_count[0][i]
        count = unique_count[1][i]
        r = count*val
        if val <= min:
            counts[0] += r
        elif val >= max:
            counts[-1] += r
        else:
            counts[int(val/2)+int(spread/2)] += r

    return counts


# In[4]:


interval = 75000
counts = [get_counts(train.acoustic_data.values[i:i+150000]) for i in tqdm_notebook(range(0,len(train),interval))]
ttfs = [train.time_to_failure.values[i] for i in range(0,len(train),interval)]
del train

labels = ["D"+str(i) for i in range(0,len(counts[0]))]

df = pd.DataFrame(counts, columns=labels)
ttf_df = pd.DataFrame(ttfs, columns=["expected"])
df = df.join(ttf_df)


# In[5]:


df.head(3)


# In[6]:


path ="../tmp"
try:
    os.makedirs(path)
except:
    pass


# # Test Data

# In[7]:


tpath = "../input/test"
files = os.listdir(tpath)
i = 0
test_id = []
test_df = pd.DataFrame(dtype=np.float64, columns=df.columns.values[:-1])
for f in tqdm_notebook(files):
    seg = pd.read_csv(f'{tpath}/{f}')
    converted = get_counts(seg.acoustic_data.values)
    test_df.loc[i] = converted
    test_id.append(f.replace(".csv", ""))
    i+=1


# In[8]:


num = len(df)
interval = int(num/100)
values = int(num/(5*100))
valid_idx = []
for i in range(0,len(df)-values,interval):
    for j in range(0,values-1):
        valid_idx.append(i+j)


# In[9]:


valid_ttfs = np.array([df.iloc[i].expected for i in valid_idx])


# In[10]:


data = TabularDataBunch.from_df(path, df, "expected", valid_idx=valid_idx, test_df=test_df, procs=[Normalize])
# data = TabularDataBunch.from_df(path, df, "expected", valid_idx=valid_idx, procs=[Normalize])


# * spread 200 - 2.02
# * spread 300 - 

# In[11]:


get_ipython().run_cell_magic('time', '', "\nbest_learn = None\nbest_mae = 9999\n\nfor i in range(0, 99):\n    learn = tabular_learner(data=data, layers=[200,100], metrics=mae, ps=0.5, y_range=(-1,15))\n    learn.callbacks = [SaveModelCallback(learn, every='improvement', mode='min', name='best')]\n    learn.fit_one_cycle(20, 1e-2)\n    gc.collect()\n\n    preds = learn.get_preds(DatasetType.Valid)[0].numpy().flatten()\n    new_mae = np.abs(valid_ttfs-preds).mean()\n    if new_mae < best_mae or not best_learn:\n        best_learn = learn\n        best_mae = new_mae\n    print(f'Run {i} - Best MAE: {best_mae}')")


# # Submission

# In[12]:


preds = best_learn.get_preds(DatasetType.Test)[0].numpy().flatten()


# In[13]:


tpath = "../input/test"
files = os.listdir(tpath)
files = [f.replace(".csv","") for f in files]
files[:3]


# In[14]:


results = pd.DataFrame({"seg_id":files, "time_to_failure":preds})
results.head()


# In[15]:


results.to_csv('submission.csv',index=False)


#!/usr/bin/env python
# coding: utf-8

# In this kernel I explain how we can estimate clip duration. 

# In[ ]:


import numpy as np
import pandas as pd
import os
import copy
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm.notebook import tqdm
import datetime
import gc
from collections import OrderedDict, Counter
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


def read_data():
    print('Reading train_labels.csv file....')
    train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')
    print('Train_labels.csv file have {} rows and {} columns'.format(train_labels.shape[0], train_labels.shape[1]))
    
    print('Reading train.csv file....')
    train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')
    print('Training.csv file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))

    print('Reading test.csv file....')
    test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')
    print('Test.csv file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))

    print('Reading specs.csv file....')
    specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')
    print('Specs.csv file have {} rows and {} columns'.format(specs.shape[0], specs.shape[1]))

    print('Reading sample_submission.csv file....')
    sample_submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')
    print('Sample_submission.csv file have {} rows and {} columns'.format(sample_submission.shape[0], sample_submission.shape[1]))
    return train, test, train_labels, specs, sample_submission


# In[ ]:


get_ipython().run_cell_magic('time', '', '# read data\ntrain, test, train_labels, specs, sample_submission = read_data()')


# In[ ]:


train


# All clips have 0 in the game_time column.

# In[ ]:


train[train['type'] == 'Clip']['game_time'].value_counts()


# To fix that we can use this code:

# In[ ]:


train['timestamp'] = pd.to_datetime(train['timestamp'])
test['timestamp'] = pd.to_datetime(test['timestamp'])
counts = [len(train[train['type'] == 'Clip']), len(test[test['type'] == 'Clip'])]
dfs = [train, test]
sizes = [train.shape[0], test.shape[0]]
results = []
for df, count, size in zip(dfs, counts, sizes):
    res = df['game_time']
    for i, row in tqdm(df[df['type'] == 'Clip'].iterrows(), total=count):
        if i < size - 1:
            res[i] = int((df.iloc[i + 1, 2] - df.iloc[i, 2]).total_seconds() * 1000) #in millisecond
    results.append(res)
for df, res in zip(dfs, results):
    df['game_time'] = res


# In[ ]:


train[train['type'] == 'Clip']['game_time'].value_counts()


# In[ ]:


train


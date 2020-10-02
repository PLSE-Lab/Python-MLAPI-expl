#!/usr/bin/env python
# coding: utf-8

# Baseline model with mean of targets in train

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import json
import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
        
path, _, files = list(os.walk('/kaggle/input'))[1]

csv_files = sorted([f for f in files if f.endswith('.csv')])
sub, test, train = [pd.read_csv(os.path.join(path, f)) for f in csv_files]

for f in files:
    if not f.endswith('.json'):
        continue
    with open(os.path.join(path, f)) as f:
        sub_map = json.load(f)


# In[ ]:


sub_map


# In[ ]:


sub_map_inv = {v: k for k, v in sub_map.items()}

mean_targets = {sub_map_inv[k]: v for k, v in train[sub_map.values()].mean().to_dict().items()}
mean_targets


# In[ ]:


sub_target = []
for rid in test.RowId.values:
    sub_target.extend(list(mean_targets.values()))


# In[ ]:


sub['Target'] = sub_target


# In[ ]:


sub.to_csv('submission.csv', index=False)


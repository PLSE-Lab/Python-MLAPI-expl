#!/usr/bin/env python
# coding: utf-8

# Previous kernel - https://www.kaggle.com/priteshshrivastava/ieee-pipeline-0-reduce-memory-size
# 
# Input - Shoretened train & test files
# 
# Output - train & val sets, test set
# 
# Next kernel(s) - Models 1, 2, 3
# 
# Note : Should validation set be time based ?

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import os
import math
import pickle

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train_df = pd.read_pickle("/kaggle/input/ieee-pipeline-0-reduce-memory-size/train.pkl")
test_df = pd.read_pickle("/kaggle/input/ieee-pipeline-0-reduce-memory-size/test.pkl")
train_df.head()


# ### Splitting into training & validation sets

# In[ ]:


y_trn = train_df['isFraud']
df_trn = train_df.drop(['isFraud'], axis = 1)


# In[ ]:


train_X, val_X, train_y, val_y = train_test_split(df_trn, y_trn, test_size=0.7, random_state=42)


# ### Storing output files

# In[ ]:


train_X.to_pickle("train_X.pkl")
val_X.to_pickle("val_X.pkl")  

train_y.to_csv("train_y.csv", index = False, header = True)   ## pandas Series
val_y.to_csv("val_y.csv", index = False, header = True)       ## pandas Series

test_df.to_pickle("test_df.pkl")


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import pandas_profiling as pp


# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df_train = pd.read_csv('../input/google-quest-challenge/train.csv')
df_test = pd.read_csv('../input/google-quest-challenge/test.csv')
df_sub = pd.read_csv('../input/google-quest-challenge/sample_submission.csv')


# ## Training data
# target labels are the last 30 columns

# In[ ]:


pp.ProfileReport(df_train)


# ## Test set
# We must predict 30 labels for each test set row

# In[ ]:


pp.ProfileReport(df_test)


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# import data

# In[ ]:


train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')
test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')
ssub = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/sample_submission.csv')


# # Simple data exploration

# train data

# In[ ]:


train.head(2)


# In[ ]:


train.info()


# In[ ]:


train.describe()


# test data

# In[ ]:


test.head(2)


# In[ ]:


test.info()


# In[ ]:


test.describe()


# Combine test and train data

# In[ ]:


combined_df = train.append(test, ignore_index=True, sort=False)
combined_df.info()


# # Feature encoding

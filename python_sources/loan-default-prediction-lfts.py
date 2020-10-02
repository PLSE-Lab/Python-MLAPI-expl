#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing, model_selection
import lightgbm as lgb
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.display.max_columns = 100


# In[8]:


get_ipython().system('ls ../input/')


# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test_bqCt9Pv.csv")
print(train_df.shape, test_df.shape)


# In[ ]:


train_df.head()


# In[ ]:


train_df["loan_default"].value_counts()


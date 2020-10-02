#!/usr/bin/env python
# coding: utf-8

# # Save your Ram!
# 
# The point of the competition is to work with categorical data. However, pandas automatically reads any string column as a `dtype = object` column, which is notoriously inefficient. However, pandas' own `dtype`, `category`, can help us to save any column. 

# In[ ]:


import numpy as np
import pandas as pd 

train_raw = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')


# Let's confirm that pandas is reading the categorical columns as object dtypes. 

# In[ ]:


train_raw.info()


# There are 17 columns with `dtype` `object`. These are very memory inefficient. We might, however, cast them into categorical types if they have a reasonably low number of unique values. Let's check that:

# In[ ]:


old_memory_usage = train_raw.memory_usage(deep = True)


# In[ ]:


ordinality_of_cats = train_raw.describe(include = [np.object]).T.sort_values('unique')
ordinality_of_cats


# Let's turn all of the columns with less than 300 unique values into categoricals.

# In[ ]:


train_less_memory = train_raw.copy()
low_card_cols = ordinality_of_cats.query('unique < 300').index.tolist()
for col in low_card_cols:
    train_less_memory[col] = train_raw[col].astype('category')


# In[ ]:


train_less_memory.dtypes


# In[ ]:


train_less_memory.memory_usage(deep = True)/old_memory_usage


# The categorical columns now take less than 3% of their original memory! 

# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# This notebook load the SQLite-format dataset as a Pandas dataframe.

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sqlite3
import spacy


# ## List files
# 

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Load data

# In[ ]:


con = sqlite3.connect('/kaggle/input/tcu-acordaos.db')
df = pd.read_sql_query("SELECT * FROM acordaos", con)


# ## Basic statistics

# In[ ]:


df.columns


# In[ ]:


df.head()


# In[ ]:


df.describe()


# ## Text ('acordao' field) length stats

# In[ ]:


lens = df.acordao.map(lambda x : 0 if x is None else (10000 if len(x) >= 10000 else len(x)))
lens.describe()


# In[ ]:


sns.distplot(lens, kde=True)
#sns.kdeplot(lens)


# ## Tokens stats

# In[ ]:


get_ipython().system('python -m spacy download pt_core_news_sm')
tok = spacy.load("pt_core_news_sm").tokenizer


# In[ ]:


numToks = df.acordao.map(lambda x : 0 if x is None else len(tok(x)))
numToks.describe()


# In[ ]:


sns.distplot(numToks.map(lambda x : 1500 if x >= 1500 else x), kde=True)
# import matplotlib.pyplot as plt
# ax = sns.distplot(numToks, kde=True)
# plt.xlim(0, 3000)
# ax


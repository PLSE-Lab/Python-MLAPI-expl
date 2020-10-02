#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import multiprocessing

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from nltk.tokenize import word_tokenize

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/train.csv')
print(df.shape)
df.head()


# In[ ]:


documents = df['question_text'].values


# Here we use only a single core.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'docs_tokenized = [word_tokenize(doc) for doc in documents]')


# Using multiprocessing, we run nltk's tokenizer on all 4 cores.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'with multiprocessing.Pool(4) as pool:\n    docs_tokenized = pool.map(word_tokenize, documents)')


# The time taken is significantly lower!

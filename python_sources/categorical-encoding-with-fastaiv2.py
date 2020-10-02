#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system(' pip install fastai2')


# In[ ]:


print(pd.__version__)


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc

from sklearn.model_selection import StratifiedKFold
from sklearn import metrics, preprocessing
from fastai2.tabular.all import *


# In[ ]:


train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')
test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')
sub = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/sample_submission.csv')


# In[ ]:


data = pd.concat([train, test]).reset_index(drop=True)

data.head()


# In[ ]:


data.shape


# In[ ]:


data.dtypes


# 
# TabularPandas
# 
# fastai2 has a new way of dealing with tabular data in a TabularPandas object. It expects some dataframe, some procs, cat_names, cont_names, y_names, block_y, and some splits. We'll walk through all of them
# 
# First we need to grab our categorical and continuous variables, along with how we want to process our data.
# 

# In[ ]:


cat_names = ['bin_3', 'bin_4', 'nom_0', 'nom_1', 'nom_2', 'nom_3','nom_4','nom_5','nom_6','nom_7','nom_8','nom_9', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']


# In[ ]:


cat = Categorify()

to = TabularPandas(data, cat, cat_names)


# In[ ]:


cats = to.procs.categorify


# In[ ]:


cats['ord_1']


# In[ ]:


to.show(max_n=10)


# In[ ]:


to.cats.head()


# In[ ]:


#names

cont_names = ['ord_0', 'bin_0', 'bin_1', 'day', 'month', ]


# In[ ]:


norm = NormalizeTab()


# In[ ]:


to = TabularPandas(data, norm, cont_names=cont_names)


# In[ ]:


norm = NormalizeTab()


# In[ ]:


norms = to.procs.normalize_tab


# In[ ]:


norms.means


# In[ ]:


norms.stds


# In[ ]:


to.conts.head()


# In[ ]:


# Fill missing

fm = FillMissing(fill_strategy=FillStrategy.median)


# In[ ]:


to = TabularPandas(data, fm, cont_names=cont_names)


# In[ ]:


to.conts.head()


# In[ ]:


to.cats.head()


# In[ ]:


splits = RandomSplitter()(range_of(data))


# In[ ]:


splits


# In[ ]:


procs = [Categorify, FillMissing, Normalize]
y_names = 'target'
block_y = CategoryBlock()


# In[ ]:


to = TabularPandas(data, procs=procs, cat_names=cat_names, cont_names=cont_names,
                   y_names=y_names, block_y=block_y, splits=splits)


# ## Dataloaders

# In[ ]:


trn_dl = TabDataLoader(to.train, bs=64, shuffle=True, drop_last=True)
val_dl = TabDataLoader(to.valid, bs=128)


# In[ ]:


dls = DataLoaders(trn_dl, val_dl)


# In[ ]:


dls.show_batch()


# In[ ]:


to._dbunch_type


# In[ ]:


dls._dbunch_type


# ## TabularLearner
# 

# In[ ]:


# Categorical variable

def get_emb_sz(to, sz, dict=None):
    return [_one_emb_sz(to.classes, n, sz_dict) for n in to.cat_names]


# In[ ]:


def _one_emb_sz(classes, n, sz_dict=None):
    "Pick an embedding size for `n` depending on `classes` if not given in `sz_dict`."
    sz_dict = ifnone(sz_dict, {})
    n_cat = len(classes[n])
    sz = sz_dict.get(n, int(emb_sz_rule(n_cat)))  # rule of thumb
    return n_cat,sz



def emb_sz_rule(n_cat):
    "Ruleof thumb to pick embedding size corresponding to `n_cat`"
    return min(600, round(1.6 * n_cat**0.56))


# In[ ]:


emb_szs = emb_sz_rule(to)


# In[ ]:


to


# In[ ]:


to.cat_names


# In[ ]:


cont_len = len(to.cont_names)


# In[ ]:


cont_len


# In[ ]:





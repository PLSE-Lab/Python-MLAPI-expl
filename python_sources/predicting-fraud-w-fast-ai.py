#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('pip install git+https://github.com/fastai/fastai@2e1ccb58121dc648751e2109fc0fbf6925aa8887')

get_ipython().system('apt update && apt install -y libsm6 libxext6')


# In[ ]:


from fastai.imports import *
from fastai.structured import *
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics


# In[ ]:


PATH = '../input'
get_ipython().system('ls {PATH}')


# In[ ]:


df_raw = pd.read_csv(f'{PATH}/creditcard.csv', low_memory=False)


# In[ ]:


df_raw.sample(10)


# In[ ]:


df_raw.describe()


# In[ ]:


def display_all(df):
    with pd.option_context('display.max_rows',1000):
        with pd.option_context('display.max_columns',1000):
            display(df)
display_all(df_raw.tail().transpose())


# In[ ]:


train_cats(df_raw)


# In[ ]:


df, y, nas = proc_df(df_raw, "Class", max_n_cat=7)


# In[ ]:


display_all(df)


# In[ ]:


def split_vals(a,n): return a[:n].copy(), a[n:].copy()

n_valid=20000
n_trn=len(df)-n_valid

X_train, X_valid = split_vals(df,n_trn)
y_train, y_valid = split_vals(y, n_trn)


X_train.shape, X_valid.shape, y_train.shape


# In[ ]:


m = RandomForestClassifier(n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
m.score(X_train, y_train), m.score(X_valid, y_valid)


# In[ ]:


def plot_fi(fi):
    fi.plot('cols', 'imp', 'barh', figsize=(10,16))

fi = rf_feat_importance(m, df)


# In[ ]:


plot_fi(fi[fi.imp > 0.008])


# In[ ]:


to_keep = fi[fi.imp >0.01].cols
df_keep = df[to_keep]


# In[ ]:


X_train, X_valid = split_vals(df_keep,n_trn)


# In[ ]:


m1 = RandomForestClassifier(n_jobs=-1)
get_ipython().run_line_magic('time', 'm1.fit(X_train, y_train)')
m1.score(X_train, y_train), m1.score(X_valid, y_valid)


# In[ ]:


from scipy.cluster import hierarchy as hc


# In[ ]:


corr=np.round(scipy.stats.spearmanr(df_keep).correlation,4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(10,16))
dendrogram = hc.dendrogram(z, labels=df_keep.columns, orientation='left', leaf_font_size=16)


# In[ ]:





# In[ ]:





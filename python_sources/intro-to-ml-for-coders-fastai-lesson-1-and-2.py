#!/usr/bin/env python
# coding: utf-8

# For working on Fast.ai [**Introduction to Machine Learning for Coders**](http://course18.fast.ai/ml) Lesson 1 and 2, based on the instructions given [here](https://forums.fast.ai/t/wiki-thread-lesson-1/6825) we need to install older version of fastai-0.7.x. Refer actual notebook by [fastai](https://github.com/fastai/fastai/blob/master/courses/ml1/lesson1-rf.ipynb). Kaggle comes with the latest version of Fastai i.e. v.1.x.
# 
# If you are stuck anywhere writing code directly from this course into the Kaggle Kernel or working on this [Lesson 1 - Kaggle kernel](https://www.kaggle.com/miwojc/fast-ai-machine-learning-lesson-1/) You have missed out the below pip installations.
# 
# Note (for beginners like me in Kaggle): Though it may look silly but please make sure you have **turned On** Internet on the right side of the kaggle kernel.
# 
# Please watch this [fastai video by Jeremy Howard](http://course18.fast.ai/lessonsml1/lesson1.html) and refer to the the extra ordinary [lesson notes](https://medium.com/@hiromi_suenaga/machine-learning-1-lesson-1-84a1dc2b5236) to get a clear understanding of how the below code works : 

# In[ ]:


get_ipython().system('pip install fastai==0.7.0')
get_ipython().system('pip install torchtext==0.2.3')


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


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn import metrics


# In[ ]:


PATH = "../input/"


# In[ ]:


get_ipython().system('ls {PATH}')


# In[ ]:


df_raw = pd.read_csv(f'{PATH}train/Train.csv', low_memory=False, parse_dates=["saledate"])


# In[ ]:


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)


# In[ ]:


display_all(df_raw.tail().T)


# In[ ]:


display_all(df_raw.describe(include='all').T)


# In[ ]:


df_raw.SalePrice = np.log(df_raw.SalePrice)


# In[ ]:


add_datepart(df_raw, 'saledate')
df_raw.saleYear.head()


# In[ ]:


train_cats(df_raw)


# In[ ]:


df_raw.UsageBand.cat.categories


# In[ ]:


df_raw.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)


# In[ ]:


df_raw.UsageBand = df_raw.UsageBand.cat.codes


# In[ ]:


display_all(df_raw.isnull().sum().sort_index()/len(df_raw))


# In[ ]:


os.makedirs('tmp', exist_ok=True)
df_raw.to_feather('tmp/bulldozers-raw')


# In[ ]:


df_raw = pd.read_feather('tmp/bulldozers-raw')


# In[ ]:


df, y, nas = proc_df(df_raw, 'SalePrice')


# In[ ]:


m = RandomForestRegressor(n_jobs=-1)
m.fit(df, y)
m.score(df,y)


# In[ ]:


def split_vals(a,n): return a[:n].copy(), a[n:].copy()
n_valid=12000
n_trn = len(df) - n_valid
raw_train, raw_valid = split_vals(df_raw, n_trn)
X_train,X_valid = split_vals(df, n_trn)
y_train,y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape


# In[ ]:


def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# In[ ]:


m = RandomForestRegressor(n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# In[ ]:


df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice', subset=30000, na_dict=nas)
X_train, _ = split_vals(df_trn, 20000)
y_train, _ = split_vals(y_trn, 20000)


# In[ ]:


m = RandomForestRegressor(n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# In[ ]:


preds = np.stack([t.predict(X_valid) for t in m.estimators_])
preds[:,0], np.mean(preds[:,0]), y_valid[0]


# In[ ]:


preds.shape


# In[ ]:


plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(10)]);


# In[ ]:


df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice')
X_train, X_valid = split_vals(df_trn, n_trn)
y_train, y_valid = split_vals(y_trn, n_trn)


# In[ ]:


set_rf_samples(20000)


# In[ ]:


m = RandomForestRegressor(n_jobs=-1, oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# In[ ]:


m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# **Tree Building Parameters**

# In[ ]:


reset_rf_samples()


# In[ ]:


def dectree_max_depth(tree):
    children_left = tree.children_left
    children_right = tree.children_right

    def walk(node_id):
        if (children_left[node_id] != children_right[node_id]):
            left_max = 1 + walk(children_left[node_id])
            right_max = 1 + walk(children_right[node_id])
            return max(left_max, right_max)
        else: # leaf
            return 1
    root_node_id = 0
    return walk(root_node_id)


# In[ ]:


m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


t=m.estimators_[0].tree_


# In[ ]:


dectree_max_depth(t)


# In[ ]:


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


t=m.estimators_[0].tree_


# In[ ]:


dectree_max_depth(t)


# In[ ]:


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# We can no longer submit to this competition - but we can at least see that we're getting similar results to the winners based on the dataset we have.

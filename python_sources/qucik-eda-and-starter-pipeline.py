#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import category_encoders as ce
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier


# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')
test = pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv')


# Let's take a look at the data.

# In[ ]:


train.head()


# No missing values.

# In[ ]:


train.isna().sum().sum()


# There are mainly two groups of variables:
# 
# * low cardinality variables: bin_{0-4}, nom_{0-4}, ord_{0-4}, day, month
# * high cardinality variables: nom_{5-9}, ord_{5}

# In[ ]:


train.nunique()


# nom_9 has a lot of categories with only one occurence.

# In[ ]:


for c in train.columns[1:]:
    print("="*5)
    print(c)
    print(train[c].value_counts().sort_values())


# In[ ]:


lc_feats = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'day', 'month', 'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']
hc_feats = ['ord_5', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']


# Some visualizations. Only the low cardinality features are displayed here.

# In[ ]:


for col in lc_feats:
    
    x = list(train.groupby(col)['target'].mean().sort_index().index)
    y1 = train.groupby(col)['target'].mean().sort_index().values
    y2 = train[col].value_counts().sort_index().values
    
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel(col)
    ax1.set_ylabel('percentage of positive in the category', color=color)
    ax1.plot(x, y1, marker="o", color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.bar(x,y2, color=color, alpha=0.2)
    ax2.set_ylabel('count', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()


# * One hot encoding of the low cardinality feats.
# * Target encoding of the high cardinality feats.
# * Logistic regression. Just the default parameters here.
# 
# It really matters imo to put the encoding process in the **pipeline**, especially the target encoding which will used the target datas in our cross validation.

# In[ ]:


pipeline = make_pipeline(
    ce.TargetEncoder(cols=hc_feats),               
    ce.OneHotEncoder(cols=lc_feats),
    LogisticRegression(solver="saga", penalty="l2", max_iter = 5000)
)


# In[ ]:


kfold = StratifiedKFold(n_splits=5, shuffle= True, random_state=42)


# In[ ]:


X_train = train.drop(["target", "id"], axis=1)
y_train = train["target"]


# Computation of the cross validation score.

# In[ ]:


get_ipython().run_cell_magic('time', '', "scores = cross_val_score(pipeline, X_train, y_train, cv=kfold, scoring='roc_auc', n_jobs=-1, verbose=1)\nprint(scores)")


# In[ ]:


scores.mean()


# Fine tuning the encoding and the estimators may improve the score above 0.800

# In[ ]:





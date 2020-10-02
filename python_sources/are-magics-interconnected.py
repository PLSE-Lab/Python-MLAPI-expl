#!/usr/bin/env python
# coding: utf-8

# # <h2>Are magics interconnected? <h2/>
# <br><br>
# Thanks Chris for his kernel https://www.kaggle.com/cdeotte/logistic-regression-0-800. 
# 
# I wondered do 'wheezy-copper-turtle-magic' categories interconnected? <br>
# And I decided to check, what would be if train on one category and evaluate AUC on another.

# In[1]:


import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm_notebook

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.head()


# In[2]:


cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]


# In[ ]:


for i in tqdm_notebook(range(512)):
    train2 = train[train['wheezy-copper-turtle-magic']==i]
    train2.reset_index(drop=True,inplace=True)
    
    clf = LogisticRegression(solver='liblinear',penalty='l1',C=0.05)
    clf.fit(train2[cols], train2['target'])
    
    for j in range(i, 512):
        val = train[train['wheezy-copper-turtle-magic']==j]
        preds = clf.predict_proba(val[cols])[:,1]
        auc = roc_auc_score(val['target'], preds)
        if auc > 0.65 or j==i:
            print(f'magic-{i}-{j}  auc={auc:.4}' )
    print('-'*40)


# ### As you can see, some predictions give us ~0.7 AUC.
# ### Does it mean something or it's just noise?

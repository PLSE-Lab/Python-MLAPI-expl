#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.drop(["unique_id"],axis=1,inplace=True)
test.drop(["unique_id"],axis=1,inplace=True)


# In[ ]:


y = train.targets
train.drop('targets', axis=1, inplace=True)


# In[ ]:


clf = LogisticRegression(class_weight='balanced')
clf.fit(train, y)


# In[ ]:


preds = clf.predict_proba(test)
sub = pd.read_csv('../input/sample_submission.csv')

for i in range(1, 10):
    sub[f'proba_{i}'] = preds[:, i-1]
    
sub.to_csv('submission.csv', index=False)


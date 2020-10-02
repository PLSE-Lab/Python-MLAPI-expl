#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import xgboost as xgb


# In[ ]:


train = pd.read_csv('../input/train/train.csv')
test = pd.read_csv('../input/test/test.csv')


# In[ ]:


y = train.AdoptionSpeed.values


# In[ ]:


train = train.drop(['Name', 'RescuerID', 'Description', 'PetID', 'AdoptionSpeed'], axis=1).values


# In[ ]:


test = test.drop(['Name', 'RescuerID', 'Description', 'PetID'], axis=1).values


# In[ ]:


clf = xgb.XGBClassifier(n_estimators=500, nthread=-1, max_depth=8, learning_rate=0.015)


# In[ ]:


clf.fit(train, y)


# In[ ]:


preds = clf.predict(test)


# In[ ]:


sample = pd.read_csv('../input/test/sample_submission.csv')


# In[ ]:


sample.AdoptionSpeed = preds


# In[ ]:


sample.to_csv('submission.csv', index=False)


# In[ ]:





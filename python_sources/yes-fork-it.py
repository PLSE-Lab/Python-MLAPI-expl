#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle


# In[15]:


with open('../input/predictions/prediction.pkl', 'rb') as f:
    preds = pickle.load(f)


# In[16]:


sub = pd.read_csv('../input/instant-gratification/sample_submission.csv')
print(preds)
preds = pd.DataFrame(preds)
preds = preds[:131074]
# if sub.shape[0] != 131073:
sub['target'] = preds
sub.to_csv('submission.csv',index=False)


# In[17]:


sub.head()


# In[4]:


sub['target'].head()


# In[10]:


preds= list(preds)


# In[11]:


preds


# In[ ]:


|


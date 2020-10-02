#!/usr/bin/env python
# coding: utf-8

# This is an example to show how to avoid your kernel running twice [as discussed here](https://www.kaggle.com/c/instant-gratification/discussion/94908). Read that discussion first. If you haven't read that already this kernel wouldn't make sense to you.

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[7]:


# write fake submission file 
sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = 0
sub.to_csv('submission.csv',index=False)


# In[3]:


test = pd.read_csv('../input/test.csv')


# In[6]:


if len(test) < 150000:
    [].shape
    
# your fancy code goes here
sleep(100000000000)


# make sure to rewrite your submission file here


# In[ ]:




